import re
import os
import sys
import warnings
from typing import Literal, Callable, Any, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from flow_planner.model.model_base import DiffusionADPlanner
from flow_planner.model.model_utils.input_preprocess import ModelInputProcessor
from flow_planner.model.model_utils.traj_tool import traj_chunking, assemble_actions
from flow_planner.data.dataset.nuplan import NuPlanDataSample

class FlowPlanner(DiffusionADPlanner):

    def __init__(
        self,
        model_encoder,
        model_decoder,

        flow_ode,
        
        model_type: Literal['x_start', 'noise', 'velocity'] = 'x_start',
        kinematic: Literal["waypoints", "velocity", "acceleration", "frenet", "va"] = 'waypoints',
    
        assemble_method='linear',
        
        data_processor: ModelInputProcessor = None,
        
        device='cuda',
        **planner_params
    ):
        
        super(FlowPlanner, self).__init__()
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self._model_type = model_type
        self.device = device
        
        self.flow_ode = flow_ode # including flow matching path and ode solver
        self.cfg_prob = planner_params['cfg_prob']
        self.cfg_weight = planner_params['cfg_weight']
        self.cfg_type = planner_params['cfg_type']

        self.kinematic = kinematic

        # CFG × Frenet partial-gate guard: when cfg_type='lanes', the lanes
        # tensor is multiplied by cfg_flags (zeroing the unconditioned half)
        # BEFORE the Frenet centerline build runs inside sample_to_model_input.
        # The centerline-selector then sees zero geometry and falls back to a
        # degenerate ray, producing nonsense (s, d) targets the model trains
        # against. Reject this combination explicitly; users with this need
        # should switch to cfg_type='neighbors' (the default) or build the
        # centerline pre-mask.
        if self.kinematic == 'frenet' and self.cfg_type == 'lanes':
            raise ValueError(
                "cfg_type='lanes' is incompatible with kinematic='frenet': "
                "the lanes mask zeroes the geometry the Frenet centerline "
                "build depends on, producing nonsense (s, d) targets. Use "
                "cfg_type='neighbors' for Frenet runs."
            )

        # Option A guard: warn loudly if Frenet is configured without a
        # centerline_encoder in the model_encoder. Without the encoder, the
        # reference_centerline produced by ModelInputProcessor is silently
        # dropped (encoder.py:126-127), and training proceeds as if Option A
        # were disabled — but the user expected it. The previous failure
        # mode was a silent state_dict mismatch at inference time (commit
        # b950ffc). This catches the same misconfiguration at construction.
        if self.kinematic == 'frenet' and getattr(
            self.model_encoder, 'centerline_encoder', None
        ) is None:
            warnings.warn(
                "kinematic='frenet' but model_encoder.centerline_encoder is None. "
                "The reference_centerline will be silently dropped by the encoder, "
                "so Option A (explicit centerline conditioning) is NOT active. "
                "If this is intentional (e.g., v4 ablation without Option A), "
                "you can ignore this. Otherwise, add the Hydra overrides:\n"
                "  +model.model_encoder.centerline_encoder._target_="
                "flow_planner.model.modules.encoder_modules.CenterlineEncoder\n"
                "  +model.model_encoder.centerline_encoder.n_points=100\n"
                "  +model.model_encoder.centerline_encoder.hidden_dim=256",
                RuntimeWarning,
                stacklevel=2,
            )

        # Symmetric guard: warn if a centerline_encoder is wired in for a
        # non-Frenet kinematic. The encoder will never receive
        # ``reference_centerline`` (only the Frenet branch of
        # ModelInputProcessor produces it), so its parameters get zero
        # gradient for the entire run, remain at random init, and bloat the
        # checkpoint. Loading that checkpoint later with kinematic='frenet'
        # silently contaminates inference with un-trained centerline weights.
        if self.kinematic != 'frenet' and getattr(
            self.model_encoder, 'centerline_encoder', None
        ) is not None:
            warnings.warn(
                f"kinematic='{self.kinematic}' but model_encoder.centerline_encoder "
                "is configured. The encoder will never receive reference_centerline, "
                "so its parameters will not train (zero gradient flow). Remove the "
                "centerline_encoder override or switch to kinematic='frenet'.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.assemble_method = assemble_method

        self.data_processor = data_processor

        self.planner_params = planner_params # including the action_len, future_len etc.
        self.action_num = (self.planner_params['future_len'] - self.planner_params['action_overlap']) // (self.planner_params['action_len'] - self.planner_params['action_overlap'])

        # Expected target dimensionality per kinematic representation. Catches
        # state_dim/kinematic mismatches at construction time instead of via a
        # cryptic shape error inside the loss block. 4-channel kinematics:
        #   waypoints  -> (x, y, cos h, sin h)
        #   frenet     -> (s, d, cos h, sin h)
        #   va         -> (v_x, v_y, a_x, a_y)        # combined V+A model
        # 3-channel kinematics:
        #   velocity     -> (v_x, v_y, ??)    legacy separate model
        #   acceleration -> (a_x, a_y, ??)    legacy separate model
        expected_state_dims = {
            'waypoints': 4,
            'velocity': 3,
            'acceleration': 3,
            'frenet': 4,
            'va': 4,
        }
        if kinematic not in expected_state_dims:
            raise ValueError(f"Unsupported kinematic representation: {kinematic}")
        expected_state_dim = expected_state_dims[kinematic]
        if self.planner_params['state_dim'] != expected_state_dim:
            raise ValueError(
                f"kinematic='{kinematic}' requires model.state_dim={expected_state_dim}, "
                f"but got {self.planner_params['state_dim']}"
            )

        # V+A combined model: dedicated weighted-MSE loss on (v, a) channels
        # plus an Euler-integrated position-consistency penalty. See VALoss /
        # VAIntegrator for the implementation. Loss weights are the same as
        # the team's 5k headline run on DagsHub (w_v=1.0, w_a=0.5, w_p=0.2).
        if kinematic == 'va':
            from flow_planner.model.modules.decoder_modules import VAIntegrator, VALoss
            self.va_integrator = VAIntegrator(dt=0.1)
            self.va_loss = VALoss(w_v=1.0, w_a=0.5, w_p=0.2, dt=0.1)

        self.basic_loss = nn.MSELoss(reduction='none')
        
    def prepare_model_input(self, cfg_flags, data: NuPlanDataSample, use_cfg, is_training):
        B = data.ego_current.shape[0]

        if is_training:
            # modify the data sample according to cfg_flags
            cfg_type = self.cfg_type
            if cfg_type == 'neighbors':
                neighbor_num = self.planner_params['neighbor_num']
                cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                mask_flags = cfg_flags.view(B, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1)
                mask_flags[:, cfg_neighbor_num:, :] = 1
                data.neighbor_past *= mask_flags
            elif cfg_type == 'lanes':
                data.lanes = data.lanes * cfg_flags.view(B, *([1] * (data.lanes.dim()-1)))

        else:
            if use_cfg:
                data = data.repeat(2)
                cfg_type = self.cfg_type
                if cfg_type == 'neighbors':
                    neighbor_num = self.planner_params['neighbor_num']
                    cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                    mask_flags = cfg_flags.view(B * 2, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1)
                    mask_flags[:, cfg_neighbor_num:, :] = 1
                    data.neighbor_past *= mask_flags
                elif cfg_type == 'lanes':
                    data.lanes = data.lanes * cfg_flags.view(B * 2, *([1] * (data.lanes.dim()-1)))
           
        model_inputs, gt = self.data_processor.sample_to_model_input(
            data, device=self.device, kinematic=self.kinematic, is_training=is_training
        )
            
        model_inputs.update({'cfg_flags': cfg_flags})
        
        return model_inputs, gt
        
    def extract_encoder_inputs(self, inputs):

        encoder_inputs = {
            'neighbors': inputs['neighbor_past'],
            'lanes': inputs['lanes'],
            'lanes_speed_limit': inputs['lanes_speedlimit'],
            'lanes_has_speed_limit': inputs['lanes_has_speedlimit'],
            'static': inputs['map_objects'],
            'routes': inputs['routes']
        }
        # Option A (Frenet only): pass the chosen reference centerline through so
        # the encoder can condition on it explicitly. Other kinematics don't add
        # this key and the encoder simply ignores the missing kwarg.
        if 'reference_centerline' in inputs:
            encoder_inputs['reference_centerline'] = inputs['reference_centerline']
        return encoder_inputs
    
    def extract_decoder_inputs(self, encoder_outputs, inputs):
        model_extra = dict(cfg_flags=inputs['cfg_flags'] if 'cfg_flags' in inputs.keys() else None,)
        model_extra.update(encoder_outputs)
        return model_extra
    
    def encoder(self, **encoder_inputs):
        return self.model_encoder(**encoder_inputs)
    
    def decoder(self, x, t, **model_extra):
        return self.model_decoder(x, t, **model_extra)
        
    def forward(self, data: NuPlanDataSample, mode='train', **params):
        if mode == 'train':
            return self.forward_train(data)
        elif mode == 'inference':
            return self.forward_inference(data, params['use_cfg'], params['cfg_weight'])
    
    def forward_train(self, data: NuPlanDataSample):
        '''
        Forward a training step and compute the training loss.
        1. generate cfg_flags
        2. preprocess (masking) according to the cfg_flags
        3. model forward
        4. compute basic mse loss
        
        Return:
            prediction: the raw prediction of the model, specified by model.prediction_type;
            loss_dict: a dict of loss containing unreduced mse loss, consistency loss and neighbor prediction loss (if one exists).
        '''
        B = data.ego_current.shape[0]
        roll_dice = torch.rand((B, 1))
        cfg_flags = (roll_dice > self.cfg_prob).to(torch.int32).to(self.device) # NOTE: 1 for conditioned (unmasked), 0 for unconditioned (masked)
        model_inputs, gt = self.prepare_model_input(cfg_flags, data, use_cfg=False, is_training=True) # note that the cfg_flags are packed into the model_inputs
        
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)

        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        B, P, T_, D = gt.shape
        
        noised_traj, target, t = self.flow_ode.sample(gt[:, :, 1:, :], self._model_type)
        noised_traj_tokens = traj_chunking(noised_traj, self.planner_params['action_len'], self.planner_params['action_overlap'])
        noised_traj_tokens = torch.cat(noised_traj_tokens, dim=1)
        target_tokens = traj_chunking(target, self.planner_params['action_len'], self.planner_params['action_overlap'])
        target_tokens = torch.cat(target_tokens, dim=1)
        
        prediction = self.decoder(noised_traj_tokens, t, **decoder_model_extra)
        
        loss_dict = {}
        batch_loss = self.basic_loss(prediction, target_tokens)
        loss_dict['batch_loss'] = batch_loss

        # Per-kinematic loss aggregation:
        #   * 'va' (combined V+A): weighted MSE on (v_x, v_y) and (a_x, a_y)
        #     channels plus a kinematic-consistency penalty on the integrated
        #     xy positions (see VALoss). VALoss assumes raw m/s and m/s^2, so
        #     the sample_to_model_input 'va' branch must NOT apply
        #     state_normalizer to the targets.
        #   * everything else: standard per-element MSE then mean.
        if self.kinematic == 'va' and self._model_type == 'x_start':
            gt_future_xy = data.ego_future[:, None, -self.planner_params['future_len']:, :2].to(self.device)
            pred_va = assemble_actions(
                prediction,
                self.planner_params['future_len'],
                self.planner_params['action_len'],
                self.planner_params['action_overlap'],
                self.planner_params['state_dim'],
                self.assemble_method,
            )[:, 0]
            target_va = assemble_actions(
                target_tokens,
                self.planner_params['future_len'],
                self.planner_params['action_len'],
                self.planner_params['action_overlap'],
                self.planner_params['state_dim'],
                self.assemble_method,
            )[:, 0]
            va_loss = self.va_loss(pred_va, target_va, xy_gt=gt_future_xy[:, 0])
            loss_dict['ego_planning_loss'] = va_loss['loss']
            loss_dict['loss_v'] = va_loss['loss_v']
            loss_dict['loss_a'] = va_loss['loss_a']
            loss_dict['loss_p'] = va_loss['loss_p']
        else:
            loss = torch.sum(batch_loss, dim=-1) # (B, action_num, action_length, dim)
            loss_dict['ego_planning_loss'] = loss.mean()

        if self.planner_params['action_overlap'] > 0 and prediction.shape[1] >= 2:
            # Audit Finding 5 fix: previously `range(0, prediction.shape[1]-2)`
            # skipped the final overlap pair (chunks 5↔6 when action_num=7).
            # The last valid pair index is action_num-2, so the exclusive
            # upper bound of range() should be action_num-1.
            # Extra guard: when action_num == 1 the list-comp is empty and
            # sum([]) / len([]) raises ZeroDivisionError. Skip cleanly.
            consistency_loss = [torch.mean(torch.sum(self.basic_loss(prediction[:, i:i+1, -self.planner_params['action_overlap']:, :], prediction[:, i+1:i+2, :self.planner_params['action_overlap'], :]), dim=-1)) for i in range(0, prediction.shape[1]-1)]
            loss_dict['consistency_loss'] = sum(consistency_loss) / len(consistency_loss)
        else:
            loss_dict['consistency_loss'] = torch.tensor(0.0, device=loss_dict['ego_planning_loss'].device)

        assert not torch.isnan(loss_dict['ego_planning_loss']).sum(), f"loss is NaN"
        
        return prediction, loss_dict
    
    def forward_inference(self, data: NuPlanDataSample, use_cfg=True, cfg_weight=None):
        B = data.ego_current.shape[0]
        if use_cfg:
            cfg_flags = torch.cat([torch.ones((B,), device=self.device), torch.zeros((B,), device=self.device)], dim=0).to(torch.int32)
        else:
            cfg_flags = torch.ones((B,), device=self.device).to(torch.int32)
        
        model_inputs, _ = self.prepare_model_input(cfg_flags, data, use_cfg, is_training=False)
        
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)
        
        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        
        x_init = torch.randn((B, self.action_num, self.planner_params['action_len'], self.planner_params['state_dim']), device=self.device)
        sample = self.flow_ode.generate(x_init, self.decoder, self._model_type, use_cfg=use_cfg, cfg_weight=cfg_weight, **decoder_model_extra)
        
        sample = assemble_actions(sample, self.planner_params['future_len'], self.planner_params['action_len'], self.planner_params['action_overlap'], self.planner_params['state_dim'], self.assemble_method)

        # V+A inference: integrate the predicted (v, a) channels to recover
        # (x, y, heading); heading is derived from the predicted velocity
        # direction (atan2(v_y, v_x)) with a current-heading fallback at low
        # speed. All other kinematics use the existing inverse-normalization
        # path; downstream code (e.g. inference_eval.py for Frenet) handles
        # any further frame conversion.
        if self.kinematic == 'va':
            sample = self.data_processor.va_to_waypoints(sample, data.ego_current)
        else:
            sample = self.data_processor.state_postprocess(sample)

        return sample
    
    @property
    def model_type(self,):
        return self._model_type
    
    def get_optimizer_params(self):
        return [
            {'params': self.model_encoder.parameters()},
            {'params': self.model_decoder.parameters()}
        ]