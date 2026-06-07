import torch
import math
from flow_planner.data.dataset.nuplan import NuPlanDataSample

class ModelInputProcessor:
    def __init__(
        self,
        future_len,
        obs_normalizer,
        state_normalizer,
        neighbor_pred_num
    ):
        self.future_len = future_len
        self.obs_normalizer = obs_normalizer
        self.state_normalizer = state_normalizer
        self.neighbor_pred_num = neighbor_pred_num

    def state_preprocess(self, x):
        return self.state_normalizer(x) if self.state_normalizer is not None else x
    
    def state_postprocess(self, x):
        return self.state_normalizer.inverse(x) if self.state_normalizer is not None else x

    def x_differentiate(self, x_future, x_current):
        x_all = torch.cat([x_current, x_future], dim=-2)
        return x_all[..., 1:, :] - x_all[..., :-1, :]

    def x_integral(self, dx_future, x_current):
        v_all = torch.cat([x_current, dx_future], dim=-2)
        return torch.cumsum(v_all, dim=-2)[..., 1:, :]

    def sample_to_model_input(
        self,
        data: NuPlanDataSample,
        device,
        kinematic,
        is_training: bool=False
    ):
        # Save raw references BEFORE obs_normalizer for the Frenet branch.
        # obs_normalizer scales xy in routes/lanes/ego_current by std=20 (after
        # mean shift), but ego_future is NOT in any norm-stats YAML and stays
        # in raw world coords. Frenet projection mixing those scales produces
        # garbage (s, d). The fix: keep raw refs and use them in the Frenet
        # branch. obs_normalizer deepcopies internally, so the original `data`
        # is not mutated by the call below.
        # See docs/research/cs269_modifications_review.md CRITICAL #1.
        raw_data = data if kinematic == 'frenet' else None

        if self.obs_normalizer is not None:
            data = self.obs_normalizer(data)

        ego_future = data.ego_future
        if ego_future.numel() != 0:
            ego_future = ego_future[..., -self.future_len:, :3] # (x, y, heading)

        model_inputs = {}
        model_inputs['ego_past'] = data.ego_past.to(device)
        model_inputs['neighbor_past'] = data.neighbor_past.to(device)
        model_inputs['lanes'] = data.lanes.to(device)
        model_inputs['lanes_speedlimit'] = data.lanes_speedlimit.to(device)
        model_inputs['lanes_has_speedlimit'] = data.lanes_has_speedlimit.to(device)
        model_inputs['routes'] = data.routes.to(device)
        model_inputs['map_objects'] = data.map_objects.to(device)


        ego_current_state = data.ego_current
        model_inputs['ego_current'] = ego_current_state
        ego_current_xy_cos_sin = ego_current_state[..., :4]
        ego_current = torch.cat([
            ego_current_xy_cos_sin[..., :2],
            torch.atan2(ego_current_xy_cos_sin[..., 3:4], ego_current_xy_cos_sin[..., 2:3])
        ], dim=-1)

        current_states = ego_current[:, None]

        if is_training:
            gt_future = ego_future[:, None, :, :]

            gt_with_current = torch.cat([
                    current_states[:, :, None, :],
                    gt_future
                ], dim=2)

            gt_with_current.to(device)
        else:
            gt_with_current = current_states[:, :, None, :].repeat(1, 1, self.future_len + 1, 1)

        if kinematic == 'waypoints':
            gt_with_current = torch.cat([
                gt_with_current[..., :2],
                torch.cat([
                    gt_with_current[..., 2:3].cos(),
                    gt_with_current[..., 2:3].sin()
                ], dim=-1)
            ], dim=-1)
            gt_with_current[..., 1:, :] = self.state_normalizer(gt_with_current[..., 1:, :])
        elif kinematic == 'velocity':
            # Transform [x, y, heading] -> [x, y, cos_h, sin_h] BEFORE differentiation so
            # the velocity targets have 4 channels matching state_normalizer's expected
            # shape (waypoints_norm_stats is sized for [x, y, cos_h, sin_h]).
            gt_with_current = torch.cat([
                gt_with_current[..., :2],
                torch.cat([
                    gt_with_current[..., 2:3].cos(),
                    gt_with_current[..., 2:3].sin()
                ], dim=-1)
            ], dim=-1)
            future_velocity = self.x_differentiate(gt_with_current[..., 1:, :], gt_with_current[..., :1, :])
            gt_with_current = torch.cat([gt_with_current[..., :1, :], future_velocity], dim=-2)
            # Symmetry fix: apply state_normalizer to the future part the same
            # way the waypoints and frenet branches do. Without this, training
            # sees raw-scale velocity targets while state_postprocess at
            # inference applies state_normalizer.inverse (multiply by std, add
            # mean), silently scaling every prediction by ~std and shifting by
            # ~mean — completely breaking ADE/FDE for the velocity baseline.
            gt_with_current[..., 1:, :] = self.state_normalizer(gt_with_current[..., 1:, :])
        elif kinematic == 'acceleration':
            # Same 4-channel transform as velocity, then differentiate twice
            gt_with_current = torch.cat([
                gt_with_current[..., :2],
                torch.cat([
                    gt_with_current[..., 2:3].cos(),
                    gt_with_current[..., 2:3].sin()
                ], dim=-1)
            ], dim=-1)
            future_velocity = self.x_differentiate(gt_with_current[..., 1:, :], gt_with_current[..., :1, :])
            current_velocity = torch.cat([
                ego_current_state[..., 4:6],
                torch.zeros_like(ego_current_state[..., 4:6])
            ], dim=-1)[:, None, None, :]
            future_acc = self.x_differentiate(future_velocity, current_velocity)
            gt_with_current = torch.cat([current_velocity, future_acc], dim=-2)
            # Symmetry fix: same as velocity branch above.
            gt_with_current[..., 1:, :] = self.state_normalizer(gt_with_current[..., 1:, :])
        elif kinematic == 'frenet':
            # NOTE: Frenet branch currently handles ONLY the ego trajectory.
            # The state_normalizer used by the production Frenet config is
            # sized for (1 + neighbor_pred_num) agents, but we only populate
            # the ego slot below — so any neighbor_pred_num > 0 in a Frenet
            # run would silently apply ego's (s, d) norm stats to Cartesian
            # neighbor predictions, propagating garbage gradients through the
            # neighbor channels. The YAML default is neighbor_pred_num=0 so
            # this is dormant today, but we warn loudly because the broadcast
            # silently succeeds without it.
            if self.neighbor_pred_num > 0:
                import warnings as _w
                _w.warn(
                    "Frenet kinematic does not yet project neighbor "
                    f"predictions to (s, d); got neighbor_pred_num="
                    f"{self.neighbor_pred_num}. Any neighbor channels will be "
                    "trained with mismatched normalization. Disable the "
                    "prediction head or extend the Frenet projection to "
                    "neighbors before relying on neighbor outputs.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            from flow_planner.data.normalization.frenet_utils import (
                select_reference_centerline,
                cartesian_to_frenet,
            )
            # The Frenet (s, d) target must be computed in RAW world scale
            # (meters), not in the post-normalizer 1/20 scale. The encoder's
            # input-embedding pipeline operates on normalized inputs, but the
            # geometric projection that defines the target must use raw coords
            # so that:
            #   - s is interpretable as physical arc length in meters
            #   - d is interpretable as physical lateral offset in meters
            #   - frenet_norm_stats.yaml means/stds are world-scale numbers
            #   - inference_eval.py (which calls select_reference_centerline
            #     on raw batch.routes / batch.lanes) produces the same
            #     centerline this branch produces, so frenet_to_cartesian
            #     decodes predictions in the trained reference frame.
            # See docs/research/cs269_modifications_review.md CRITICAL #1, #2.
            raw_routes = raw_data.routes.to(device)
            raw_lanes = raw_data.lanes.to(device)
            # v8 Frenet-fix: pass ego_past so select_reference_centerline can use
            # smart-centerline mode (anchored by recent past motion). Inactive
            # unless FRENET_SMART_CENTERLINE=1 in the environment.
            raw_ego_past_xy = raw_data.ego_past[..., :2].to(device)  # (B, T_past, 2)
            centerline = select_reference_centerline(
                route_lanes=raw_routes,
                lanes=raw_lanes,
                ego_past_xy=raw_ego_past_xy,
            )  # (B, N_points, 2) in raw world coords

            # Rebuild gt_with_current in raw world scale. ego_current was
            # normalized into 1/20 scale by obs_normalizer; ego_future was
            # untouched (not in any norm-stats YAML). Recompute both from
            # raw_data so the entire trajectory lives in one consistent frame.
            raw_ego_current = raw_data.ego_current[..., :4].to(device)        # (B, 4)
            raw_ego_current_xy = raw_ego_current[..., :2]                     # (B, 2)
            raw_ego_current_h = torch.atan2(
                raw_ego_current[..., 3:4], raw_ego_current[..., 2:3]
            )                                                                 # (B, 1)
            raw_current_state = torch.cat(
                [raw_ego_current_xy, raw_ego_current_h], dim=-1
            )                                                                 # (B, 3)

            if is_training:
                raw_ego_future = raw_data.ego_future
                if raw_ego_future.numel() != 0:
                    raw_ego_future = raw_ego_future[..., -self.future_len:, :3].to(device)
                else:
                    raw_ego_future = torch.zeros(
                        (raw_current_state.shape[0], self.future_len, 3), device=device
                    )
                gt_raw = torch.cat([
                    raw_current_state[:, None, None, :],                      # (B, 1, 1, 3)
                    raw_ego_future[:, None, :, :],                            # (B, 1, T, 3)
                ], dim=2)                                                     # (B, 1, T+1, 3)
            else:
                gt_raw = raw_current_state[:, None, None, :].repeat(
                    1, 1, self.future_len + 1, 1
                )                                                             # (B, 1, T+1, 3)

            B = gt_raw.shape[0]
            T_plus_1 = gt_raw.shape[2]
            xy = gt_raw[..., :2].view(B, T_plus_1, 2)        # (B, T+1, 2) raw
            sd = cartesian_to_frenet(xy, centerline)         # (B, T+1, 2) raw meters
            sd = sd.view(B, 1, T_plus_1, 2)                  # (B, 1, T+1, 2)
            heading = gt_raw[..., 2:3]
            cos_sin = torch.cat([heading.cos(), heading.sin()], dim=-1)  # (B, 1, T+1, 2)
            gt_with_current = torch.cat([sd, cos_sin], dim=-1)           # (B, 1, T+1, 4)
            gt_with_current[..., 1:, :] = self.state_normalizer(gt_with_current[..., 1:, :])

            # Option A: pass the chosen centerline (raw scale) to the model so
            # CenterlineEncoder can condition on the reference frame. Picked
            # up by FlowPlanner.extract_encoder_inputs and FlowPlannerEncoder.
            model_inputs['reference_centerline'] = centerline

        return model_inputs, gt_with_current