import math
import torch
import torch.nn as nn
from timm.layers import Mlp

from functools import partial
from flow_planner.model.modules.decoder_modules import MixerBlock, SelfAttentionBlock
from flow_planner.model.model_utils.tool_func import lanes_to_route_mask


class AgentFusionEncoder(nn.Module):
    def __init__(self, past_time_len, drop_path_rate=0.3, hidden_dim=192, layer_num=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8+1,  hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=past_time_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(layer_num)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        pos = x[:, :, -1, :7].clone() # x, y, cos, sin
        # neighbor: [1,0,0]
        pos[..., -3:] = 0.0
        pos[..., -3] = 1.0
        
        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0 # for mask_v==0, this indicates that the corresponding x is padded with 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        x = torch.cat([x, (~mask_v).float().unsqueeze(-1)], dim=-1)
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)

        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)  

        x = torch.mean(x, dim=1)


        neighbor_type = neighbor_type.view(B * P, -1)

        neighbor_type = neighbor_type[valid_indices]
        type_embedding = self.type_emb(neighbor_type)  # Traffic light embedding for valid data
        x = x + type_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

class StaticFusionEncoder(nn.Module):
    def __init__(self, static_objects_state_dim, drop_path_rate=0.3, hidden_dim=192):
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(in_features=static_objects_state_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        B, P, _ = x.shape

        pos = x[:, :, :7].clone() # x, y, cos, sin
        # static: [0,1,0]
        pos[..., -3:] = 0.0
        pos[..., -2] = 1.0

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        valid_indices = ~mask_p.view(-1) 

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)
  
class LaneFusionEncoder(nn.Module):
    def __init__(self, lane_points_num, drop_path_rate=0.3, hidden_dim=192, layer_num=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()
        self._lane_points_num = lane_points_num
        self._channel = channels_mlp_dim
        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=lane_points_num, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(layer_num)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''

        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_points_num / 2), :7].clone() # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        # lane: [0,0,1]
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0

        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1) 
        x = x[valid_indices].type(torch.float32)

        x = self.channel_pre_project(x)

        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)

        for block in self.blocks:
            x = block(x)  

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1
            )
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit
        
        speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
        speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
            (~has_speed_limit).sum().item(), -1
        )
        speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices].type(torch.float32)
        traffic_light_embedding = self.traffic_emb(traffic)  # Traffic light embedding for valid data

        D = x.shape[-1]

        x = x + speed_limit_embedding + traffic_light_embedding

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)

class RouteEncoder(nn.Module):
    def __init__(self, route_num, route_points_num, drop_path_rate=0.3, hidden_dim=192, tokens_mlp_dim=32, channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * route_points_num, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1) 
        x = x[valid_indices] 

        x = self.channel_pre_project(x)

        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)

        x = self.Mixer(x)
        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, -1)


class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, layer_num=3):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(layer_num)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):

        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)

class CenterlineEncoder(nn.Module):
    """Encode the horizon-covering reference centerline into a conditioning vector.

    The Frenet (s, d) target is defined relative to ONE specific reference
    centerline (chosen by select_reference_centerline). Without this module,
    the model conditions only on a pooled summary of *all* route lanes
    (routes_cond) and cannot tell which one was the (s, d) reference — which
    is the root cause documented in docs/frenet_representation_analysis.md.

    This module encodes the chosen centerline polyline explicitly. Its output
    is added to routes_cond so the decoder sees the reference frame.

    Per-point features: (x, y, tangent_x, tangent_y, arc_length_normalized) — 5d.

    Input:  (B, N_points, 2) raw centerline coordinates in ego frame.
    Output: (B, hidden_dim).
    """

    def __init__(self, n_points: int = 100, hidden_dim: int = 256,
                 mlp_hidden: int = 128, drop_path_rate: float = 0.1):
        super().__init__()
        self._n_points = n_points
        self.point_proj = Mlp(
            in_features=5, hidden_features=mlp_hidden, out_features=mlp_hidden,
            act_layer=nn.GELU, drop=0.,
        )
        self.token_pre_project = Mlp(
            in_features=n_points, hidden_features=32, out_features=32,
            act_layer=nn.GELU, drop=0.,
        )
        self.mixer = MixerBlock(32, mlp_hidden, drop_path_rate)
        self.norm = nn.LayerNorm(mlp_hidden)
        self.out_proj = Mlp(
            in_features=mlp_hidden, hidden_features=hidden_dim, out_features=hidden_dim,
            act_layer=nn.GELU, drop=drop_path_rate,
        )

    def forward(self, centerline: torch.Tensor) -> torch.Tensor:
        """centerline: (B, N_points, 2) -> (B, hidden_dim).

        Note (2026-05-29 audit Finding 2): the raw centerline arrives in
        world meters while every sibling encoder receives 1/20-scaled input
        via `obs_normalizer`. We rescale by /20 (matching the routes/lanes
        xy convention) so `point_proj` activations match the magnitude
        regime the rest of the model is initialized for. Tangent (unit
        vector) and arclen ([0, 1]) are already scale-free.
        """
        B, N, _ = centerline.shape
        # Tangent computed on RAW centerline so its unit-vector direction is
        # numerically identical to the unscaled version (avoid precision loss
        # on the divisor for short segments).
        tangent = torch.zeros_like(centerline)
        tangent[:, :-1, :] = centerline[:, 1:, :] - centerline[:, :-1, :]
        tangent[:, -1, :] = tangent[:, -2, :]
        tangent_norm = torch.linalg.vector_norm(tangent, dim=-1, keepdim=True).clamp(min=1e-6)
        tangent = tangent / tangent_norm
        # Scale xy to the 1/20 regime to match obs_normalizer (mean=10, std=20).
        centerline_scaled = centerline / 20.0
        # Normalized arc length parameter in [0, 1] so the model can tell where
        # along the centerline each point lies.
        arclen = torch.linspace(0.0, 1.0, N, device=centerline.device, dtype=centerline.dtype)
        arclen = arclen[None, :, None].expand(B, N, 1)
        feat = torch.cat([centerline_scaled, tangent, arclen], dim=-1)  # (B, N, 5)

        x = self.point_proj(feat)                                       # (B, N, C)
        # MLP-Mixer style: mix tokens (points) then channels.
        x = x.permute(0, 2, 1)                                          # (B, C, N)
        x = self.token_pre_project(x)                                   # (B, C, 32)
        x = x.permute(0, 2, 1)                                          # (B, 32, C)
        x = self.mixer(x)                                               # (B, 32, C)
        x = self.norm(x).mean(dim=1)                                    # (B, C)
        return self.out_proj(x)                                         # (B, hidden_dim)
