"""Regression test for the encoder.token_dist shape contract.

Pre-fix bug: encoder.all_loc unconditionally appended pred_neighbor_loc
(``neighbors[:, :neighbor_pred_num, -1, :2]``) regardless of whether the
decoder's JointAttention treated those tokens as actual attention tokens.
The decoder freezes token_num = neighbor_num + static_num + lane_num +
action_num (excluding neighbor_pred_num), so the two sides diverged the
moment a user enabled neighbor_pred_num > 0 — silent broadcast / shape
mismatch in BiasedAttention.

Fix: drop pred_neighbor_loc from the all_loc cat so token_dist's middle
dim equals the decoder's frozen token_num regardless of neighbor_pred_num.
"""
from __future__ import annotations

import sys
import pathlib

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FP_PKG = REPO_ROOT / 'flow_planner'
if str(FP_PKG) not in sys.path:
    sys.path.insert(0, str(FP_PKG))

einops = pytest.importorskip('einops')


def _build_encoder(neighbor_pred_num: int):
    from flow_planner.model.flow_planner_model.encoder import FlowPlannerEncoder
    from flow_planner.model.modules.encoder_modules import (
        AgentFusionEncoder, StaticFusionEncoder, LaneFusionEncoder, RouteEncoder,
    )

    hidden_dim = 192
    neighbor = AgentFusionEncoder(past_time_len=21, hidden_dim=hidden_dim,
                                  layer_num=1, tokens_mlp_dim=32, channels_mlp_dim=64)
    static = StaticFusionEncoder(static_objects_state_dim=10, hidden_dim=hidden_dim)
    lane = LaneFusionEncoder(lane_points_num=20, hidden_dim=hidden_dim,
                             layer_num=1, tokens_mlp_dim=32, channels_mlp_dim=64)
    route = RouteEncoder(route_num=25, route_points_num=20, hidden_dim=256,
                         tokens_mlp_dim=32, channels_mlp_dim=64)
    return FlowPlannerEncoder(
        encoder_hidden_dim=hidden_dim,
        with_ego_history=False,
        neighbor_encoder=neighbor, static_encoder=static, lane_encoder=lane,
        route_encoder=route,
        neighbor_agent_num=32, static_objects_num=5, lane_num=70,
        neighbor_pred_num=neighbor_pred_num,
        action_length=20, action_overlap=10, future_len=80,
    )


@pytest.mark.parametrize("neighbor_pred_num", [0, 5, 10])
def test_token_dist_shape_invariant_to_neighbor_pred_num(neighbor_pred_num):
    """token_dist's middle dim must equal neighbor_num + static_num + lane_num
    + action_num (= 32 + 5 + 70 + 7 = 114 with defaults), regardless of
    neighbor_pred_num. Pre-fix this depended on neighbor_pred_num."""
    enc = _build_encoder(neighbor_pred_num=neighbor_pred_num)
    enc.eval()

    B = 2
    neighbors = torch.zeros((B, 32, 21, 11))
    static = torch.zeros((B, 5, 10))
    lanes = torch.zeros((B, 70, 20, 12))
    lanes_speed = torch.zeros((B, 70, 1))
    lanes_has = torch.zeros((B, 70, 1), dtype=torch.bool)
    routes = torch.zeros((B, 25, 20, 12))

    with torch.no_grad():
        out = enc(
            neighbors=neighbors, static=static, lanes=lanes,
            lanes_speed_limit=lanes_speed, lanes_has_speed_limit=lanes_has,
            routes=routes,
        )

    expected = 32 + 5 + 70 + enc.action_num  # 114 with defaults
    assert out['token_dist'].shape == (B, expected, expected), (
        f"token_dist shape {out['token_dist'].shape} depends on "
        f"neighbor_pred_num={neighbor_pred_num} — partial-gate bug regressed."
    )


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
