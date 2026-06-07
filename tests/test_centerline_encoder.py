"""Tests for Option A: CenterlineEncoder + encoder integration.

Run with: pytest tests/test_centerline_encoder.py -v
"""
from __future__ import annotations

import math
import sys
import pathlib

import pytest
import torch

# Allow running from repo root without installing the package
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FP_PKG = REPO_ROOT / 'flow_planner'
if str(FP_PKG) not in sys.path:
    sys.path.insert(0, str(FP_PKG))


# ----------------------------- CenterlineEncoder unit tests -----------------------------

def test_centerline_encoder_basic_shape():
    """CenterlineEncoder produces (B, hidden_dim) from (B, N, 2) input."""
    from flow_planner.model.modules.encoder_modules import CenterlineEncoder

    enc = CenterlineEncoder(n_points=100, hidden_dim=256)
    enc.eval()
    x = torch.randn(4, 100, 2)
    with torch.no_grad():
        out = enc(x)
    assert out.shape == (4, 256), f'expected (4, 256), got {out.shape}'


def test_centerline_encoder_deterministic_in_eval():
    """Same input → same output in eval mode (no dropout etc.)."""
    from flow_planner.model.modules.encoder_modules import CenterlineEncoder

    enc = CenterlineEncoder(n_points=100, hidden_dim=256)
    enc.eval()
    x = torch.randn(2, 100, 2)
    with torch.no_grad():
        a = enc(x)
        b = enc(x)
    assert torch.allclose(a, b), 'CenterlineEncoder is non-deterministic in eval'


def test_centerline_encoder_distinguishes_different_centerlines():
    """Two different centerlines should produce different embeddings.

    This is the core requirement: the encoder must encode the centerline's
    actual geometry so the decoder can condition on which frame it's in.
    """
    from flow_planner.model.modules.encoder_modules import CenterlineEncoder

    enc = CenterlineEncoder(n_points=100, hidden_dim=256)
    enc.eval()

    # Straight centerline along +x at y=0
    straight = torch.zeros(1, 100, 2)
    straight[0, :, 0] = torch.linspace(0.0, 100.0, 100)

    # Same but offset by 20m in y — different reference frame
    offset = straight.clone()
    offset[0, :, 1] = 20.0

    with torch.no_grad():
        e_straight = enc(straight)
        e_offset = enc(offset)

    diff = (e_straight - e_offset).abs().mean().item()
    # At random init the absolute difference is small but must be > 0; after
    # training this will be much larger. We just verify the encoder is
    # input-sensitive (not pooling away the geometry).
    assert diff > 1e-4, f'CenterlineEncoder produces identical outputs for very different centerlines (diff={diff:.2e})'


def test_centerline_encoder_uses_tangent_information():
    """Two centerlines with the same points but reversed direction should
    differ — the tangent feature should pick up the direction.

    Seeded because the assertion margin (diff > 0.01) is small enough that
    ~10% of random initializations of the freshly-constructed encoder
    happen to produce a smaller diff. The test is checking that the
    encoder is input-sensitive in the tangent dimension, not that any
    particular initialization passes — a fixed seed gives a deterministic
    pass/fail signal.
    """
    from flow_planner.model.modules.encoder_modules import CenterlineEncoder

    torch.manual_seed(0)
    enc = CenterlineEncoder(n_points=100, hidden_dim=256)
    enc.eval()

    forward = torch.zeros(1, 100, 2)
    forward[0, :, 0] = torch.linspace(0.0, 100.0, 100)
    reverse = forward.flip(dims=[1])

    with torch.no_grad():
        e_fwd = enc(forward)
        e_rev = enc(reverse)

    # Should differ — tangent points opposite ways
    diff = (e_fwd - e_rev).abs().mean().item()
    assert diff > 0.01, f'Tangent direction not encoded (diff={diff:.6f})'


# ----------------------------- Backwards-compat tests -----------------------------

def test_flow_planner_encoder_centerline_none_no_extra_params():
    """When centerline_encoder=None (default), the encoder has zero extra
    parameters compared to the original — existing waypoints checkpoints
    must still load cleanly."""
    from flow_planner.model.flow_planner_model.encoder import FlowPlannerEncoder
    from flow_planner.model.modules.encoder_modules import (
        AgentFusionEncoder, StaticFusionEncoder, LaneFusionEncoder, RouteEncoder,
    )
    encoder_hidden_dim = 192
    neighbor = AgentFusionEncoder(past_time_len=21, hidden_dim=encoder_hidden_dim,
                                  layer_num=1, tokens_mlp_dim=32, channels_mlp_dim=64)
    static = StaticFusionEncoder(static_objects_state_dim=10, hidden_dim=encoder_hidden_dim)
    lane = LaneFusionEncoder(lane_points_num=20, hidden_dim=encoder_hidden_dim,
                             layer_num=1, tokens_mlp_dim=32, channels_mlp_dim=64)
    route = RouteEncoder(route_num=25, route_points_num=20, hidden_dim=256,
                         tokens_mlp_dim=32, channels_mlp_dim=64)

    enc_without = FlowPlannerEncoder(
        encoder_hidden_dim=encoder_hidden_dim,
        with_ego_history=False,
        neighbor_encoder=neighbor, static_encoder=static, lane_encoder=lane,
        route_encoder=route,
        action_length=20, action_overlap=10,
        centerline_encoder=None,  # explicit None
    )

    # Sanity: count parameter tensors. No new ones should be added beyond the
    # standard encoder components (neighbor + static + lane + route + pos_emb).
    n_params_modules = sum(1 for _ in enc_without.named_parameters())
    assert n_params_modules > 0
    # The key property: state_dict has no 'centerline_encoder.*' keys.
    centerline_keys = [k for k in enc_without.state_dict().keys() if 'centerline_encoder' in k]
    assert len(centerline_keys) == 0, (
        f'Expected no centerline_encoder.* state dict keys when centerline_encoder=None, '
        f'found {len(centerline_keys)}: {centerline_keys[:3]}'
    )


# ----------------------------- Integration smoke test -----------------------------

def test_flow_planner_encoder_with_centerline_encoder_runs():
    """End-to-end smoke: encoder with a real CenterlineEncoder produces output
    of the expected shapes when given a reference_centerline."""
    from flow_planner.model.flow_planner_model.encoder import FlowPlannerEncoder
    from flow_planner.model.modules.encoder_modules import (
        AgentFusionEncoder, StaticFusionEncoder, LaneFusionEncoder, RouteEncoder,
        CenterlineEncoder,
    )

    encoder_hidden_dim = 192
    decoder_hidden_dim = 256
    neighbor = AgentFusionEncoder(past_time_len=21, hidden_dim=encoder_hidden_dim,
                                  layer_num=1, tokens_mlp_dim=32, channels_mlp_dim=64)
    static = StaticFusionEncoder(static_objects_state_dim=10, hidden_dim=encoder_hidden_dim)
    lane = LaneFusionEncoder(lane_points_num=20, hidden_dim=encoder_hidden_dim,
                             layer_num=1, tokens_mlp_dim=32, channels_mlp_dim=64)
    route = RouteEncoder(route_num=25, route_points_num=20, hidden_dim=decoder_hidden_dim,
                         tokens_mlp_dim=32, channels_mlp_dim=64)
    centerline = CenterlineEncoder(n_points=100, hidden_dim=decoder_hidden_dim)

    enc = FlowPlannerEncoder(
        encoder_hidden_dim=encoder_hidden_dim,
        with_ego_history=False,
        neighbor_encoder=neighbor, static_encoder=static, lane_encoder=lane,
        route_encoder=route, centerline_encoder=centerline,
        action_length=20, action_overlap=10,
    )
    enc.eval()

    B = 2
    neighbors = torch.zeros((B, 32, 21, 11))
    static_objs = torch.zeros((B, 5, 10))
    lanes = torch.zeros((B, 70, 20, 12))
    # mark a couple of lanes as valid so the lane encoder doesn't degenerate
    lanes[0, 0, :, 0] = torch.linspace(0.0, 20.0, 20)
    lanes[0, 0, :, 1] = 1.0
    lanes[0, 0, :-1, 2:4] = lanes[0, 0, 1:, :2] - lanes[0, 0, :-1, :2]
    speed = torch.zeros((B, 70, 1))
    has_speed = torch.zeros((B, 70, 1), dtype=torch.bool)
    routes = torch.zeros((B, 25, 20, 12))
    routes[0, 0, :, 0] = torch.linspace(0.0, 20.0, 20)
    routes[0, 0, :-1, 2:4] = routes[0, 0, 1:, :2] - routes[0, 0, :-1, :2]
    reference = torch.randn((B, 100, 2))

    with torch.no_grad():
        out_with = enc(neighbors=neighbors, static=static_objs, lanes=lanes,
                       lanes_speed_limit=speed, lanes_has_speed_limit=has_speed,
                       routes=routes, reference_centerline=reference)
        out_without = enc(neighbors=neighbors, static=static_objs, lanes=lanes,
                          lanes_speed_limit=speed, lanes_has_speed_limit=has_speed,
                          routes=routes)

    # routes_cond should be (B, decoder_hidden_dim) in both cases
    assert out_with['routes_cond'].shape == (B, decoder_hidden_dim)
    assert out_without['routes_cond'].shape == (B, decoder_hidden_dim)

    # Post audit Finding 2 fix (commit applying audit_patches.md), the
    # centerline injection is gated by a zero-init learnable scalar
    # `centerline_gate`. At init the gate is 0 so the centerline branch
    # is a no-op, which means out_with == out_without bit-for-bit at
    # step 0. The model learns to enable the branch during training.
    #
    # Verify: (a) zero-init identity at step 0, and (b) the wiring works
    # once the gate is non-zero.
    diff_init = (out_with['routes_cond'] - out_without['routes_cond']).abs().mean().item()
    assert diff_init < 1e-6, (
        f'Expected centerline injection to be identity at init (gate=0); '
        f'got diff={diff_init:.2e}. Either the gate was not initialized to '
        f'zero or _basic_init clobbered it.'
    )

    # Now manually set the gate to a non-zero value and re-run; routes_cond
    # MUST differ — this confirms the wiring still flows when enabled.
    with torch.no_grad():
        enc.centerline_gate.fill_(1.0)
    with torch.no_grad():
        out_with_enabled = enc(neighbors=neighbors, static=static_objs, lanes=lanes,
                               lanes_speed_limit=speed, lanes_has_speed_limit=has_speed,
                               routes=routes, reference_centerline=reference)
    diff_enabled = (out_with_enabled['routes_cond'] - out_without['routes_cond']).abs().mean().item()
    assert diff_enabled > 1e-4, (
        f'With centerline_gate=1.0, routes_cond did not change when '
        f'reference_centerline was added (diff={diff_enabled:.2e}). '
        f'Option A wiring is broken — encoder did not use the centerline.'
    )


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
