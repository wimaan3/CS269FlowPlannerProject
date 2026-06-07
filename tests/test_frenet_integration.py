"""Integration tests for the Frenet kinematic path in ModelInputProcessor.

These tests cover the normalization-frame contracts surfaced in the
2026-05-28 code review:
  - Bug 1: reference centerline must be in raw world scale (not 1/20 scale
    introduced by obs_normalizer).
  - Bug 2: the centerline used by inference_eval.py (built externally from
    the raw batch) must match the centerline the model uses internally.
  - HIGH: silent-drop when kinematic='frenet' but model_encoder.centerline_encoder
    is None.

Run with: pytest tests/test_frenet_integration.py -v
"""
from __future__ import annotations

import sys
import pathlib
import warnings

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FP_PKG = REPO_ROOT / 'flow_planner'
if str(FP_PKG) not in sys.path:
    sys.path.insert(0, str(FP_PKG))

from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.data.normalization.obs_normalize import ObservationNormalizer
from flow_planner.model.model_utils.input_preprocess import ModelInputProcessor
from flow_planner.data.normalization.frenet_utils import select_reference_centerline


# ----------------------------- helpers -----------------------------

def realistic_obs_stats():
    """Match waypoints_norm_stats.yaml exactly: xy scaled by std=20 after mean shift."""
    return {
        'ego':            {'log': {}, 'uniform': {'mean': [10, 0, 0, 0], 'std': [20, 20, 1, 1]}},
        'neighbor':       {'log': {}, 'uniform': {'mean': [10, 0, 0, 0], 'std': [20, 20, 1, 1]}},
        'ego_past':       {'mean': [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           'std':  [20, 20, 1, 1, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1]},
        'ego_current':    {'mean': [10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           'std':  [20, 20, 1, 1, 5, 1, 1, 1, 1, 1, 20, 20, 1, 1, 1, 1]},
        'neighbor_past':  {'mean': [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           'std':  [20, 20, 1, 1, 20, 20, 20, 20, 1, 1, 1]},
        'map_objects':    {'mean': [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           'std':  [20, 20, 1, 1, 20, 20, 1, 1, 1, 1]},
        'lanes':          {'mean': [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           'std':  [20, 20, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1]},
        'lanes_speedlimit':  {'mean': [0], 'std': [20]},
        'routes':         {'mean': [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           'std':  [20, 20, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1]},
        'routes_speedlimit': {'mean': [0], 'std': [20]},
    }


def make_sample_with_known_geometry(B=1, future_len=80, lane_pts=20):
    """Build a NuPlanDataSample with raw world geometry:
        - ego_current at world (0, 0), facing +x
        - ego_future is a straight line from (1, 0) to (80, 0) in world meters
        - one route_lane covers (0, 0) → (50, 0)
        - one route_lane covers (50, 0) → (100, 0)

    The obs_normalizer (xy → (x-10)/20, y/20) will be applied by
    ModelInputProcessor; the test asserts that the Frenet projection still
    uses raw world coordinates internally.
    """
    ego_past = torch.zeros((B, 21, 14))
    ego_current = torch.zeros((B, 16))
    ego_current[:, 0] = 0.0   # x
    ego_current[:, 1] = 0.0   # y
    ego_current[:, 2] = 1.0   # cos_h
    ego_current[:, 3] = 0.0   # sin_h

    ego_future = torch.zeros((B, future_len, 3))
    ego_future[:, :, 0] = torch.linspace(1.0, 80.0, future_len)
    ego_future[:, :, 1] = 0.0
    ego_future[:, :, 2] = 0.0

    # Two adjacent route lanes covering x in [0, 100]
    routes = torch.zeros((B, 25, lane_pts, 12))
    routes[:, 0, :, 0] = torch.linspace(0.0, 50.0, lane_pts)
    routes[:, 0, :-1, 2:4] = routes[:, 0, 1:, :2] - routes[:, 0, :-1, :2]
    routes[:, 1, :, 0] = torch.linspace(50.0, 100.0, lane_pts)
    routes[:, 1, :-1, 2:4] = routes[:, 1, 1:, :2] - routes[:, 1, :-1, :2]

    lanes = torch.zeros((B, 70, lane_pts, 12))
    neighbor_past = torch.zeros((B, 32, 21, 11))
    map_objects = torch.zeros((B, 5, 10))
    lanes_speedlimit = torch.zeros((B, 70, 1))
    lanes_has_speedlimit = torch.zeros((B, 70, 1), dtype=torch.bool)
    routes_speedlimit = torch.zeros((B, 25, 1))
    routes_has_speedlimit = torch.zeros((B, 25, 1), dtype=torch.bool)

    return NuPlanDataSample(
        batched=True,
        ego_past=ego_past,
        ego_current=ego_current,
        ego_future=ego_future,
        neighbor_past=neighbor_past,
        neighbor_future=torch.zeros((B, 10, future_len, 4)),
        neighbor_future_observed=torch.zeros((B, 10, future_len, 1)),
        lanes=lanes,
        lanes_speedlimit=lanes_speedlimit,
        lanes_has_speedlimit=lanes_has_speedlimit,
        routes=routes,
        routes_speedlimit=routes_speedlimit,
        routes_has_speedlimit=routes_has_speedlimit,
        map_objects=map_objects,
    )


class _IdentityStateNorm:
    """State normalizer stub that's a no-op, so tests can read raw (s, d) targets."""
    def __call__(self, x): return x
    def inverse(self, x): return x


# ----------------------------- Bug 1: centerline in raw world scale -----------------------------

def test_reference_centerline_is_in_raw_world_scale():
    """The centerline stored in model_inputs['reference_centerline'] must be
    in RAW world scale (meters), NOT post-normalizer scale.

    Pre-fix bug: select_reference_centerline was called on POST-normalizer
    routes/lanes (xy scaled by 1/20 after mean shift), producing a centerline
    in 1/20 scale. ego_future is NOT in the norm-stats YAML so it stayed in
    raw scale; the projection mixed scales and produced garbage (s, d).
    """
    obs_norm = ObservationNormalizer(realistic_obs_stats())
    processor = ModelInputProcessor(
        future_len=80,
        obs_normalizer=obs_norm,
        state_normalizer=_IdentityStateNorm(),
        neighbor_pred_num=10,
    )
    sample = make_sample_with_known_geometry(B=1, future_len=80)

    model_inputs, _ = processor.sample_to_model_input(
        sample, device='cpu', kinematic='frenet', is_training=True,
    )

    centerline = model_inputs['reference_centerline']
    # Two route lanes covering x in [0, 100] should produce a centerline
    # with max_x ~ 100 in raw scale, or ~ 4.5 in post-normalizer scale.
    max_x = centerline[0, :, 0].max().item()
    assert max_x > 80.0, (
        f'centerline appears to be in post-normalizer scale (max_x = {max_x:.2f}). '
        f'Expected raw world scale with max_x ~ 100. Frame mismatch bug returned.'
    )


def test_frenet_target_s_is_raw_arc_length():
    """When ego drives straight forward 0→80m along a centerline that also
    runs straight forward, the (s, d) target should report s ≈ 80m (raw
    meters) at the final timestep and d ≈ 0 throughout. Pre-fix bug
    produced |d| > 10m because of frame mismatch."""
    obs_norm = ObservationNormalizer(realistic_obs_stats())
    processor = ModelInputProcessor(
        future_len=80,
        obs_normalizer=obs_norm,
        state_normalizer=_IdentityStateNorm(),
        neighbor_pred_num=10,
    )
    sample = make_sample_with_known_geometry(B=1, future_len=80)

    _, gt_with_current = processor.sample_to_model_input(
        sample, device='cpu', kinematic='frenet', is_training=True,
    )

    # gt_with_current shape: (B, 1, T+1, 4) — (s, d, cos_h, sin_h)
    s = gt_with_current[0, 0, :, 0]
    d = gt_with_current[0, 0, :, 1]

    assert s[-1].item() > 70.0, (
        f's at final timestep is {s[-1].item():.2f}, expected ~80m (raw scale). '
        f'Frenet target appears to be computed in a scaled frame.'
    )
    assert d.abs().max().item() < 1.0, (
        f'max |d| = {d.abs().max().item():.2f}m. Expected <1m since ego stays on '
        f'the centerline. Large |d| indicates frame mismatch.'
    )


# ----------------------------- Bug 2: training/inference centerline match -----------------------------

def test_inference_eval_centerline_matches_model_centerline():
    """Contract: inference_eval.py builds centerline = select_reference_centerline(
    batch.routes, batch.lanes) on the raw batch. The model's internal call inside
    sample_to_model_input must yield the SAME centerline.

    Pre-fix bug: model called select_reference_centerline on POST-normalizer
    routes/lanes; eval driver called it on RAW routes/lanes. The two centerlines
    lived in different frames, so frenet_to_cartesian at eval-time decoded
    predictions in a frame the model was not trained in.
    """
    obs_norm = ObservationNormalizer(realistic_obs_stats())
    processor = ModelInputProcessor(
        future_len=80,
        obs_normalizer=obs_norm,
        state_normalizer=_IdentityStateNorm(),
        neighbor_pred_num=10,
    )
    sample = make_sample_with_known_geometry(B=2, future_len=80)

    model_inputs, _ = processor.sample_to_model_input(
        sample, device='cpu', kinematic='frenet', is_training=False,
    )
    centerline_model = model_inputs['reference_centerline']

    # Reproduce inference_eval.py's external call on the raw batch.
    centerline_eval = select_reference_centerline(
        route_lanes=sample.routes,
        lanes=sample.lanes,
    )

    assert torch.allclose(centerline_model, centerline_eval, atol=1e-4), (
        f'Centerline mismatch between model-internal and inference-eval-external. '
        f'Max diff: {(centerline_model - centerline_eval).abs().max().item():.4f}m. '
        f'frenet_to_cartesian at eval would use a different reference frame than '
        f'the model trained against.'
    )


# ----------------------------- HIGH: silent-drop warning -----------------------------

def test_silent_drop_warning_when_frenet_without_centerline_encoder():
    """If kinematic='frenet' is set on FlowPlanner but the encoder has
    centerline_encoder=None, training silently proceeds without Option A
    (the centerline is built and passed but the encoder drops it). The
    user must be warned so they can either add the Hydra overrides or
    explicitly acknowledge a v4 (no Option A) ablation.
    """
    from flow_planner.model.flow_planner_model.flow_planner import FlowPlanner
    from flow_planner.model.flow_planner_model.encoder import FlowPlannerEncoder
    from flow_planner.model.modules.encoder_modules import (
        AgentFusionEncoder, StaticFusionEncoder, LaneFusionEncoder, RouteEncoder,
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
    model_encoder = FlowPlannerEncoder(
        encoder_hidden_dim=encoder_hidden_dim,
        with_ego_history=False,
        neighbor_encoder=neighbor, static_encoder=static, lane_encoder=lane,
        route_encoder=route, centerline_encoder=None,
        action_length=20, action_overlap=10,
    )

    class _StubDecoder(torch.nn.Module):
        def forward(self, x, t, **kw): return x

    class _StubODE:
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = FlowPlanner(
            model_encoder=model_encoder,
            model_decoder=_StubDecoder(),
            flow_ode=_StubODE(),
            kinematic='frenet',
            cfg_prob=0.1,
            cfg_weight=1.0,
            cfg_type='neighbors',
            future_len=80,
            action_len=20,
            action_overlap=10,
            state_dim=4,
            neighbor_num=32,
            cfg_neighbor_num=10,
        )
        relevant = [str(w.message) for w in caught
                    if 'centerline_encoder' in str(w.message).lower()
                    or 'option a' in str(w.message).lower()]
        assert relevant, (
            f'No warning emitted when kinematic=frenet but centerline_encoder=None. '
            f'All warnings: {[str(w.message) for w in caught]}'
        )


def test_no_warning_when_frenet_with_centerline_encoder():
    """Sanity: when kinematic='frenet' AND centerline_encoder is provided,
    no Option-A warning should fire."""
    from flow_planner.model.flow_planner_model.flow_planner import FlowPlanner
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
    centerline_encoder = CenterlineEncoder(n_points=100, hidden_dim=decoder_hidden_dim)
    model_encoder = FlowPlannerEncoder(
        encoder_hidden_dim=encoder_hidden_dim,
        with_ego_history=False,
        neighbor_encoder=neighbor, static_encoder=static, lane_encoder=lane,
        route_encoder=route, centerline_encoder=centerline_encoder,
        action_length=20, action_overlap=10,
    )

    class _StubDecoder(torch.nn.Module):
        def forward(self, x, t, **kw): return x

    class _StubODE:
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = FlowPlanner(
            model_encoder=model_encoder,
            model_decoder=_StubDecoder(),
            flow_ode=_StubODE(),
            kinematic='frenet',
            cfg_prob=0.1,
            cfg_weight=1.0,
            cfg_type='neighbors',
            future_len=80,
            action_len=20,
            action_overlap=10,
            state_dim=4,
            neighbor_num=32,
            cfg_neighbor_num=10,
        )
        relevant = [str(w.message) for w in caught
                    if 'option a' in str(w.message).lower()]
        assert not relevant, (
            f'Spurious Option A warning when centerline_encoder is provided: {relevant}'
        )


def test_no_warning_when_waypoints():
    """Sanity: waypoints kinematic with centerline_encoder=None should not
    trigger the Option A warning."""
    from flow_planner.model.flow_planner_model.flow_planner import FlowPlanner
    from flow_planner.model.flow_planner_model.encoder import FlowPlannerEncoder
    from flow_planner.model.modules.encoder_modules import (
        AgentFusionEncoder, StaticFusionEncoder, LaneFusionEncoder, RouteEncoder,
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
    model_encoder = FlowPlannerEncoder(
        encoder_hidden_dim=encoder_hidden_dim,
        with_ego_history=False,
        neighbor_encoder=neighbor, static_encoder=static, lane_encoder=lane,
        route_encoder=route, centerline_encoder=None,
        action_length=20, action_overlap=10,
    )

    class _StubDecoder(torch.nn.Module):
        def forward(self, x, t, **kw): return x

    class _StubODE:
        pass

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _ = FlowPlanner(
            model_encoder=model_encoder,
            model_decoder=_StubDecoder(),
            flow_ode=_StubODE(),
            kinematic='waypoints',
            cfg_prob=0.1,
            cfg_weight=1.0,
            cfg_type='neighbors',
            future_len=80,
            action_len=20,
            action_overlap=10,
            state_dim=4,
            neighbor_num=32,
            cfg_neighbor_num=10,
        )
        relevant = [str(w.message) for w in caught
                    if 'option a' in str(w.message).lower()]
        assert not relevant, (
            f'Spurious Option A warning on waypoints kinematic: {relevant}'
        )


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
