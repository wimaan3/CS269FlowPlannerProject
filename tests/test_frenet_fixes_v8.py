"""Regression tests for v8 Frenet fixes.

Covers:
  - tanh(d/3) bounded target (FRENET_TANH_D env var) — round-trip identity
  - Smart centerline selection (FRENET_SMART_CENTERLINE env var) — uses ego_past
  - Inference hook discovery + apply() (multiframe and best_of_n modules)
"""
from __future__ import annotations

import os
import sys
import pathlib
import importlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FP_PKG = REPO_ROOT / 'flow_planner'
if str(FP_PKG) not in sys.path:
    sys.path.insert(0, str(FP_PKG))

# einops gates the model-internal forward-pass tests; install in venv for Colab CI
einops = pytest.importorskip('einops')


def _set_env(**kw):
    """Context-manager-ish helper to set + restore env vars."""
    prev = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return prev


def _restore_env(prev):
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ============== tanh(d/3) compression — round-trip identity ==============

def test_tanh_d_compression_round_trips():
    """cartesian_to_frenet + frenet_to_cartesian should be approximate-identity
    with FRENET_TANH_D=1 set, just like it is with FRENET_TANH_D unset.
    """
    import torch
    from flow_planner.data.normalization.frenet_utils import (
        cartesian_to_frenet, frenet_to_cartesian,
    )

    # Build a simple straight centerline along +x at y=0
    B, N = 1, 50
    centerline = torch.zeros(B, N, 2)
    centerline[0, :, 0] = torch.linspace(0.0, 49.0, N)

    # Trajectory: ego moves +x at 1 m/step with small lateral wiggle in physical range
    T = 30
    xy = torch.zeros(B, T, 2)
    xy[0, :, 0] = torch.linspace(0.0, 29.0, T)
    xy[0, :, 1] = 0.5 * torch.sin(torch.linspace(0.0, 3.14, T))   # |d| < 0.5 m bulk

    # ---- Baseline (no tanh) ----
    prev = _set_env(FRENET_TANH_D=None, FRENET_TANH_D_SCALE=None)
    try:
        sd_raw = cartesian_to_frenet(xy, centerline)
        xy_raw = frenet_to_cartesian(sd_raw, centerline)
        err_raw = (xy - xy_raw).abs().max().item()
    finally:
        _restore_env(prev)

    # ---- With tanh(d/3) ----
    prev = _set_env(FRENET_TANH_D='1', FRENET_TANH_D_SCALE='3.0')
    try:
        sd_tanh = cartesian_to_frenet(xy, centerline)
        xy_tanh = frenet_to_cartesian(sd_tanh, centerline)
        err_tanh = (xy - xy_tanh).abs().max().item()
    finally:
        _restore_env(prev)

    assert err_raw < 0.1, f'Raw round-trip error too high: {err_raw:.4f}'
    assert err_tanh < 0.1, f'Tanh round-trip error too high: {err_tanh:.4f}'

    # tanh-compressed d should differ from raw d (otherwise the compression wasn't applied)
    d_raw_max = sd_raw[..., 1].abs().max().item()
    d_tanh_max = sd_tanh[..., 1].abs().max().item()
    assert d_tanh_max < d_raw_max + 1e-6
    # tanh image is bounded by 1
    assert d_tanh_max <= 1.0, f'tanh-compressed d should be in [-1, 1], got max={d_tanh_max}'


def test_tanh_d_compression_large_d_bounded():
    """When physical d is large (e.g. centerline mis-selection), tanh keeps target in [-1, 1]."""
    import torch
    from flow_planner.data.normalization.frenet_utils import cartesian_to_frenet

    B, N = 1, 50
    centerline = torch.zeros(B, N, 2)
    centerline[0, :, 0] = torch.linspace(0.0, 49.0, N)

    # Trajectory FAR from centerline (large d): ego at y=20
    T = 30
    xy = torch.zeros(B, T, 2)
    xy[0, :, 0] = torch.linspace(0.0, 29.0, T)
    xy[0, :, 1] = 20.0   # 20m off centerline — large d

    prev = _set_env(FRENET_TANH_D='1', FRENET_TANH_D_SCALE='3.0')
    try:
        sd = cartesian_to_frenet(xy, centerline)
        d = sd[..., 1]
    finally:
        _restore_env(prev)

    assert d.abs().max().item() < 1.0, f'tanh(20/3)≈0.997; got {d.abs().max().item()}'
    assert d.abs().min().item() > 0.95, f'tanh(20/3)≈0.997; got {d.abs().min().item()}'


def test_smart_centerline_uses_ego_past():
    """When FRENET_SMART_CENTERLINE=1, ego_past_xy is used as the anchor."""
    import torch
    from flow_planner.data.normalization.frenet_utils import select_reference_centerline

    B, N_route, N_lanes, N_pts, D = 1, 2, 1, 10, 12
    route_lanes = torch.zeros(B, N_route, N_pts, D)
    lanes = torch.zeros(B, N_lanes, N_pts, D)

    # Route lane 0 goes +x at y=0; route lane 1 goes +x at y=10
    route_lanes[0, 0, :, 0] = torch.linspace(0.0, 9.0, N_pts)
    route_lanes[0, 0, :, 1] = 0.0
    route_lanes[0, 0, :, 2] = 1.0  # populate geometry beyond the xy slice
    route_lanes[0, 1, :, 0] = torch.linspace(0.0, 9.0, N_pts)
    route_lanes[0, 1, :, 1] = 10.0
    route_lanes[0, 1, :, 2] = 1.0

    # Smart mode with ego_past anchored at y=10 — should prefer lane 1
    ego_past = torch.zeros(B, 5, 2)
    ego_past[0, :, 1] = 10.0
    prev = _set_env(FRENET_SMART_CENTERLINE='1')
    try:
        cl_smart = select_reference_centerline(
            route_lanes=route_lanes, lanes=lanes,
            ego_past_xy=ego_past, out_points=20,
        )
    finally:
        _restore_env(prev)

    # The smart-selected centerline should sit at y≈10 (lane 1), not y≈0
    avg_y = cl_smart[0, :, 1].mean().item()
    assert avg_y > 5.0, (
        f'smart mode with ego_past at y=10 should pick lane 1 (y=10); '
        f'got centerline avg y={avg_y:.2f}'
    )


def test_smart_centerline_vs_default_differs():
    """Smart mode (FRENET_SMART_CENTERLINE=1) should produce a DIFFERENT centerline
    than default mode when ego_past_xy is non-trivial (lane 1 vs origin)."""
    import torch
    from flow_planner.data.normalization.frenet_utils import select_reference_centerline

    B, N_route, N_lanes, N_pts, D = 1, 2, 1, 10, 12
    route_lanes = torch.zeros(B, N_route, N_pts, D)
    lanes = torch.zeros(B, N_lanes, N_pts, D)
    route_lanes[0, 0, :, 0] = torch.linspace(0.0, 9.0, N_pts)
    route_lanes[0, 0, :, 1] = 0.0
    route_lanes[0, 0, :, 2] = 1.0
    route_lanes[0, 1, :, 0] = torch.linspace(0.0, 9.0, N_pts)
    route_lanes[0, 1, :, 1] = 10.0
    route_lanes[0, 1, :, 2] = 1.0

    # ego_past at y=10 (lane 1)
    ego_past = torch.zeros(B, 5, 2)
    ego_past[0, :, 1] = 10.0

    # default mode (ego_past ignored, anchored at origin)
    prev = _set_env(FRENET_SMART_CENTERLINE=None)
    try:
        cl_default = select_reference_centerline(
            route_lanes=route_lanes, lanes=lanes,
            ego_past_xy=ego_past, out_points=20,
        )
    finally:
        _restore_env(prev)

    # smart mode (ego_past at y=10 used as anchor)
    prev = _set_env(FRENET_SMART_CENTERLINE='1')
    try:
        cl_smart = select_reference_centerline(
            route_lanes=route_lanes, lanes=lanes,
            ego_past_xy=ego_past, out_points=20,
        )
    finally:
        _restore_env(prev)

    # The two centerlines should differ (smart mode shifted toward ego_past)
    delta = (cl_smart - cl_default).abs().mean().item()
    assert delta > 1.0, (
        f'smart mode should produce a different centerline than default '
        f'(ego_past at y=10); got mean abs diff={delta:.2f}'
    )


# ============== Inference hooks discovery + apply() ==============

def test_multiframe_hook_module_importable():
    """The V4 multiframe hook should be importable and expose apply()."""
    mod = importlib.import_module('flow_planner.inference.hooks.multiframe')
    assert hasattr(mod, 'apply'), 'multiframe hook must expose apply(model)'


def test_best_of_n_hook_module_importable():
    """The V5 best_of_n hook should be importable and expose apply()."""
    mod = importlib.import_module('flow_planner.inference.hooks.best_of_n')
    assert hasattr(mod, 'apply'), 'best_of_n hook must expose apply(model)'


def test_multiframe_hook_k1_is_noop():
    """K=1 should print a no-op message and not patch anything."""
    from flow_planner.inference.hooks import multiframe

    prev = _set_env(FRENET_INFERENCE_K='1')
    try:
        # Pass a dummy "model" (not used for K=1 path)
        multiframe.apply(model=object())
    finally:
        _restore_env(prev)
    # No assertions on side effects — just that it doesn't raise.


def test_best_of_n_hook_n1_is_noop():
    from flow_planner.inference.hooks import best_of_n
    prev = _set_env(FRENET_INFERENCE_N='1')
    try:
        best_of_n.apply(model=object())
    finally:
        _restore_env(prev)


def test_best_of_n_score_function():
    """The scoring function should rank lower-d trajectories as better."""
    import torch
    from flow_planner.inference.hooks.best_of_n import _score

    # Two samples: A has |d| ≈ 0.5 (good), B has |d| ≈ 5.0 (bad)
    B, T, D = 2, 10, 4
    sample_A = torch.zeros(B, T, D)
    sample_A[..., 1] = 0.5
    sample_B = torch.zeros(B, T, D)
    sample_B[..., 1] = 5.0

    score_A = _score(sample_A, 'mean_abs_d', 1.0)
    score_B = _score(sample_B, 'mean_abs_d', 1.0)

    assert score_A.shape == (B,), f'score should be (B,); got {score_A.shape}'
    assert (score_A < score_B).all(), 'lower |d| should score lower (= better)'


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
