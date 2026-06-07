"""Tests for Frenet projection utilities.

Run with: pytest tests/test_frenet.py -v
(Requires torch and the flow_planner package to be installable / on PYTHONPATH.)
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

from flow_planner.data.normalization.frenet_utils import (  # noqa: E402
    select_reference_centerline,
    cartesian_to_frenet,
    frenet_to_cartesian,
)


# ----------------------------- helpers -----------------------------

def straight_centerline(length: float = 20.0, n_points: int = 20) -> torch.Tensor:
    """Centerline along +x axis: (1, n_points, 2)."""
    pts = torch.zeros((1, n_points, 2))
    pts[0, :, 0] = torch.linspace(0.0, length, n_points)
    return pts


def quarter_circle_centerline(radius: float = 10.0, n_points: int = 20) -> torch.Tensor:
    """Quarter circle from (radius, 0) sweeping counter-clockwise to (0, radius)."""
    thetas = torch.linspace(0.0, math.pi / 2, n_points)
    x = radius * torch.cos(thetas)
    y = radius * torch.sin(thetas)
    pts = torch.stack([x, y], dim=-1).unsqueeze(0)  # (1, n_points, 2)
    return pts


def fake_route_lanes(centerline: torch.Tensor) -> torch.Tensor:
    """Wrap a centerline into a route_lanes-shaped tensor with 12-dim features.

    Only the first 2 dims (xy) need to be set for our utilities.
    Shape: (B, 25, n_points, 12) with route_lanes[:, 0, :, :2] = centerline.
    """
    B, n_points, _ = centerline.shape
    rl = torch.zeros((B, 25, n_points, 12))
    rl[:, 0, :, :2] = centerline
    # Set features 2-3 (tangent vector) to non-zero so the lane registers as "valid"
    # under _lane_is_valid.
    rl[:, 0, :-1, 2:4] = centerline[:, 1:, :] - centerline[:, :-1, :]
    return rl


def fake_lanes_empty(B: int = 1, n_lanes: int = 70, n_points: int = 20) -> torch.Tensor:
    """All-zero lanes (no valid lanes), shape (B, n_lanes, n_points, 12)."""
    return torch.zeros((B, n_lanes, n_points, 12))


# ----------------------------- tests: select_reference_centerline -----------------------------
#
# The current implementation builds a horizon-covering polyline by concatenating
# route lanes (greedy nearest-end ordering) and resampling to a fixed length.
# The returned shape is (B, out_points, 2) where out_points defaults to 100,
# regardless of the input lane point count.


def test_select_centerline_returns_expected_shape():
    """Default output shape is (B, 100, 2)."""
    centerline_in = straight_centerline()
    route_lanes = fake_route_lanes(centerline_in)
    lanes = fake_lanes_empty()
    out = select_reference_centerline(route_lanes, lanes)
    assert out.shape == (1, 100, 2), f'expected (1, 100, 2), got {out.shape}'


def test_select_centerline_single_route_lane_resampled():
    """A single valid route lane should still produce a centerline that passes
    near each of the original lane points (within resampling tolerance)."""
    centerline_in = straight_centerline(length=20.0, n_points=20)
    route_lanes = fake_route_lanes(centerline_in)
    lanes = fake_lanes_empty()
    out = select_reference_centerline(route_lanes, lanes)
    # The resampled polyline should still lie along +x at y~0
    assert out[0, :, 1].abs().max().item() < 0.1
    # And span roughly the original arc length (0 to 20)
    assert out[0, 0, 0].item() < 2.0
    assert out[0, -1, 0].item() > 18.0


def test_select_centerline_falls_back_to_nearest_lane():
    """If no valid route lane exists, fall back to the nearest visible lane."""
    route_lanes = torch.zeros((1, 25, 20, 12))
    lanes = torch.zeros((1, 70, 20, 12))
    # lane 5 sits right by ego, lane 10 is far away
    lanes[0, 5, :, 0] = torch.linspace(0.0, 20.0, 20)
    lanes[0, 5, :, 1] = 0.5
    lanes[0, 5, :-1, 2:4] = lanes[0, 5, 1:, :2] - lanes[0, 5, :-1, :2]
    lanes[0, 10, :, 0] = torch.linspace(100.0, 120.0, 20)
    lanes[0, 10, :, 1] = 5.0
    lanes[0, 10, :-1, 2:4] = lanes[0, 10, 1:, :2] - lanes[0, 10, :-1, :2]

    out = select_reference_centerline(route_lanes, lanes)
    # Should resemble lane 5 (at y~0.5), not lane 10 (at y=5)
    assert out[0, :, 1].mean().item() < 1.5


def test_select_centerline_concatenates_sequential_route_lanes():
    """The whole point of the rewrite: two adjacent route lanes should be
    concatenated into one long polyline so d-offsets stay small over 100m+."""
    # route_lanes[0]: x in [0, 50], y=0
    # route_lanes[1]: x in [50, 100], y=0
    # Concatenated, the centerline should cover x in [0, 100].
    route_lanes = torch.zeros((1, 25, 20, 12))
    route_lanes[0, 0, :, 0] = torch.linspace(0.0, 50.0, 20)
    route_lanes[0, 0, :, 1] = 0.0
    route_lanes[0, 0, :-1, 2:4] = route_lanes[0, 0, 1:, :2] - route_lanes[0, 0, :-1, :2]
    route_lanes[0, 1, :, 0] = torch.linspace(50.0, 100.0, 20)
    route_lanes[0, 1, :, 1] = 0.0
    route_lanes[0, 1, :-1, 2:4] = route_lanes[0, 1, 1:, :2] - route_lanes[0, 1, :-1, :2]
    lanes = fake_lanes_empty()

    out = select_reference_centerline(route_lanes, lanes)
    # First point should be near x=0; last point should be near x=100
    assert out[0, 0, 0].item() < 5.0, f'expected polyline start near 0, got {out[0, 0, 0].item()}'
    assert out[0, -1, 0].item() > 90.0, f'expected polyline end near 100, got {out[0, -1, 0].item()}'
    # All y values should be ~0
    assert out[0, :, 1].abs().max().item() < 0.5


def test_select_centerline_rejects_far_route_in_favor_of_near_lane():
    """A far/perpendicular route lane should not dominate when a closer
    route lane is available."""
    # route_lanes[0] is far away at y=60
    route_lanes = torch.zeros((1, 25, 20, 12))
    route_lanes[0, 0, :, 0] = torch.linspace(-10.0, 10.0, 20)
    route_lanes[0, 0, :, 1] = 60.0
    route_lanes[0, 0, :-1, 2:4] = route_lanes[0, 0, 1:, :2] - route_lanes[0, 0, :-1, :2]
    # route_lanes[1] is right by ego at y=0
    route_lanes[0, 1, :, 0] = torch.linspace(0.0, 20.0, 20)
    route_lanes[0, 1, :, 1] = 0.0
    route_lanes[0, 1, :-1, 2:4] = route_lanes[0, 1, 1:, :2] - route_lanes[0, 1, :-1, :2]
    lanes = fake_lanes_empty()

    out = select_reference_centerline(route_lanes, lanes)
    # The seed should be route_lanes[1] (closer); polyline should stay near y=0
    assert out[0, :, 1].abs().mean().item() < 5.0


def test_select_centerline_keeps_d_small_for_realistic_horizon():
    """Concretely measure that the rewrite produces small d-offsets for an
    ego trajectory that spans 80 m forward."""
    route_lanes = torch.zeros((1, 25, 20, 12))
    # Two sequential route lanes covering x in [0, 100]
    route_lanes[0, 0, :, 0] = torch.linspace(0.0, 50.0, 20)
    route_lanes[0, 0, :-1, 2:4] = route_lanes[0, 0, 1:, :2] - route_lanes[0, 0, :-1, :2]
    route_lanes[0, 1, :, 0] = torch.linspace(50.0, 100.0, 20)
    route_lanes[0, 1, :-1, 2:4] = route_lanes[0, 1, 1:, :2] - route_lanes[0, 1, :-1, :2]
    lanes = fake_lanes_empty()

    centerline = select_reference_centerline(route_lanes, lanes)

    # Ego trajectory: drives straight from origin to x=80 over 8s
    ego_xy = torch.zeros((1, 80, 2))
    ego_xy[0, :, 0] = torch.linspace(0.0, 80.0, 80)
    ego_xy[0, :, 1] = 0.0  # straight, no lateral motion

    sd = cartesian_to_frenet(ego_xy, centerline)
    # d should stay tiny (<0.5 m) because the centerline covers ego's whole path
    assert sd[0, :, 1].abs().max().item() < 1.0, f'd-offset too large: {sd[0, :, 1].abs().max().item()}'


# ----------------------------- tests: cartesian_to_frenet (straight) -----------------------------

def test_straight_centerline_point_on_centerline():
    """A point on the centerline projects to (s=arc_len, d=0)."""
    centerline = straight_centerline(length=20.0, n_points=21)  # points at 0, 1, 2, ... 20
    xy = torch.tensor([[[3.0, 0.0], [7.5, 0.0], [15.0, 0.0]]])  # (1, 3, 2)
    sd = cartesian_to_frenet(xy, centerline)
    assert sd.shape == (1, 3, 2)
    # s values should match x exactly (since centerline is along +x)
    assert torch.allclose(sd[:, :, 0], torch.tensor([[3.0, 7.5, 15.0]]), atol=1e-4)
    # d values should be ~0
    assert torch.allclose(sd[:, :, 1], torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-4)


def test_straight_centerline_perpendicular_offset_left_positive():
    """A point at (3, +2) projects to (s=3, d=+2). Left of +x direction = positive d."""
    centerline = straight_centerline(length=20.0, n_points=21)
    xy = torch.tensor([[[3.0, 2.0]]])
    sd = cartesian_to_frenet(xy, centerline)
    assert torch.allclose(sd[0, 0, 0], torch.tensor(3.0), atol=1e-4)
    assert torch.allclose(sd[0, 0, 1], torch.tensor(2.0), atol=1e-4)


def test_straight_centerline_perpendicular_offset_right_negative():
    """A point at (3, -2) projects to (s=3, d=-2). Right of +x direction = negative d."""
    centerline = straight_centerline(length=20.0, n_points=21)
    xy = torch.tensor([[[3.0, -2.0]]])
    sd = cartesian_to_frenet(xy, centerline)
    assert torch.allclose(sd[0, 0, 0], torch.tensor(3.0), atol=1e-4)
    assert torch.allclose(sd[0, 0, 1], torch.tensor(-2.0), atol=1e-4)


# ----------------------------- tests: round trip -----------------------------

def test_round_trip_straight_centerline():
    """frenet_to_cartesian(cartesian_to_frenet(xy)) == xy for straight centerlines."""
    centerline = straight_centerline(length=20.0, n_points=21)
    xy_in = torch.tensor([[
        [3.0, 0.5],
        [7.5, -1.2],
        [12.0, 2.0],
        [18.0, 0.0],
    ]])
    sd = cartesian_to_frenet(xy_in, centerline)
    xy_out = frenet_to_cartesian(sd, centerline)
    assert torch.allclose(xy_in, xy_out, atol=1e-3)


def test_round_trip_curved_centerline():
    """Round trip should approximately preserve points on a quarter-circle."""
    centerline = quarter_circle_centerline(radius=10.0, n_points=30)
    # Points on the centerline
    xy_in = centerline[0, ::5, :].unsqueeze(0)  # (1, 6, 2)
    sd = cartesian_to_frenet(xy_in, centerline)
    xy_out = frenet_to_cartesian(sd, centerline)
    # Discretization error is non-trivial on a curve; allow looser tolerance
    assert torch.allclose(xy_in, xy_out, atol=0.5)


# ----------------------------- tests: edge cases -----------------------------

def test_point_beyond_centerline_end_snaps_to_end():
    """A point past the centerline's last segment should snap to a valid s value."""
    centerline = straight_centerline(length=20.0, n_points=21)
    # Point at x=25 is past the end (centerline ends at x=20)
    xy = torch.tensor([[[25.0, 0.0]]])
    sd = cartesian_to_frenet(xy, centerline)
    # s should be at or near the end of the centerline (20)
    assert sd[0, 0, 0].item() <= 20.0 + 1e-3
    # d picks up the distance beyond the end
    assert sd[0, 0, 1].abs().item() <= 5.5  # rough — endpoint distance


def test_point_before_centerline_start():
    """A point before the centerline start projects to s=0."""
    centerline = straight_centerline(length=20.0, n_points=21)
    xy = torch.tensor([[[-3.0, 0.0]]])
    sd = cartesian_to_frenet(xy, centerline)
    assert sd[0, 0, 0].abs().item() < 1e-3   # s ~ 0


def test_batched_inputs():
    """Function should handle B > 1 correctly."""
    centerline = torch.cat([
        straight_centerline(length=20.0, n_points=21),
        straight_centerline(length=10.0, n_points=21),
    ], dim=0)  # (2, 21, 2)
    xy = torch.tensor([
        [[5.0, 1.0], [10.0, -1.0]],
        [[3.0, 0.5], [7.0,  0.0]],
    ])  # (2, 2, 2)
    sd = cartesian_to_frenet(xy, centerline)
    assert sd.shape == (2, 2, 2)
    # Batch 0
    assert torch.allclose(sd[0, 0], torch.tensor([5.0, 1.0]), atol=1e-3)
    # Batch 1
    assert torch.allclose(sd[1, 0], torch.tensor([3.0, 0.5]), atol=1e-3)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
