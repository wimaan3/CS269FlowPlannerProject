"""Regression tests for the CFG batch-alignment fix.

Before the fix three independent batching operations used different orderings:
  - ``NuPlanDataSample.repeat(2)`` was ``repeat_interleave`` (``[a,a,b,b,...]``)
  - ``VelocityModel.forward`` does ``x.repeat(2, ...)`` (tile / cat — ``[a,b,a,b]``)
  - ``cfg_flags`` is built with ``cat([ones(B), zeros(B)])`` (block — ``[1,1,...,0,0]``)
At B>1 the three orderings collide: each sample denoises against another
scene's encoder context, and the CFG mix combines predictions for different
scenes. Invisible at B=1 (where repeat_interleave == cat). The fix makes
``NuPlanDataSample.repeat`` tile-style so every consumer agrees.
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

from flow_planner.data.dataset.nuplan import NuPlanDataSample


def _make_batched_sample(B: int = 3) -> NuPlanDataSample:
    """Build a batched sample where every field's batch axis stores the
    batch index as the first feature, so the ordering after `.repeat(2)` is
    directly readable from the tensor values."""
    ego_past = torch.arange(B, dtype=torch.float32).view(B, 1, 1).expand(B, 21, 14).contiguous()
    ego_current = torch.arange(B, dtype=torch.float32).view(B, 1).expand(B, 16).contiguous()
    ego_future = torch.arange(B, dtype=torch.float32).view(B, 1, 1).expand(B, 80, 3).contiguous()
    neighbor_past = torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1).expand(B, 32, 21, 11).contiguous()
    neighbor_future = torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1).expand(B, 10, 80, 4).contiguous()
    neighbor_future_observed = torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1).expand(B, 10, 80, 1).contiguous()
    lanes = torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1).expand(B, 70, 20, 12).contiguous()
    lanes_speedlimit = torch.arange(B, dtype=torch.float32).view(B, 1, 1).expand(B, 70, 1).contiguous()
    lanes_has_speedlimit = torch.zeros((B, 70, 1), dtype=torch.bool)
    routes = torch.arange(B, dtype=torch.float32).view(B, 1, 1, 1).expand(B, 25, 20, 12).contiguous()
    routes_speedlimit = torch.arange(B, dtype=torch.float32).view(B, 1, 1).expand(B, 25, 1).contiguous()
    routes_has_speedlimit = torch.zeros((B, 25, 1), dtype=torch.bool)
    map_objects = torch.arange(B, dtype=torch.float32).view(B, 1, 1).expand(B, 5, 10).contiguous()
    return NuPlanDataSample(
        batched=True,
        ego_past=ego_past,
        ego_current=ego_current,
        ego_future=ego_future,
        neighbor_past=neighbor_past,
        neighbor_future=neighbor_future,
        neighbor_future_observed=neighbor_future_observed,
        lanes=lanes,
        lanes_speedlimit=lanes_speedlimit,
        lanes_has_speedlimit=lanes_has_speedlimit,
        routes=routes,
        routes_speedlimit=routes_speedlimit,
        routes_has_speedlimit=routes_has_speedlimit,
        map_objects=map_objects,
    )


def test_repeat_is_tile_not_interleave():
    """For B=3, NuPlanDataSample.repeat(2) must produce TILE ordering
    [0, 1, 2, 0, 1, 2], NOT interleave [0, 0, 1, 1, 2, 2]. This matches
    the layout that cfg_flags = cat([ones(B), zeros(B)]) and
    VelocityModel.x.repeat(2, ...) use, so all three consumers of the
    doubled batch dim agree on which sample is at index k.
    """
    B = 3
    sample = _make_batched_sample(B=B)
    doubled = sample.repeat(2)

    # Each tensor's batch axis should be [0, 1, 2, 0, 1, 2]
    expected_ego_current = torch.tensor(
        [[i] * 16 for i in [0, 1, 2, 0, 1, 2]], dtype=torch.float32
    )
    assert doubled.ego_current.shape[0] == 2 * B
    assert torch.equal(doubled.ego_current, expected_ego_current), (
        f"ego_current ordering wrong: got first column "
        f"{doubled.ego_current[:, 0].tolist()}, expected [0,1,2,0,1,2]"
    )

    # Sanity: same ordering should appear on lanes (a different rank tensor)
    lanes_first_col = doubled.lanes[:, 0, 0, 0].tolist()
    assert lanes_first_col == [0, 1, 2, 0, 1, 2], (
        f"lanes batch ordering wrong: {lanes_first_col}"
    )


def test_repeat_at_b1_is_identical_to_interleave():
    """At B=1, tile and interleave coincide — this is why the pre-fix bug
    was invisible from the only call site that defaults use_cfg=True
    (nuplan_simulation/planner.py at B=1). Pin the equivalence as a
    sanity check."""
    sample = _make_batched_sample(B=1)
    doubled = sample.repeat(2)
    assert doubled.ego_current.shape[0] == 2
    # Both rows should be sample 0
    assert torch.equal(doubled.ego_current[0], doubled.ego_current[1])


def test_cfg_layout_consumers_agree_at_b2():
    """End-to-end ordering check: with B=2 and use_cfg=True the three
    competing layouts (data.repeat(2), cfg_flags via cat, x.repeat(2, ...))
    must yield the SAME batch identity at every index k in [0, 2B).
    """
    B = 2
    sample = _make_batched_sample(B=B)
    doubled = sample.repeat(2)

    # cfg_flags layout (as built in FlowPlanner.forward_inference)
    cfg_flags = torch.cat([torch.ones(B), torch.zeros(B)], dim=0)
    # x layout (as built in VelocityModel.forward)
    x = torch.arange(B, dtype=torch.float32).view(B, 1)
    x_doubled = x.repeat(2, 1)

    # Both should match the data ordering: [s0, s1, s0, s1]
    data_ids = doubled.ego_current[:, 0].tolist()
    x_ids = x_doubled[:, 0].tolist()
    assert data_ids == [0, 1, 0, 1] == x_ids, (
        f"layouts disagree: data={data_ids}, x={x_ids}"
    )
    # cfg_flags should be [1, 1, 0, 0] — first B conditioned, second B unconditioned
    assert cfg_flags.tolist() == [1.0, 1.0, 0.0, 0.0]


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
