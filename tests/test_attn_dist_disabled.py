"""Regression tests for the enable_attn_dist=False code path.

The v7-AUDIT Frenet training run hit a crash: when enable_attn_dist=False
(audit Patch 5), JointAttention sets gen_taus=None → taus=None at forward
time, but the encoder still forwards token_dist as attn_dist, so attn_bias
arrived non-None at Attend.forward. The original guard
`if attn_bias is not None:` then tried `None * Tensor.unsqueeze(-1)` → TypeError.

These tests pin the fixed behavior so the same crash can't regress.
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

# global_attention.py imports einops, which is installed inside the Colab
# flow_planner venv but not necessarily in the local pre-commit env. Skip
# this module's tests cleanly when einops is unavailable rather than failing
# them — Colab CI is the source of truth for the model-internal regression
# checks (it has einops + torch + the full nuplan stack).
einops = pytest.importorskip('einops')


def test_attend_handles_taus_none_with_attn_bias_present():
    """Attend.forward must not crash when taus=None even if attn_bias is given."""
    from flow_planner.model.flow_planner_model.global_attention import BiasedAttention

    attend = BiasedAttention(dropout=0.0)
    B, H, N, D = 2, 4, 8, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    attn_bias = torch.randn(B, N, N)  # token_dist style — symmetric distances

    # taus=None reproduces the v7-AUDIT failure mode. Pre-fix this raised
    # TypeError: unsupported operand type(s) for *: 'NoneType' and 'Tensor'.
    out = attend(q=q, k=k, v=v, mask=None, taus=None, attn_bias=attn_bias)
    assert out.shape == (B, H, N, D), f'expected (B,H,N,D), got {out.shape}'
    assert not torch.isnan(out).any(), 'NaNs in Attend output'


def test_attend_still_uses_taus_when_both_present():
    """Sanity: when both taus and attn_bias are given, the bias modulates sim."""
    from flow_planner.model.flow_planner_model.global_attention import BiasedAttention

    torch.manual_seed(0)
    attend = BiasedAttention(dropout=0.0)
    B, H, N, D = 2, 4, 8, 16
    q = torch.randn(B, H, N, D)
    k = torch.randn(B, H, N, D)
    v = torch.randn(B, H, N, D)
    attn_bias = torch.randn(B, N, N)
    taus_zero = torch.zeros(B, N, N, H)
    taus_one = torch.ones(B, N, N, H)

    out_zero = attend(q=q, k=k, v=v, mask=None, taus=taus_zero, attn_bias=attn_bias)
    out_one = attend(q=q, k=k, v=v, mask=None, taus=taus_one, attn_bias=attn_bias)
    # Different taus → different output (the bias actually flows through)
    assert not torch.allclose(out_zero, out_one, atol=1e-6), (
        'Attend output is identical for taus=0 vs taus=1 — bias path is dead'
    )


def test_joint_attention_disabled_full_forward():
    """End-to-end: JointAttention with enable_attn_dist=False must succeed
    when the caller forwards a non-None attn_dist (mirrors what the
    FlowPlannerDecoder does in production — see decoder.py:117,143).
    """
    from flow_planner.model.flow_planner_model.global_attention import JointAttention

    torch.manual_seed(0)
    dim_inputs = (64, 64, 64)
    token_num = 30  # 10 tokens per modality so total = 30
    ja = JointAttention(
        dim_inputs=dim_inputs,
        dim_head=16,
        heads=4,
        enable_attn_dist=False,
        token_num=token_num,
    )
    ja.eval()

    # Three modality inputs, 10 tokens each
    inputs = tuple(torch.randn(2, 10, d) for d in dim_inputs)
    # token_dist is computed unconditionally upstream and passed through
    attn_dist = torch.randn(2, token_num, token_num)

    with torch.no_grad():
        outs = ja(inputs=inputs, masks=None, attn_dist=attn_dist)

    assert len(outs) == len(dim_inputs)
    for out, dim in zip(outs, dim_inputs):
        assert out.shape == (2, 10, dim), f'got {out.shape}, expected (2, 10, {dim})'
        assert not torch.isnan(out).any(), 'NaNs in JointAttention output'


def test_joint_attention_enabled_still_works():
    """Sanity: enable_attn_dist=True (the YAML default) keeps working."""
    from flow_planner.model.flow_planner_model.global_attention import JointAttention

    torch.manual_seed(0)
    dim_inputs = (64, 64, 64)
    token_num = 30
    ja = JointAttention(
        dim_inputs=dim_inputs,
        dim_head=16,
        heads=4,
        enable_attn_dist=True,
        token_num=token_num,
    )
    ja.eval()
    inputs = tuple(torch.randn(2, 10, d) for d in dim_inputs)
    attn_dist = torch.randn(2, token_num, token_num)
    with torch.no_grad():
        outs = ja(inputs=inputs, masks=None, attn_dist=attn_dist)
    for out, dim in zip(outs, dim_inputs):
        assert out.shape == (2, 10, dim)
        assert not torch.isnan(out).any()


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
