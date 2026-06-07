"""V4: Multi-frame inference ensemble.

At inference, sample K trajectories with different noise seeds, then average
(or softmax-weight) them. The intuition: a single noise draw can pick an
unlucky mode of the flow-matching distribution; averaging K draws moves the
sample toward the conditional mean. Cheaper analogue of best-of-N (V5) — no
verifier needed, just averaging.

Env vars:
    FRENET_INFERENCE_K          int, default 4. Number of inference samples.
    FRENET_INFERENCE_TEMPERATURE
        float, default 1.0. If softmax-weighting is enabled, this controls
        how peaked the weighting is over per-sample mean(|d|) scores. Lower
        temperature = sharper weighting (closer to argmin). Set to 0.0 to
        use uniform averaging.

Activation:
    Set FRENET_INFERENCE_HOOK=flow_planner.inference.hooks.multiframe
    before running inference_eval.py. The hook then monkey-patches the
    FlowPlanner's forward_inference at import time.
"""
from __future__ import annotations

import os
import torch


def _get_k() -> int:
    raw = os.environ.get('FRENET_INFERENCE_K', '4')
    try:
        return max(1, int(raw))
    except ValueError:
        print(f'[multiframe hook] WARNING: invalid FRENET_INFERENCE_K={raw!r}; falling back to default 4.')
        return 4


def _get_temperature() -> float:
    raw = os.environ.get('FRENET_INFERENCE_TEMPERATURE', '1.0')
    try:
        return max(0.0, float(raw))
    except ValueError:
        print(f'[multiframe hook] WARNING: invalid FRENET_INFERENCE_TEMPERATURE={raw!r}; falling back to default 1.0.')
        return 1.0


def apply(model) -> None:
    """Monkey-patch FlowPlanner.forward_inference on the given model.

    Each call to forward_inference will now sample K trajectories internally
    and return the weighted average. This is transparent to inference_eval —
    the returned shape is the same as the single-sample version.
    """
    from flow_planner.model.flow_planner_model.flow_planner import FlowPlanner

    K = _get_k()
    T = _get_temperature()
    if K <= 1:
        print('[multiframe hook] K=1; no-op (single sample).')
        return

    original_inference = FlowPlanner.forward_inference

    def multiframe_inference(self, data, use_cfg=True, cfg_weight=None):
        samples = []
        # We rely on Python-level RNG seeding to vary draws. The model's
        # forward_inference reads from torch.randn for its noise init, so
        # advancing the global RNG between calls suffices.
        for k in range(K):
            # Use distinct seed per draw so K samples are independent.
            torch.manual_seed(269 + k * 1000)
            sample = original_inference(self, data, use_cfg=use_cfg, cfg_weight=cfg_weight)
            samples.append(sample)
        stacked = torch.stack(samples, dim=0)   # (K, B, T, D) or (K, B, ...)

        if T <= 1e-6:
            # Uniform average over K samples.
            return stacked.mean(dim=0)

        # Softmax-weighted average over per-sample mean(|d|). Lower |d| =
        # closer to centerline = higher weight. Channel 1 is the d-channel
        # in (s, d, cos_h, sin_h) Frenet target. This shape assumption is
        # specific to Frenet kinematic; non-Frenet runs should not enable
        # this hook (it would still work but the weighting becomes a
        # pseudo-confidence over channel 1, whatever that is).
        if stacked.dim() < 3 or stacked.shape[-1] < 2:
            return stacked.mean(dim=0)
        d_abs_mean = stacked[..., 1].abs().mean(dim=tuple(range(2, stacked.dim() - 1)))  # (K, B)
        # Negate so smaller |d| → larger logit. Divide by T for sharpness.
        logits = -d_abs_mean / max(T, 1e-3)
        weights = torch.softmax(logits, dim=0)  # (K, B)
        # Broadcast weights to match stacked: (K, B, 1, ..., 1)
        while weights.dim() < stacked.dim():
            weights = weights.unsqueeze(-1)
        return (stacked * weights).sum(dim=0)

    FlowPlanner.forward_inference = multiframe_inference
    print(f'[multiframe hook] FlowPlanner.forward_inference patched: K={K}, T={T:.2f}')
