"""V5: Best-of-N=16 verifier-reranked inference.

At inference, sample N trajectories with different noise seeds, then SELECT
(argmin/argmax) the one with the best verifier score rather than averaging.
The verifier here is mean(|d|): smaller lateral offset from the chosen
centerline = better trajectory.

This is the closest no-retrain analog to DAgger's "correct off-distribution
behavior" — instead of training the model to avoid bad samples, we sample
many and discard the bad ones.

Env vars:
    FRENET_INFERENCE_N         int, default 16. Number of samples per scenario.
    FRENET_INFERENCE_VERIFIER  str, default 'mean_abs_d'. Verifier choice:
        - 'mean_abs_d'  — pick sample with smallest mean(|d_pred|)
        - 'max_abs_d'   — pick sample with smallest max(|d_pred|) (strictest)
        - 'arc_length'  — pick sample with largest **peak** s_pred (i.e.
          `s_pred.amax` over all timesteps, not `s_pred[-1]`). For a
          predicted s that is non-monotonic — entirely possible from a
          flow-matching sampler that briefly overshoots and retreats —
          this rewards the sample that poked furthest forward, NOT the
          sample whose final position is furthest along the centerline.
          If you want last-timestep semantics instead, change the
          implementation in `_score`.
    FRENET_INFERENCE_VERIFIER_ALPHA
        float, default 1.0. Mixing coefficient if verifier='mean_abs_d':
        score = alpha * mean(|d|) + (1 - alpha) * max(|d|).

Activation:
    Set FRENET_INFERENCE_HOOK=flow_planner.inference.hooks.best_of_n
    before running inference_eval.py.
"""
from __future__ import annotations

import os
import torch


def _get_n() -> int:
    raw = os.environ.get('FRENET_INFERENCE_N', '16')
    try:
        return max(1, int(raw))
    except ValueError:
        print(f'[best_of_n hook] WARNING: invalid FRENET_INFERENCE_N={raw!r}; falling back to default 16.')
        return 16


def _get_verifier() -> str:
    return os.environ.get('FRENET_INFERENCE_VERIFIER', 'mean_abs_d')


def _get_alpha() -> float:
    raw = os.environ.get('FRENET_INFERENCE_VERIFIER_ALPHA', '1.0')
    try:
        # Clamped to [0, 1] — see docstring at module top.
        return min(1.0, max(0.0, float(raw)))
    except ValueError:
        print(f'[best_of_n hook] WARNING: invalid FRENET_INFERENCE_VERIFIER_ALPHA={raw!r}; falling back to default 1.0.')
        return 1.0


def _score(sample: torch.Tensor, verifier: str, alpha: float) -> torch.Tensor:
    """Compute a per-batch-element score where LOWER = better.

    Args:
        sample: (B, ..., D) where D >= 2 (channel 0 = s, channel 1 = d).

    Returns:
        (B,) tensor of scores. Lower = better trajectory.
    """
    # Reduce over all dims except batch (dim 0) and channel (dim -1).
    reduce_dims = tuple(range(1, sample.dim() - 1))
    if verifier == 'mean_abs_d':
        mean_d = sample[..., 1].abs().mean(dim=reduce_dims)
        max_d = sample[..., 1].abs().amax(dim=reduce_dims)
        return alpha * mean_d + (1.0 - alpha) * max_d
    elif verifier == 'max_abs_d':
        return sample[..., 1].abs().amax(dim=reduce_dims)
    elif verifier == 'arc_length':
        # We want the LARGEST s magnitude, so negate.
        return -sample[..., 0].amax(dim=reduce_dims)
    else:
        raise ValueError(f'Unknown verifier: {verifier!r}')


def apply(model) -> None:
    """Monkey-patch FlowPlanner.forward_inference to sample N + argmin verifier."""
    from flow_planner.model.flow_planner_model.flow_planner import FlowPlanner

    N = _get_n()
    verifier = _get_verifier()
    alpha = _get_alpha()
    if N <= 1:
        print('[best_of_n hook] N=1; no-op.')
        return

    original_inference = FlowPlanner.forward_inference

    def best_of_n_inference(self, data, use_cfg=True, cfg_weight=None):
        samples = []
        scores = []
        for n in range(N):
            torch.manual_seed(269 + n * 1000)
            sample = original_inference(self, data, use_cfg=use_cfg, cfg_weight=cfg_weight)
            samples.append(sample)
            scores.append(_score(sample, verifier, alpha))
        stacked = torch.stack(samples, dim=0)           # (N, B, ..., D)
        score_tensor = torch.stack(scores, dim=0)       # (N, B)
        best_idx = score_tensor.argmin(dim=0)           # (B,)

        # Gather per-batch the chosen sample.
        B = best_idx.shape[0]
        chosen = torch.stack([stacked[best_idx[b], b] for b in range(B)], dim=0)
        return chosen

    FlowPlanner.forward_inference = best_of_n_inference
    print(f'[best_of_n hook] FlowPlanner.forward_inference patched: N={N}, verifier={verifier!r}, alpha={alpha:.2f}')
