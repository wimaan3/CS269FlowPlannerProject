"""Measure Frenet (s, d, cos_h, sin_h) target distribution from a preprocessed cache.

For Phase 2 / v6 retrain we need `frenet_norm_stats.yaml` values measured
from the actual training data under the post-PR-#1 fix (raw-frame
projection). The previous values were measured under the buggy frame and
gave d-std ~16 m instead of the sub-metre values we expect.

For every `.npz` in CACHE_DIR (or NUM_FILES sample), this script:
  1. Reads `ego_agent_future`, `lanes`, `route_lanes` (same fields that
     ModelInputProcessor reads at training time).
  2. Calls the same `select_reference_centerline` + `cartesian_to_frenet`
     that runs in production inside the model.
  3. Aggregates per-timestep (s, d) values across all files.
  4. Writes a JSON file with the per-channel statistics.

The (cos_h, sin_h) target channels are kept at the same unit-std convention
as `waypoints_norm_stats.yaml` (`mean=[1, 0]`, `std=[1.0, 1.0]`). Using
std=0.3 (as in the original Frenet YAML) inflates the heading gradient
~11x relative to the (s, d) channels in the unweighted sum-over-channels
MSE. These are
geometric properties of the ego-centric coordinate convention and do not
change with the Frenet fix.

Parameters via environment variables (cell 38 sets these):
    FP_DIR        Flow Planner source dir (added to sys.path)
    CACHE_DIR     Directory containing preprocessed .npz files
    OUTPUT_JSON   Where to write the measured stats
    NUM_FILES     Optional cap on number of files (0 or unset = all files)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make flow_planner importable from the editable checkout.
sys.path.insert(0, os.environ["FP_DIR"])

import numpy as np
import torch

from flow_planner.data.normalization.frenet_utils import (  # noqa: E402
    cartesian_to_frenet,
    select_reference_centerline,
)

CACHE_DIR = Path(os.environ["CACHE_DIR"])
OUTPUT_JSON = Path(os.environ["OUTPUT_JSON"])
NUM_FILES = int(os.environ.get("NUM_FILES", "0") or "0")

npz_files = sorted(CACHE_DIR.glob("*.npz"))
if NUM_FILES > 0:
    npz_files = npz_files[:NUM_FILES]

if not npz_files:
    print(f"ERROR: no .npz files in {CACHE_DIR}", file=sys.stderr)
    sys.exit(1)

print(f"Measuring Frenet stats over {len(npz_files)} scenarios "
      f"(raw-frame projection, post-PR-#1 code path)...")

all_s: list[np.ndarray] = []
all_d: list[np.ndarray] = []
n_skipped = 0

for p in npz_files:
    try:
        data = np.load(p)
        ego_future = torch.from_numpy(data["ego_agent_future"]).to(torch.float32).unsqueeze(0)
        lanes = torch.from_numpy(data["lanes"]).to(torch.float32).unsqueeze(0)
        routes = torch.from_numpy(data["route_lanes"]).to(torch.float32).unsqueeze(0)
        centerline = select_reference_centerline(routes, lanes)
        sd = cartesian_to_frenet(ego_future[..., :2], centerline)
        all_s.append(sd[0, :, 0].numpy())
        all_d.append(sd[0, :, 1].numpy())
    except Exception as e:
        n_skipped += 1
        if n_skipped <= 5:
            print(f"  skipped {p.name}: {e}", file=sys.stderr)

if not all_s:
    print(f"ERROR: every file failed to process ({n_skipped} skipped)", file=sys.stderr)
    sys.exit(1)

s = np.concatenate(all_s)
d = np.concatenate(all_d)

s_stats = {
    "mean": float(s.mean()),
    "std":  float(s.std()),
    "min":  float(s.min()),
    "max":  float(s.max()),
}
d_stats = {
    "mean": float(d.mean()),
    "std":  float(d.std()),
    "min":  float(d.min()),
    "max":  float(d.max()),
}

stats = {
    "n_files":       len(npz_files),
    "n_skipped":     n_skipped,
    "n_timesteps":   int(len(s)),
    "s":             s_stats,
    "d":             d_stats,
    # Heading channels — geometric properties of the ego-centric frame.
    # Use std=1.0 to match waypoints_norm_stats.yaml convention; std=0.3
    # (the previous value) inflated the heading gradient ~11x under
    # sum-over-channels MSE (audit Finding 1).
    "cos_h_mean":    1.0,
    "cos_h_std":     1.0,
    "sin_h_mean":    0.0,
    "sin_h_std":     1.0,
}

print()
print(f"s: mean={s_stats['mean']:.2f}, std={s_stats['std']:.2f}, "
      f"min={s_stats['min']:.2f}, max={s_stats['max']:.2f}")
print(f"d: mean={d_stats['mean']:.2f}, std={d_stats['std']:.2f}, "
      f"min={d_stats['min']:.2f}, max={d_stats['max']:.2f}")
print(f"skipped: {n_skipped}/{len(npz_files)} files")

if d_stats["std"] > 8.0:
    print()
    print("WARNING: d-std is still high (>8 m). The PR #1 fix may not be in effect.")
    print("         Verify scripts/check_sanity.py passes before trusting these stats.")
elif d_stats["std"] > 3.0:
    print()
    print(f"NOTE: d-std = {d_stats['std']:.2f} m. Higher than expected for a clean")
    print("      log-disjoint train cache; investigate which scenarios contribute.")
else:
    print()
    print(f"d-std = {d_stats['std']:.2f} m looks reasonable for the post-fix code.")

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON.write_text(json.dumps(stats, indent=2))
print()
print(f"wrote {OUTPUT_JSON}")
