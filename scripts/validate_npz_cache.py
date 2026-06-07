"""Validate a preprocessed nuPlan .npz cache and write a training manifest.

For every `.npz` in CACHE_DIR, this script:
  1. checks numpy can parse it
  2. checks all expected schema keys are present
  3. checks no NaN / Inf in the agent + lane channels
  4. checks ego is at exact origin (the preprocessor centres ego by
     construction — anything > EGO_ORIGIN_TOL m signals a coordinate-frame
     mismatch)
  5. checks ego future displacement < MAX_FUTURE_DISPLACEMENT m over 8 s
     (catches order-of-magnitude scale errors)

Files that fail are EXCLUDED from the output manifest (quarantined). If
more than FAIL_THRESHOLD_FRACTION of files fail, the script exits non-zero
so the notebook stops before training on a corrupt cache.

Used by Section 7c (train cache, cell 25) and Section 8c (held-out cache,
cell 32) — single source of truth for what counts as a valid scenario.

Parameters via environment variables:
    CACHE_DIR         Directory containing preprocessed .npz files
    OUTPUT_MANIFEST   Path to write the JSON manifest of good files
    FAIL_THRESHOLD    Fraction of files allowed to fail (default 0.05)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

# Schema as the .npz files actually use it. Ground truth: NuPlanDataset.__getitem__
# at flow_planner/data/dataset/nuplan.py:244 reads exactly these keys.
EXPECTED_KEYS = {
    "ego_agent_past", "ego_agent_future", "ego_current_state",
    "neighbor_agents_past", "neighbor_agents_future",
    "lanes", "lanes_speed_limit", "lanes_has_speed_limit",
    "route_lanes", "route_lanes_speed_limit", "route_lanes_has_speed_limit",
    "static_objects",
}

# The preprocessor centres ego at exactly (0, 0) by construction (see
# Diffusion-Planner / Flow Planner data_processor). Allow only numerical-noise
# slack — a larger value would silently pass coordinate-frame mismatches.
EGO_ORIGIN_TOL = 0.01

# nuPlan future horizon is 8 s; ~25 m/s practical speed cap implies ~200 m.
# Above this implies a scale error.
MAX_FUTURE_DISPLACEMENT = 200.0


CACHE_DIR = Path(os.environ["CACHE_DIR"])
OUTPUT_MANIFEST = Path(os.environ["OUTPUT_MANIFEST"])
FAIL_THRESHOLD_FRACTION = float(os.environ.get("FAIL_THRESHOLD", "0.05"))

npz_files = sorted(CACHE_DIR.glob("*.npz"))
print(f"Validating ALL {len(npz_files)} .npz files in {CACHE_DIR}...")
print()

if not npz_files:
    print(f"ERROR: no .npz files in {CACHE_DIR}", file=sys.stderr)
    sys.exit(1)

n_bad_load = 0
n_missing_key = 0
n_nan_inf = 0
n_weird_scale = 0
n_origin_off = 0
n_route_empty = 0  # soft warning — not quarantined
bad_files: set[str] = set()
bad_examples: list[tuple[str, str]] = []
route_empty_examples: list[str] = []


def _flag(name: str, reason: str) -> None:
    bad_files.add(name)
    if len(bad_examples) < 10:
        bad_examples.append((name, reason))


for f in npz_files:
    try:
        d = np.load(f, allow_pickle=True)
    except Exception as e:
        n_bad_load += 1
        _flag(f.name, f"load failed: {e}")
        continue

    missing = EXPECTED_KEYS - set(d.files)
    if missing:
        n_missing_key += 1
        _flag(f.name, f"missing keys: {sorted(missing)}")
        continue

    has_nan_or_inf = False
    for k in ["ego_agent_past", "ego_agent_future", "ego_current_state",
              "neighbor_agents_past", "lanes", "route_lanes"]:
        arr = d[k]
        if np.isnan(arr).any() or np.isinf(arr).any():
            has_nan_or_inf = True
            _flag(f.name, f"NaN/Inf in {k}")
            break
    if has_nan_or_inf:
        n_nan_inf += 1
        continue

    current = d["ego_current_state"]
    if abs(current[0]) > EGO_ORIGIN_TOL or abs(current[1]) > EGO_ORIGIN_TOL:
        n_origin_off += 1
        _flag(f.name, f"ego_current[:2] = {tuple(current[:2])}, expected near (0,0)")

    future = d["ego_agent_future"]
    max_dist = float(np.linalg.norm(future[:, :2], axis=-1).max())
    if max_dist > MAX_FUTURE_DISPLACEMENT:
        n_weird_scale += 1
        _flag(f.name, f"max future displacement = {max_dist:.1f}m (>{MAX_FUTURE_DISPLACEMENT})")

    # SOFT WARNING: all-zero route_lanes signals nuPlan returned no
    # route assignment (route_roadblock_ids == ['']) and map_process
    # produced an all-zero route_lanes tensor. For Frenet, the
    # centerline selector then falls back to nearest non-route lane
    # (or worse, a synthetic +x ray), which makes the training target
    # degenerate. We do NOT quarantine because the model may still
    # learn a useful waypoints target — but we want to know how much
    # of the cache is in this state so it can be filtered separately
    # if Frenet d-std looks inflated.
    route_lanes = d["route_lanes"]
    if not np.any(route_lanes):
        n_route_empty += 1
        if len(route_empty_examples) < 10:
            route_empty_examples.append(f.name)


print(f"Results across {len(npz_files)} files:")
print(f"  Failed to load:       {n_bad_load}")
print(f"  Missing keys:         {n_missing_key}")
print(f"  Contained NaN/Inf:    {n_nan_inf}")
print(f"  Weird scale:          {n_weird_scale}")
print(f"  Ego not at origin:    {n_origin_off}")
print(f"  Total bad:            {len(bad_files)}")
print(f"  All-zero route_lanes: {n_route_empty} (soft warning — not quarantined)")
if route_empty_examples:
    print("    Examples:")
    for name in route_empty_examples:
        print(f"      {name}")

if bad_examples:
    print("\nFirst bad files / reasons:")
    for name, reason in bad_examples:
        print(f"  {name}: {reason}")

good_files = [p.name for p in npz_files if p.name not in bad_files]
OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_MANIFEST.write_text(json.dumps(good_files))
print()
print(f"Wrote {OUTPUT_MANIFEST} with {len(good_files)} good entries "
      f"({len(bad_files)} quarantined)")

# Detailed view of the first known-good file (one sample is enough for a
# smoke check; production reads many).
if good_files:
    sample = np.load(CACHE_DIR / good_files[0], allow_pickle=True)
    print(f"\n=== Detailed view of {good_files[0]} ===")
    for k in sorted(sample.files):
        a = sample[k]
        # Only print range/mean stats for numeric arrays. Some cache fields
        # are string arrays (e.g. scenario_token as np.dtype('<U12')) and
        # numpy's min/max ufuncs don't have a loop for those types.
        if a.dtype.kind in 'biufc':  # bool / int / uint / float / complex
            print(f"  {k:30s} shape={str(a.shape):20s} dtype={a.dtype} "
                  f"range=[{a.min():.2f}, {a.max():.2f}] mean={a.mean():.2f}")
        else:
            print(f"  {k:30s} shape={str(a.shape):20s} dtype={a.dtype}")

fail_threshold = max(1, int(FAIL_THRESHOLD_FRACTION * len(npz_files)))
if len(bad_files) > fail_threshold:
    print(file=sys.stderr)
    print(
        f"ERROR: too many bad files: {len(bad_files)} / {len(npz_files)} "
        f"(threshold: {fail_threshold} = {FAIL_THRESHOLD_FRACTION:.0%}). "
        f"Stop and investigate before training.",
        file=sys.stderr,
    )
    sys.exit(1)
