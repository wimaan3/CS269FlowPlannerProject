"""Update flow_planner/script/normalization_stats/frenet_norm_stats.yaml from
measured Frenet statistics.

Reads stats from STATS_JSON (produced by scripts/measure_frenet_stats.py),
updates the `ego.uniform` and `neighbor.uniform` blocks with the measured
(s, d) values, leaves every other block (ego_past, ego_current,
neighbor_past, map_objects, lanes, lanes_speedlimit, routes,
routes_speedlimit) UNCHANGED — those are obs-side normalization values
that don't depend on the Frenet target.

Backs the original YAML up to `<name>.yaml.bak` before overwriting so the
previous values can be recovered if the measurement turns out wrong.

Parameters via environment variables:
    FP_DIR       Flow Planner source dir
    STATS_JSON   Path to the JSON file produced by measure_frenet_stats.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import yaml

FP_DIR = Path(os.environ["FP_DIR"])
STATS_JSON = Path(os.environ["STATS_JSON"])

YAML_PATH = FP_DIR / "flow_planner" / "script" / "normalization_stats" / "frenet_norm_stats.yaml"
BACKUP_PATH = YAML_PATH.with_suffix(".yaml.bak")

if not STATS_JSON.is_file():
    print(f"ERROR: STATS_JSON={STATS_JSON} does not exist", file=sys.stderr)
    sys.exit(1)

if not YAML_PATH.is_file():
    print(f"ERROR: frenet_norm_stats.yaml not found at {YAML_PATH}", file=sys.stderr)
    sys.exit(1)

stats = json.loads(STATS_JSON.read_text())
s_mean = round(float(stats["s"]["mean"]), 2)
s_std  = round(float(stats["s"]["std"]),  2)
d_mean = round(float(stats["d"]["mean"]), 2)
d_std  = round(float(stats["d"]["std"]),  2)
cos_mean = float(stats["cos_h_mean"])
cos_std  = float(stats["cos_h_std"])
sin_mean = float(stats["sin_h_mean"])
sin_std  = float(stats["sin_h_std"])

# Guardrails: refuse to write nonsense values. Catches a botched measurement
# (e.g., running on a buggy cache) before it corrupts the YAML.
if not (0.1 < s_std < 200) or not (0.01 < d_std < 100):
    print(f"ERROR: measured s_std={s_std} or d_std={d_std} out of plausible range",
          file=sys.stderr)
    print("Refusing to write YAML. Re-measure with a healthy cache.", file=sys.stderr)
    sys.exit(1)

# Read existing YAML, update only the two uniform blocks.
with YAML_PATH.open() as f:
    doc = yaml.safe_load(f)

new_block = {
    "log": {},
    "uniform": {
        "mean": [s_mean, d_mean, cos_mean, sin_mean],
        "std":  [s_std,  d_std,  cos_std,  sin_std],
    },
}

doc["ego"] = new_block
doc["neighbor"] = new_block

# Backup before overwrite.
if not BACKUP_PATH.exists():
    shutil.copy2(YAML_PATH, BACKUP_PATH)
    print(f"backed up original to {BACKUP_PATH.name}")
else:
    print(f"backup already exists: {BACKUP_PATH.name} (not overwriting)")

# Write atomically: write to a sibling temp then rename.
tmp = YAML_PATH.with_suffix(".yaml.tmp")
with tmp.open("w") as f:
    yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=None)
tmp.replace(YAML_PATH)

print(f"updated {YAML_PATH.name}:")
print(f"  ego.uniform.mean: {new_block['uniform']['mean']}")
print(f"  ego.uniform.std:  {new_block['uniform']['std']}")
print(f"  (neighbor.uniform set to same values)")
print(f"\nfull YAML after update:")
print(YAML_PATH.read_text())
