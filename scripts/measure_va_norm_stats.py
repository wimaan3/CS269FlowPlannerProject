"""Measure velocity / acceleration normalization stats from a preprocessed nuPlan cache
and produce the missing `va_norm_stats.yaml` Hydra config.

Why this exists
---------------
The team's `motion_representations.ipynb` trains separate Velocity and Acceleration
models whose 4-channel target is `(channel1, channel2, cos h, sin h)` where:
- velocity model:     channels = (v_x, v_y)   computed as finite-difference of position
- acceleration model: channels = (a_x, a_y)   computed as finite-difference of velocity

The training script references a `va_norm_stats.yaml` Hydra config that holds the
per-channel mean and std used to normalize the target before the MSE loss. That
config file was never committed to the repository, which blocks reproducing the
Velocity / Acceleration runs.

This script measures the empirical channel statistics from a preprocessed `.npz`
cache directly. The other sections of the YAML (ego_past, ego_current, neighbor_past,
map_objects, lanes, routes, etc.) are scene-input features whose statistics are
representation-agnostic; they are copied verbatim from `waypoints_norm_stats.yaml`.

Usage
-----
    python scripts/measure_va_norm_stats.py \
        --cache-dir   /path/to/preprocessed_cache \
        --output-yaml flow_planner/flow_planner/script/normalization_stats/va_norm_stats.yaml \
        [--sample-limit 500]

The optional `--sample-limit` caps the number of npz files used for the stats
measurement (default 500). The mean/std numbers should converge after ~500
scenarios.

What gets written
-----------------
A YAML file with the same top-level structure as the other norm-stats configs:
- ego / neighbor — measured mean/std for the target channels
- All other sections — copied from waypoints_norm_stats.yaml

How the measurement is done
---------------------------
For each .npz scenario in the cache:
1. Read `ego_future_xy` (shape [T, 2]) — the ground-truth future positions
2. Compute velocity via finite-difference: v_t = (x_{t+1} - x_t) / dt
3. Compute acceleration via finite-difference: a_t = (v_{t+1} - v_t) / dt
4. Accumulate per-channel sums and sum-of-squares
5. After all scenarios, compute mean and std and write the YAML
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import yaml


# nuPlan default future horizon is 8 seconds sampled at 1 Hz post-downsampling.
# A typical preprocessed cache stores 8 timesteps in ego_future_xy.
DT = 1.0


# These sections are scene-input features whose statistics are
# representation-agnostic. We copy them verbatim from waypoints_norm_stats.yaml.
SCENE_FEATURE_SECTIONS = {
    "ego_past": {
        "mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "std":  [20, 20, 1, 1, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1],
    },
    "ego_current": {
        "mean": [10, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "std":  [20, 20, 1, 1, 5, 1, 1, 1, 1, 1, 20, 20, 1, 1, 1, 1],
    },
    "neighbor_past": {
        "mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "std":  [20, 20, 1, 1, 20, 20, 20, 20, 1, 1, 1],
    },
    "map_objects": {
        "mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "std":  [20, 20, 1, 1, 20, 20, 1, 1, 1, 1],
    },
    "lanes": {
        "mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "std":  [20, 20, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1],
    },
    "lanes_speedlimit": {"mean": [0], "std": [20]},
    "routes": {
        "mean": [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "std":  [20, 20, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1],
    },
    "routes_speedlimit": {"mean": [0], "std": [20]},
}


def measure_velocity_acceleration_stats(
    cache_dir: pathlib.Path, sample_limit: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-channel mean and std for both velocity and acceleration targets
    from the npz cache. Returns (vel_mean, vel_std, accel_mean, accel_std), each
    of shape (2,) covering (channel_x, channel_y).
    """
    npz_files = sorted(cache_dir.glob("*.npz"))[:sample_limit]
    if not npz_files:
        raise SystemExit(f"No .npz files found in {cache_dir}")

    vel_samples: list[np.ndarray] = []
    accel_samples: list[np.ndarray] = []

    for path in npz_files:
        scene = np.load(path, allow_pickle=True)
        if "ego_future_xy" not in scene.files:
            continue
        xy = scene["ego_future_xy"]  # shape: [T, 2]
        if xy.ndim != 2 or xy.shape[0] < 3 or xy.shape[1] != 2:
            continue

        vel = np.diff(xy, axis=0) / DT  # shape: [T-1, 2]
        accel = np.diff(vel, axis=0) / DT  # shape: [T-2, 2]
        vel_samples.append(vel)
        accel_samples.append(accel)

    if not vel_samples:
        raise SystemExit(
            f"None of the {len(npz_files)} scenarios contained usable ego_future_xy"
        )

    vel_all = np.concatenate(vel_samples, axis=0)  # shape: [N*(T-1), 2]
    accel_all = np.concatenate(accel_samples, axis=0)  # shape: [N*(T-2), 2]

    return (
        vel_all.mean(axis=0),
        vel_all.std(axis=0),
        accel_all.mean(axis=0),
        accel_all.std(axis=0),
    )


def build_va_norm_stats_yaml(
    vel_mean: np.ndarray,
    vel_std: np.ndarray,
    accel_mean: np.ndarray,
    accel_std: np.ndarray,
    measurement_n: int,
) -> str:
    """Build the YAML text for va_norm_stats.yaml.

    The team's per-representation training uses a SHARED target schema across
    Velocity and Acceleration models: 4 channels = (chan_x, chan_y, cos h, sin h).
    For the shared YAML we average velocity and acceleration channel stats so a
    single config covers both representations. If you need representation-specific
    configs, split this into va_velocity_norm_stats.yaml and va_accel_norm_stats.yaml.
    """
    # Take the larger std as a conservative choice (covers both cases)
    chan_mean = np.maximum(np.abs(vel_mean), np.abs(accel_mean))
    chan_std = np.maximum(vel_std, accel_std)

    # Heading channels remain (cos h, sin h) with same conventions as waypoints
    heading_mean = [0, 0]
    heading_std = [1.0, 1.0]

    ego_mean = [round(float(chan_mean[0]), 2), round(float(chan_mean[1]), 2), *heading_mean]
    ego_std = [round(float(chan_std[0]), 2), round(float(chan_std[1]), 2), *heading_std]

    lines = [
        "# Velocity / Acceleration normalization stats.",
        "#",
        "# Target schema: (channel_x, channel_y, cos h, sin h) where channel_x/y is either",
        "#   - velocity (v_x, v_y) — for the velocity representation",
        "#   - acceleration (a_x, a_y) — for the acceleration representation",
        "#",
        f"# Stats measured from {measurement_n} preprocessed scenarios via",
        "# scripts/measure_va_norm_stats.py.",
        "#",
        f"# velocity (m/s):     mean = ({vel_mean[0]:.3f}, {vel_mean[1]:.3f}), std = ({vel_std[0]:.3f}, {vel_std[1]:.3f})",
        f"# acceleration (m/s²): mean = ({accel_mean[0]:.3f}, {accel_mean[1]:.3f}), std = ({accel_std[0]:.3f}, {accel_std[1]:.3f})",
        "#",
        "# The YAML below uses the conservative envelope (max |mean|, max std) so a single",
        "# config can serve both representations. For split configs, see the docstring.",
        "",
        "ego:",
        "  log: {}",
        "  uniform:",
        f"    mean: {ego_mean}",
        f"    std: {ego_std}",
        "",
        "neighbor:",
        "  log: {}",
        "  uniform:",
        f"    mean: {ego_mean}",
        f"    std: {ego_std}",
        "",
    ]

    # Append scene-input feature sections (unchanged from waypoints)
    for section_name, stats in SCENE_FEATURE_SECTIONS.items():
        lines.append(f"{section_name}:")
        if "log" in stats:
            lines.append("  log: {}")
            lines.append("  uniform:")
            lines.append(f"    mean: {stats['uniform']['mean']}")
            lines.append(f"    std: {stats['uniform']['std']}")
        else:
            lines.append(f"  mean: {stats['mean']}")
            lines.append(f"  std: {stats['std']}")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure velocity/acceleration channel statistics from a preprocessed "
            "nuPlan cache and produce va_norm_stats.yaml."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing preprocessed nuPlan .npz files.",
    )
    parser.add_argument(
        "--output-yaml",
        type=pathlib.Path,
        default=pathlib.Path("flow_planner/flow_planner/script/normalization_stats/va_norm_stats.yaml"),
        help="Where to write the generated va_norm_stats.yaml.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=500,
        help="Maximum number of .npz files to sample for the measurement.",
    )
    args = parser.parse_args()

    if not args.cache_dir.exists():
        raise SystemExit(f"Cache directory not found: {args.cache_dir}")

    n_npz = len(sorted(args.cache_dir.glob("*.npz")))
    print(f"Found {n_npz} .npz files in {args.cache_dir}")
    print(f"Measuring channel statistics from up to {args.sample_limit} scenarios...")

    vel_mean, vel_std, accel_mean, accel_std = measure_velocity_acceleration_stats(
        args.cache_dir, args.sample_limit
    )

    print(f"\nMeasured statistics:")
    print(f"  velocity     mean = ({vel_mean[0]:.3f}, {vel_mean[1]:.3f}) m/s")
    print(f"               std  = ({vel_std[0]:.3f}, {vel_std[1]:.3f}) m/s")
    print(f"  acceleration mean = ({accel_mean[0]:.3f}, {accel_mean[1]:.3f}) m/s²")
    print(f"               std  = ({accel_std[0]:.3f}, {accel_std[1]:.3f}) m/s²")

    yaml_text = build_va_norm_stats_yaml(
        vel_mean, vel_std, accel_mean, accel_std, min(n_npz, args.sample_limit)
    )

    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml_text)
    print(f"\nWrote {args.output_yaml}")
    print(f"  ({len(yaml_text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
