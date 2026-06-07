"""CLI driver for Flow Planner's own DataProcessor.

This version implements the v2 methodology defined in
docs/preprocessing_methodology.md:
- Matches the PlanTF / Diffusion-Planner / Flow Planner training-filter
  convention (random, no stratification, shuffled).
- Supports per-log filtering via `--log_names_json` for log-disjoint train /
  held-out construction.
- Defaults to processing ALL available scenarios; use `--total_scenarios`
  only as a safety cap.

Usage:
    # Generate train split (54 logs)
    python -m flow_planner.run_script.preprocess \
        --data_path /content/work/nuplan/data/cache/mini \
        --map_path  /content/work/nuplan/maps \
        --save_path /content/work/preprocessed_cache_train \
        --log_names_json docs/log_split_mini_seed42.json \
        --log_names_key train

    # Generate held-out split (10 logs)
    python -m flow_planner.run_script.preprocess \
        --data_path /content/work/nuplan/data/cache/mini \
        --map_path  /content/work/nuplan/maps \
        --save_path /content/work/preprocessed_cache_heldout \
        --log_names_json docs/log_split_mini_seed42.json \
        --log_names_key val
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from flow_planner.data.data_process.data_processor import DataProcessor

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor


def get_scenario_filter(
    log_names=None,
    limit_total_scenarios=None,
    shuffle=True,
):
    """Build a ScenarioFilter matching the PlanTF / Flow Planner training convention.

    Defaults:
      - scenario_types: None (all types — NO stratification)
      - num_scenarios_per_type: None (NO per-type cap)
      - shuffle: True (randomized order)
      - expand_scenarios: True (matches PlanTF training_scenarios_1M.yaml)
      - remove_invalid_goals: True

    Args:
        log_names: optional list[str] of log file basenames (without `.db`) to
            restrict sampling to. None = all available logs.
        limit_total_scenarios: int safety cap. None = no cap (use all).
        shuffle: randomize scenario order before truncation.

    Source: PlanTF training_scenarios_1M.yaml
    https://github.com/jchengai/planTF/blob/main/config/scenario_filter/training_scenarios_1M.yaml
    """
    return ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=log_names,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=limit_total_scenarios,
        timestamp_threshold_s=None,
        ego_displacement_minimum_m=None,
        expand_scenarios=True,
        remove_invalid_goals=True,
        shuffle=shuffle,
        ego_start_speed_threshold=None,
        ego_stop_speed_threshold=None,
        speed_noise_tolerance=None,
    )


def resolve_log_names(args) -> list | None:
    """Decide which log_names to pass to ScenarioFilter.

    Priority order:
        1. --log_names_json + --log_names_key  (preferred for train/val splits)
        2. --log_names_csv                     (explicit comma-separated list)
        3. None (use all logs)
    """
    if args.log_names_json:
        with open(args.log_names_json) as f:
            splits = json.load(f)
        key = args.log_names_key
        if key not in splits:
            raise ValueError(
                f"log_names_json {args.log_names_json} has no key '{key}'. "
                f"Available keys: {sorted(splits.keys())}"
            )
        log_names = list(splits[key])
        print(f"Using log_names from {args.log_names_json}[{key!r}]: {len(log_names)} logs")
        return log_names
    if args.log_names_csv:
        log_names = [x.strip() for x in args.log_names_csv.split(",") if x.strip()]
        print(f"Using log_names from --log_names_csv: {len(log_names)} logs")
        return log_names
    return None


def main(args):
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    print(f"data_path:        {args.data_path}")
    print(f"map_path:         {args.map_path}")
    print(f"save_path:        {save_path}")
    print(f"total_scenarios:  {args.total_scenarios or 'unlimited'}")
    print(f"seed:             {args.seed}")

    map_version = "nuplan-maps-v1.0"

    scenario_mapping = ScenarioMapping(scenario_map={}, subsample_ratio_override=0.5)

    builder = NuPlanScenarioBuilder(
        data_root=args.data_path,
        map_root=args.map_path,
        sensor_root=None,
        db_files=None,
        map_version=map_version,
        scenario_mapping=scenario_mapping,
    )

    # Seed numpy + python RNGs BEFORE enumerating scenarios so that the
    # ScenarioFilter's internal `shuffle=True` is reproducible across runs
    # and the truncation under --total_scenarios picks a deterministic
    # subset. Without this, the manifest's `seed` field is misleading —
    # the file-list is non-deterministic from run to run.
    random.seed(args.seed)
    np.random.seed(args.seed)

    log_names = resolve_log_names(args)

    scenario_filter = get_scenario_filter(
        log_names=log_names,
        limit_total_scenarios=args.total_scenarios,
        shuffle=True,
    )

    worker = SingleMachineParallelExecutor(use_process_pool=False)

    print("\nEnumerating scenarios...")
    scenarios = builder.get_scenarios(scenario_filter, worker)
    # Deterministic-order pass for reproducibility: sort by (log_name,
    # map_name, token) so that even non-deterministic worker order from
    # SingleMachineParallelExecutor doesn't affect the on-disk file list.
    def _sort_key(s):
        ln = getattr(s, "log_name", None) or getattr(s, "_log_name", "")
        mn = getattr(s, "_map_name", "")
        tk = getattr(s, "token", "")
        return (str(ln), str(mn), str(tk))
    scenarios = sorted(scenarios, key=_sort_key)
    print(f"Enumerated {len(scenarios)} scenarios (sorted deterministically)")

    if len(scenarios) == 0:
        print("ERROR: filter returned 0 scenarios.")
        print("       Check that --data_path contains .db files and that any")
        print("       supplied log_names match files that exist there.")
        sys.exit(1)

    print("\nRunning Flow Planner DataProcessor...")
    processor = DataProcessor(str(save_path))
    failed = processor.work(scenarios) or []

    # Write the JSON file list AND a sidecar manifest describing how this cache was built.
    npz_files = sorted(p.name for p in save_path.glob("*.npz"))
    json_path = save_path / "diffusion_planner_training.json"
    json_path.write_text(json.dumps(npz_files))
    print(f"\nWrote {json_path} with {len(npz_files)} entries")
    if npz_files:
        print(f"  First: {npz_files[0]}")

    n_enumerated = len(scenarios)
    n_produced = len(npz_files)
    n_failed = len(failed)
    # Loud surface of the gap. Some attrition is expected (route correction
    # raises, etc.), but a large gap warrants investigation. The exact
    # equality assertion `enumerated == produced + failed` would be too
    # strict because per-scenario try/except can also fall through the
    # collision-guard branch in save_to_disk; we just report the math.
    print(
        f"  enumerated={n_enumerated}, produced={n_produced}, failed={n_failed}, "
        f"unaccounted={n_enumerated - n_produced - n_failed}"
    )
    if n_failed:
        print(f"  See preprocess_failures.json for the per-scenario error list.")

    manifest = {
        "schema_version": 3,
        "scenarios_produced": n_produced,
        "scenarios_enumerated": n_enumerated,
        "scenarios_failed": n_failed,
        "log_names_json": args.log_names_json,
        "log_names_key": args.log_names_key,
        "log_names_csv": args.log_names_csv,
        "resolved_log_names": log_names,
        "total_scenarios_cap": args.total_scenarios,
        "shuffle": True,
        "expand_scenarios": True,
        "remove_invalid_goals": True,
        "num_scenarios_per_type": None,
        "scenario_types": None,
        "seed": args.seed,
        "seed_applied_to_numpy_and_random": True,
        "deterministic_sort": True,
        "data_path": args.data_path,
        "map_path": args.map_path,
    }
    (save_path / "preprocess_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  Wrote preprocess_manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True,
                        help="dir containing nuPlan .db files")
    parser.add_argument("--map_path", type=str, required=True,
                        help="nuPlan maps dir (contains nuplan-maps-v1.0.json)")
    parser.add_argument("--save_path", type=str, required=True,
                        help="output dir for .npz files")
    parser.add_argument("--total_scenarios", type=int, default=None,
                        help="safety cap on number of scenarios to process. "
                             "None (default) = no cap, process all available.")
    parser.add_argument("--log_names_json", type=str, default=None,
                        help="path to a JSON file mapping split name -> list of log basenames. "
                             "Use with --log_names_key to choose a split.")
    parser.add_argument("--log_names_key", type=str, default="train",
                        help="key in --log_names_json to use (e.g. 'train' or 'val').")
    parser.add_argument("--log_names_csv", type=str, default=None,
                        help="comma-separated list of log basenames (alternative to --log_names_json).")
    parser.add_argument("--seed", type=int, default=42,
                        help="seed for the ScenarioFilter shuffle, for reproducibility.")
    args = parser.parse_args()
    main(args)
