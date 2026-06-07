"""Generate a deterministic, seeded log-disjoint train/val split.

The held-out validation cache must be built from a set of nuPlan logs that
NEVER appear in the training cache. Otherwise the "held-out" scenarios share
drivers / weather / scene geometry with training and our generalization
numbers understate the real gap.

This script:
  1. enumerates the `.db` files in a nuPlan logs directory
  2. seeded-shuffles them
  3. splits into train / val using --train_frac (default 54/64 ≈ 0.84375)
  4. writes a JSON file with the form:
       {
         "seed": 42,
         "train_frac": 0.84375,
         "train": ["log_a", "log_b", ...],   # 54 entries for nuPlan mini
         "val":   ["log_x", "log_y", ...],   # 10 entries
       }

The same seed + same set of logs always produce the same split. Anyone
checking out the repo and pointing this script at the same nuPlan mini
release will reproduce the same JSON byte-for-byte.

Once generated, `docs/log_split_mini_seed42.json` should be committed so
the split is part of the repo contract. The notebook then reads the JSON
and passes it to `flow_planner.run_script.preprocess` via the
`--log_names_json` + `--log_names_key` CLI flags.

Usage:
    python scripts/generate_log_split.py \\
        --logs_dir /content/work/nuplan/data/cache/mini \\
        --output   docs/log_split_mini_seed42.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


def generate_split(log_names: list[str], seed: int, train_frac: float) -> dict:
    """Pure function: split a list of log basenames into train/val.

    Args:
        log_names: list of log basenames (without `.db` extension).
        seed: random seed for the shuffle.
        train_frac: fraction of logs to put in train (0 < train_frac < 1).

    Returns:
        Dict with keys: seed, train_frac, train (sorted), val (sorted),
        n_train, n_val, n_total.
    """
    if not log_names:
        raise ValueError("log_names is empty")
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")

    rng = random.Random(seed)
    shuffled = sorted(log_names)  # canonical input order
    rng.shuffle(shuffled)

    n_train = max(1, int(round(len(shuffled) * train_frac)))
    if n_train >= len(shuffled):
        # ensure at least one val log
        n_train = len(shuffled) - 1

    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train:])

    # Sanity: disjoint
    assert not (set(train) & set(val)), "train and val share logs"

    return {
        "seed": seed,
        "train_frac": train_frac,
        "n_total": len(shuffled),
        "n_train": len(train),
        "n_val": len(val),
        "train": train,
        "val": val,
    }


def enumerate_logs(logs_dir: Path) -> list[str]:
    """List .db file basenames (sans extension) in the logs directory."""
    files = sorted(p.stem for p in logs_dir.glob("*.db"))
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs_dir", type=Path, required=True,
                        help="Directory containing nuPlan log .db files.")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write the JSON split file.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--train_frac", type=float, default=54.0 / 64.0,
                        help="Fraction of logs in the train split (default: 54/64 ≈ 0.84375).")
    parser.add_argument("--expected_total", type=int, default=64,
                        help="Expected number of logs (default: 64 for nuPlan mini). "
                             "If 0, the check is skipped.")
    args = parser.parse_args(argv)

    if not args.logs_dir.is_dir():
        print(f"ERROR: logs_dir does not exist or is not a directory: {args.logs_dir}",
              file=sys.stderr)
        return 1

    log_names = enumerate_logs(args.logs_dir)
    if not log_names:
        print(f"ERROR: no .db files found in {args.logs_dir}", file=sys.stderr)
        return 1

    if args.expected_total > 0 and len(log_names) != args.expected_total:
        print(f"WARNING: expected {args.expected_total} logs (nuPlan mini), "
              f"found {len(log_names)} in {args.logs_dir}", file=sys.stderr)

    split = generate_split(log_names, seed=args.seed, train_frac=args.train_frac)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(split, indent=2) + "\n")

    print(f"Wrote {args.output}")
    print(f"  seed        {split['seed']}")
    print(f"  train_frac  {split['train_frac']:.6f}")
    print(f"  n_total     {split['n_total']}")
    print(f"  n_train     {split['n_train']}")
    print(f"  n_val       {split['n_val']}")
    print(f"  first train {split['train'][0] if split['train'] else '(none)'}")
    print(f"  first val   {split['val'][0]   if split['val']   else '(none)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
