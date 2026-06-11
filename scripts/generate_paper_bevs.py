"""Generate paper-ready BEV visualizations from trained Flow Planner checkpoints.

Environment-agnostic. Works on any machine (local, Colab, cluster) with:
- Python 3.10+
- GPU (recommended; CPU will work but takes ~5 min/checkpoint)
- The dependencies pinned in flow_planner/requirements.txt installed

What this script does
---------------------
1. Discovers every `.ckpt` in CHECKPOINTS_DIR
2. Infers per-checkpoint metadata (representation, seed, audit-config, train scale)
   from the filename and file size
3. Loads each eval JSON in EVAL_DIRS to attach ground-truth ADE/FDE numbers
4. For each Waypoints/Frenet checkpoint, invokes scripts/visualize_bev.py
   with identical settings (same 4 scene indices via BEV_SEED, same Hydra config)
   so all rendered figures are visually comparable
5. Writes a `.json` metadata sidecar next to each rendered `.png` containing:
   - representation, seed, config, scale, source checkpoint path
   - ADE/FDE from the matched eval JSON
   - render timestamp
6. Builds per-representation comparison grids by stacking the rendered BEVs

Inputs expected on disk
-----------------------
- CHECKPOINTS_DIR    one or more `.ckpt` files (any subdirectory layout)
- CACHE_DIR          preprocessed nuPlan `.npz` cache + diffusion_planner_training.json
                     manifest. Roughly 50 scenarios is sufficient.
- EVAL_DIRS          (optional) directories containing eval_*.json sidecar files
                     so the per-BEV metadata can include ADE/FDE numbers
- REPO_DIR           path to a clone of CS269FlowPlannerProject so the script can find
                     the inner flow_planner package and visualize_bev.py

Setup steps (when running on Colab)
-----------------------------------
1. Mount Google Drive at /content/drive
2. Mount DagsHub jialic/dagshub-drive (provides the 80k npz cache)
3. Copy ~50 npz scenarios from DagsHub to local Colab storage
   (the FUSE mount is too slow for direct reads from inference)
4. Clone this repository to /content/CS269FlowPlannerProject
5. Run this script with arguments pointing at the above paths

Setup steps (when running locally)
----------------------------------
1. Clone this repository
2. Install dependencies: `pip install -r flow_planner/requirements.txt`
3. Place a small preprocessed npz cache + manifest at CACHE_DIR
   (contact the team or use scripts/generate_log_split.py to build one)
4. Place trained `.ckpt` files in CHECKPOINTS_DIR
5. Run: `python scripts/generate_paper_bevs.py --checkpoints-dir <path> ...`

Usage
-----
    python scripts/generate_paper_bevs.py \
        --checkpoints-dir /path/to/checkpoints \
        --cache-dir       /path/to/preprocessed_cache \
        --output-dir      /path/to/paper_bevs \
        --eval-dirs       /path/to/eval_jsons /path/to/more/eval_jsons \
        --repo-dir        /path/to/CS269FlowPlannerProject \
        --bev-seed        42

All arguments except `--cache-dir` and `--checkpoints-dir` have sensible defaults.

Known limitations
-----------------
- Velocity and Acceleration representations require `va_norm_stats.yaml`,
  which is not currently in the repository. See QUICKSTART.md §6.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional


NORM_STATS_MAP = {
    "waypoints": "waypoints_norm_stats",
    "frenet": "frenet_norm_stats_v1",  # paper-headline Frenet config
    "velocity": "va_norm_stats",
    "acceleration": "va_norm_stats",
}


@dataclass
class CheckpointMeta:
    """Per-checkpoint metadata inferred from filename + filesystem stat."""
    name: str
    path: str
    representation: str
    seed: Optional[int]
    config: str
    train_scale: str
    size_mb: float
    ade_mean: Optional[float] = None
    fde_mean: Optional[float] = None
    ade_std: Optional[float] = None
    fde_std: Optional[float] = None
    eval_source: Optional[str] = None


def infer_metadata(path: pathlib.Path) -> CheckpointMeta:
    """Parse the filename and file size to extract representation/config/seed/scale."""
    name = path.name.lower()
    if "waypoint" in name:
        rep = "waypoints"
    elif "frenet" in name:
        rep = "frenet"
    elif "velocity" in name:
        rep = "velocity"
    elif "acceleration" in name:
        rep = "acceleration"
    else:
        rep = "unknown"

    seed_match = re.search(r"seed(\d+)", name)
    seed = int(seed_match.group(1)) if seed_match else None

    if "fixedabc" in name:
        config = "fix_a1bc"
    elif "fixedcb" in name:
        config = "fix_bc"
    elif "v7audit" in name:
        config = "v7AUDIT"
    elif "v7paper" in name:
        config = "v7PAPER"
    elif "paper_baseline" in name:
        config = "paper_baseline"
    elif "best" in name or path.stem.lower() == "frenet_seed42":
        config = "best_ever_original"
    else:
        config = "default"

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > 230:
        scale = "5000"
    elif 205 <= size_mb <= 230:
        scale = "1500"
    else:
        scale = "unknown"

    return CheckpointMeta(
        name=path.name,
        path=str(path),
        representation=rep,
        seed=seed,
        config=config,
        train_scale=scale,
        size_mb=round(size_mb, 1),
    )


def discover_checkpoints(checkpoint_dir: pathlib.Path) -> list[CheckpointMeta]:
    """Walk checkpoint_dir recursively and infer metadata for each .ckpt found."""
    metas = [infer_metadata(p) for p in sorted(checkpoint_dir.rglob("*.ckpt"))]

    # De-dupe by (representation, seed, config, scale)
    seen: set[tuple] = set()
    unique: list[CheckpointMeta] = []
    for m in metas:
        key = (m.representation, m.seed, m.config, m.train_scale)
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    return unique


def load_eval_jsons(eval_dirs: list[pathlib.Path]) -> dict[str, dict]:
    """Read every eval_*.json in eval_dirs and key by basename without extension."""
    evals: dict[str, dict] = {}
    for d in eval_dirs:
        if not d.exists():
            continue
        for p in d.rglob("eval_*.json"):
            try:
                evals[p.stem] = json.loads(p.read_text())
                evals[p.stem]["_source_path"] = str(p)
            except (OSError, json.JSONDecodeError):
                continue
    return evals


def attach_eval(meta: CheckpointMeta, evals: dict[str, dict]) -> None:
    """Try several conventions to map a checkpoint to its eval JSON."""
    rep = meta.representation
    cfg = meta.config
    seed = meta.seed

    candidates = [
        f"eval_motion_{rep}_seed{seed}",
        f"eval_{rep}_val_{cfg}",
        f"eval_{rep}_val_fixedCB" if cfg == "fix_bc" else None,
        f"eval_{rep}_val_fixedABC" if cfg == "fix_a1bc" else None,
        f"eval_{rep}_val",
        f"eval_paper_{rep}",
    ]
    for key in candidates:
        if key and key in evals:
            e = evals[key]
            meta.ade_mean = e.get("ade_mean", e.get("ade"))
            meta.fde_mean = e.get("fde_mean", e.get("fde"))
            meta.ade_std = e.get("ade_std")
            meta.fde_std = e.get("fde_std")
            meta.eval_source = e.get("_source_path")
            return


def render_bev(
    meta: CheckpointMeta,
    repo_dir: pathlib.Path,
    cache_dir: pathlib.Path,
    out_path: pathlib.Path,
    bev_seed: int,
) -> tuple[bool, str]:
    """Invoke scripts/visualize_bev.py as a subprocess with the required env vars."""
    fp_dir = repo_dir / "flow_planner"
    rep = meta.representation
    if rep not in ("waypoints", "frenet"):
        return False, "skip (representation not supported by visualize_bev.py)"

    env = {
        **os.environ,
        "FP_DIR": str(fp_dir),
        "CKPT_PATH": meta.path,
        "CACHE_DIR": str(cache_dir),
        "OUTPUT_PNG": str(out_path),
        "KINEMATIC": rep,
        "NORM_STATS": NORM_STATS_MAP[rep],
        "PLOT_TITLE": f"{rep} seed={meta.seed} {meta.config} ({meta.train_scale}-scen)",
        "BEV_SEED": str(bev_seed),
    }
    proc = subprocess.run(
        [sys.executable, str(repo_dir / "scripts" / "visualize_bev.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if proc.returncode == 0 and out_path.exists():
        return True, "ok"
    return False, f"exit={proc.returncode}: {proc.stderr.strip()[-200:]}"


def build_comparison_grid(images: list[pathlib.Path], out_path: pathlib.Path) -> None:
    """Stack a list of PNG paths vertically into a single comparison figure."""
    from PIL import Image

    imgs = [Image.open(p).convert("RGB") for p in images]
    target_w = min(i.size[0] for i in imgs)
    resized = [
        i.resize((target_w, int(i.size[1] * target_w / i.size[0])), Image.LANCZOS)
        for i in imgs
    ]
    total_h = sum(i.size[1] for i in resized)
    grid = Image.new("RGB", (target_w, total_h), (255, 255, 255))
    y = 0
    for img in resized:
        grid.paste(img, (0, y))
        y += img.size[1]
    grid.save(out_path, optimize=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready BEV figures from Flow Planner checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=pathlib.Path,
        required=True,
        help="Directory containing .ckpt files (walked recursively).",
    )
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        required=True,
        help="Preprocessed npz cache + manifest used for inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        required=True,
        help="Where to write rendered .png files and metadata sidecars.",
    )
    parser.add_argument(
        "--eval-dirs",
        type=pathlib.Path,
        nargs="*",
        default=[],
        help="One or more directories containing eval_*.json files.",
    )
    parser.add_argument(
        "--repo-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parent.parent,
        help="Path to the CS269FlowPlannerProject repository root.",
    )
    parser.add_argument(
        "--bev-seed",
        type=int,
        default=42,
        help="Sampling seed for the 4 scene indices. Same seed across all checkpoints "
        "ensures the rendered figures are paired/comparable.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Discover checkpoints
    checkpoints = discover_checkpoints(args.checkpoints_dir)
    print(f"Discovered {len(checkpoints)} unique checkpoints in {args.checkpoints_dir}")
    for m in checkpoints:
        print(
            f"  {m.representation:13s} seed={m.seed} scale={m.train_scale} "
            f"config={m.config:25s} {m.name}"
        )

    # 2. Load eval JSONs
    evals = load_eval_jsons(args.eval_dirs)
    print(f"\nLoaded {len(evals)} eval JSONs from {len(args.eval_dirs)} directory(ies)")

    # 3. Render each checkpoint + attach eval metadata
    rendered: dict[str, list[pathlib.Path]] = {}
    for m in checkpoints:
        attach_eval(m, evals)
        out_png = args.output_dir / (
            f"{m.representation}_seed{m.seed}_{m.config}_{m.train_scale}.png"
        )
        if out_png.exists():
            print(f"  cached {out_png.name}")
            success, status = True, "cached"
        else:
            print(f"\n=== {out_png.name} ===")
            success, status = render_bev(
                m, args.repo_dir, args.cache_dir, out_png, args.bev_seed
            )
            print(f"  {status}")

        if not success:
            continue

        # Write metadata sidecar
        meta_dict = asdict(m)
        meta_dict["rendered_at"] = datetime.now(timezone.utc).isoformat()
        meta_dict["render_status"] = status
        out_json = out_png.with_suffix(".json")
        out_json.write_text(json.dumps(meta_dict, indent=2))

        rendered.setdefault(m.representation, []).append(out_png)

    # 4. Build per-representation comparison grids
    print()
    for rep, pngs in rendered.items():
        if not pngs:
            continue
        grid_path = args.output_dir / f"_grid_{rep}.png"
        build_comparison_grid(pngs, grid_path)
        print(f"  built {grid_path.name} ({len(pngs)} BEVs stacked)")

    print(f"\nAll outputs at: {args.output_dir}")
    total_pngs = sum(len(v) for v in rendered.values())
    print(f"  {total_pngs} BEV PNGs + {total_pngs} metadata JSONs + {len(rendered)} grid(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
