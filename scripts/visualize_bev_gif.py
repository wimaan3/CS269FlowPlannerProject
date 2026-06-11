"""Animated BEV (bird's-eye view) rollout visualization for a trained checkpoint.

Companion to ``scripts/visualize_bev.py`` (which produces static PNGs) — this
version emits one animated ``.gif`` per scene where frame N of the animation
shows the predicted (and ground-truth) trajectory drawn up to step N, like the
pilot's ``rollout_*.gif`` artifacts.

Used by ``notebooks/v8_frenet_fixes.ipynb`` Section 12 to produce per-variant
rollout GIFs on a fixed set of held-out scenes so the report can show the
qualitative failure modes recover (or not) under each Frenet fix.

Environment-variable interface (same convention as ``visualize_bev.py`` so the
two can share a wrapper):

    FP_DIR        Flow Planner source directory (added to sys.path)
    CKPT_PATH     Path to the .ckpt file to load
    CACHE_DIR     Directory containing preprocessed .npz + manifest JSON
    OUTPUT_DIR    Directory in which to write rollout_{scene}.gif files
    PLOT_TITLE    Optional figure title prefix (default: KINEMATIC name)
    KINEMATIC     'frenet' (default) | 'waypoints' — decode path + Hydra overrides
    NORM_STATS    Hydra config name for normalization stats. Defaults to
                  frenet_norm_stats for Frenet, waypoints_norm_stats otherwise.
    SCENE_INDICES Comma-separated list of dataset indices to render. Defaults
                  to '0,1,2,3' (deterministic; same scenes across variants so
                  the report can compare like-for-like).
    GIF_FPS       Output frame rate (default: 10).

Output: ``${OUTPUT_DIR}/rollout_{scene_idx}.gif`` for each scene index.
"""
from __future__ import annotations

import os
import pathlib
import sys

# Make flow_planner importable from the editable checkout.
sys.path.insert(0, os.environ["FP_DIR"])

import matplotlib
matplotlib.use("Agg")  # headless backends only — Colab + CI
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from flow_planner.data.dataset.nuplan import NuPlanDataset
from flow_planner.data.utils.collect import collect_batch
from flow_planner.data.normalization.frenet_utils import (
    frenet_to_cartesian,
    select_reference_centerline,
)

CKPT_PATH = os.environ["CKPT_PATH"]
CACHE_DIR = os.environ["CACHE_DIR"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
KINEMATIC = os.environ.get("KINEMATIC", "frenet")
if KINEMATIC not in {"frenet", "waypoints"}:
    raise SystemExit(f"KINEMATIC must be 'frenet' or 'waypoints', got {KINEMATIC!r}")
DEFAULT_NORM_STATS = "frenet_norm_stats" if KINEMATIC == "frenet" else "waypoints_norm_stats"
NORM_STATS = os.environ.get("NORM_STATS", DEFAULT_NORM_STATS)
PLOT_TITLE = os.environ.get("PLOT_TITLE", f"{KINEMATIC.title()} rollout")
SCENE_INDICES = [
    int(x) for x in os.environ.get("SCENE_INDICES", "0,1,2,3").split(",") if x.strip()
]
GIF_FPS = int(os.environ.get("GIF_FPS", "10"))

pathlib.Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# --- Hydra config (mirrors visualize_bev.py exactly so we never introduce
# silent train/eval architecture drift between PNG and GIF artifacts). ---
overrides = [
    f"++model.kinematic={KINEMATIC}",
    f"normalization_stats={NORM_STATS}",
    "ddp.distributed=false",
]
if KINEMATIC == "frenet":
    overrides += [
        "+model.model_encoder.centerline_encoder._target_="
        "flow_planner.model.modules.encoder_modules.CenterlineEncoder",
        "+model.model_encoder.centerline_encoder.n_points=100",
        "+model.model_encoder.centerline_encoder.hidden_dim=256",
        "++model.model_decoder.enable_attn_dist=false",
    ]

config_dir = os.path.join(os.environ["FP_DIR"], "flow_planner/script")
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name="flow_planner_standard", overrides=overrides)

model = instantiate(cfg.model)
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    ckpt = torch.load(CKPT_PATH, weights_only=True, map_location=device)
except Exception:
    # Fallback for legacy checkpoints that embed non-tensor Python objects
    # (e.g. third-party or pre-2.0 checkpoints). Only use on trusted sources.
    import warnings
    warnings.warn(
        f"torch.load(weights_only=True) failed for {CKPT_PATH}; "
        "falling back to weights_only=False. Ensure the checkpoint file is from a trusted source.",
        stacklevel=1,
    )
    ckpt = torch.load(CKPT_PATH, weights_only=False, map_location=device)
sd = ckpt.get("ema_state_dict", ckpt.get("state_dict", ckpt))
sd = {k.replace("module.", ""): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing or unexpected:
    print(
        f"WARNING: state_dict mismatch on a {KINEMATIC} checkpoint. "
        f"missing[:5]={list(missing)[:5]} unexpected[:5]={list(unexpected)[:5]}"
    )
# Backward-compat with pre-Patch-4 Frenet checkpoints (centerline_gate added later).
if (
    KINEMATIC == "frenet"
    and any("centerline_gate" in k for k in missing)
    and any("centerline_encoder" in k for k in sd)
):
    with torch.no_grad():
        model.model_encoder.centerline_gate.data.fill_(1.0)
    print("Initialised missing centerline_gate to 1.0 (pre-Patch-4 ckpt).")
model = model.to(device).eval()

# --- Dataset: load exactly the requested scenes (deterministic across variants). ---
ds = NuPlanDataset(
    data_dir=CACHE_DIR,
    data_list=os.path.join(CACHE_DIR, "diffusion_planner_training.json"),
    past_neighbor_num=cfg.model.neighbor_num,
    predicted_neighbor_num=cfg.model.neighbor_pred_num,
    future_len=cfg.model.future_len,
    future_downsampling_method="uniform",
)
n_total = len(ds)
indices = [i for i in SCENE_INDICES if 0 <= i < n_total]
if not indices:
    raise SystemExit(
        f"No valid SCENE_INDICES in {SCENE_INDICES} for a dataset of size {n_total}."
    )
subset = torch.utils.data.Subset(ds, indices)
loader = DataLoader(subset, batch_size=len(indices), shuffle=False, collate_fn=collect_batch)
batch = next(iter(loader)).to(device)

with torch.no_grad():
    preds = model(batch, mode="inference", use_cfg=False, cfg_weight=cfg.model.cfg_weight)

# --- Decode predictions to Cartesian (x, y) — same contract as visualize_bev.py. ---
assert batch.ego_future.shape[1] >= preds.shape[2], (
    f"ego_future too short ({batch.ego_future.shape[1]}) for prediction "
    f"horizon ({preds.shape[2]})"
)
gt_xy = batch.ego_future[:, -preds.shape[2]:, :2].cpu().numpy()
if KINEMATIC == "frenet":
    centerline = select_reference_centerline(route_lanes=batch.routes, lanes=batch.lanes)
    pred_xy = frenet_to_cartesian(preds[:, 0, :, :2], centerline).cpu().numpy()
    centerline_np = centerline.cpu().numpy()
else:
    pred_xy = preds[:, 0, :, :2].cpu().numpy()
    centerline_np = None


def _render_gif(scene_local_idx: int, scene_global_idx: int, out_path: str) -> None:
    """Render one rollout gif for a single scene.

    Each frame N draws the predicted trajectory + GT trajectory truncated to
    the first N+1 timesteps, with the current ego as a triangle marker at the
    head of the rendered path. Centerline (if any) and origin are static
    decorations.
    """
    gt = gt_xy[scene_local_idx]
    pred = pred_xy[scene_local_idx]
    n_steps = min(len(gt), len(pred))

    fig, ax = plt.subplots(figsize=(6, 6))
    if centerline_np is not None:
        ax.plot(
            centerline_np[scene_local_idx, :, 0],
            centerline_np[scene_local_idx, :, 1],
            color="grey",
            lw=1.5,
            alpha=0.5,
            label="centerline",
        )
    # Bounds: union of GT, pred, and centerline if present, padded.
    xs = list(gt[:, 0]) + list(pred[:, 0])
    ys = list(gt[:, 1]) + list(pred[:, 1])
    if centerline_np is not None:
        xs += list(centerline_np[scene_local_idx, :, 0])
        ys += list(centerline_np[scene_local_idx, :, 1])
    pad = 5.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"{PLOT_TITLE} — scene {scene_global_idx}")
    ax.scatter([0], [0], c="blue", s=80, marker="s", label="ego origin", zorder=5)

    (gt_line,) = ax.plot([], [], color="green", lw=2.5, label="GT future")
    (pred_line,) = ax.plot([], [], color="red", lw=2.5, ls="--", label="Predicted")
    (ego_marker,) = ax.plot([], [], color="red", marker="^", ms=12, ls="", zorder=6)
    ax.legend(loc="upper left", fontsize=8)

    def init():
        gt_line.set_data([], [])
        pred_line.set_data([], [])
        ego_marker.set_data([], [])
        return gt_line, pred_line, ego_marker

    def update(frame_idx):
        head = frame_idx + 1
        gt_line.set_data(gt[:head, 0], gt[:head, 1])
        pred_line.set_data(pred[:head, 0], pred[:head, 1])
        ego_marker.set_data([pred[frame_idx, 0]], [pred[frame_idx, 1]])
        return gt_line, pred_line, ego_marker

    anim = animation.FuncAnimation(
        fig, update, init_func=init, frames=n_steps, interval=int(1000 / GIF_FPS), blit=True
    )
    writer = animation.PillowWriter(fps=GIF_FPS)
    anim.save(out_path, writer=writer)
    plt.close(fig)
    print(f"saved {out_path}")


for local_idx, global_idx in enumerate(indices):
    out_path = os.path.join(OUTPUT_DIR, f"rollout_{global_idx}.gif")
    _render_gif(local_idx, global_idx, out_path)

print(f"done — wrote {len(indices)} rollout gif(s) to {OUTPUT_DIR}")
