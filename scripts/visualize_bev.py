"""BEV (bird's-eye view) visualization for a trained checkpoint.

Loads a checkpoint, runs one inference batch on the preprocessed cache,
decodes predictions back to Cartesian (x, y) — via the reference
centerline for Frenet, identity for waypoints — and saves a side-by-side
comparison of GT vs prediction for the first four scenes in the batch.

Parameters via environment variables (set by the notebook cell):
    FP_DIR       Flow Planner source directory (added to sys.path)
    CKPT_PATH    Path to the .ckpt file to load
    CACHE_DIR    Directory containing preprocessed .npz + manifest JSON
    OUTPUT_PNG   Where to save the resulting PNG
    PLOT_TITLE   Optional figure suptitle (default: "BEV")
    KINEMATIC    'frenet' (default) | 'waypoints' — which decode path
                 to use and which Hydra overrides to apply
    NORM_STATS   Hydra config name for normalization stats. Defaults to
                 frenet_norm_stats for Frenet, waypoints_norm_stats otherwise.

Previously lived inline in notebook cell 50 as a triple-quoted f-string
written to /tmp/viz_v3.py. Extracted because the nested-Python-in-Python
pattern was a render-time hotspot for GitHub's notebook viewer.

Prior to the v7 audit, this script hard-coded the Frenet code path. When
called with a waypoints checkpoint, it loaded the waypoint weights into
a Frenet-shaped model with strict=False (most encoder weights line up by
name; the Frenet-only centerline_encoder weights are left random) and
then interpreted the waypoint (x, y) predictions as Frenet (s, d),
applying frenet_to_cartesian to them — producing meaningless trajectories
that were silently plotted as if valid.
"""
from __future__ import annotations

import os
import sys

# Make flow_planner importable from the editable checkout.
sys.path.insert(0, os.environ["FP_DIR"])

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
OUTPUT_PNG = os.environ["OUTPUT_PNG"]
KINEMATIC = os.environ.get("KINEMATIC", "frenet")
if KINEMATIC not in {"frenet", "waypoints"}:
    raise SystemExit(f"KINEMATIC must be 'frenet' or 'waypoints', got {KINEMATIC!r}")
DEFAULT_NORM_STATS = "frenet_norm_stats" if KINEMATIC == "frenet" else "waypoints_norm_stats"
NORM_STATS = os.environ.get("NORM_STATS", DEFAULT_NORM_STATS)
PLOT_TITLE = os.environ.get("PLOT_TITLE", f"{KINEMATIC.title()} BEV")

# --- Build Hydra overrides. For Frenet we also wire in Option A
# (centerline_encoder) and force-disable the Cartesian-calibrated
# enable_attn_dist gating so the architecture matches v7-AUDIT training
# exactly. For waypoints we do NOT add either override — the checkpoint
# was trained against the YAML defaults. ---
overrides = [
    f"model.kinematic={KINEMATIC}",
    f"normalization_stats={NORM_STATS}",
    "ddp.distributed=false",
]
if KINEMATIC == "frenet":
    overrides += [
        "+model.model_encoder.centerline_encoder._target_="
        "flow_planner.model.modules.encoder_modules.CenterlineEncoder",
        "+model.model_encoder.centerline_encoder.n_points=100",
        "+model.model_encoder.centerline_encoder.hidden_dim=256",
        # `++` because the key IS in the YAML (default true). v7-AUDIT
        # training disables it for Frenet; eval must mirror that or the
        # JointAttention.gen_taus layers exist only at eval time and are
        # initialised randomly, corrupting attention bias.
        "++model.model_decoder.enable_attn_dist=false",
    ]

config_dir = os.path.join(os.environ["FP_DIR"], "flow_planner/script")
with initialize_config_dir(version_base=None, config_dir=config_dir):
    cfg = compose(config_name="flow_planner_standard", overrides=overrides)

model = instantiate(cfg.model)
device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load(CKPT_PATH, weights_only=False, map_location=device)
sd = ckpt.get("ema_state_dict", ckpt.get("state_dict", ckpt))
sd = {k.replace("module.", ""): v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd, strict=False)
if missing or unexpected:
    # Loud warning so a train/eval architecture mismatch is never silent.
    print(
        f"WARNING: state_dict mismatch on a {KINEMATIC} checkpoint. "
        f"missing[:5]={list(missing)[:5]} unexpected[:5]={list(unexpected)[:5]}"
    )
# Backward-compat: pre-Patch-4 Frenet checkpoints have centerline_encoder
# weights but no `centerline_gate` parameter (Patch 4 added it). The new
# gate constructor initialises to 0.0, which silently zeros the centerline
# contribution at inference. Restore pre-gate semantics by setting the gate
# to 1.0 — matches the behavior the checkpoint was trained against.
if (
    KINEMATIC == "frenet"
    and any("centerline_gate" in k for k in missing)
    and any("centerline_encoder" in k for k in sd)
):
    with torch.no_grad():
        model.model_encoder.centerline_gate.data.fill_(1.0)
    print(
        "Initialised missing centerline_gate to 1.0 for backward "
        "compatibility with pre-Patch-4 Frenet checkpoints."
    )
model = model.to(device).eval()

# --- Dataset + one batch ---
# Earlier versions used shuffle=False which always plotted the first 4 scenes
# alphabetically by scenario token (cherry-picked, unrepresentative).
# Now: random sample with a seed (BEV_SEED env var, default 42) so the figure
# is reproducible per run-name but spans the actual scenario distribution.
ds = NuPlanDataset(
    data_dir=CACHE_DIR,
    data_list=os.path.join(CACHE_DIR, "diffusion_planner_training.json"),
    past_neighbor_num=cfg.model.neighbor_num,
    predicted_neighbor_num=cfg.model.neighbor_pred_num,
    future_len=cfg.model.future_len,
    future_downsampling_method="uniform",
)
import random
_bev_seed = int(os.environ.get("BEV_SEED", "42"))
random.seed(_bev_seed)
torch.manual_seed(_bev_seed)
_n_total = len(ds)
_n_to_sample = min(4, _n_total)
_indices = random.sample(range(_n_total), _n_to_sample)
print(f"BEV sampling {_n_to_sample} of {_n_total} scenarios (BEV_SEED={_bev_seed}): {_indices}")
_subset = torch.utils.data.Subset(ds, _indices)
loader = DataLoader(_subset, batch_size=_n_to_sample, shuffle=False, collate_fn=collect_batch)
batch = next(iter(loader)).to(device)

with torch.no_grad():
    preds = model(batch, mode="inference", use_cfg=False, cfg_weight=cfg.model.cfg_weight)

# --- Decode predictions to Cartesian (x, y). For Frenet we use the same
# raw-scale centerline the model trained against (Bug 2 contract); for
# waypoints the model already outputs (x, y) so the decode is identity. ---
# Match the training-time slice in ModelInputProcessor (input_preprocess.py:54):
# `ego_future[..., -self.future_len:, :3]`. Slicing from index 0 instead of -N
# silently compares predictions to a different time window when the cache stores
# more timesteps than future_len.
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
    # Waypoints: predictions are already in ego-frame Cartesian.
    pred_xy = preds[:, 0, :, :2].cpu().numpy()
    centerline_np = None

# --- Plot ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i in range(4):
    ax = axes[i]
    if centerline_np is not None:
        ax.plot(
            centerline_np[i, :, 0],
            centerline_np[i, :, 1],
            "k--",
            lw=1,
            alpha=0.5,
            label="centerline",
        )
    ax.plot(gt_xy[i, :, 0], gt_xy[i, :, 1], "g-", lw=2, label="GT")
    ax.plot(pred_xy[i, :, 0], pred_xy[i, :, 1], "r--", lw=2, label="Pred")
    ax.scatter([0], [0], c="b", s=100, zorder=5)
    ax.set_title(f"Scene {i}")
    ax.axis("equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle(PLOT_TITLE)
plt.savefig(OUTPUT_PNG, dpi=100, bbox_inches="tight")
plt.close()
print(f"saved {OUTPUT_PNG}")
