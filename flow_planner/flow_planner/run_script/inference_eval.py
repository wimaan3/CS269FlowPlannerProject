"""Inference + open-loop eval driver.

Runs under the same venv as training. Loads a checkpoint, runs inference on
the preprocessed scenarios, computes ADE/FDE, writes a JSON results summary.

Usage:
    python -m flow_planner.run_script.inference_eval \
        --checkpoint /content/drive/MyDrive/cs269/checkpoints/waypoints_seed42.ckpt \
        --data_dir /content/work/preprocessed_cache \
        --data_list /content/work/preprocessed_cache/diffusion_planner_training.json \
        --kinematic waypoints \
        --norm_stats waypoints_norm_stats \
        --output_json /content/work/eval_results.json \
        --num_batches 10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from flow_planner.data.dataset.nuplan import NuPlanDataset
from flow_planner.data.utils.collect import collect_batch
from flow_planner.data.normalization.frenet_utils import (
    frenet_to_cartesian,
    select_reference_centerline,
)


def main(args):
    # Pin every RNG we know about BEFORE instantiating the model. The
    # flow-ODE solver in FlowPlanner seeds itself from torch.randn at
    # each forward pass; with no global seed pinned, two re-runs of the
    # same checkpoint+data produce different ADE/FDE. That inflates the
    # noise floor and silently masks small representation differences
    # in paired comparisons (e.g. Frenet vs waypoints). Pin everything.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:  # noqa: BLE001
        print(f"  (could not enable deterministic algorithms: {e})")

    cfg_dir = str(Path(__file__).resolve().parents[1] / "script")
    print(f"Hydra config dir: {cfg_dir}")

    # Option A: if evaluating a Frenet checkpoint trained with explicit
    # reference-centerline conditioning, we must instantiate the SAME model
    # architecture (with CenterlineEncoder) so the checkpoint's centerline_encoder
    # weights are actually loaded. Without these overrides, the inference model
    # has no centerline_encoder, those weights become "unexpected", and Option A
    # has no effect at inference (silently degrading to v4 behavior).
    base_overrides = [
        f"model.kinematic={args.kinematic}",
        f"normalization_stats={args.norm_stats}",
        "ddp.distributed=false",
    ]
    if args.kinematic == "frenet" and not args.no_centerline_encoder:
        # `++` for enable_attn_dist because the key IS in the YAML
        # (model/flow_planner.yaml:118, default true). v7-AUDIT training
        # disables it for Frenet so the decoder doesn't see a
        # Cartesian-calibrated geometric bias. Eval must mirror that
        # override exactly — otherwise the model is instantiated with
        # enable_attn_dist=true (extra JointAttention.gen_taus Linear
        # layers per block), the checkpoint has no weights for those
        # layers, strict=False initialises them at random, and every
        # Frenet ADE/FDE silently reflects random attention bias.
        base_overrides += [
            "+model.model_encoder.centerline_encoder._target_=flow_planner.model.modules.encoder_modules.CenterlineEncoder",
            "+model.model_encoder.centerline_encoder.n_points=100",
            "+model.model_encoder.centerline_encoder.hidden_dim=256",
            "++model.model_decoder.enable_attn_dist=false",
        ]

    # v8 gap-fillers (CFG sweep, sample_steps sweep): allow caller-provided
    # Hydra overrides to vary inference-time hyperparameters like
    # `model.cfg_weight` or `model.flow_ode.sample_steps`. Appended AFTER
    # the base overrides so they win on any conflict.
    if getattr(args, "extra_overrides", None):
        base_overrides = list(base_overrides) + list(args.extra_overrides)
        print(f"  extra Hydra overrides: {args.extra_overrides}")

    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        cfg = compose(config_name="flow_planner_standard", overrides=base_overrides)

    print("Instantiating model...")
    model = instantiate(cfg.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # v8 Frenet-fix: optional inference-time hook (V4 multi-frame, V5 best-of-N, etc.)
    # The notebook sets FRENET_INFERENCE_HOOK=<module.path> before invoking this script.
    # The hook's `apply(model)` function monkey-patches FlowPlanner.forward_inference
    # to implement the inference-time technique. Applied AFTER instantiation and
    # BEFORE checkpoint load so the patched method picks up the loaded weights.
    _hook_path = os.environ.get("FRENET_INFERENCE_HOOK", "").strip()
    if _hook_path:
        import importlib
        try:
            hook_mod = importlib.import_module(_hook_path)
            hook_mod.apply(model)
            print(f"  inference hook applied: {_hook_path}")
        except Exception as e:
            # Hard fail rather than silently degrade to vanilla inference.
            # A mistyped FRENET_INFERENCE_HOOK or a runtime apply() crash
            # would otherwise produce "success" ADE/FDE numbers that
            # reflect single-sample inference rather than the requested
            # best-of-N or multi-frame hook — and the eval JSON would not
            # record the silent fallback. Mirror the strict state_dict
            # guard a few lines below.
            print(f"  ERROR: failed to apply inference hook {_hook_path}: {e}")
            raise RuntimeError(
                f"FRENET_INFERENCE_HOOK={_hook_path!r} could not be imported "
                "or applied. Refusing to silently degrade to vanilla inference. "
                "Unset FRENET_INFERENCE_HOOK to run un-patched inference "
                "intentionally."
            ) from e

    print(f"Loading checkpoint: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location=device)
    except Exception:
        # Fallback for legacy checkpoints that embed non-tensor Python objects
        # (e.g. third-party or pre-2.0 checkpoints). Only use on trusted sources.
        import warnings
        warnings.warn(
            f"torch.load(weights_only=True) failed for {args.checkpoint}; "
            "falling back to weights_only=False. Ensure the checkpoint file is from a trusted source.",
            stacklevel=1,
        )
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    if "ema_state_dict" in ckpt:
        sd = {k.replace("module.", ""): v for k, v in ckpt["ema_state_dict"].items()}
    elif "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    n_params_M = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded ({n_params_M:.2f}M params, {len(missing)} missing keys, {len(unexpected)} unexpected)")
    # Mirror the Frenet hardening for ALL kinematics. Previously the
    # warning was gated on `args.kinematic == "frenet"` only, which let
    # waypoints silently absorb the exact bug the Frenet branch loudly
    # rejected: if a v7 waypoints ckpt is loaded into a model whose
    # architecture has drifted (e.g. someone changes
    # `enable_attn_dist` or `cfg_type` defaults in the YAML), strict=False
    # silently randomly-init's the mismatched parameters and ADE/FDE
    # reflect random-init weights. Paired Frenet-vs-waypoints comparison
    # becomes meaningless because only one side has the guard.
    if missing or unexpected:
        # Centerline-gate-only mismatch on Frenet checkpoints is benign
        # (handled by the backward-compat init below). Suppress just
        # that one case so the loud warning is not noisy.
        ignorable = (
            args.kinematic == "frenet"
            and not unexpected
            and all("centerline_gate" in k for k in missing)
        )
        if not ignorable:
            print(f"  WARNING: state_dict mismatch on a {args.kinematic} checkpoint.")
            print(f"  Missing[:10]: {list(missing)[:10]}")
            print(f"  Unexpected[:10]: {list(unexpected)[:10]}")
            if not args.allow_state_dict_mismatch:
                raise RuntimeError(
                    "Refusing to evaluate with state_dict mismatch. Re-run with "
                    "--allow_state_dict_mismatch to override (numbers may reflect "
                    "random-init weights for the missing parameters)."
                )
            print("  --allow_state_dict_mismatch was set; proceeding anyway.")
    # Backward-compat: pre-Patch-4 Frenet checkpoints have centerline_encoder
    # weights but no `centerline_gate` parameter. The new gate constructor
    # initialises it to 0.0 (Patch 4 "identity at init"), which silently
    # ZEROS the centerline contribution at inference for old checkpoints.
    # Detect this exact case and restore pre-gate semantics by setting the
    # gate to 1.0. Controlled by --centerline_gate_init.
    if (
        args.kinematic == "frenet"
        and any("centerline_gate" in k for k in missing)
        and any("centerline_encoder" in k for k in sd)
    ):
        mode = getattr(args, "centerline_gate_init", "auto")
        if mode in ("auto", "one"):
            with torch.no_grad():
                model.model_encoder.centerline_gate.data.fill_(1.0)
            print(
                "  Initialised missing centerline_gate to 1.0 for backward "
                "compatibility with pre-Patch-4 Frenet checkpoints "
                f"(centerline_gate_init={mode})."
            )
        else:
            print(
                "  Skipped centerline_gate backward-compat init "
                f"(centerline_gate_init={mode}); Option A contribution will "
                "be zeroed at inference."
            )

    print(f"\nLoading dataset from {args.data_dir}")
    dataset = NuPlanDataset(
        data_dir=args.data_dir,
        data_list=args.data_list,
        past_neighbor_num=cfg.model.neighbor_num,
        predicted_neighbor_num=cfg.model.neighbor_pred_num,
        future_len=cfg.model.future_len,
        future_downsampling_method="uniform",
    )
    print(f"Dataset size: {len(dataset)}")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collect_batch,
    )

    all_ade, all_fde = [], []
    print(f"\nRunning inference on up to {args.num_batches} batches...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.to(device)
            preds = model(
                batch,
                mode="inference",
                use_cfg=False,
                cfg_weight=cfg.model.cfg_weight,
            )

            if args.kinematic == "frenet":
                # Pass ego_past_xy so select_reference_centerline can use
                # smart-centerline mode (anchored by recent past motion).
                # Inactive unless FRENET_SMART_CENTERLINE=1 in the
                # environment. CRITICAL: training-time call in
                # input_preprocess.py:159-163 DOES pass raw_ego_past_xy.
                # Without this argument, the two paths build different
                # centerlines under FRENET_SMART_CENTERLINE=1 (~9 m delta
                # in realistic off-route scenarios), and frenet_to_cartesian
                # decodes predictions in a reference frame the model was
                # NOT trained against — silently corrupting ADE/FDE.
                centerline = select_reference_centerline(
                    route_lanes=batch.routes,
                    lanes=batch.lanes,
                    ego_past_xy=batch.ego_past[..., :2],
                )
                pred_xy = frenet_to_cartesian(preds[:, 0, :, :2], centerline)
            elif args.kinematic in ("velocity", "acceleration"):
                pred_dxy = preds[:, 0, :, :2]
                ego_xy0 = batch.ego_current[:, :2].unsqueeze(1)
                pred_xy = ego_xy0 + pred_dxy.cumsum(dim=1)
            else:  # waypoints
                pred_xy = preds[:, 0, :, :2]

            # Match the training-time slice in ModelInputProcessor.sample_to_model_input
            # (input_preprocess.py:54): `ego_future[..., -self.future_len:, :3]`.
            # The on-disk cache may store more timesteps than future_len; the
            # model is trained on the LAST `future_len` of those, so eval must
            # compare against the SAME slice — otherwise ADE/FDE reflect a
            # comparison between two non-overlapping time windows.
            assert batch.ego_future.shape[1] >= preds.shape[2], (
                f"ego_future too short ({batch.ego_future.shape[1]}) for prediction "
                f"horizon ({preds.shape[2]})"
            )
            gt_xy = batch.ego_future[:, -preds.shape[2]:, :2]
            disp = torch.linalg.vector_norm(pred_xy - gt_xy, dim=-1)
            all_ade.append(disp.mean(dim=-1).cpu())
            all_fde.append(disp[:, -1].cpu())

            print(f"  batch {i}: ADE={disp.mean(dim=-1).mean().item():.3f}, "
                  f"FDE={disp[:, -1].mean().item():.3f}")

            if i + 1 >= args.num_batches:
                break

    all_ade = torch.cat(all_ade).numpy()
    all_fde = torch.cat(all_fde).numpy()

    # Record the configuration that drives behavior so paired comparison
    # results stored to disk can be unambiguously matched back to their
    # source configuration. Without this snapshot, two runs of the same
    # checkpoint with different FRENET_* env-var values (best-of-N N=16
    # vs N=1, smart centerline on/off, etc.) write identical-looking
    # results and the reader cannot tell which condition produced the
    # number. Pull all known FRENET_* keys plus the script-level args
    # that affect output.
    _frenet_env = {
        k: v for k, v in os.environ.items()
        if k.startswith("FRENET_")
    }
    results = {
        "kinematic": args.kinematic,
        "num_scenarios_evaluated": int(len(all_ade)),
        "ade_mean": float(all_ade.mean()),
        "ade_std": float(all_ade.std()),
        "fde_mean": float(all_fde.mean()),
        "fde_std": float(all_fde.std()),
        "checkpoint": args.checkpoint,
        "seed": int(args.seed),
        "norm_stats": args.norm_stats,
        "inference_hook": _hook_path or None,
        "frenet_env": _frenet_env,
        "num_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
    }

    print("\n=== Results ===")
    for k, v in results.items():
        print(f"  {k}: {v}")

    out = Path(args.output_json)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--data_list", required=True)
    parser.add_argument("--kinematic", default="waypoints",
                        choices=["waypoints", "velocity", "acceleration", "frenet"])
    parser.add_argument("--norm_stats", default="waypoints_norm_stats")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--no_centerline_encoder", action="store_true",
                        help="(Frenet only) skip the Option A centerline encoder override. "
                             "Use this when evaluating a v4-or-earlier Frenet checkpoint "
                             "that was trained without Option A.")
    parser.add_argument("--centerline_gate_init",
                        default="auto", choices=["auto", "zero", "one"],
                        help="(Frenet only) how to initialise centerline_gate when it is "
                             "missing from the checkpoint. 'auto'/'one' = restore pre-Patch-4 "
                             "semantics (gate=1.0); 'zero' = keep the constructor default "
                             "(gate=0.0, silently disabling Option A at inference).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for python/numpy/torch RNGs. Pins the flow-ODE x_init "
                             "draw so paired runs differ only in checkpoint+arch, not noise.")
    parser.add_argument("--allow_state_dict_mismatch", action="store_true",
                        help="Override the hard-stop on missing/unexpected state_dict keys. "
                             "Use only when intentionally evaluating a checkpoint whose "
                             "architecture differs from the inference model.")
    parser.add_argument("--extra_overrides", nargs="*", default=None,
                        help="Extra Hydra overrides appended to base_overrides "
                             "(e.g. 'model.cfg_weight=1.0' 'model.flow_ode.sample_steps=8'). "
                             "Used by the CFG_weight and sample_steps sweep cells in "
                             "v8_frenet_fixes.ipynb to vary inference-time hyperparameters.")
    args = parser.parse_args()
    main(args)
