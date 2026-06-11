# `results/` — eval-JSON sidecars for paper-reported numbers

This directory ships the verbatim `eval_*.json` outputs produced by
`flow_planner/run_script/inference_eval.py` for every figure / table row in
the CS269 final report. Each file is the raw evaluator output (ADE/FDE +
metadata) — nothing post-processed.

## Index — paper claim → JSON file

### Paper Table 1 (5,000-scenario headline)

| Row | Representation | n_eval | ADE | FDE | JSON |
|---|---|---:|---:|---:|---|
| 1 | Frenet — Original | 4,992 | 22.66 | 27.50 | [`headline_5k_frenet/eval_frenet_val.json`](headline_5k_frenet/eval_frenet_val.json) |
| 2 | Frenet — Fix B+C | 4,992 | **21.00** | 26.42 | [`headline_5k_frenet/eval_frenet_val_fixedCB.json`](headline_5k_frenet/eval_frenet_val_fixedCB.json) |
| 3 | V+A combined (5k) | — | — | — | **NOT IN REPO** — eval JSON never produced locally; see `notes` below. |
| 4 | Waypoints (5k) | — | — | — | **NOT IN REPO** — comparison run pending; the 1.5k waypoints number in `motion_representations/` is the surrogate used in the paper. |

### Paper Table 5 (1,500-scenario zoo, seed=269)

| Representation | n_eval | ADE | FDE | JSON |
|---|---:|---:|---:|---|
| Waypoints | 300 | 4.19 | 8.63 | [`motion_representations/eval_motion_waypoints_seed269.json`](motion_representations/eval_motion_waypoints_seed269.json) |
| Acceleration | 300 | 19.98 | 35.60 | [`motion_representations/eval_motion_acceleration_seed269.json`](motion_representations/eval_motion_acceleration_seed269.json) |
| Velocity | 300 | 22.30 | 39.38 | [`motion_representations/eval_motion_velocity_seed269.json`](motion_representations/eval_motion_velocity_seed269.json) |
| Frenet | 300 | 25.77 | 37.77 | [`motion_representations/eval_motion_frenet_seed269.json`](motion_representations/eval_motion_frenet_seed269.json) |

### Paper baseline (pristine upstream Flow Planner)

| Run | n_eval | ADE | FDE | JSON |
|---|---:|---:|---:|---|
| Waypoints (paper-faithful, seed=269) | 300 | 4.21 | 8.78 | [`paper_baseline/eval_paper_waypoints.json`](paper_baseline/eval_paper_waypoints.json) |

### Paper §6 — Frenet inference-variant ablation (V0–V11)

All 14 runs are 300-scenario evals of the Frenet checkpoint with different
inference-time configurations. Used to populate the V0–V11 matrix figure.

| Variant | ADE | FDE | JSON |
|---|---:|---:|---|
| V0 baseline | 26.55 | 42.40 | [`v8_frenet_fixes/eval_V0_baseline.json`](v8_frenet_fixes/eval_V0_baseline.json) |
| V1 stacked | 35.84 | 45.61 | [`v8_frenet_fixes/eval_V1_stacked.json`](v8_frenet_fixes/eval_V1_stacked.json) |
| V4 multiframe v1 | 35.40 | 44.42 | [`v8_frenet_fixes/eval_V4_multiframe_v1.json`](v8_frenet_fixes/eval_V4_multiframe_v1.json) |
| V5 best-of-N v1 | 30.84 | 40.51 | [`v8_frenet_fixes/eval_V5_best_of_N_v1.json`](v8_frenet_fixes/eval_V5_best_of_N_v1.json) |
| V7 (seed 1337) | 33.09 | 42.91 | [`v8_frenet_fixes/eval_V7_seed1337.json`](v8_frenet_fixes/eval_V7_seed1337.json) |
| V8 (CFG ω=1.0) | 26.55 | 42.40 | [`v8_frenet_fixes/eval_V8_cfg_1.0.json`](v8_frenet_fixes/eval_V8_cfg_1.0.json) |
| V8 (CFG ω=1.4) | 26.55 | 42.40 | [`v8_frenet_fixes/eval_V8_cfg_1.4.json`](v8_frenet_fixes/eval_V8_cfg_1.4.json) |
| V8 (CFG ω=1.8) | 26.55 | 42.40 | [`v8_frenet_fixes/eval_V8_cfg_1.8.json`](v8_frenet_fixes/eval_V8_cfg_1.8.json) |
| V8 (CFG ω=2.5) | 26.55 | 42.40 | [`v8_frenet_fixes/eval_V8_cfg_2.5.json`](v8_frenet_fixes/eval_V8_cfg_2.5.json) |
| V9 (4 ODE steps) | 26.55 | 42.40 | [`v8_frenet_fixes/eval_V9_steps_4.json`](v8_frenet_fixes/eval_V9_steps_4.json) |
| V9 (8 ODE steps) | 26.77 | 42.57 | [`v8_frenet_fixes/eval_V9_steps_8.json`](v8_frenet_fixes/eval_V9_steps_8.json) |
| V9 (16 ODE steps) | 26.92 | 42.68 | [`v8_frenet_fixes/eval_V9_steps_16.json`](v8_frenet_fixes/eval_V9_steps_16.json) |
| V9 (32 ODE steps) | 27.04 | 42.80 | [`v8_frenet_fixes/eval_V9_steps_32.json`](v8_frenet_fixes/eval_V9_steps_32.json) |
| V11 velocity | 25.94 | 43.22 | [`v8_frenet_fixes/eval_V11_velocity.json`](v8_frenet_fixes/eval_V11_velocity.json) |

### Paper §6.3 B-check — Frenet centerline gate ablation

300-scenario evals of forced-gate-value variants used to verify the
centerline gate is actually learning and not just being silently ignored.

| Variant | Gate value | ADE | FDE | JSON |
|---|---|---:|---:|---|
| Trained gate (default) | learned | 26.55 | 42.40 | [`force_gate_sweep/eval_gate_auto.json`](force_gate_sweep/eval_gate_auto.json) |
| Forced 0.0 | 0.0 | 26.55 | 42.40 | [`force_gate_sweep/eval_gate_zero.json`](force_gate_sweep/eval_gate_zero.json) |
| Forced +1.0 | +1.0 | 26.55 | 42.40 | [`force_gate_sweep/eval_gate_one.json`](force_gate_sweep/eval_gate_one.json) |
| Forced −1.0 | −1.0 | 26.55 | 42.40 | [`force_gate_sweep/eval_gate_neg_one.json`](force_gate_sweep/eval_gate_neg_one.json) |
| v3 forced 0.0 | 0.0 | 31.86 | 44.19 | [`force_gate_sweep/eval_zero_v3.json`](force_gate_sweep/eval_zero_v3.json) |
| v3 forced +1.0 | +1.0 | 48.87 | 63.66 | [`force_gate_sweep/eval_one_v3.json`](force_gate_sweep/eval_one_v3.json) |
| v3 forced −1.0 | −1.0 | 32.29 | 41.16 | [`force_gate_sweep/eval_neg_one_v3.json`](force_gate_sweep/eval_neg_one_v3.json) |
| v3 EMA gate | learned (EMA) | 23.53 | 32.04 | [`force_gate_sweep/eval_auto_ema_v3.json`](force_gate_sweep/eval_auto_ema_v3.json) |

## How to regenerate any of these

1. Load the appropriate checkpoint (see `flow_planner/script/...yaml` for
   matching Hydra config).
2. Run `python flow_planner/run_script/inference_eval.py
   --checkpoint <ckpt> --kinematic <k> [other overrides]`.
3. Compare the resulting JSON against the file under the matching
   subdirectory here.

## Notes

- **V+A 5k eval is missing.** The combined V+A model was trained on Jiali's
  DagsHub fork (`s3://dagshub-drive/Flow-Planner-main-2/`); the eval JSON
  for the headline 5k V+A row never made it back to this machine. With the
  V+A code port now in `main` (commit `c1e9665`), the run can be reproduced
  by setting `RUN_KINEMATIC = 'va'` in
  [`notebooks/cs269_dagshub_best_ever_frenet.ipynb`](../notebooks/cs269_dagshub_best_ever_frenet.ipynb).
- **5k Waypoints comparison eval is missing.** Same notebook can run it
  with `RUN_KINEMATIC = 'waypoints'`.
- **The eight `force_gate_sweep` files marked `seed=?`** were produced
  before the eval pipeline added the `seed` field; treat them as seed=0.
- **`num_scenarios_evaluated = 300`** for ablation runs — these were
  intentionally sampled to keep iteration time short. The 4,992 in the
  5k Frenet headline rows is full nuPlan-mini val.
