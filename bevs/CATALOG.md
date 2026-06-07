# BEV Catalog

Detailed per-file metadata for every bird's-eye-view artifact under `bevs/`. The machine-readable form is [`catalog.json`](./catalog.json); this document is its human-readable mirror.

## Coverage summary

All four motion representations (Waypoints, Frenet, Velocity, Acceleration) are covered at seed=269 / 1.5k via the `motion_representations` checkpoints — Waypoints and Frenet additionally have per-scene rollout GIFs (scenes 12, 57, 125, 140), and Velocity appears as the V11 control inside the v8 sweep. Frenet has a full audit progression (v0 → v1 oldnorm → v2 fixednorm → v3 fixedcenterline → v7AUDIT) at seed=42, plus a V0–V11 inference-variant sweep (stacked past, multi-frame, best-of-N, seed swap, CFG 1.0/1.4/1.8/2.5, Euler steps 4/8/16/32, velocity control) at seed=269 against the v7AUDIT checkpoint. Waypoints has the legacy seed=42 baseline, the v7AUDIT seed=269 headline result (ADE 4.36 m), and a paper-baseline multi-seed reproduction matrix. Acceleration is represented in the motion-matrix summary figure only (no rollouts retained locally). Local preprocessed `.npz` cache is unavailable, so none of these can be regenerated from the listed checkpoints without re-running the dataset pipeline.

**Totals:** 76 catalog entries (12 PNG, 64 GIF) — 7 four-scene static, 2 sixteen-scene multi, 3 matrices, 64 rollouts.

---

## By Representation

### Waypoints (8)

| File | Config | Seed | Scenes | Kind | ADE (m) | Source checkpoint |
|---|---|---|---|---|---|---|
| `bevs/static/waypoints_seed42_4scenes.png` | seed42_baseline | 42 | 4 | 4-scene static | 5.54 | (legacy seed=42 1.5k; not retained locally) |
| `bevs/static/waypoints_v7AUDIT_seed269_4scenes.png` | v7AUDIT | 269 | 4 | 4-scene static | 4.36 | `v8_frenet_fixes/v7AUDIT_waypoints_seed269.ckpt` |
| `bevs/static/waypoints_v7AUDIT_seed269_16scenes.png` | v7AUDIT | 269 | 16 | 16-scene multi | 4.36 | `v8_frenet_fixes/v7AUDIT_waypoints_seed269.ckpt` |
| `bevs/rollouts/waypoints_seed269_scene12.gif` | motion_waypoints | 269 | 1 | rollout | — | `motion_representations/checkpoints/motion_waypoints_seed269.ckpt` |
| `bevs/rollouts/waypoints_seed269_scene57.gif` | motion_waypoints | 269 | 1 | rollout | — | `motion_representations/checkpoints/motion_waypoints_seed269.ckpt` |
| `bevs/rollouts/waypoints_seed269_scene125.gif` | motion_waypoints | 269 | 1 | rollout | — | `motion_representations/checkpoints/motion_waypoints_seed269.ckpt` |
| `bevs/rollouts/waypoints_seed269_scene140.gif` | motion_waypoints | 269 | 1 | rollout | — | `motion_representations/checkpoints/motion_waypoints_seed269.ckpt` |
| `bevs/matrices/paper_baseline_waypoints_seeds.png` | paper_baseline | multi-seed | — | matrix | — | `paper_baseline/checkpoints/paper_baseline_waypoints_seed269.ckpt` |

### Frenet (63)

| File | Config | Seed | Scenes | Kind | ADE (m) | Source checkpoint |
|---|---|---|---|---|---|---|
| `bevs/static/frenet_seed42_v0_4scenes.png` | v0_no_fixes | 42 | 4 | 4-scene static | ~40 | (legacy seed=42 Frenet 1.5k; not retained locally) |
| `bevs/static/frenet_seed42_v1_oldnorm_4scenes.png` | v1_oldnorm | 42 | 4 | 4-scene static | — | (legacy seed=42 Frenet 1.5k) |
| `bevs/static/frenet_seed42_v2_fixednorm_4scenes.png` | v2_fixednorm | 42 | 4 | 4-scene static | — | (legacy seed=42 Frenet 1.5k) |
| `bevs/static/frenet_seed42_v3_fixedcenterline_4scenes.png` | v3_fixedcenterline | 42 | 4 | 4-scene static | — | (legacy seed=42 Frenet 1.5k) |
| `bevs/static/frenet_v7AUDIT_seed269_4scenes.png` | v7AUDIT | 269 | 4 | 4-scene static | ~22 | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` |
| `bevs/static/frenet_v7AUDIT_seed269_16scenes.png` | v7AUDIT | 269 | 16 | 16-scene multi | ~22 | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` |
| `bevs/matrices/frenet_inference_variants_v0_to_v11.png` | v8_inference_variants | 269 | — | matrix | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (varied inference knobs) |
| `bevs/rollouts/frenet_seed269_scene{12,57,125,140}.gif` (4) | motion_frenet | 269 | 1 | rollout | — | `motion_representations/checkpoints/motion_frenet_seed269.ckpt` |
| `bevs/rollouts/frenet_V0_baseline_scene{12,57,125,140}.gif` (4) | v8_V0_baseline | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` |
| `bevs/rollouts/frenet_V1_stacked_scene{12,57,125,140}.gif` (4) | v8_V1_stacked | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` |
| `bevs/rollouts/frenet_V4_multiframe_v1_scene{12,57,125,140}.gif` (4) | v8_V4_multiframe | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` |
| `bevs/rollouts/frenet_V5_best_of_N_v1_scene{12,57,125,140}.gif` (4) | v8_V5_best_of_N | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` |
| `bevs/rollouts/frenet_V7_seed1337_scene{12,57,125,140}.gif` (4) | v8_V7_seed1337 | 1337 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (inference seed=1337) |
| `bevs/rollouts/frenet_V8_cfg_1.0_scene{12,57,125,140}.gif` (4) | v8_V8_cfg_1.0 | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (omega=1.0) |
| `bevs/rollouts/frenet_V8_cfg_1.4_scene{12,57,125,140}.gif` (4) | v8_V8_cfg_1.4 | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (omega=1.4) |
| `bevs/rollouts/frenet_V8_cfg_1.8_scene{12,57,125,140}.gif` (4) | v8_V8_cfg_1.8 | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (omega=1.8) |
| `bevs/rollouts/frenet_V8_cfg_2.5_scene{12,57,125,140}.gif` (4) | v8_V8_cfg_2.5 | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (omega=2.5) |
| `bevs/rollouts/frenet_V9_steps_4_scene{12,57,125,140}.gif` (4) | v8_V9_steps_4 | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (Euler steps=4) |
| `bevs/rollouts/frenet_V9_steps_8_scene{12,57,125,140}.gif` (4) | v8_V9_steps_8 | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (Euler steps=8) |
| `bevs/rollouts/frenet_V9_steps_16_scene{12,57,125,140}.gif` (4) | v8_V9_steps_16 | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (Euler steps=16) |
| `bevs/rollouts/frenet_V9_steps_32_scene{12,57,125,140}.gif` (4) | v8_V9_steps_32 | 269 | 1 | rollout | — | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (Euler steps=32, default) |

### Velocity (4)

These ship under the `frenet_V11_velocity_*` filename prefix because they were captured inside the v8 Frenet sweep as a representation control. The underlying model is the V11 velocity checkpoint, not a Frenet model.

| File | Config | Seed | Scenes | Kind | Source checkpoint |
|---|---|---|---|---|---|
| `bevs/rollouts/frenet_V11_velocity_scene12.gif` | v8_V11_velocity | 269 | 1 | rollout | `v8_frenet_fixes/checkpoints/v8_V11_velocity_seed269.ckpt` |
| `bevs/rollouts/frenet_V11_velocity_scene57.gif` | v8_V11_velocity | 269 | 1 | rollout | `v8_frenet_fixes/checkpoints/v8_V11_velocity_seed269.ckpt` |
| `bevs/rollouts/frenet_V11_velocity_scene125.gif` | v8_V11_velocity | 269 | 1 | rollout | `v8_frenet_fixes/checkpoints/v8_V11_velocity_seed269.ckpt` |
| `bevs/rollouts/frenet_V11_velocity_scene140.gif` | v8_V11_velocity | 269 | 1 | rollout | `v8_frenet_fixes/checkpoints/v8_V11_velocity_seed269.ckpt` |

### Acceleration

No standalone BEV PNGs or rollout GIFs are retained locally for Acceleration. It appears only as one bar in `bevs/matrices/motion_representation_4kinematics_ade.png` (~20 m ADE at seed=269 / 1.5k), trained from `motion_representations/checkpoints/motion_acceleration_seed269.ckpt`.

### Mixed (1)

| File | Config | Seed | Kind | Notes |
|---|---|---|---|---|
| `bevs/matrices/motion_representation_4kinematics_ade.png` | motion_representations | 269 | matrix | Bar chart spanning all four representations. |

---

## By Configuration

### `seed42_baseline` (legacy Waypoints, pre-audit)

| File | Representation | ADE (m) | FDE (m) |
|---|---|---|---|
| `bevs/static/waypoints_seed42_4scenes.png` | waypoints | 5.54 | 10.86 |

### `v0_no_fixes` (pre-audit Frenet)

| File | Representation | Notes |
|---|---|---|
| `bevs/static/frenet_seed42_v0_4scenes.png` | frenet | ADE >40 m — failure-mode motivator. |

### `v1_oldnorm` (audit step 1)

| File | Representation | Notes |
|---|---|---|
| `bevs/static/frenet_seed42_v1_oldnorm_4scenes.png` | frenet | Regenerated with the original (incorrect) normalization YAML. |

### `v2_fixednorm` (audit step 2)

| File | Representation | Notes |
|---|---|---|
| `bevs/static/frenet_seed42_v2_fixednorm_4scenes.png` | frenet | Re-measured normalization statistics applied at decode time. |

### `v3_fixedcenterline` (audit step 3 — paper Figure 3)

| File | Representation | Notes |
|---|---|---|
| `bevs/static/frenet_seed42_v3_fixedcenterline_4scenes.png` | frenet | Smart centerline picker; surfaces dramatic lateral-collapse failure modes. |

### `v7AUDIT` (headline paper checkpoint)

| File | Representation | Scenes | ADE (m) |
|---|---|---|---|
| `bevs/static/waypoints_v7AUDIT_seed269_4scenes.png` | waypoints | 4 | 4.36 |
| `bevs/static/waypoints_v7AUDIT_seed269_16scenes.png` | waypoints | 16 | 4.36 |
| `bevs/static/frenet_v7AUDIT_seed269_4scenes.png` | frenet | 4 | ~22 |
| `bevs/static/frenet_v7AUDIT_seed269_16scenes.png` | frenet | 16 | ~22 |

### `motion_representations` (4-kinematics baselines)

| File | Representation | Notes |
|---|---|---|
| `bevs/matrices/motion_representation_4kinematics_ade.png` | mixed | Bar chart: Waypoints ~4 m, Acceleration ~20 m, Velocity ~22 m, Frenet ~26 m (seed=269 / 1.5k). |
| `bevs/rollouts/motion_waypoints_seed269_scene*.gif` (4) | waypoints | One rollout per scene 12/57/125/140. |
| `bevs/rollouts/motion_frenet_seed269_scene*.gif` (4) | frenet | One rollout per scene 12/57/125/140. |

### `v8_*` (Frenet inference variants, paper `v8_frenet_fixes.ipynb`)

All rollouts share the same trained Frenet checkpoint (`v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt`) except V11, which uses the velocity ckpt.

| Variant | Inference knob | # rollouts |
|---|---|---|
| V0_baseline | Default (CFG=1.4, 32 Euler steps, single-frame past) | 4 |
| V1_stacked | Stacked past-frame features | 4 |
| V4_multiframe_v1 | Multi-frame past conditioning | 4 |
| V5_best_of_N_v1 | Best-of-32 sampling | 4 |
| V7_seed1337 | Sampling seed 1337 instead of 269 | 4 |
| V8_cfg_1.0 | CFG omega = 1.0 (no guidance) | 4 |
| V8_cfg_1.4 | CFG omega = 1.4 (default) | 4 |
| V8_cfg_1.8 | CFG omega = 1.8 (stronger guidance) | 4 |
| V8_cfg_2.5 | CFG omega = 2.5 (much stronger) | 4 |
| V9_steps_4 | ODE Euler steps = 4 | 4 |
| V9_steps_8 | ODE Euler steps = 8 | 4 |
| V9_steps_16 | ODE Euler steps = 16 | 4 |
| V9_steps_32 | ODE Euler steps = 32 (default) | 4 |
| V11_velocity | Velocity-representation control | 4 |

### `paper_baseline` (Waypoints reproducibility across seeds)

| File | Representation | Notes |
|---|---|---|
| `bevs/matrices/paper_baseline_waypoints_seeds.png` | waypoints | Bar chart of paper-baseline Waypoints reproductions across multiple seeds at 1.5k scenarios. |

---

## Matrices and Summary Figures

| File | Description | Source checkpoint(s) |
|---|---|---|
| `bevs/matrices/motion_representation_4kinematics_ade.png` | Held-out ADE bar chart across Waypoints / Velocity / Acceleration / Frenet at seed=269 / 1.5k. | `motion_representations/checkpoints/motion_{waypoints,frenet,velocity,acceleration}_seed269.ckpt` |
| `bevs/matrices/frenet_inference_variants_v0_to_v11.png` | Grid matrix of the 14 Frenet inference variants (V0–V11). | `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` (single base, varied inference knobs) |
| `bevs/matrices/paper_baseline_waypoints_seeds.png` | Paper-baseline Waypoints reproductions across multiple seeds. | `paper_baseline/checkpoints/paper_baseline_waypoints_seed269.ckpt` + seed reproductions |

---

## Rollout GIFs

64 GIFs total: 4 scenes (12, 57, 125, 140) × 16 model configurations.

### Motion-representation baseline rollouts (seed=269, 1.5k, 50 epochs)

| Group | Files | Source checkpoint |
|---|---|---|
| Waypoints | `bevs/rollouts/waypoints_seed269_scene{12,57,125,140}.gif` | `motion_representations/checkpoints/motion_waypoints_seed269.ckpt` |
| Frenet | `bevs/rollouts/frenet_seed269_scene{12,57,125,140}.gif` | `motion_representations/checkpoints/motion_frenet_seed269.ckpt` |

### Frenet inference-variant rollouts (V0–V11)

All against `v8_frenet_fixes/v7AUDIT_frenet_seed269.ckpt` unless otherwise noted.

| Variant | File pattern | Notes |
|---|---|---|
| V0_baseline | `bevs/rollouts/frenet_V0_baseline_scene{12,57,125,140}.gif` | Default inference settings. |
| V1_stacked | `bevs/rollouts/frenet_V1_stacked_scene{12,57,125,140}.gif` | Stacked past-frame features. |
| V4_multiframe_v1 | `bevs/rollouts/frenet_V4_multiframe_v1_scene{12,57,125,140}.gif` | Multi-frame past conditioning. |
| V5_best_of_N_v1 | `bevs/rollouts/frenet_V5_best_of_N_v1_scene{12,57,125,140}.gif` | Best-of-32 sampling. |
| V7_seed1337 | `bevs/rollouts/frenet_V7_seed1337_scene{12,57,125,140}.gif` | Inference seed swap (269 → 1337). |
| V8_cfg_1.0 | `bevs/rollouts/frenet_V8_cfg_1.0_scene{12,57,125,140}.gif` | omega = 1.0 (no guidance). |
| V8_cfg_1.4 | `bevs/rollouts/frenet_V8_cfg_1.4_scene{12,57,125,140}.gif` | omega = 1.4 (default). |
| V8_cfg_1.8 | `bevs/rollouts/frenet_V8_cfg_1.8_scene{12,57,125,140}.gif` | omega = 1.8. |
| V8_cfg_2.5 | `bevs/rollouts/frenet_V8_cfg_2.5_scene{12,57,125,140}.gif` | omega = 2.5. |
| V9_steps_4 | `bevs/rollouts/frenet_V9_steps_4_scene{12,57,125,140}.gif` | Euler steps = 4. |
| V9_steps_8 | `bevs/rollouts/frenet_V9_steps_8_scene{12,57,125,140}.gif` | Euler steps = 8. |
| V9_steps_16 | `bevs/rollouts/frenet_V9_steps_16_scene{12,57,125,140}.gif` | Euler steps = 16. |
| V9_steps_32 | `bevs/rollouts/frenet_V9_steps_32_scene{12,57,125,140}.gif` | Euler steps = 32 (default). |
| V11_velocity | `bevs/rollouts/frenet_V11_velocity_scene{12,57,125,140}.gif` | Velocity-rep control; uses `v8_V11_velocity_seed269.ckpt`. |

---

## Provenance

Originals live under `/home/imaansol/cs269_archive/` (not in the repo). Each entry in `catalog.json` carries an `archive_origin` field pointing back to the source path. Local checkpoints used to produce these BEVs are listed in the corresponding `source_checkpoint` field; the preprocessed `.npz` cache required to re-run inference is not available on this machine.
