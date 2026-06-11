# CS 269 Modifications to Vendored Flow Planner

This directory contains a **vendored fork** of the official Flow Planner code
([Tan et al., NeurIPS 2025](https://arxiv.org/abs/2510.11083), MIT licensed),
initially added at repo-root commit `006b9a5`.

The original upstream README is preserved at [README.md](README.md) — **do not
modify it**. This file (`CS269_NOTES.md`) catalogues every change we have made
to the vendored code for our CS 269 project.

If reviewers want to see exactly what we changed (vs the paper authors'
release), this is the place.

---

## How to diff against upstream

Upstream: https://github.com/DiffusionAD/Flow-Planner

To see our changes:
```bash
# In a fresh clone of the upstream repo
git diff <upstream-HEAD>..<our-vendored-HEAD> -- flow_planner/
```

Or compare our file list against upstream's tree to find:
- **Modified files** (paper's file, with our edits)
- **New files** (files we added that upstream doesn't have)

---

## Catalogue of modifications

### 1. `flow_planner/data/normalization/frenet_utils.py` — **NEW FILE**

**Purpose:** Frenet coordinate math (centerline construction, cartesian↔Frenet
projection). Upstream has no Frenet support.

**Key entries:**
- `select_reference_centerline(route_lanes, lanes, ...) -> (B, 100, 2)` —
  builds a horizon-covering reference polyline by greedy nearest-end
  concatenation of route lanes. Falls back to nearest visible lane if no
  route. **This is Option B** of our two-stage fix (see
  [../docs/frenet_representation_analysis.md](../docs/frenet_representation_analysis.md)).
- `cartesian_to_frenet(xy, centerline) -> (s, d)` — projects each xy point
  onto the centerline, picks the closest segment, returns signed arc-length
  and signed perpendicular offset.
- `frenet_to_cartesian(sd, centerline) -> (x, y)` — inverse projection. Round-
  trip tested in `../../tests/test_frenet.py`.

**Why this lives here:** in upstream Flow Planner's architecture, normalization
utilities live at `flow_planner/data/normalization/`. We follow the same
convention.

---

### 2. `flow_planner/model/model_utils/input_preprocess.py` — **MODIFIED**

**Upstream version:** Implements three kinematic modes — `'waypoints'`,
`'velocity'`, `'acceleration'` — as branches inside
`ModelInputProcessor.sample_to_model_input(...)`.

**Our modifications:**
- Added a fourth branch: `elif kinematic == 'frenet':` in our version.
- Inside that branch: capture **raw** route_lanes / lanes / ego_current /
  ego_future references BEFORE `obs_normalizer` rescales them, then call
  `select_reference_centerline(raw_routes, raw_lanes)` and
  `cartesian_to_frenet(raw_xy, centerline)` so the (s, d) target lives in
  world-scale meters. Working in raw scale is non-negotiable: the previous
  version called these on the post-normalizer tensors, which produced a
  centerline at 1/20 scale while ego_future stayed in raw scale (ego_future
  is not in either norm-stats YAML, so `obs_normalizer` skips it). The
  resulting (s, d) targets were nonsense and silently inflated `d` std to
  ~16 m. See [../docs/research/cs269_modifications_review.md](../docs/research/cs269_modifications_review.md)
  CRITICAL #1.
- Construct the 4-D Frenet target as `(s, d, cos_h, sin_h)`.
- **Store the chosen raw-scale `centerline` in `model_inputs['reference_centerline']`**
  so the encoder can condition on it (this is the Option A plumbing — without
  this, the encoder has no way to identify the reference frame).

**Backwards-compatible:** non-Frenet runs (waypoints, velocity, acceleration)
go through the upstream code paths unchanged. The new key
`model_inputs['reference_centerline']` is only added when `kinematic=='frenet'`.
The raw-data references are also only captured for `'frenet'`.

**Test contract:** `tests/test_frenet_integration.py` asserts the centerline
xy spans ~100 m (not ~5 m post-normalizer scale) and that the model's
internal centerline matches what `inference_eval.py` builds externally from
the raw batch.

---

### 3. `flow_planner/model/modules/encoder_modules.py` — **MODIFIED (new class added)**

**Upstream version:** Implements `AgentFusionEncoder`, `StaticFusionEncoder`,
`LaneFusionEncoder`, `RouteEncoder`, `FusionEncoder`, and `MixerBlock`. Each
processes one type of scene element into hidden-dim embeddings.

**Our modifications:**
- Added a new class `CenterlineEncoder` (lines ~277-330 of our version).
- Architecture: MLP-Mixer style — per-point projection of `(x, y, tan_x, tan_y, arc_len)` features → MLP-Mixer block → mean pool over points → MLP projection to hidden_dim.
- Input shape: `(B, 100, 2)` (the chosen centerline polyline).
- Output shape: `(B, hidden_dim)`.
- Used only when `kinematic=='frenet'`. For other reps, this class is never instantiated.

**Why this class:** the only architectural enabling-modification we promised
in the proposal (proposal §1.2: "We modify the representation encoders…").
The Frenet (s, d) target is defined relative to a per-scenario reference
centerline. Without explicit encoder conditioning on that centerline, the
model has no way to identify which coordinate system the target is in. See
[../docs/frenet_representation_analysis.md](../docs/frenet_representation_analysis.md)
for the full root-cause analysis.

---

### 4. `flow_planner/model/flow_planner_model/encoder.py` — **MODIFIED**

**Upstream version:** `FlowPlannerEncoder` class with `__init__` and `forward`.
`forward` returns a dict containing `routes_cond = self.route_encoder(routes)`,
a pooled summary of route lanes.

**Our modifications:**
- Added optional `centerline_encoder` argument to `__init__` (default `None`).
- When `centerline_encoder is None` (default — non-Frenet runs), zero extra
  parameters are added to the model. Existing waypoints / velocity / acceleration
  checkpoints load cleanly with no missing/unexpected keys.
- Modified `forward` to accept an optional `reference_centerline` kwarg.
- If both `centerline_encoder` and `reference_centerline` are present, compute
  `routes_cond += self.centerline_encoder(reference_centerline)`, adding the
  centerline embedding to the pooled routes summary. The DiT decoder then
  conditions on this enriched `routes_cond` through the rest of the network.

**Backwards-compatibility:** verified by a test in
`../../tests/test_centerline_encoder.py::test_flow_planner_encoder_centerline_none_no_extra_params`
which asserts `centerline_encoder=None` yields **zero new state_dict keys**.

---

### 5. `flow_planner/model/flow_planner_model/flow_planner.py` — **MODIFIED**

**Upstream version:** `FlowPlanner` class wraps encoder + decoder + flow ODE.
The `kinematic` argument is `Literal["waypoints", "velocity", "acceleration"]`.

**Our modifications:**
- Extended the `kinematic` Literal to include `"frenet"` (line ~24).
- Modified `extract_encoder_inputs(self, inputs)` to thread
  `reference_centerline` through if present in `inputs`. This lets the
  encoder receive the chosen centerline when running Frenet.
- Added an Option-A guard in `__init__`: if `kinematic == 'frenet'` and the
  encoder has no `centerline_encoder`, emit a `RuntimeWarning` describing
  exactly which Hydra `+` overrides are missing. Prevents the silent-drop
  failure mode where the centerline is plumbed in but the encoder drops it
  (the same misconfiguration that caused the 26-unexpected-key state-dict
  mismatch in commit `b950ffc`).

---

### 6. `flow_planner/script/model/flow_planner.yaml` — **MODIFIED**

**Upstream version:** Hydra config for the `FlowPlanner` model instantiation.
Defines the encoder + decoder hyperparameters.

**Our modifications:**
- Added top-level `kinematic: waypoints` (line ~21) so the kinematic can be
  selected via CLI override (`model.kinematic=frenet`).
- Documented the Hydra `+` prefix overrides that enable the centerline encoder
  for Frenet runs:
  ```
  +model.model_encoder.centerline_encoder._target_=flow_planner.model.modules.encoder_modules.CenterlineEncoder
  +model.model_encoder.centerline_encoder.n_points=100
  +model.model_encoder.centerline_encoder.hidden_dim=256
  ```
- Did not add `centerline_encoder: null` to the YAML itself — that caused
  Hydra config-composition errors because OmegaConf reads `null` as a leaf
  and rejects sub-key overrides. The Python `__init__` default (`None`)
  handles the non-Frenet case cleanly.

---

### 7. `flow_planner/script/normalization_stats/frenet_norm_stats.yaml` — **NEW FILE (legacy v0)**

**Purpose:** Per-feature mean/std for the Frenet `(s, d, cos_h, sin_h)`
target. Used by the `StateNormalizer` during training (target normalization)
and inference (target denormalization). Mirrors upstream's `waypoints_norm_stats.yaml`.

**Currently:** `mean: [31, 0, 1, 0]`, `std: [26, 16, 1.0, 1.0]`. The `s` and
`d` channel stats were measured empirically from 1500 mini-split scenarios.
The `cos_h`/`sin_h` stds were bumped from the originally-measured `0.3` to
`1.0` to match `waypoints_norm_stats.yaml` convention (audit Finding 1:
`std=0.3` inflated the heading gradient ~11× relative to the (s, d) channels
under the sum-over-channels MSE). The `d` std of `16 m` is a pre-bugfix
artifact of the cartesian-to-Frenet frame mismatch; the post-bugfix `d`
distribution is much tighter, which motivated the v1 config below.

### 7b. `flow_planner/script/normalization_stats/frenet_norm_stats_v1.yaml` — **NEW FILE (paper-headline config)**

**Purpose:** The Frenet norm-stats config used by **all paper-headline 5{,}000-scenario Frenet runs** (`frenet_seed42.ckpt`, `frenet_seed42_fixedCB.ckpt`).

**Differences from the v0 config above:** the `d` channel is now in the
bounded range `[-1, +1]` (image of `tanh(d/3)`), so the `d`-std is reduced
from `16` to `0.5`. The `cos_h`/`sin_h` stds remain `1.0` (matching the v0
fix). This config is **paired** with the v8 Frenet env vars set in the
training notebook:

- `FRENET_TANH_D=1` (cartesian_to_frenet returns `tanh(d/3)`)
- `FRENET_TANH_D_SCALE=3.0`
- `FRENET_SMART_CENTERLINE=1` (smart-centerline picker uses `ego_past` + all lanes)

**Stats:** `ego.uniform.mean = [31, 0, 1, 0]`, `ego.uniform.std = [26, 0.5, 1.0, 1.0]`.

**Which notebooks use which:** Five of the eight notebooks (`paper_baseline`,
`motion_representations`, `recover_all`, `v8_team`, `v8_frenet_fixes`) use
`frenet_norm_stats` (legacy v0) for the small-scale 1{,}500-scenario runs.
The two paper-headline 5{,}000-scenario notebooks
(`cs269_dagshub_best_ever_frenet`, `cs269_frenet_every_advantage`) use
`frenet_norm_stats_v1`.

---

### 8. `flow_planner/trainer.py` — **MODIFIED (one-line fix)**

**Upstream version:** at the end of training, calls
`torch.distributed.destroy_process_group()` unconditionally.

**Our modification:** wrapped that call in `if cfg.ddp.distributed:` so
single-GPU (non-DDP) runs don't crash with `AssertionError: Default process
group has not been initialized`. Single-line guard, no functional change to
distributed runs.

---

### 9. `flow_planner/run_script/preprocess.py` — **NEW FILE**

**Purpose:** Standalone CLI for nuPlan scenario preprocessing. Wraps Flow
Planner's `DataProcessor` with proper argparse, log-disjoint train/val split
support via `--log_names_json`, and a sidecar manifest that records exactly
how each cache was built.

**Why we added this:** the upstream Flow Planner repo provides only
`run_script/launch_*.sh` bash scripts assuming the user has already
preprocessed data via Diffusion-Planner's CLI. Diffusion-Planner's CLI
produces `.npz` files that our Flow Planner code can no longer read (the
preprocessor's schema has diverged). Our CLI uses Flow Planner's own
`DataProcessor` to produce the correct schema.

**Methodology compliance:** the default `ScenarioFilter` matches PlanTF's
`training_scenarios_1M.yaml` convention (random sampling, no stratification,
shuffled). See [../docs/preprocessing_methodology.md](../docs/preprocessing_methodology.md).

---

### 10. `flow_planner/run_script/inference_eval.py` — **NEW FILE**

**Purpose:** Standalone CLI for loading a checkpoint, running inference on a
preprocessed `.npz` cache, computing ADE/FDE, and writing a JSON summary.
Used by the notebook's eval cells.

**Why we added this:** the upstream Flow Planner repo's only eval path is
the nuBoard simulation runner, which (a) requires the full nuPlan trainval
split and ~6 GPU-hours per pass, and (b) reports closed-loop scores, not
open-loop ADE/FDE. For our open-loop comparison metric (matching Flow
Planner Table 3, not Table 1) we need a focused CLI.

**Option A wiring:** when `--kinematic frenet`, automatically adds the
`centerline_encoder` Hydra overrides so the instantiated inference model
matches the trained-with-Option-A checkpoint. Without this, the
state_dict load would silently drop the centerline_encoder weights and
the inference model would degrade to v4 (no Option A) behavior, masking
the very effect we want to measure.

---

## Files we did NOT modify

For completeness — every file in the vendored Flow Planner package that we
did NOT touch:

- `flow_planner/core/` — flow matching ODE core
- `flow_planner/data/data_process/data_processor.py` — the actual per-scenario feature extractor
- `flow_planner/data/dataset/nuplan.py` — `NuPlanDataset` (we did temporarily modify this during early debugging but reverted)
- `flow_planner/data/utils/collect.py`
- `flow_planner/model/model_base.py`
- `flow_planner/model/flow_planner_model/decoder.py` — DiT decoder
- `flow_planner/model/flow_planner_model/flow_utils/flow_ode.py`
- `flow_planner/model/flow_planner_model/global_attention.py`
- `flow_planner/model/modules/decoder_modules.py`
- `flow_planner/nuplan_simulation/` — closed-loop sim adapters (we don't use these for open-loop eval)
- `flow_planner/planner.py` — nuBoard `AbstractPlanner` adapter
- `flow_planner/recorder/`
- `flow_planner/script/core/`, `script/data/`, `script/ema/`, `script/optimizer/`, `script/recorder/`, `script/scheduler/` — Hydra configs
- `flow_planner/train_utils/`
- `flow_planner/setup.py`
- All `__init__.py` files

The fact that we didn't touch these — particularly the decoder, flow ODE,
and training loop — is what backs the proposal claim that "the architecture
stays as released."

---

## Diff summary (for the report)

For the report's implementation section, the one-line summary is:

> "Our implementation extends the public Flow Planner reference code (Tan et
> al., NeurIPS 2025), vendored into our team repository at commit `006b9a5`.
> We added two new files (`frenet_utils.py`, `frenet_norm_stats.yaml`) and
> modified four existing files (`input_preprocess.py`, `encoder_modules.py`,
> `flow_planner_model/encoder.py`, `flow_planner_model/flow_planner.py`)
> totaling ~400 lines of net additions. The flow-matching loss, ODE
> sampler, DiT decoder, and training loop are all unmodified upstream code."

---

**Last update:** 2026-05-29 (Imaan + post-overnight-research bug fixes — two
CRITICAL normalization-frame bugs in the Frenet pipeline closed; silent-drop
guard added to `FlowPlanner.__init__`; six new tests landed in
`tests/test_frenet_integration.py`; `frenet_norm_stats.yaml` flagged for
re-measurement before v6 training).

---

### Upstream demo assets removed

The upstream `flow_planner/assets/` directory (rollout GIFs and case-study PNGs
shipped with Tan et al.'s public release) has been removed from this vendored
fork to keep `git clone` size minimal. Diff against the upstream commit
`006b9a5` (or the upstream public release) to recover them if needed for
side-by-side qualitative comparison.
