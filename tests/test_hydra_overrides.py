"""Regression tests for Hydra override coherence between train + eval + viz.

These tests guard the v7-AUDIT class of bugs:

1. The training cell in notebooks/v7_retrain.ipynb adds Hydra overrides
   that include `enable_attn_dist=false`. That key IS defined in the
   YAML, so the prefix MUST be `++` (force-override) rather than `+`
   (which Hydra reserves for keys absent from the YAML). A `+` prefix
   here HARD-FAILS the training launch.

2. inference_eval.py and scripts/visualize_bev.py must apply the SAME
   override set for Frenet so the eval / viz model architectures match
   the trained checkpoint. If eval omits `enable_attn_dist=false` while
   training included it, the eval model has extra
   JointAttention.gen_taus Linear layers that load with random weights
   under strict=False — silently corrupting every Frenet metric.

3. visualize_bev.py must branch on KINEMATIC: applying the Frenet-only
   centerline_encoder overrides + frenet_to_cartesian decode to a
   waypoints checkpoint produces meaningless trajectories that get
   silently plotted as if valid.

Run with: pytest tests/test_hydra_overrides.py -v
"""
from __future__ import annotations

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "v7_retrain.ipynb"
INFERENCE_EVAL = REPO_ROOT / "flow_planner" / "flow_planner" / "run_script" / "inference_eval.py"
VISUALIZE_BEV = REPO_ROOT / "scripts" / "visualize_bev.py"
MODEL_YAML = (
    REPO_ROOT
    / "flow_planner"
    / "flow_planner"
    / "script"
    / "model"
    / "flow_planner.yaml"
)


def _read_cell(idx: int) -> str:
    """Return the joined source of cell `idx` from the v7 notebook."""
    nb = json.loads(NOTEBOOK.read_text())
    cell = nb["cells"][idx]
    src = cell["source"]
    if isinstance(src, list):
        return "".join(src)
    return src


# ----------------------------- enable_attn_dist override coherence -----


def test_training_uses_double_plus_for_enable_attn_dist():
    """Cell 60 (training launch, was 58 before v7-AUDIT verification fix
    inserted the centerline-gate diagnostic + waypoints-purity markdown
    above) must use `++` for enable_attn_dist.

    `+` would hard-fail with ConfigCompositionException because the key
    is defined in flow_planner/script/model/flow_planner.yaml.
    """
    src = _read_cell(60)
    assert "++model.model_decoder.enable_attn_dist=false" in src, (
        "Cell 60 must use `++model.model_decoder.enable_attn_dist=false` "
        "(force-override). Found `+...` instead, which Hydra rejects "
        "because enable_attn_dist IS in model/flow_planner.yaml."
    )
    # Negative: the single-prefix form must not appear.
    assert " +model.model_decoder.enable_attn_dist=" not in src, (
        "Cell 60 still contains the single-`+` form of enable_attn_dist, "
        "which Hydra rejects (ConfigCompositionException)."
    )


def test_inference_eval_uses_double_plus_for_enable_attn_dist():
    """inference_eval.py Frenet branch must mirror training's override."""
    src = INFERENCE_EVAL.read_text()
    assert "++model.model_decoder.enable_attn_dist=false" in src, (
        "inference_eval.py must add `++model.model_decoder.enable_attn_dist=false` "
        "to its Frenet override list so the eval model matches the trained "
        "architecture (otherwise gen_taus layers load with random weights)."
    )


def test_visualize_bev_uses_double_plus_for_enable_attn_dist():
    src = VISUALIZE_BEV.read_text()
    assert "++model.model_decoder.enable_attn_dist=false" in src, (
        "scripts/visualize_bev.py must add "
        "`++model.model_decoder.enable_attn_dist=false` for Frenet so the "
        "viz model matches the trained architecture."
    )


def test_enable_attn_dist_is_actually_in_yaml():
    """Sanity check: confirms `enable_attn_dist` IS defined in the YAML.

    This is the precondition that forces `++` over `+`. If this ever
    flips (YAML deletes the default), the other tests in this module
    need to be revisited.
    """
    yaml_text = MODEL_YAML.read_text()
    assert "enable_attn_dist:" in yaml_text, (
        "enable_attn_dist no longer in model/flow_planner.yaml — review "
        "whether `++` is still required (vs `+`) in train/eval/viz."
    )


# ----------------------------- visualize_bev kinematic branch ----------


def test_visualize_bev_branches_on_kinematic_env_var():
    """visualize_bev.py must read KINEMATIC env var (not hardcode frenet)."""
    src = VISUALIZE_BEV.read_text()
    assert 'os.environ.get("KINEMATIC"' in src or 'os.environ["KINEMATIC"]' in src, (
        "scripts/visualize_bev.py must read KINEMATIC from env so it can "
        "skip the Frenet-only overrides + frenet_to_cartesian decode "
        "when called on a waypoints checkpoint."
    )


def test_visualize_bev_skips_frenet_decode_for_waypoints():
    """The frenet_to_cartesian call must be gated by KINEMATIC=='frenet'.

    A waypoints checkpoint outputs (x, y) directly; applying
    frenet_to_cartesian to (x, y) interpreted as (s, d) produces
    meaningless trajectories.
    """
    src = VISUALIZE_BEV.read_text()
    # The decode path must be inside a frenet-only branch.
    assert 'if KINEMATIC == "frenet":' in src or "KINEMATIC == 'frenet':" in src, (
        "scripts/visualize_bev.py must wrap the frenet_to_cartesian "
        "decode in an `if KINEMATIC == 'frenet'` branch."
    )


def test_notebook_cell_65_passes_kinematic_env_var():
    """Notebook BEV-viz cell (now 67 after v7-AUDIT verification fix) must
    forward KINEMATIC to visualize_bev.py.
    """
    src = _read_cell(67)
    assert "'KINEMATIC'" in src and "RUN_KINEMATIC" in src, (
        "BEV-viz cell must pass `KINEMATIC: RUN_KINEMATIC` in the subprocess "
        "env so visualize_bev.py can branch correctly."
    )


def test_notebook_cell_65_asserts_returncode():
    """BEV-viz cell (now 67) must fail loud on visualize_bev.py errors.

    Otherwise the next cell's display(Image(bev_png)) can show a stale PNG
    from a prior successful run (local /content/work survives runtime
    restart), masquerading as fresh output.
    """
    src = _read_cell(67)
    assert "assert r.returncode == 0" in src, (
        "BEV-viz cell must `assert r.returncode == 0` so a viz failure "
        "halts rather than silently letting the next cell show a stale PNG."
    )


# ----------------------------- checkpoint naming coherence -------------


def test_notebook_cell_72_uses_v7_naming_for_waypoints_ckpt():
    """Waypoints comparison eval (now cell 74) must look for the v7-named
    waypoints checkpoint that the save cell actually writes
    (v7{V7_VARIANT}_waypoints_seed{RUN_SEED}.ckpt), not the legacy
    waypoints_seed{RUN_SEED}.ckpt path that never matches a v7 run.
    """
    src = _read_cell(74)
    assert "v7{V7_VARIANT}_waypoints_seed{RUN_SEED}" in src or (
        "v7" in src and "V7_VARIANT" in src and "waypoints" in src
    ), (
        "Waypoints comparison eval must look for the v7 RUN_NAME pattern, "
        "not the legacy waypoints_seed{RUN_SEED}.ckpt path that never "
        "matches v7 runs."
    )


def test_notebook_cell_50_backup_uses_v7_naming():
    """Cell 51 (was 50; the +1 shift is from the diagnostic cell inserted at
    index 8) backup must match the save cell's naming convention.
    """
    src = _read_cell(51)
    assert "v7" in src and "V7_VARIANT" in src, (
        "Backup cell's source path must use the v7 naming pattern "
        "(v7{V7_VARIANT}_{kinematic}_seed{SEED}.ckpt) to actually match "
        "what the save cell writes, otherwise the defensive backup is "
        "dead code."
    )


def test_notebook_cell_60_uses_shutil_copy_not_shell_cp():
    """Checkpoint-save cell (now 62) must use shutil.copy + size assertion,
    not `!cp` magic.

    `!cp` swallows non-zero exit codes, so a Drive blip would silently
    leave ckpt_drive_path pointing at the prior run's checkpoint and
    downstream eval would report numbers from the wrong weights.
    """
    src = _read_cell(62)
    assert "shutil.copy" in src, (
        "Checkpoint-save cell must use shutil.copy (not `!cp` magic) so "
        "Drive copy failures raise rather than silently leaving stale "
        "checkpoints."
    )
    assert "size" in src.lower() and "assert" in src, (
        "Checkpoint-save cell must assert dst size matches src size to "
        "catch truncated / failed copies."
    )


# ----------------------------- defensive cache + CSV guards ------------


def test_notebook_cell_45_heldout_diagnostic_guarded():
    """Held-out diagnostic cell (now 46) must only fire when HELDOUT_AVAILABLE."""
    src = _read_cell(46)
    assert "if HELDOUT_AVAILABLE and _npz_count == 0" in src, (
        "Held-out diagnostic cell must guard the 0-npz raise with "
        "`if HELDOUT_AVAILABLE` — held-out is documented as best-effort "
        "(Section 8 header)."
    )


def test_notebook_cell_75_guards_csv_read():
    """Results-display cell (now 77) must not raise FileNotFoundError on a
    fresh DRIVE_RESULTS dir.
    """
    src = _read_cell(77)
    assert "pathlib.Path(results_csv).exists()" in src or "exists()" in src, (
        "Results-display cell must guard the CSV read with an exists() "
        "check so it doesn't raise FileNotFoundError on a fresh-Drive setup."
    )
    assert "mkdir" in src and "DRIVE_RESULTS" in src, (
        "Results-display cell must mkdir DRIVE_RESULTS for consistency "
        "with the other write paths."
    )


def test_notebook_cell_26_copies_preprocess_manifest():
    """Drive->local copy cell (now 27) must copy preprocess_manifest.json
    so the methodology-compliance check doesn't always fire a false warning.
    """
    src = _read_cell(27)
    assert "preprocess_manifest.json" in src and "shutil" in src.lower() or "cp" in src, (
        "Drive->local copy cell must copy preprocess_manifest.json from "
        "Drive to local so the methodology-compliance check reads the "
        "correct manifest."
    )


def test_notebook_cell_4_splits_wipe_flags():
    """Cell 4 must offer WIPE_LOCAL_CACHE / WIPE_DRIVE_CACHE separately so
    local-only debugging never accidentally destroys the Drive cache.
    """
    src = _read_cell(4)
    assert "WIPE_LOCAL_CACHE" in src and "WIPE_DRIVE_CACHE" in src, (
        "Cell 4 must split FORCE_REPREPROCESS into WIPE_LOCAL_CACHE and "
        "WIPE_DRIVE_CACHE so users don't accidentally nuke the 30-min "
        "Drive rebuild."
    )
