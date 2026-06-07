"""Tests for scripts/validate_npz_cache.py.

Synthesises minimal-but-valid + intentionally-broken .npz files, then runs
the script under each condition and checks the manifest contents + exit
code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "validate_npz_cache.py"


def _write_valid(path: Path, future_len: int = 80) -> None:
    """Write a minimal .npz that passes every validation check."""
    np.savez(
        path,
        ego_agent_past=np.zeros((21, 14), dtype=np.float32),
        ego_agent_future=np.zeros((future_len, 3), dtype=np.float32),
        ego_current_state=np.zeros(16, dtype=np.float32),
        neighbor_agents_past=np.zeros((32, 21, 11), dtype=np.float32),
        neighbor_agents_future=np.zeros((10, future_len, 4), dtype=np.float32),
        lanes=np.zeros((70, 20, 12), dtype=np.float32),
        lanes_speed_limit=np.zeros((70, 1), dtype=np.float32),
        lanes_has_speed_limit=np.zeros((70, 1), dtype=bool),
        route_lanes=np.zeros((25, 20, 12), dtype=np.float32),
        route_lanes_speed_limit=np.zeros((25, 1), dtype=np.float32),
        route_lanes_has_speed_limit=np.zeros((25, 1), dtype=bool),
        static_objects=np.zeros((5, 10), dtype=np.float32),
    )


def _write_broken_missing_key(path: Path) -> None:
    np.savez(
        path,
        ego_agent_past=np.zeros((21, 14), dtype=np.float32),
        # ego_agent_future intentionally missing
    )


def _write_broken_nan(path: Path) -> None:
    fut = np.zeros((80, 3), dtype=np.float32)
    fut[0, 0] = float("nan")
    _write_valid(path)  # write a valid one first
    # Then re-save with nan in ego_agent_future
    valid = np.load(path)
    arrays = {k: valid[k] for k in valid.files}
    arrays["ego_agent_future"] = fut
    np.savez(path, **arrays)


def _write_broken_ego_off_origin(path: Path) -> None:
    _write_valid(path)
    valid = np.load(path)
    arrays = {k: valid[k] for k in valid.files}
    ego = arrays["ego_current_state"].copy()
    ego[0] = 5.0  # 5 m off origin
    ego[1] = 0.0
    arrays["ego_current_state"] = ego
    np.savez(path, **arrays)


def _run(cache_dir: Path, manifest: Path,
         fail_threshold: float = 0.05) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "CACHE_DIR": str(cache_dir),
        "OUTPUT_MANIFEST": str(manifest),
        "FAIL_THRESHOLD": str(fail_threshold),
    }
    return subprocess.run([sys.executable, str(SCRIPT)],
                          env=env, capture_output=True, text=True)


# ---------------------------------------------------------------------------


def test_all_valid_files_pass(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(5):
        _write_valid(cache / f"scene_{i}.npz")

    manifest = tmp_path / "manifest.json"
    r = _run(cache, manifest)
    assert r.returncode == 0, r.stderr
    assert manifest.exists()
    good = json.loads(manifest.read_text())
    assert len(good) == 5
    assert all("scene_" in f for f in good)


def test_missing_key_files_are_quarantined(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    # 20 valid + 1 broken (5% threshold passes)
    for i in range(20):
        _write_valid(cache / f"valid_{i}.npz")
    _write_broken_missing_key(cache / "broken.npz")

    manifest = tmp_path / "manifest.json"
    r = _run(cache, manifest)
    assert r.returncode == 0, r.stderr
    good = json.loads(manifest.read_text())
    assert "broken.npz" not in good
    assert len(good) == 20


def test_too_many_bad_files_exits_nonzero(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    # 1 valid + 4 broken — 80% bad, far over 5% threshold
    _write_valid(cache / "valid.npz")
    for i in range(4):
        _write_broken_missing_key(cache / f"broken_{i}.npz")

    manifest = tmp_path / "manifest.json"
    r = _run(cache, manifest)
    assert r.returncode != 0
    assert "too many bad files" in r.stderr.lower()


def test_nan_in_ego_future_is_quarantined(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(20):
        _write_valid(cache / f"valid_{i}.npz")
    _write_broken_nan(cache / "nanny.npz")

    manifest = tmp_path / "manifest.json"
    r = _run(cache, manifest)
    assert r.returncode == 0, r.stderr
    good = json.loads(manifest.read_text())
    assert "nanny.npz" not in good


def test_ego_off_origin_is_quarantined(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(20):
        _write_valid(cache / f"valid_{i}.npz")
    _write_broken_ego_off_origin(cache / "offset.npz")

    manifest = tmp_path / "manifest.json"
    r = _run(cache, manifest)
    assert r.returncode == 0, r.stderr
    good = json.loads(manifest.read_text())
    assert "offset.npz" not in good


def test_fails_loud_on_empty_cache(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    manifest = tmp_path / "manifest.json"
    r = _run(cache, manifest)
    assert r.returncode != 0
    assert "no .npz files" in r.stderr


def test_handles_string_dtype_fields_in_diagnostic(tmp_path: Path):
    """Real nuPlan caches contain string-typed fields (e.g. scenario_token
    as np.dtype('<U12')). The post-validation diagnostic printout must not
    crash trying to call .min()/.max() on those — numpy's min/max ufunc
    raises _UFuncNoLoopError for string dtypes.

    Regression test for the failure observed on Colab 2026-05-29:
    `ufunc 'minimum' did not contain a loop with signature matching types
    (dtype('<U12'), dtype('<U12'))`.
    """
    cache = tmp_path / "cache"
    cache.mkdir()
    # Write a valid .npz then add a string field at savez time.
    extras = {
        "ego_agent_past":           np.zeros((21, 14), dtype=np.float32),
        "ego_agent_future":         np.zeros((80, 3), dtype=np.float32),
        "ego_current_state":        np.zeros(16, dtype=np.float32),
        "neighbor_agents_past":     np.zeros((32, 21, 11), dtype=np.float32),
        "neighbor_agents_future":   np.zeros((10, 80, 4), dtype=np.float32),
        "lanes":                    np.zeros((70, 20, 12), dtype=np.float32),
        "lanes_speed_limit":        np.zeros((70, 1), dtype=np.float32),
        "lanes_has_speed_limit":    np.zeros((70, 1), dtype=bool),
        "route_lanes":              np.zeros((25, 20, 12), dtype=np.float32),
        "route_lanes_speed_limit":  np.zeros((25, 1), dtype=np.float32),
        "route_lanes_has_speed_limit": np.zeros((25, 1), dtype=bool),
        "static_objects":           np.zeros((5, 10), dtype=np.float32),
        "scenario_token":           np.array("abc123def456", dtype="<U12"),
    }
    np.savez(cache / "with_string.npz", **extras)

    manifest = tmp_path / "manifest.json"
    r = _run(cache, manifest)
    assert r.returncode == 0, r.stderr
    # The string field is reported in the diagnostic block but without
    # range/mean (which would crash).
    assert "scenario_token" in r.stdout
    assert "<U12" in r.stdout


def test_threshold_is_configurable(tmp_path: Path):
    """A more permissive threshold lets more files fail without aborting."""
    cache = tmp_path / "cache"
    cache.mkdir()
    # 6 valid + 4 broken — 40% bad. 5% threshold would fail; 50% would pass.
    for i in range(6):
        _write_valid(cache / f"valid_{i}.npz")
    for i in range(4):
        _write_broken_missing_key(cache / f"broken_{i}.npz")

    manifest = tmp_path / "manifest.json"
    r = _run(cache, manifest, fail_threshold=0.5)
    assert r.returncode == 0, r.stderr
    good = json.loads(manifest.read_text())
    assert len(good) == 6
