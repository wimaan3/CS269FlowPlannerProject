"""Tests for scripts/measure_frenet_stats.py and scripts/update_frenet_norm_stats.py.

The measurement script needs nuplan-style .npz files to run; we synthesise a
minimal but realistic single-scenario cache for the test. The update script
is tested in isolation against a synthetic stats JSON.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


# ---------------------------------------------------------------------------
# update_frenet_norm_stats.py — pure file IO, no Flow Planner deps
# ---------------------------------------------------------------------------

def _seed_yaml(target: Path) -> None:
    """Write a minimal valid frenet_norm_stats.yaml at `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "ego:\n"
        "  log: {}\n"
        "  uniform:\n"
        "    mean: [42, 0, 1, 0]\n"
        "    std:  [63, 9, 0.3, 0.3]\n"
        "neighbor:\n"
        "  log: {}\n"
        "  uniform:\n"
        "    mean: [42, 0, 1, 0]\n"
        "    std:  [63, 9, 0.3, 0.3]\n"
        "ego_past:\n"
        "  mean: [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\n"
        "  std:  [20, 20, 1, 1, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1]\n"
    )


def _make_stats(s_mean=31.18, s_std=25.63, d_mean=-0.53, d_std=1.42) -> dict:
    return {
        "n_files": 1500, "n_skipped": 0, "n_timesteps": 120000,
        "s": {"mean": s_mean, "std": s_std, "min": 0.45, "max": 123.95},
        "d": {"mean": d_mean, "std": d_std, "min": -5.0,  "max": 5.0},
        "cos_h_mean": 1.0, "cos_h_std": 0.3,
        "sin_h_mean": 0.0, "sin_h_std": 0.3,
    }


def _run_update(tmp_path: Path, stats: dict, fp_dir: Path) -> subprocess.CompletedProcess:
    stats_json = tmp_path / "stats.json"
    stats_json.write_text(json.dumps(stats))
    env = {**os.environ, "FP_DIR": str(fp_dir), "STATS_JSON": str(stats_json)}
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "update_frenet_norm_stats.py")],
        env=env, capture_output=True, text=True,
    )


def test_update_writes_measured_values(tmp_path: Path):
    fp_dir = tmp_path / "fp"
    yaml_path = fp_dir / "flow_planner" / "script" / "normalization_stats" / "frenet_norm_stats.yaml"
    _seed_yaml(yaml_path)

    r = _run_update(tmp_path, _make_stats(), fp_dir)
    assert r.returncode == 0, r.stderr

    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["ego"]["uniform"]["mean"] == [31.18, -0.53, 1.0, 0.0]
    assert doc["ego"]["uniform"]["std"]  == [25.63, 1.42, 0.3, 0.3]
    assert doc["neighbor"]["uniform"]["mean"] == doc["ego"]["uniform"]["mean"]
    assert doc["neighbor"]["uniform"]["std"]  == doc["ego"]["uniform"]["std"]


def test_update_preserves_other_blocks(tmp_path: Path):
    """ego_past block (and any other obs-side normalisation block) must be
    left untouched by the script."""
    fp_dir = tmp_path / "fp"
    yaml_path = fp_dir / "flow_planner" / "script" / "normalization_stats" / "frenet_norm_stats.yaml"
    _seed_yaml(yaml_path)

    r = _run_update(tmp_path, _make_stats(), fp_dir)
    assert r.returncode == 0, r.stderr

    doc = yaml.safe_load(yaml_path.read_text())
    assert doc["ego_past"]["mean"] == [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert doc["ego_past"]["std"]  == [20, 20, 1, 1, 20, 20, 20, 20, 20, 20, 1, 1, 1, 1]


def test_update_creates_backup(tmp_path: Path):
    fp_dir = tmp_path / "fp"
    yaml_path = fp_dir / "flow_planner" / "script" / "normalization_stats" / "frenet_norm_stats.yaml"
    _seed_yaml(yaml_path)
    original = yaml_path.read_text()

    r = _run_update(tmp_path, _make_stats(), fp_dir)
    assert r.returncode == 0, r.stderr

    backup = yaml_path.with_suffix(".yaml.bak")
    assert backup.exists()
    assert backup.read_text() == original


def test_update_refuses_implausible_d_std(tmp_path: Path):
    """A d_std > 100 m is implausible and should not be written."""
    fp_dir = tmp_path / "fp"
    yaml_path = fp_dir / "flow_planner" / "script" / "normalization_stats" / "frenet_norm_stats.yaml"
    _seed_yaml(yaml_path)
    original = yaml_path.read_text()

    r = _run_update(tmp_path, _make_stats(d_std=500), fp_dir)
    assert r.returncode != 0
    assert "out of plausible range" in r.stderr
    # YAML must NOT have been modified
    assert yaml_path.read_text() == original


def test_update_refuses_implausible_s_std(tmp_path: Path):
    fp_dir = tmp_path / "fp"
    yaml_path = fp_dir / "flow_planner" / "script" / "normalization_stats" / "frenet_norm_stats.yaml"
    _seed_yaml(yaml_path)
    original = yaml_path.read_text()

    r = _run_update(tmp_path, _make_stats(s_std=0.0), fp_dir)
    assert r.returncode != 0
    # YAML untouched
    assert yaml_path.read_text() == original


def test_update_fails_loud_on_missing_yaml(tmp_path: Path):
    """If the YAML doesn't exist, the script must fail cleanly."""
    fp_dir = tmp_path / "fp"
    # Note: deliberately don't call _seed_yaml; the YAML is absent
    r = _run_update(tmp_path, _make_stats(), fp_dir)
    assert r.returncode != 0
    assert "frenet_norm_stats.yaml not found" in r.stderr


def test_update_fails_loud_on_missing_stats(tmp_path: Path):
    """If STATS_JSON points at a missing file, fail."""
    fp_dir = tmp_path / "fp"
    yaml_path = fp_dir / "flow_planner" / "script" / "normalization_stats" / "frenet_norm_stats.yaml"
    _seed_yaml(yaml_path)

    env = {**os.environ, "FP_DIR": str(fp_dir),
           "STATS_JSON": str(tmp_path / "missing.json")}
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "update_frenet_norm_stats.py")],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert "does not exist" in r.stderr


# ---------------------------------------------------------------------------
# measure_frenet_stats.py — requires nuplan-format .npz files
# Marked optional because building a sample npz requires the flow_planner
# package which we depend on indirectly. The script reads only the three
# fields select_reference_centerline + cartesian_to_frenet need.
# ---------------------------------------------------------------------------

def _make_minimal_npz(path: Path, future_len=80, lane_pts=20):
    """Write a single .npz mimicking the DataProcessor output that the
    measure script actually reads."""
    routes = np.zeros((25, lane_pts, 12), dtype=np.float32)
    routes[0, :, 0] = np.linspace(0.0, 100.0, lane_pts)
    routes[0, :-1, 2:4] = routes[0, 1:, :2] - routes[0, :-1, :2]

    lanes = np.zeros((70, lane_pts, 12), dtype=np.float32)

    ego_future = np.zeros((future_len, 3), dtype=np.float32)
    ego_future[:, 0] = np.linspace(1.0, 80.0, future_len)

    np.savez(path,
             ego_agent_future=ego_future,
             lanes=lanes,
             route_lanes=routes)


def test_measure_runs_on_synthetic_cache(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    for i in range(3):
        _make_minimal_npz(cache / f"scene_{i}.npz")

    out_json = tmp_path / "stats.json"
    env = {
        **os.environ,
        "FP_DIR":      str(REPO_ROOT / "flow_planner"),
        "CACHE_DIR":   str(cache),
        "OUTPUT_JSON": str(out_json),
        "NUM_FILES":   "0",
    }
    r = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "measure_frenet_stats.py")],
        env=env, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    stats = json.loads(out_json.read_text())
    assert stats["n_files"] == 3
    # Straight-line ego on a straight-line route -> d should be very close to 0
    assert abs(stats["d"]["mean"]) < 1.0
    assert stats["d"]["std"] < 1.0
    # s should span ~80 m (the ego trajectory length)
    assert stats["s"]["max"] > 70.0
