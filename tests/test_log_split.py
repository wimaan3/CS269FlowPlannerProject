"""Tests for scripts/generate_log_split.py.

The generator must be:
  - deterministic given the same seed + log list
  - produce disjoint train / val sets
  - hit the expected counts for the nuPlan-mini case (54 / 10)
  - handle edge cases (single log, train_frac near 0 or 1) without crashing
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_log_split import generate_split, enumerate_logs, main  # noqa: E402


# Fixture: 64 fake log names mimicking the nuPlan mini set
NUPLAN_MINI_FAKE_LOGS = [f"2021.{m:02d}.{d:02d}.{h:02d}.{i:02d}.{s:02d}_veh-{v:02d}_{tok:05d}_{tok2:05d}"
                        for v in range(1, 5)
                        for m in range(7, 9)
                        for d in range(1, 9)
                        for h in [12, 14]
                        for i in [0, 30]
                        for s in [0]
                        for tok in [100, 200]
                        for tok2 in [500]][:64]


# ---------------------------------------------------------------------------
# generate_split — determinism + correctness
# ---------------------------------------------------------------------------

def test_split_is_deterministic_given_seed():
    """Same seed + same inputs → same output, every time."""
    s1 = generate_split(NUPLAN_MINI_FAKE_LOGS, seed=42, train_frac=54/64)
    s2 = generate_split(NUPLAN_MINI_FAKE_LOGS, seed=42, train_frac=54/64)
    assert s1 == s2, "split is non-deterministic"


def test_split_changes_when_seed_changes():
    """Different seeds → different splits (almost surely)."""
    s1 = generate_split(NUPLAN_MINI_FAKE_LOGS, seed=42, train_frac=54/64)
    s2 = generate_split(NUPLAN_MINI_FAKE_LOGS, seed=1337, train_frac=54/64)
    assert s1["train"] != s2["train"], "different seeds produced identical train sets"


def test_split_is_disjoint():
    """train ∩ val = ∅."""
    split = generate_split(NUPLAN_MINI_FAKE_LOGS, seed=42, train_frac=54/64)
    assert not (set(split["train"]) & set(split["val"]))


def test_split_covers_all_logs():
    """train ∪ val = inputs."""
    split = generate_split(NUPLAN_MINI_FAKE_LOGS, seed=42, train_frac=54/64)
    assert set(split["train"]) | set(split["val"]) == set(NUPLAN_MINI_FAKE_LOGS)


def test_split_hits_expected_mini_counts():
    """For nuPlan mini (64 logs) at default train_frac, expect 54 train / 10 val."""
    split = generate_split(NUPLAN_MINI_FAKE_LOGS, seed=42, train_frac=54/64)
    assert split["n_train"] == 54, f"expected 54 train logs, got {split['n_train']}"
    assert split["n_val"] == 10,   f"expected 10 val logs,   got {split['n_val']}"


def test_train_and_val_are_sorted():
    """Output train/val should be sorted so the JSON file is byte-stable across
    different filesystems that enumerate in different orders."""
    split = generate_split(NUPLAN_MINI_FAKE_LOGS, seed=42, train_frac=54/64)
    assert split["train"] == sorted(split["train"])
    assert split["val"] == sorted(split["val"])


# ---------------------------------------------------------------------------
# generate_split — edge cases
# ---------------------------------------------------------------------------

def test_split_rejects_empty_logs():
    with pytest.raises(ValueError, match="empty"):
        generate_split([], seed=42, train_frac=0.5)


def test_split_rejects_invalid_train_frac():
    with pytest.raises(ValueError, match="train_frac"):
        generate_split(NUPLAN_MINI_FAKE_LOGS, seed=42, train_frac=0.0)
    with pytest.raises(ValueError, match="train_frac"):
        generate_split(NUPLAN_MINI_FAKE_LOGS, seed=42, train_frac=1.0)


def test_split_with_two_logs_guarantees_one_val():
    """Even with train_frac very close to 1.0, val must have at least 1 log."""
    split = generate_split(["log_a", "log_b"], seed=42, train_frac=0.99)
    assert split["n_train"] == 1
    assert split["n_val"] == 1


# ---------------------------------------------------------------------------
# enumerate_logs
# ---------------------------------------------------------------------------

def test_enumerate_logs_returns_basenames(tmp_path: Path):
    """enumerate_logs strips the .db extension and sorts."""
    for name in ["b.db", "a.db", "c.db"]:
        (tmp_path / name).touch()
    logs = enumerate_logs(tmp_path)
    assert logs == ["a", "b", "c"]


def test_enumerate_logs_ignores_non_db_files(tmp_path: Path):
    (tmp_path / "real.db").touch()
    (tmp_path / "junk.txt").touch()
    (tmp_path / "README.md").touch()
    logs = enumerate_logs(tmp_path)
    assert logs == ["real"]


# ---------------------------------------------------------------------------
# CLI main()
# ---------------------------------------------------------------------------

def test_cli_writes_expected_json(tmp_path: Path):
    """End-to-end: CLI writes a JSON file with the expected structure."""
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    for name in NUPLAN_MINI_FAKE_LOGS:
        (logs_dir / f"{name}.db").touch()
    output = tmp_path / "split.json"

    rc = main(["--logs_dir", str(logs_dir),
               "--output", str(output),
               "--seed", "42",
               "--train_frac", str(54/64)])

    assert rc == 0, "CLI returned non-zero"
    assert output.exists()
    split = json.loads(output.read_text())
    assert split["n_train"] == 54
    assert split["n_val"] == 10
    assert not (set(split["train"]) & set(split["val"]))


def test_cli_fails_loudly_on_missing_logs_dir(tmp_path: Path, capsys):
    rc = main(["--logs_dir", str(tmp_path / "nonexistent"),
               "--output", str(tmp_path / "out.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "logs_dir does not exist" in err


def test_cli_fails_loudly_on_empty_logs_dir(tmp_path: Path, capsys):
    (tmp_path / "logs").mkdir()
    rc = main(["--logs_dir", str(tmp_path / "logs"),
               "--output", str(tmp_path / "out.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no .db files" in err
