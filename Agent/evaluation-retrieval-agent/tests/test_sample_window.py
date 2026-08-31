"""Tests for ERA Phase C-2.3 — random N-sample window per iter."""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from era.cli import main
from era.orchestration.sample_window import (
    _sha256_seed,
    pick_random_window,
)
from era.workspace import Workspace


def _scaffold_dataset(tmp_path: Path, n_samples: int = 12) -> Path:
    """Build a dataset with two methods, each carrying ``n_samples`` leaf
    sample dirs (mirrored sample structure)."""
    data_root = tmp_path / "data"
    for method in ("method_a", "method_b"):
        for i in range(n_samples):
            sample_dir = data_root / method / f"sample_{i:03d}"
            sample_dir.mkdir(parents=True)
            (sample_dir / "out.png").write_bytes(b"\x89PNG\r\n")
    return data_root


def _make_workspace(
    base: Path, data_root: Path, *,
    name: str = "sw-demo",
    iter_sample_count: int = 5,
    sample_count: int = 12,
    iteration: int = 1,
) -> Workspace:
    """Workspace with two mirrored methods + a fixed iter_sample_count."""
    ws = Workspace(base, name)
    ws.scaffold()
    ws.create_iteration(iteration)
    ws.set_current(iteration)
    ws.write_status({"project_name": name, "stage": "experiment_plan",
                     "stage_index": 5, "iteration": iteration,
                     "run_state": "running"})
    (ws.root / "config.yaml").write_text(yaml.safe_dump({
        "project_name": name, "task_family": "editing", "task_adapter": "x",
        "hardware": {"visible_gpu_ids": [0], "max_gpus_per_run": 1,
                     "per_gpu_memory_gb": 10.0},
        "data": {"data_root": str(data_root), "layout": "per_sample_dirs",
                 "methods": [
                    {"method_id": "method_a",
                     "path": str(data_root / "method_a"),
                     "output_file": "out.png"},
                    {"method_id": "method_b",
                     "path": str(data_root / "method_b"),
                     "output_file": "out.png"},
                 ],
                 "sample_glob": "*", "sample_count": sample_count,
                 "iter_sample_count": iter_sample_count,
                 "input_roles": {"src": "src.png"},
                 "sample_key": "relpath"},
    }), encoding="utf-8")
    return ws


# ---- _sha256_seed -------------------------------------------------------

def test_sha256_seed_is_stable_across_invocations():
    """Same inputs → same seed across Python interpreter sessions
    (Python's built-in hash() is salted; sha256 is not)."""
    a = _sha256_seed("project-x", 1)
    b = _sha256_seed("project-x", 1)
    assert a == b
    # 32-bit
    assert 0 <= a < 2**32


def test_sha256_seed_changes_with_iteration():
    s1 = _sha256_seed("project-x", 1)
    s2 = _sha256_seed("project-x", 2)
    assert s1 != s2


def test_sha256_seed_changes_with_project_name():
    s1 = _sha256_seed("project-a", 1)
    s2 = _sha256_seed("project-b", 1)
    assert s1 != s2


# ---- pick_random_window -------------------------------------------------

def test_deterministic_seed_same_iter_same_keys(tmp_path: Path):
    """Re-running the same iter produces the same sample list."""
    data_root = _scaffold_dataset(tmp_path, n_samples=12)
    ws = _make_workspace(tmp_path, data_root, iter_sample_count=5, iteration=1)
    a = pick_random_window(str(ws.root))
    b = pick_random_window(str(ws.root))
    assert a["status"] == "ok"
    assert b["status"] == "ok"
    assert a["sample_keys"] == b["sample_keys"]
    assert a["seed"] == b["seed"]
    assert a["count"] == 5
    assert a["total"] == 12
    assert a["source"] == "random"
    # Sorted lex (so runner iteration order is deterministic)
    assert a["sample_keys"] == sorted(a["sample_keys"])


def test_different_iters_pick_different_keys(tmp_path: Path):
    """Different iteration numbers → different sample sets (the whole
    point of Phase C-2.3 random-N rotation)."""
    data_root = _scaffold_dataset(tmp_path, n_samples=12)
    base = tmp_path / "iter1"
    ws1 = _make_workspace(base, data_root, name="rot1",
                          iter_sample_count=5, iteration=1)
    # Bump iteration to 2 for the second call
    ws2_base = tmp_path / "iter2"
    ws2 = _make_workspace(ws2_base, data_root, name="rot1",
                          iter_sample_count=5, iteration=2)
    a = pick_random_window(str(ws1.root), iteration=1)
    b = pick_random_window(str(ws2.root), iteration=2)
    # Same N
    assert a["count"] == b["count"] == 5
    # Different SETS (with 12-choose-5 = 792 possibilities, collision is
    # vanishingly unlikely; assert at least one key differs).
    assert a["sample_keys"] != b["sample_keys"]
    assert a["seed"] != b["seed"]


def test_n_exceeds_total_returns_all_samples(tmp_path: Path):
    """When N >= total samples, return every key (no random picking)."""
    data_root = _scaffold_dataset(tmp_path, n_samples=4)
    ws = _make_workspace(tmp_path, data_root,
                         iter_sample_count=10, sample_count=4)
    result = pick_random_window(str(ws.root))
    assert result["status"] == "ok"
    assert result["count"] == 4
    assert result["total"] == 4
    assert result["source"] == "all"
    assert result["sample_keys"] == sorted(result["sample_keys"])


def test_explicit_seed_overrides_auto(tmp_path: Path):
    """Caller can pin an explicit seed to reproduce a sample set."""
    data_root = _scaffold_dataset(tmp_path, n_samples=12)
    ws = _make_workspace(tmp_path, data_root, iter_sample_count=5)
    a = pick_random_window(str(ws.root), seed=42)
    b = pick_random_window(str(ws.root), seed=42)
    c = pick_random_window(str(ws.root), seed=999)
    assert a["sample_keys"] == b["sample_keys"]
    assert a["sample_keys"] != c["sample_keys"]
    assert a["seed"] == 42
    assert c["seed"] == 999


def test_rejects_non_workspace(tmp_path: Path):
    result = pick_random_window(tmp_path / "nope")
    assert result["error"] == "not_a_workspace"


def test_rejects_empty_methods(tmp_path: Path):
    data_root = _scaffold_dataset(tmp_path)
    ws = _make_workspace(tmp_path, data_root)
    # Manually wipe methods after scaffold to provoke the error.
    cfg_path = ws.root / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["data"]["methods"] = []
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    result = pick_random_window(str(ws.root))
    assert result["error"] == "no_methods"


def test_reports_no_samples_when_directory_is_empty(tmp_path: Path):
    """method[0].path exists but contains no leaf dirs → error."""
    data_root = tmp_path / "data"
    (data_root / "method_a").mkdir(parents=True)
    (data_root / "method_b").mkdir(parents=True)
    ws = _make_workspace(tmp_path, data_root, iter_sample_count=5)
    result = pick_random_window(str(ws.root))
    assert result["error"] == "no_samples"


def test_iteration_defaults_to_status_json(tmp_path: Path):
    """When iteration is omitted, the helper reads it from status.json."""
    data_root = _scaffold_dataset(tmp_path, n_samples=12)
    ws = _make_workspace(tmp_path, data_root,
                         iter_sample_count=5, iteration=3)
    result = pick_random_window(str(ws.root))
    assert result["status"] == "ok"
    assert result["iteration"] == 3


# ---- CLI round-trip ----------------------------------------------------

def test_cli_sample_window_round_trip(tmp_path: Path, monkeypatch, capsys):
    data_root = _scaffold_dataset(tmp_path, n_samples=12)
    ws = _make_workspace(tmp_path, data_root, iter_sample_count=5)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root)})))
    rc = main(["sample-window"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["count"] == 5
    assert payload["total"] == 12
    assert payload["source"] == "random"
    assert len(payload["sample_keys"]) == 5


def test_cli_sample_window_missing_workspace_path(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    rc = main(["sample-window"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["error"] == "missing_workspace_path"


def test_cli_sample_window_explicit_iteration_seed(
    tmp_path: Path, monkeypatch, capsys,
):
    data_root = _scaffold_dataset(tmp_path, n_samples=12)
    ws = _make_workspace(tmp_path, data_root, iter_sample_count=5)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root),
         "iteration": 7, "seed": 1234})))
    rc = main(["sample-window"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["iteration"] == 7
    assert payload["seed"] == 1234


def test_cli_sample_window_rejects_bad_iteration(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": "/tmp/anything",
         "iteration": "two"})))
    rc = main(["sample-window"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["error"] == "bad_iteration"
