"""Tests for the Workspace scaffolder."""

from __future__ import annotations

import json
from pathlib import Path

from era.workspace import (
    GLOBAL_DIRS,
    ITER_DIRS,
    Workspace,
    resolve_workspace_root,
)


def test_scaffold_creates_global_dirs(tmp_path: Path):
    ws = Workspace(tmp_path, "proj")
    ws.scaffold()
    for rel in GLOBAL_DIRS:
        assert (ws.root / rel).is_dir()


def test_create_iteration_and_current(tmp_path: Path):
    ws = Workspace(tmp_path, "proj")
    ws.scaffold()
    iter_dir = ws.create_iteration(1)
    ws.set_current(1)

    assert iter_dir.name == "iter_001"
    for rel in ITER_DIRS:
        assert (iter_dir / rel).is_dir()

    iteration_json = json.loads(
        (iter_dir / "iteration.json").read_text(encoding="utf-8")
    )
    assert iteration_json["iteration"] == 1
    assert "parent_feedback" in iteration_json
    # lifecycle stage lives in status.json, not iteration.json
    assert "stage" not in iteration_json

    # current resolves to iter_001
    assert ws.iter_path().resolve() == iter_dir.resolve()
    assert ws.iter_path().name == "iter_001"


def test_create_iteration_records_parent_feedback(tmp_path: Path):
    """create_iteration(parent_feedback=...) writes the carry-forward atomically."""
    ws = Workspace(tmp_path, "proj")
    ws.scaffold()
    pf = {
        "human_labels": "iter_001/human/human_labels.json",
        "evolution_state": "iter_001/react/evolution_state.json",
        "literature_update_brief": "iter_001/react/literature_update_brief.md",
    }
    ws.create_iteration(2, parent_feedback=pf)
    payload = json.loads(
        (ws.root / "iter_002" / "iteration.json").read_text(encoding="utf-8")
    )
    assert payload["parent_feedback"] == pf


def test_create_iteration_back_fills_parent_feedback(tmp_path: Path):
    """Re-calling create_iteration on an existing dir back-fills parent_feedback."""
    ws = Workspace(tmp_path, "proj")
    ws.scaffold()
    ws.create_iteration(2)  # parent_feedback starts None
    pf = {"human_labels": "iter_001/human/human_labels.json"}
    ws.create_iteration(2, parent_feedback=pf)
    payload = json.loads(
        (ws.root / "iter_002" / "iteration.json").read_text(encoding="utf-8")
    )
    assert payload["parent_feedback"] == pf


def test_set_current_repoints(tmp_path: Path):
    ws = Workspace(tmp_path, "proj")
    ws.scaffold()
    ws.create_iteration(1)
    ws.create_iteration(2)
    ws.set_current(1)
    assert ws.iter_path().name == "iter_001"
    ws.set_current(2)
    assert ws.iter_path().name == "iter_002"


def test_status_atomic_round_trip(tmp_path: Path):
    ws = Workspace(tmp_path, "proj")
    ws.scaffold()
    assert ws.exists() is False
    ws.write_status({"stage": "task_init", "iteration": 1})
    assert ws.exists() is True
    assert ws.read_status()["stage"] == "task_init"
    ws.update_stage("research")
    status = ws.read_status()
    assert status["stage"] == "research"
    assert "updated_at" in status


def test_write_json_no_tmp_left(tmp_path: Path):
    ws = Workspace(tmp_path, "proj")
    ws.scaffold()
    ws.write_json("probe/gpu_inventory.json", {"probe_ok": True})
    assert (ws.root / "probe" / "gpu_inventory.json").is_file()
    assert not (ws.root / "probe" / "gpu_inventory.json.tmp").exists()


# ---- resolve_workspace_root ---------------------------------------------

def test_resolve_workspace_root_from_root(tmp_path: Path):
    ws = Workspace(tmp_path, "proj")
    ws.scaffold()
    ws.write_status({"stage": "task_init"})
    assert resolve_workspace_root(ws.root) == ws.root.resolve()


def test_resolve_workspace_root_from_current_pointer(tmp_path: Path):
    ws = Workspace(tmp_path, "proj")
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({"stage": "task_init"})
    # the `current` iteration pointer normalizes back to the workspace root
    assert resolve_workspace_root(ws.root / "current") == ws.root.resolve()


def test_resolve_workspace_root_leaves_non_workspace_current(tmp_path: Path):
    # a directory literally named `current` with no sibling status.json is not
    # a workspace pointer — it is returned resolved, unchanged.
    target = tmp_path / "current"
    target.mkdir()
    assert resolve_workspace_root(target) == target.resolve()
