"""Tests for ERA Phase C-1 — auto-revise replaces pre-Stage-8 blocks.

Cover the deterministic helper :mod:`era.orchestration.auto_revise` and the
``era.cli auto-revise`` subcommand. The contract: any pre-Stage-8 stage that
would otherwise have set ``run_state: blocked`` now routes through
``auto_revise``, which records the failure context and triggers a Stage 9
``REVISE_SKIP_STAGE1`` (or forces ADVANCE at the iter cap).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from era.cli import main
from era.orchestration.auto_revise import (
    TRIGGER_REL,
    auto_revise,
    read_trigger,
)
from era.workspace import Workspace


def _make_workspace(
    base: Path, *, iteration: int = 1, max_iterations: int = 5,
    name: str = "auto-revise-demo",
) -> Workspace:
    """Build a minimal workspace at the given iter with status=Stage 6/running."""
    ws = Workspace(base, name)
    ws.scaffold()
    for n in range(1, iteration + 1):
        ws.create_iteration(n)
    ws.set_current(iteration)
    ws.write_status({
        "project_name": name, "stage": "full_experiment",
        "stage_index": 6, "iteration": iteration, "run_state": "running",
    })
    (ws.root / "config.yaml").write_text(yaml.safe_dump({
        "project_name": name, "task_family": "editing", "task_adapter": "x",
        "hardware": {"visible_gpu_ids": [0], "max_gpus_per_run": 1,
                     "per_gpu_memory_gb": 10.0},
        "data": {"data_root": "/tmp/x", "layout": "per_sample_dirs",
                 "methods": [{"method_id": "m", "path": "/tmp/m",
                              "output_file": "out.png"}],
                 "sample_glob": "*", "sample_count": 1,
                 "input_roles": {"input": "in.png"},
                 "sample_key": "relpath"},
        "react": {"max_iterations": max_iterations,
                  "endorsement_threshold": 0.80,
                  "min_alignment_samples": 20},
    }), encoding="utf-8")
    return ws


def _run_cli(cmd: str, payload: dict, capsys) -> tuple[int, dict]:
    import sys
    saved_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps(payload))
    try:
        rc = main([cmd])
    finally:
        sys.stdin = saved_stdin
    captured = capsys.readouterr()
    return rc, json.loads(captured.out)


# ---- happy path: under-cap REVISE_SKIP_STAGE1 ---------------------------

def test_auto_revise_under_cap_scaffolds_next_iter(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=1, max_iterations=5)
    diag = {
        "missing_configs": ["b1", "b2", "h1", "h3"],
        "scored_configs": ["a1", "a2", "a3"],
        "failed_tasks": [],
        "in_progress_tasks": [],
    }
    out = auto_revise(
        ws.root, reason="stage6_incomplete", source_stage=6,
        blocker_summary="4/9 configs missing after one heal pass",
        diagnostic=diag,
    )
    assert out["status"] == "ok"
    assert out["decision"] == "REVISE_SKIP_STAGE1"
    assert out["forced_advance"] is False
    assert out["iteration"] == 1
    assert out["next_iter"] == 2

    # trigger.json written under prior iter
    trigger_path = Path(out["trigger_path"])
    assert trigger_path == ws.root / "iter_001" / TRIGGER_REL
    assert trigger_path.is_file()
    trigger = json.loads(trigger_path.read_text(encoding="utf-8"))
    assert trigger["reason"] == "stage6_incomplete"
    assert trigger["source_stage"] == 6
    assert trigger["diagnostic"]["missing_configs"] == ["b1", "b2", "h1", "h3"]

    # new iter exists and is current
    iter2 = ws.root / "iter_002"
    assert iter2.is_dir()
    status = ws.read_status()
    assert status["iteration"] == 2

    # parent_feedback carries the auto_revise_trigger pointer
    iter_json = json.loads((iter2 / "iteration.json").read_text(encoding="utf-8"))
    pf = iter_json["parent_feedback"]
    assert pf["auto_revise_trigger"] == "iter_001/auto_revise/trigger.json"


def test_auto_revise_at_cap_returns_forced_advance(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=5, max_iterations=5)
    out = auto_revise(
        ws.root, reason="stage6_incomplete", source_stage=6,
        blocker_summary="hit cap",
    )
    assert out["status"] == "ok"
    assert out["decision"] == "ADVANCE"
    assert out["forced_advance"] is True
    assert out["iteration"] == 5
    # trigger.json still written for the audit trail
    trigger_path = Path(out["trigger_path"])
    assert trigger_path == ws.root / "iter_005" / TRIGGER_REL
    assert trigger_path.is_file()
    # No iter_006 scaffolded
    assert not (ws.root / "iter_006").exists()
    # status.json untouched (loop owns stage_index advance)
    assert ws.read_status()["iteration"] == 5


def test_auto_revise_is_idempotent(tmp_path: Path):
    """Calling auto_revise twice on the same iter returns the prior trigger
    and does NOT scaffold a second next-iter."""
    ws = _make_workspace(tmp_path, iteration=1, max_iterations=5)
    first = auto_revise(
        ws.root, reason="stage6_incomplete", source_stage=6,
        blocker_summary="first call",
    )
    assert first["decision"] == "REVISE_SKIP_STAGE1"

    # iter_002 exists after the first call
    assert (ws.root / "iter_002").is_dir()

    # A second auto-revise call on the same prior iter (we manually rewind
    # status.json's iteration to 1 to simulate a duplicate trigger from
    # two different ralph-loop stages on the same pass).
    status = ws.read_status()
    status["iteration"] = 1
    ws.write_status(status)
    ws.set_current(1)

    second = auto_revise(
        ws.root, reason="stage7_comparison_missing", source_stage=7,
        blocker_summary="second call (should be ignored)",
    )
    assert second["status"] == "ok"
    assert second.get("already_triggered") is True
    assert second["reason"] == "stage6_incomplete"
    # No iter_003 scaffolded
    assert not (ws.root / "iter_003").exists()


def test_auto_revise_rejects_bad_source_stage(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=1)
    out = auto_revise(
        ws.root, reason="oops", source_stage=8,
        blocker_summary="Stage 8 has its own handoff path",
    )
    assert out["error"] == "bad_source_stage"


def test_auto_revise_rejects_missing_reason(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=1)
    out = auto_revise(
        ws.root, reason="", source_stage=6, blocker_summary="x",
    )
    assert out["error"] == "missing_reason"


def test_auto_revise_rejects_non_workspace(tmp_path: Path):
    out = auto_revise(
        tmp_path / "nope", reason="stage6_incomplete", source_stage=6,
        blocker_summary="not a workspace",
    )
    assert out["error"] == "not_a_workspace"


# ---- read_trigger -------------------------------------------------------

def test_read_trigger_returns_dict_when_present(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=1)
    auto_revise(
        ws.root, reason="stage4_brief_invalid", source_stage=4,
        blocker_summary="brief failed Rule 5",
        diagnostic={"problems": ["one", "two"]},
    )
    iter_dir = ws.root / "iter_001"
    trigger = read_trigger(iter_dir)
    assert trigger is not None
    assert trigger["reason"] == "stage4_brief_invalid"
    assert trigger["diagnostic"]["problems"] == ["one", "two"]


def test_read_trigger_returns_none_when_absent(tmp_path: Path):
    iter_dir = tmp_path / "iter_999"
    iter_dir.mkdir()
    assert read_trigger(iter_dir) is None


# ---- CLI subcommand -----------------------------------------------------

def test_cli_auto_revise_under_cap(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path, iteration=1, max_iterations=5)
    rc, payload = _run_cli("auto-revise", {
        "workspace_path": str(ws.root),
        "reason": "stage6_incomplete",
        "source_stage": 6,
        "blocker_summary": "CLI smoke",
        "diagnostic": {"missing_configs": ["b1"]},
    }, capsys)
    assert rc == 0
    assert payload["decision"] == "REVISE_SKIP_STAGE1"
    assert payload["next_iter"] == 2


def test_cli_auto_revise_missing_fields(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path, iteration=1)
    rc, payload = _run_cli("auto-revise", {
        "workspace_path": str(ws.root),
        # no reason, no source_stage, no blocker_summary
    }, capsys)
    assert rc == 1
    assert payload["error"] in (
        "missing_params", "missing_reason", "bad_source_stage",
    )


def test_cli_auto_revise_at_cap(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path, iteration=5, max_iterations=5)
    rc, payload = _run_cli("auto-revise", {
        "workspace_path": str(ws.root),
        "reason": "stage6_incomplete",
        "source_stage": 6,
        "blocker_summary": "cap",
    }, capsys)
    assert rc == 0
    assert payload["forced_advance"] is True
    assert payload["decision"] == "ADVANCE"
