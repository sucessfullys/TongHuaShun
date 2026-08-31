"""Tests for the Stage 2-4 debate-loop counter (era/orchestration/debate.py)."""

from __future__ import annotations

import io
import json
from pathlib import Path

from era.cli import main
from era.orchestration.debate import debate_state, debate_tick
from era.workspace import Workspace


def _make_workspace(base: Path, max_rounds: int | None = None,
                    name: str = "demo-eval") -> Workspace:
    ws = Workspace(base, name)
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({
        "project_name": name, "stage": "plan_decision",
        "stage_index": 4, "iteration": 1, "run_state": "running",
    })
    if max_rounds is not None:
        ws.write_file("config.yaml", f"debate:\n  max_rounds: {max_rounds}\n")
    return ws


# ---- debate_state -------------------------------------------------------

def test_debate_state_initializes(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = debate_state(ws.root)
    assert result["status"] == "ok"
    state = result["state"]
    assert state["round"] == 1
    assert state["max_rounds"] == 4   # the default
    assert state["history"] == []
    assert (ws.iter_path() / "design" / "debate_state.json").is_file()


def test_debate_state_honors_config_max_rounds(tmp_path: Path):
    ws = _make_workspace(tmp_path, max_rounds=2)
    assert debate_state(ws.root)["state"]["max_rounds"] == 2


def test_debate_state_not_a_workspace(tmp_path: Path):
    assert debate_state(tmp_path / "nope")["error"] == "not_a_workspace"


def test_debate_state_atomic(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    debate_state(ws.root)
    design = ws.iter_path() / "design"
    assert not (design / "debate_state.json.tmp").exists()


# ---- debate_tick --------------------------------------------------------

def test_debate_tick_advance(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    result = debate_tick(ws.root, "ADVANCE")
    assert result["action"] == "advance"
    assert result["forced"] is False


def test_debate_tick_revise_under_cap(tmp_path: Path):
    ws = _make_workspace(tmp_path, max_rounds=3)
    result = debate_tick(ws.root, "REVISE")
    assert result["action"] == "revise"
    assert result["forced"] is False
    assert result["round"] == 2


def test_debate_tick_revise_at_cap_forces_advance(tmp_path: Path):
    ws = _make_workspace(tmp_path, max_rounds=2)
    debate_tick(ws.root, "REVISE")            # round 1 -> 2
    result = debate_tick(ws.root, "REVISE")   # round 2 == cap -> forced advance
    assert result["action"] == "advance"
    assert result["forced"] is True


def test_debate_tick_records_history(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    debate_tick(ws.root, "REVISE")
    debate_tick(ws.root, "ADVANCE")
    assert len(debate_state(ws.root)["state"]["history"]) == 2


def test_debate_tick_bad_verdict(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    assert debate_tick(ws.root, "MAYBE")["error"] == "bad_verdict"


def test_debate_tick_accepts_lowercase_verdict(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    assert debate_tick(ws.root, "advance")["action"] == "advance"


def test_debate_tick_not_a_workspace(tmp_path: Path):
    assert debate_tick(tmp_path / "nope", "ADVANCE")["error"] == "not_a_workspace"


# ---- debate_tick: transient-dir clearing on REVISE ----------------------

def _seed_transient(ws: Workspace) -> tuple[Path, Path]:
    """Drop a stale per-persona file into design/candidates/ + design/reviews/."""
    design = ws.iter_path() / "design"
    stale_candidate = design / "candidates" / "stale-persona.md"
    stale_review = design / "reviews" / "stale-critic.md"
    stale_candidate.write_text("stale round-N candidate", encoding="utf-8")
    stale_review.write_text("stale round-N review", encoding="utf-8")
    return stale_candidate, stale_review


def test_debate_tick_revise_clears_transient_dirs(tmp_path: Path):
    ws = _make_workspace(tmp_path, max_rounds=3)
    stale_candidate, stale_review = _seed_transient(ws)
    result = debate_tick(ws.root, "REVISE")
    assert result["action"] == "revise"
    # the stale per-persona files are gone — but the dirs survive, empty
    assert not stale_candidate.exists()
    assert not stale_review.exists()
    assert (ws.iter_path() / "design" / "candidates").is_dir()
    assert (ws.iter_path() / "design" / "reviews").is_dir()
    assert result["cleared"] == ["design/candidates", "design/reviews"]


def test_debate_tick_advance_keeps_transient_dirs(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    stale_candidate, stale_review = _seed_transient(ws)
    result = debate_tick(ws.root, "ADVANCE")
    assert result["action"] == "advance"
    assert result["cleared"] == []
    # ADVANCE ends the loop — nothing is cleared
    assert stale_candidate.exists()
    assert stale_review.exists()


def test_debate_tick_forced_advance_keeps_transient_dirs(tmp_path: Path):
    ws = _make_workspace(tmp_path, max_rounds=2)
    debate_tick(ws.root, "REVISE")            # round 1 -> 2 (clears)
    stale_candidate, stale_review = _seed_transient(ws)
    result = debate_tick(ws.root, "REVISE")   # round 2 == cap -> forced advance
    assert result["action"] == "advance" and result["forced"] is True
    assert result["cleared"] == []
    # a forced advance terminates the loop — the seeded files survive
    assert stale_candidate.exists()
    assert stale_review.exists()


def test_debate_tick_records_reason(tmp_path: Path):
    ws = _make_workspace(tmp_path, max_rounds=3)
    debate_tick(ws.root, "REVISE", reason="metric-baseline missing a pilot gate")
    history = debate_state(ws.root)["state"]["history"]
    assert history[-1]["reason"] == "metric-baseline missing a pilot gate"


def test_debate_tick_omits_reason_when_absent(tmp_path: Path):
    ws = _make_workspace(tmp_path)
    debate_tick(ws.root, "ADVANCE")
    history = debate_state(ws.root)["state"]["history"]
    assert "reason" not in history[-1]


# ---- CLI round-trips ----------------------------------------------------

def test_cli_debate_state_round_trip(tmp_path: Path, monkeypatch, capsys):
    ws = _make_workspace(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root)})))
    code = main(["debate-state"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["state"]["round"] == 1


def test_cli_debate_tick_round_trip(tmp_path: Path, monkeypatch, capsys):
    ws = _make_workspace(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "verdict": "ADVANCE"})))
    code = main(["debate-tick"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["action"] == "advance"


def test_cli_debate_tick_forwards_reason(tmp_path: Path, monkeypatch, capsys):
    ws = _make_workspace(tmp_path, max_rounds=3)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "verdict": "REVISE",
         "reason": "rigor-critic flagged an unbounded slot"})))
    code = main(["debate-tick"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["action"] == "revise"
    assert result["cleared"] == ["design/candidates", "design/reviews"]
    history = debate_state(ws.root)["state"]["history"]
    assert history[-1]["reason"] == "rigor-critic flagged an unbounded slot"
