"""Tests for ERA Stage 9 — the ReAct iteration gate.

Cover the deterministic backbone (:mod:`era.orchestration.react`) and the four
CLI subcommands that drive it. The advisor sub-agent is out of scope — its
output is JSON, and we exercise the *consumer* of that JSON
(``check_evolution_state``) plus the *response* to its verdict
(``react_tick`` + ``create_next_iteration``).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from era.cli import main
from era.orchestration.react import (
    PROPOSAL_TYPES,
    VERDICTS,
    aggregate_cumulative_feedback,
    check_evolution_state,
    create_next_iteration,
    react_state,
    react_tick,
)
from era.workspace import Workspace


# ---- fixtures -----------------------------------------------------------

def _make_workspace(
    base: Path, *, iteration: int = 1, max_iterations: int = 5,
    name: str = "react-demo",
) -> Workspace:
    """Build a minimal workspace with status.json + config.yaml at the given iter."""
    ws = Workspace(base, name)
    ws.scaffold()
    for n in range(1, iteration + 1):
        ws.create_iteration(n)
    ws.set_current(iteration)
    ws.write_status({
        "project_name": name, "stage": "human_feedback",
        "stage_index": 8, "iteration": iteration, "run_state": "running",
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


def _write_iter_feedback(
    ws: Workspace, n: int, *,
    config_summary: list[dict] | None = None,
    general_feedback: str = "",
    item_marks: list[dict] | None = None,
) -> None:
    """Write finalized feedback.json + human_labels.json for iter_NNN."""
    iter_dir = ws.root / f"iter_{n:03d}"
    (iter_dir / "human").mkdir(parents=True, exist_ok=True)
    feedback = {
        "schema_version": "1.0", "iteration": n,
        "iteration_dir": f"iter_{n:03d}", "mode": "full",
        "status": "finalized", "reviewer": "test",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "finalized_at": "2026-01-01T00:00:00+00:00",
        "item_marks": item_marks or [],
        "comparison_marks": [],
        "general_feedback": general_feedback,
    }
    labels = {
        "schema_version": "1.0", "iteration": n,
        "iteration_dir": f"iter_{n:03d}",
        "finalized_at": "2026-01-01T00:00:00+00:00",
        "task_family": "editing", "mode": "full",
        "default_rule": "unmarked_is_correct",
        "flagging_configs": [], "comparison_configs": [],
        "labels": [], "comparison_labels": [],
        "config_summary": config_summary or [],
        "general_feedback": general_feedback,
    }
    (iter_dir / "human" / "feedback.json").write_text(
        json.dumps(feedback), encoding="utf-8")
    (iter_dir / "human" / "human_labels.json").write_text(
        json.dumps(labels), encoding="utf-8")


def _good_state(iteration: int, decision: str) -> dict:
    """Return a minimal, schema-valid evolution_state.json payload."""
    return {
        "schema_version": "1.0", "iteration": iteration,
        "created_at": "2026-01-01T00:00:00+00:00",
        "cumulative_feedback": {
            "iters_analyzed": [iteration],
            "per_config_trajectory": [],
            "general_feedback_themes": [],
        },
        "exclude_list": [],
        "evolution_proposals": [],
        "general_failure_modes": [],
        "decision": decision,
        "forced": False, "round": iteration,
        "rationale": "test",
        "literature_update_requested": decision == "REVISE_RERUN_STAGE1",
    }


def _run_cli(cmd: str, payload: dict,
             capsys) -> tuple[int, dict]:
    """Invoke ``python -m era.cli <cmd>`` with a fake stdin and capture stdout."""
    import sys
    saved_stdin = sys.stdin
    sys.stdin = io.StringIO(json.dumps(payload))
    try:
        rc = main([cmd])
    finally:
        sys.stdin = saved_stdin
    captured = capsys.readouterr()
    return rc, json.loads(captured.out)


# ---- aggregate_cumulative_feedback --------------------------------------

def test_aggregate_skips_iters_without_human_labels(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=2)
    # iter_001 is finalized; iter_002 is not.
    _write_iter_feedback(ws, 1, general_feedback="iter 1 was OK")
    out = aggregate_cumulative_feedback(ws.root, 2)
    assert out["iters_analyzed"] == [1]
    assert "iter_001: iter 1 was OK" in out["general_feedback_themes"]


def test_aggregate_builds_per_config_trajectory(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=2)
    cs1 = [
        {"combination_id": "vlm-7b", "family": "A", "modality": "flag",
         "total": 40, "endorsed": 14, "flagged_wrong": 26,
         "endorsement_rate": 0.35},
        {"combination_id": "vlm-72b", "family": "A", "modality": "flag",
         "total": 40, "endorsed": 30, "flagged_wrong": 10,
         "endorsement_rate": 0.75},
    ]
    cs2 = [
        {"combination_id": "vlm-7b", "family": "A", "modality": "flag",
         "total": 40, "endorsed": 16, "flagged_wrong": 24,
         "endorsement_rate": 0.40},
        {"combination_id": "vlm-72b", "family": "A", "modality": "flag",
         "total": 40, "endorsed": 34, "flagged_wrong": 6,
         "endorsement_rate": 0.85},
    ]
    _write_iter_feedback(ws, 1, config_summary=cs1,
                         general_feedback="7B underperforms on fidelity")
    _write_iter_feedback(ws, 2, config_summary=cs2,
                         general_feedback="all 7B is not good",
                         item_marks=[{"sample_key": "s1", "method_id": "m_a",
                                      "combination_id": "vlm-7b", "label": "wrong",
                                      "comment": "missed obvious garment defect"}])

    out = aggregate_cumulative_feedback(ws.root, 2)
    assert out["iters_analyzed"] == [1, 2]

    by_id = {row["combination_id"]: row for row in out["per_config_trajectory"]}
    assert set(by_id) == {"vlm-7b", "vlm-72b"}
    seven = by_id["vlm-7b"]["endorsement_rate_by_iter"]
    assert [r["iteration"] for r in seven] == [1, 2]
    assert [r["rate"] for r in seven] == [0.35, 0.40]
    assert [r["n_samples"] for r in seven] == [40, 40]
    # item_marks comments are surfaced on the right config.
    assert any("missed obvious garment defect" in t
               for t in by_id["vlm-7b"]["wrong_themes"])
    # general_feedback is bulleted per-iter.
    themes = out["general_feedback_themes"]
    assert "iter_001: 7B underperforms on fidelity" in themes
    assert "iter_002: all 7B is not good" in themes


def test_aggregate_returns_correct_rate_for_family_b(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=1)
    _write_iter_feedback(ws, 1, config_summary=[
        {"combination_id": "region-baseline", "family": "B",
         "modality": "comparison", "total": 30, "correct": 24, "wrong": 6,
         "correct_rate": 0.80},
    ])
    out = aggregate_cumulative_feedback(ws.root, 1)
    assert out["per_config_trajectory"][0]["endorsement_rate_by_iter"][0]["rate"] == 0.80


def _write_iter_runtime_failures(
    ws: Workspace, n: int,
    runtime_failed: list[tuple[str, str]],
) -> None:
    """Write an experiment_state.json with the given (task_id, category) pairs
    marked outcome=runtime_failed. Used to feed the React aggregator."""
    iter_dir = ws.root / f"iter_{n:03d}"
    (iter_dir / "experiments").mkdir(parents=True, exist_ok=True)
    tasks: dict = {}
    for task_id, cat in runtime_failed:
        tasks[task_id] = {
            "kind": "eval", "status": "failed",
            "outcome": "runtime_failed", "failure_category": cat,
        }
    (iter_dir / "experiments" / "experiment_state.json").write_text(
        json.dumps({"schema_version": 1, "mode": "full", "tasks": tasks}),
        encoding="utf-8")


def test_aggregate_surfaces_runtime_failure_signal(tmp_path: Path):
    """Runtime-failed eval tasks from prior iters flow into the cumulative
    block so Stage 9 React's advisor can spot infra-cluster signals."""
    ws = _make_workspace(tmp_path, iteration=2)
    # iter_001 — three serve-class infra failures
    _write_iter_runtime_failures(ws, 1, [
        ("eval-pilot-cfg-a", "serving"),
        ("eval-pilot-cfg-b", "serving"),
        ("eval-pilot-cfg-c", "oom"),
    ])
    _write_iter_feedback(ws, 1, general_feedback="iter 1 surveyed")
    # iter_002 — one hung runner
    _write_iter_runtime_failures(ws, 2, [("eval-full-cfg-d", "hung")])
    _write_iter_feedback(ws, 2, general_feedback="iter 2 surveyed")

    out = aggregate_cumulative_feedback(ws.root, 2)
    # Per-event list
    assert len(out["runtime_failure_signal"]) == 4
    by_iter = {row["iteration"]: row for row in out["runtime_failure_signal"]
               if row["task_id"] == "eval-pilot-cfg-a"}
    assert by_iter[1]["failure_category"] == "serving"
    # Category counts make a cluster visible
    counts = out["runtime_failure_category_counts"]
    assert counts == {"serving": 2, "oom": 1, "hung": 1}


def test_aggregate_runtime_failure_signal_independent_of_feedback(
    tmp_path: Path,
):
    """Runtime failures from an iter whose Stage 8 human feedback never
    finalized must still surface — they're the strongest signal that the
    iter never got that far for a reason the next iter should know."""
    ws = _make_workspace(tmp_path, iteration=1)
    _write_iter_runtime_failures(ws, 1, [("eval-pilot-cfg-x", "oom")])
    # no _write_iter_feedback — iter 1 has no human_labels.json
    out = aggregate_cumulative_feedback(ws.root, 1)
    assert out["iters_analyzed"] == []  # feedback not finalized
    # But runtime failure is still surfaced
    assert out["runtime_failure_category_counts"] == {"oom": 1}
    assert out["runtime_failure_signal"][0]["combination_id"] == "cfg-x"


# ---- react_state / react_tick -------------------------------------------

def test_react_state_reports_cap_and_priors(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=3, max_iterations=5)
    state = react_state(ws.root)
    assert state["status"] == "ok"
    assert state["iteration"] == 3
    assert state["max_iterations"] == 5
    assert state["at_cap"] is False
    assert state["prior_decisions"] == []


def test_react_tick_advance_pass_through(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=1)
    out = react_tick(ws.root, "ADVANCE", rationale="best config aligned")
    assert out["status"] == "ok"
    assert out["decision"] == "ADVANCE"
    assert out["forced"] is False
    decision_path = Path(out["decision_path"])
    assert decision_path.is_file()
    recorded = json.loads(decision_path.read_text())
    assert recorded["decision"] == "ADVANCE"
    assert recorded["rationale"] == "best config aligned"


def test_react_tick_revise_under_cap_passes_through(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=2, max_iterations=5)
    out = react_tick(ws.root, "REVISE_SKIP_STAGE1")
    assert out["decision"] == "REVISE_SKIP_STAGE1"
    assert out["forced"] is False


def test_react_tick_at_cap_forces_advance(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=5, max_iterations=5)
    out = react_tick(ws.root, "REVISE_RERUN_STAGE1")
    assert out["decision"] == "ADVANCE"
    assert out["forced"] is True
    # The history file records the original verdict, not just the forced decision.
    hist_lines = (ws.iter_path() / "react" / "history.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(hist_lines) == 1
    row = json.loads(hist_lines[0])
    assert row["verdict"] == "REVISE_RERUN_STAGE1"
    assert row["decision"] == "ADVANCE"
    assert row["forced"] is True


def test_react_tick_rejects_unknown_verdict(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=1)
    out = react_tick(ws.root, "MAYBE")
    assert out["error"] == "bad_verdict"


def test_react_tick_overwrites_decision_keeps_history(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=2, max_iterations=5)
    react_tick(ws.root, "REVISE_SKIP_STAGE1", rationale="first try")
    react_tick(ws.root, "REVISE_RERUN_STAGE1", rationale="actually we need lit")
    decision = json.loads(
        (ws.iter_path() / "react" / "decision.json").read_text())
    assert decision["decision"] == "REVISE_RERUN_STAGE1"
    hist = (ws.iter_path() / "react" / "history.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(hist) == 2


# ---- create_next_iteration ----------------------------------------------

def test_create_next_iteration_skip_stage1(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=1)
    # Drop carry-forward fixtures so parent_feedback gets populated.
    iter1 = ws.root / "iter_001"
    (iter1 / "human" / "human_labels.json").parent.mkdir(parents=True, exist_ok=True)
    (iter1 / "human" / "human_labels.json").write_text("{}", encoding="utf-8")
    (iter1 / "react" / "evolution_state.json").parent.mkdir(parents=True, exist_ok=True)
    (iter1 / "react" / "evolution_state.json").write_text(
        json.dumps(_good_state(1, "REVISE_SKIP_STAGE1")), encoding="utf-8")

    out = create_next_iteration(ws.root, rerun_stage1=False)
    assert out["status"] == "ok"
    assert out["iteration"] == 2
    # stage_index stores the *last completed* stage. SKIP_STAGE1 means
    # Stage 1 (research) is treated as carried-forward from the prior iter,
    # so the next ralph pass dispatches Stage 2 (plan_brainstorm).
    assert out["stage"] == "research"
    assert out["stage_index"] == 1
    assert out["rerun_stage1"] is False

    iter2 = ws.root / "iter_002"
    assert iter2.is_dir()

    # parent_feedback has resolved paths to the carry-forward files.
    iter_json = json.loads((iter2 / "iteration.json").read_text())
    pf = iter_json["parent_feedback"]
    assert pf["human_labels"] == "iter_001/human/human_labels.json"
    assert pf["evolution_state"] == "iter_001/react/evolution_state.json"
    assert "literature_update_brief" not in pf  # not present on disk

    # status.json is on the new iter at stage_index=1 (research completed,
    # plan_brainstorm runs next).
    status = ws.read_status()
    assert status["iteration"] == 2
    assert status["stage"] == "research"
    assert status["stage_index"] == 1
    assert status["run_state"] == "running"


def test_create_next_iteration_carries_auto_revise_trigger(tmp_path: Path):
    """When the prior iter wrote auto_revise/trigger.json, the new iter's
    parent_feedback exposes a workspace-relative pointer to it."""
    ws = _make_workspace(tmp_path, iteration=1)
    iter1 = ws.root / "iter_001"
    (iter1 / "human").mkdir(parents=True, exist_ok=True)
    (iter1 / "human" / "human_labels.json").write_text("{}", encoding="utf-8")
    (iter1 / "react").mkdir(parents=True, exist_ok=True)
    (iter1 / "react" / "evolution_state.json").write_text(
        json.dumps(_good_state(1, "REVISE_SKIP_STAGE1")), encoding="utf-8")
    # NEW: prior iter has a Phase C-1 auto-revise trigger
    (iter1 / "auto_revise").mkdir(parents=True, exist_ok=True)
    (iter1 / "auto_revise" / "trigger.json").write_text(json.dumps({
        "iteration": 1, "reason": "stage6_incomplete", "source_stage": 6,
        "blocker_summary": "test", "diagnostic": {"missing_configs": ["b1"]},
        "at": "2026-01-01T00:00:00+00:00",
    }), encoding="utf-8")

    out = create_next_iteration(ws.root, rerun_stage1=False)
    assert out["status"] == "ok"

    iter2 = ws.root / "iter_002"
    iter_json = json.loads((iter2 / "iteration.json").read_text())
    pf = iter_json["parent_feedback"]
    assert pf["auto_revise_trigger"] == "iter_001/auto_revise/trigger.json"


def test_create_next_iteration_omits_auto_revise_when_absent(tmp_path: Path):
    """No trigger on disk → no pointer. Purely additive guarantee."""
    ws = _make_workspace(tmp_path, iteration=1)
    iter1 = ws.root / "iter_001"
    (iter1 / "human").mkdir(parents=True, exist_ok=True)
    (iter1 / "human" / "human_labels.json").write_text("{}", encoding="utf-8")
    (iter1 / "react").mkdir(parents=True, exist_ok=True)
    (iter1 / "react" / "evolution_state.json").write_text(
        json.dumps(_good_state(1, "REVISE_SKIP_STAGE1")), encoding="utf-8")

    create_next_iteration(ws.root, rerun_stage1=False)
    iter_json = json.loads(
        (ws.root / "iter_002" / "iteration.json").read_text())
    assert "auto_revise_trigger" not in iter_json["parent_feedback"]


def test_create_next_iteration_rerun_stage1_with_brief(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=1)
    iter1 = ws.root / "iter_001"
    (iter1 / "human" / "human_labels.json").parent.mkdir(parents=True, exist_ok=True)
    (iter1 / "human" / "human_labels.json").write_text("{}", encoding="utf-8")
    (iter1 / "react").mkdir(parents=True, exist_ok=True)
    (iter1 / "react" / "evolution_state.json").write_text(
        json.dumps(_good_state(1, "REVISE_RERUN_STAGE1")), encoding="utf-8")
    (iter1 / "react" / "literature_update_brief.md").write_text(
        "## Add\n- EA / iterative auto-eval refinement\n", encoding="utf-8")

    out = create_next_iteration(ws.root, rerun_stage1=True)
    # stage_index stores the *last completed* stage. RERUN_STAGE1 wants
    # Stage 1 (research) to run again, so the new iter's last-completed is
    # Stage 0 (task_init — workspace-global). The next ralph pass reads
    # stage_index=0 and dispatches Stage 1 (research).
    assert out["stage_index"] == 0
    assert out["stage"] == "task_init"
    iter_json = json.loads(
        (ws.root / "iter_002" / "iteration.json").read_text())
    pf = iter_json["parent_feedback"]
    assert pf["literature_update_brief"] == "iter_001/react/literature_update_brief.md"


def test_create_next_iteration_stage_index_drives_correct_dispatch(
    tmp_path: Path,
):
    """The strongest contract test: STAGES[status.stage_index + 1] must be
    the stage the user wants to run next. Pinned directly against STAGES,
    so re-introducing the off-by-one would fail this even if literal-value
    assertions in the sibling tests get "fixed" back to the old values."""
    from era.orchestration.lifecycle import STAGES

    # RERUN_STAGE1 — the next ralph pass must dispatch Stage 1 (research).
    ws_r = _make_workspace(tmp_path / "rerun", iteration=1, name="rerun-demo")
    (ws_r.root / "iter_001" / "human").mkdir(parents=True, exist_ok=True)
    (ws_r.root / "iter_001" / "human" / "human_labels.json").write_text(
        "{}", encoding="utf-8")
    (ws_r.root / "iter_001" / "react").mkdir(parents=True, exist_ok=True)
    (ws_r.root / "iter_001" / "react" / "evolution_state.json").write_text(
        json.dumps(_good_state(1, "REVISE_RERUN_STAGE1")), encoding="utf-8")
    create_next_iteration(ws_r.root, rerun_stage1=True)
    status_r = ws_r.read_status()
    assert STAGES[status_r["stage_index"] + 1] == "research"

    # SKIP_STAGE1 — the next ralph pass must dispatch Stage 2 (plan_brainstorm).
    ws_s = _make_workspace(tmp_path / "skip", iteration=1, name="skip-demo")
    (ws_s.root / "iter_001" / "human").mkdir(parents=True, exist_ok=True)
    (ws_s.root / "iter_001" / "human" / "human_labels.json").write_text(
        "{}", encoding="utf-8")
    (ws_s.root / "iter_001" / "react").mkdir(parents=True, exist_ok=True)
    (ws_s.root / "iter_001" / "react" / "evolution_state.json").write_text(
        json.dumps(_good_state(1, "REVISE_SKIP_STAGE1")), encoding="utf-8")
    create_next_iteration(ws_s.root, rerun_stage1=False)
    status_s = ws_s.read_status()
    assert STAGES[status_s["stage_index"] + 1] == "plan_brainstorm"


def test_create_next_iteration_refuses_at_cap(tmp_path: Path):
    ws = _make_workspace(tmp_path, iteration=5, max_iterations=5)
    out = create_next_iteration(ws.root, rerun_stage1=False)
    assert out["error"] == "at_cap"
    # status.json untouched.
    assert ws.read_status()["iteration"] == 5


# ---- Phase C-2.5 — full elitism wire integration -----------------------

def _brief_with_combination_ids(combination_ids: list[str]) -> dict:
    """A minimal valid Stage 4 brief with the given combination_ids.

    One metric_baseline (Family B) + N-1 distinct vlm_scale (Family A)
    so Rule-4 + Rule-5 invariants pass while letting callers pick which
    cids appear.
    """
    if not combination_ids:
        combination_ids = ["metric-baseline"]
    configs = []
    for i, cid in enumerate(combination_ids):
        if i == 0:
            configs.append({
                "combination_id": cid, "slot": "metric_baseline",
                "family": "B", "judge": None,
                "metric_subfamily": "clip-dino", "prompt": None,
                "scope": "whole", "inputs_needed": ["output"],
                "gpu_estimate": "1xH100", "cost_estimate_usd": 0.0,
                "hypothesis_id": f"H{i}", "feasible": True,
            })
        else:
            configs.append({
                "combination_id": cid, "slot": "vlm_scale",
                "family": "A", "judge": f"qwen-vl-{30 + i}b",
                "metric_subfamily": None, "prompt": "pairwise-rubric",
                "scope": "whole",
                "inputs_needed": ["input_cloth", "output"],
                "gpu_estimate": "4xH100", "cost_estimate_usd": 0.0,
                "hypothesis_id": f"H{i}", "feasible": True,
            })
    return {
        "iteration": 2, "debate_round": 1,
        "evaluation_goal": "test",
        "candidate_configs": configs,
        "validation": {"human_rating_set": "test",
                       "alignment_metric": "kendall_tau",
                       "sample_size": 50, "debiasing": "order-swap"},
        "pilot": {"sample_count": 10, "go_no_go": "tau visible"},
        "scale_comparison": "two scales",
        "resource_estimate": {"gpu_hours": 1.0, "wallclock_hours": 1.0,
                              "api_cost_usd": 0.0},
        "pivot_matrix": [{"pilot_outcome": "all tau<0.3",
                          "action": "revise"}],
    }


def test_must_include_propagates_through_create_next_iteration(
    tmp_path: Path, capsys
):
    """Phase C-2.5 integration — the full elitism wire:

    iter_001/react/evolution_state.json carries
    must_include_configs → create_next_iteration scaffolds iter_002 with
    parent_feedback.evolution_state pointer → the Stage 4 skill resolves
    that pointer and passes it to check-experiment-brief → the brief
    gate rejects any candidate_configs that drops a must-include and
    surfaces the dropped IDs in structured ``missing_must_include``.

    Catches a future regression in any of:
    - create_next_iteration's parent_feedback assembly
    - the workspace-relative pointer resolution
    - check-experiment-brief evolution_state_path parsing
    - validate_experiment_brief elitism enforcement
    """
    ws = _make_workspace(tmp_path, iteration=1)
    iter1 = ws.root / "iter_001"

    # Stage 9 of iter_001 wrote an evolution_state with must_include_configs
    # (partial-pass on the C-2 gate — 1 of N configs survived, the rest
    # failed, so we elitism-carry-forward the survivor).
    (iter1 / "react").mkdir(parents=True, exist_ok=True)
    es_state = _good_state(1, "REVISE_SKIP_STAGE1")
    es_state["must_include_configs"] = ["vlm-qwen35b-proven"]
    es_state["hall_of_fame"] = [{
        "combination_id": "vlm-qwen35b-proven",
        "fitness_composite": 0.78,
        "pass_rate": 0.85, "recall_rate": 0.72,
        "human_endorsement_rate": 0.80,
        "iter_introduced": 1, "last_iter_seen": 1,
    }]
    (iter1 / "react" / "evolution_state.json").write_text(
        json.dumps(es_state), encoding="utf-8",
    )
    # human_labels.json must exist for the parent_feedback pointer to be
    # populated — empty dict is fine for this test.
    (iter1 / "human").mkdir(parents=True, exist_ok=True)
    (iter1 / "human" / "human_labels.json").write_text("{}", encoding="utf-8")

    # Scaffold iter_002 from iter_001's evolution_state.
    out = create_next_iteration(ws.root, rerun_stage1=False)
    assert out["status"] == "ok"
    assert out["iteration"] == 2

    # The new iter's iteration.json must point at the prior
    # evolution_state via a workspace-relative path.
    iter2_path = Path(out["iter_path"])
    iter2_json = json.loads((iter2_path / "iteration.json").read_text())
    pf = iter2_json["parent_feedback"]
    assert pf["evolution_state"] == "iter_001/react/evolution_state.json"

    # Resolve the pointer the same way the Stage 4 skill does (the prompt
    # spells this out: workspace-relative → join with ws.root).
    es_abs = ws.root / pf["evolution_state"]
    assert es_abs.is_file()
    # Sanity-check: the resolved file actually carries the must-include.
    loaded = json.loads(es_abs.read_text())
    assert loaded["must_include_configs"] == ["vlm-qwen35b-proven"]

    # --- Case A: brief includes the must-include → gate passes ---
    good_brief = _brief_with_combination_ids(
        ["metric-baseline", "vlm-qwen35b-proven", "vlm-gemma31b-new"]
    )
    rc, payload = _run_cli(
        "check-experiment-brief",
        {"brief": good_brief, "evolution_state_path": str(es_abs)},
        capsys,
    )
    assert rc == 0
    assert payload["valid"] is True
    assert payload["must_include_configs_checked"] is True
    assert payload["missing_must_include"] == []

    # --- Case B: brief drops the must-include → gate rejects ---
    bad_brief = _brief_with_combination_ids(
        ["metric-baseline", "vlm-different-a", "vlm-different-b"]
    )
    rc, payload = _run_cli(
        "check-experiment-brief",
        {"brief": bad_brief, "evolution_state_path": str(es_abs)},
        capsys,
    )
    assert rc == 0
    assert payload["valid"] is False
    # The structured field carries ONLY the dropped IDs — Stage 4's
    # revision brief reads it directly without parsing prose.
    assert payload["missing_must_include"] == ["vlm-qwen35b-proven"]
    # The prose problem still appears for human-readable logs.
    assert any(
        "vlm-qwen35b-proven" in p and "C-2.5 elitism" in p
        for p in payload["problems"]
    )


# ---- check_evolution_state ---------------------------------------------

def test_check_evolution_state_accepts_good_payload():
    problems = check_evolution_state(_good_state(1, "ADVANCE"))
    assert problems == []


def test_check_evolution_state_rejects_unknown_decision():
    state = _good_state(1, "ADVANCE")
    state["decision"] = "MAYBE"
    problems = check_evolution_state(state)
    assert any("decision must be one of" in p for p in problems)


def test_check_evolution_state_requires_lit_flag_for_rerun():
    state = _good_state(1, "REVISE_RERUN_STAGE1")
    state["literature_update_requested"] = False
    problems = check_evolution_state(state)
    assert any("literature_update_requested=true" in p for p in problems)


def test_check_evolution_state_rejects_advance_with_lit_request():
    state = _good_state(1, "ADVANCE")
    state["literature_update_requested"] = True
    problems = check_evolution_state(state)
    assert any("literature_update_requested=false" in p for p in problems)


def test_check_evolution_state_rejects_bad_proposal_type():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["evolution_proposals"] = [
        {"id": "evo_1", "type": "nuke_it", "for_combination": "x",
         "proposal": "..."},
    ]
    problems = check_evolution_state(state)
    assert any("type must be one of" in p for p in problems)


def test_check_evolution_state_rejects_duplicate_proposal_ids():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["evolution_proposals"] = [
        {"id": "evo_1", "type": "model_upsize", "for_combination": "x",
         "proposal": "a"},
        {"id": "evo_1", "type": "prompt_rewrite", "for_combination": "y",
         "proposal": "b"},
    ]
    problems = check_evolution_state(state)
    assert any("not unique" in p for p in problems)


def test_check_evolution_state_rejects_missing_exclude_fields():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["exclude_list"] = [{"scope": "judge_size"}]  # missing match, reason
    problems = check_evolution_state(state)
    assert sum("exclude_list[0] is missing" in p for p in problems) == 2


def test_check_evolution_state_rejects_non_dict():
    assert check_evolution_state("nope") == [
        "evolution_state must be a JSON object"
    ]


# ---- Phase C-2.5 EA primitives -----------------------------------------

def test_check_evolution_state_accepts_must_include_configs():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["must_include_configs"] = ["vlm-qwen35b", "vlm-pairwise"]
    # When must_include is set, hall_of_fame must contain those cids
    state["hall_of_fame"] = [
        {"combination_id": "vlm-qwen35b", "fitness_composite": 0.75,
         "pass_rate": 0.8, "recall_rate": 0.7,
         "human_endorsement_rate": 0.72,
         "iter_introduced": 1, "last_iter_seen": 1},
        {"combination_id": "vlm-pairwise", "fitness_composite": 0.7,
         "pass_rate": 0.75, "recall_rate": 0.65,
         "human_endorsement_rate": 0.7,
         "iter_introduced": 1, "last_iter_seen": 1},
    ]
    assert check_evolution_state(state) == []


def test_check_evolution_state_rejects_must_include_not_in_hof():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["must_include_configs"] = ["vlm-stray"]
    state["hall_of_fame"] = [
        {"combination_id": "vlm-other", "fitness_composite": 0.7,
         "pass_rate": 0.7, "recall_rate": 0.7,
         "human_endorsement_rate": 0.7,
         "iter_introduced": 1, "last_iter_seen": 1},
    ]
    problems = check_evolution_state(state)
    assert any(
        "must_include_configs" in p and "not in hall_of_fame" in p
        for p in problems
    )


def test_check_evolution_state_rejects_duplicate_must_include():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["must_include_configs"] = ["dup", "dup"]
    problems = check_evolution_state(state)
    assert any("duplicated" in p for p in problems)


def test_check_evolution_state_rejects_bad_must_include_type():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["must_include_configs"] = "not a list"
    problems = check_evolution_state(state)
    assert any(
        "must_include_configs must be a list" in p for p in problems
    )


def test_check_evolution_state_accepts_lessons_learned():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["lessons_learned"] = {
        "success_patterns": [
            {"pattern": "≥30B VLM + region scope clears C-2",
             "evidence": ["iter1:vlm-qwen35b-region"],
             "confidence": "low", "since_iter": 1},
        ],
        "failure_patterns": [
            {"pattern": "pointwise judges miss geometric distortion",
             "evidence": ["iter1:vlm-pointwise"],
             "confidence": "low",
             "remedy_proposed": "pairwise with source visible",
             "since_iter": 1},
        ],
        "open_questions": [
            "Does hybrid composition help when both children fail?",
        ],
    }
    assert check_evolution_state(state) == []


def test_check_evolution_state_rejects_bad_confidence():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["lessons_learned"] = {
        "success_patterns": [
            {"pattern": "x", "confidence": "extremely-high", "since_iter": 1},
        ],
    }
    problems = check_evolution_state(state)
    assert any("confidence must be one of" in p for p in problems)


def test_check_evolution_state_rejects_bad_lesson_shape():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["lessons_learned"] = {
        "success_patterns": [{"pattern": ""}],  # empty pattern
    }
    problems = check_evolution_state(state)
    assert any("pattern must be a non-empty string" in p for p in problems)


def test_check_evolution_state_rejects_open_questions_non_list():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["lessons_learned"] = {"open_questions": "should be a list"}
    problems = check_evolution_state(state)
    assert any("open_questions must be a list" in p for p in problems)


def test_check_evolution_state_accepts_hall_of_fame():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["hall_of_fame"] = [
        {"combination_id": "vlm-a", "fitness_composite": 0.85,
         "pass_rate": 0.9, "recall_rate": 0.8,
         "human_endorsement_rate": 0.85,
         "iter_introduced": 1, "last_iter_seen": 2,
         "lineage": ["iter1:vlm-a"]},
        {"combination_id": "vlm-b", "fitness_composite": 0.72,
         "pass_rate": 0.75, "recall_rate": 0.7,
         "human_endorsement_rate": 0.71,
         "iter_introduced": 2, "last_iter_seen": 2},
    ]
    assert check_evolution_state(state) == []


def test_check_evolution_state_rejects_bad_hof_fitness():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["hall_of_fame"] = [
        {"combination_id": "x", "fitness_composite": 1.5,
         "pass_rate": 0.5, "recall_rate": 0.5,
         "human_endorsement_rate": 0.5,
         "iter_introduced": 1, "last_iter_seen": 1},
    ]
    problems = check_evolution_state(state)
    assert any("fitness_composite" in p for p in problems)


def test_check_evolution_state_rejects_duplicate_hof_cid():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["hall_of_fame"] = [
        {"combination_id": "dup", "fitness_composite": 0.7,
         "pass_rate": 0.7, "recall_rate": 0.7,
         "human_endorsement_rate": 0.7,
         "iter_introduced": 1, "last_iter_seen": 1},
        {"combination_id": "dup", "fitness_composite": 0.6,
         "pass_rate": 0.6, "recall_rate": 0.6,
         "human_endorsement_rate": 0.6,
         "iter_introduced": 1, "last_iter_seen": 1},
    ]
    problems = check_evolution_state(state)
    assert any("duplicated" in p for p in problems)


def test_check_evolution_state_rejects_bad_hof_lineage_type():
    state = _good_state(1, "REVISE_SKIP_STAGE1")
    state["hall_of_fame"] = [
        {"combination_id": "x", "fitness_composite": 0.7,
         "pass_rate": 0.7, "recall_rate": 0.7,
         "human_endorsement_rate": 0.7,
         "iter_introduced": 1, "last_iter_seen": 1,
         "lineage": [42, "wrong"]},
    ]
    problems = check_evolution_state(state)
    assert any("lineage must be a list of strings" in p for p in problems)


# ---- CLI surface --------------------------------------------------------

def test_cli_react_aggregate_returns_cumulative_block(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path, iteration=1)
    _write_iter_feedback(ws, 1, general_feedback="seems fine")
    rc, payload = _run_cli(
        "react-aggregate", {"workspace_path": str(ws.root)}, capsys)
    assert rc == 0
    assert payload["status"] == "ok"
    assert payload["cumulative_feedback"]["iters_analyzed"] == [1]


def test_cli_react_tick_returns_decision(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path, iteration=1)
    rc, payload = _run_cli("react-tick", {
        "workspace_path": str(ws.root), "verdict": "ADVANCE",
        "rationale": "shipping the 72B baseline",
    }, capsys)
    assert rc == 0
    assert payload["decision"] == "ADVANCE"


def test_cli_create_next_iteration(tmp_path: Path, capsys):
    ws = _make_workspace(tmp_path, iteration=1, max_iterations=3)
    rc, payload = _run_cli("create-next-iteration", {
        "workspace_path": str(ws.root), "rerun_stage1": True,
    }, capsys)
    assert rc == 0
    # rerun_stage1=True → stage_index=0 (task_init last-completed) so the
    # next ralph pass dispatches Stage 1 (research).
    assert payload["stage_index"] == 0
    assert payload["iteration"] == 2


def test_cli_check_evolution_state_inline(tmp_path: Path, capsys):
    state = _good_state(1, "REVISE_RERUN_STAGE1")
    rc, payload = _run_cli(
        "check-evolution-state", {"state": state}, capsys)
    assert rc == 0
    assert payload["valid"] is True
    assert payload["problems"] == []


def test_cli_check_evolution_state_path(tmp_path: Path, capsys):
    state = _good_state(1, "ADVANCE")
    path = tmp_path / "evolution_state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    rc, payload = _run_cli(
        "check-evolution-state", {"state_path": str(path)}, capsys)
    assert rc == 0
    assert payload["valid"] is True


def test_cli_check_evolution_state_missing_params(capsys):
    rc, payload = _run_cli("check-evolution-state", {}, capsys)
    assert rc == 1
    assert payload["error"] == "missing_params"


# ---- module-level constants are surfaced --------------------------------

def test_constants_exposed():
    assert "ADVANCE" in VERDICTS
    assert "REVISE_SKIP_STAGE1" in VERDICTS
    assert "REVISE_RERUN_STAGE1" in VERDICTS
    assert "prompt_rewrite" in PROPOSAL_TYPES
    assert "model_upsize" in PROPOSAL_TYPES
