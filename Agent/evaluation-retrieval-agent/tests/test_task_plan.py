"""Tests for Stage 5 task-plan validation (era/orchestration/task_plan.py)."""

from __future__ import annotations

import io
import json
from pathlib import Path

from era.cli import main
from era.orchestration.task_plan import validate_task_plan


def _brief() -> dict:
    """The Stage 4 brief the plan expands: 1 metric baseline + 2 VLM scales."""
    return {
        "candidate_configs": [
            {"combination_id": "metric-baseline", "family": "B"},
            {"combination_id": "vlm-32b", "family": "A"},
            {"combination_id": "vlm-72b", "family": "A"},
        ],
    }


def _serve(tid: str, judge: str, mode: str, depends_on: list[str],
           serves: list[str]) -> dict:
    return {
        "id": tid, "type": "serve", "family": "A", "mode": mode,
        "depends_on": depends_on, "gpu_count": 4, "estimated_minutes": 12,
        "judge": judge, "serves": serves, "teardown_after": serves,
        "expected_output": f"experiments/logs/{tid}.done.json",
        "serve": {"model_path": f"/mnt/model/{judge}",
                  "served_model_name": judge, "tensor_parallel": 4,
                  "gpu_memory_utilization": 0.9},
    }


def _eval(tid: str, cid: str, family: str, mode: str,
          depends_on: list[str], hyp: str, **extra) -> dict:
    task = {
        "id": tid, "type": "eval", "family": family, "mode": mode,
        "depends_on": depends_on, "estimated_minutes": 8,
        "combination_id": cid, "hypothesis_id": hyp, "inputs": ["output"],
        "expected_output": f"experiments/results/{mode}/{cid}/scores.jsonl",
        "pilot": {"samples": 30, "seed": 42, "timeout": 900,
                  "pass_criteria": "a tau gap is visible"},
    }
    task.update(extra)
    return task


def _agg(tid: str, cid: str, mode: str, dep: str) -> dict:
    return {
        "id": tid, "type": "aggregate", "mode": mode, "depends_on": [dep],
        "gpu_count": 0, "estimated_minutes": 1, "combination_id": cid,
        "expected_output": f"experiments/results/{mode}/{cid}/aggregate.json",
    }


def _metric_eval(tid: str, mode: str, depends_on: list[str]) -> dict:
    return _eval(tid, "metric-baseline", "B", mode, depends_on, "H0",
                 gpu_count=1,
                 eval={"judge": None, "metric_subfamily": "clip-dino",
                       "prompt": None, "scope": "whole"})


def _judge_eval(tid: str, cid: str, judge: str, mode: str,
                serve_id: str, hyp: str) -> dict:
    return _eval(tid, cid, "A", mode, [serve_id], hyp, gpu_count=0,
                 judge_task_id=serve_id,
                 eval={"judge": judge, "metric_subfamily": None,
                       "prompt": "pairwise-rubric", "scope": "whole"})


def _valid_plan() -> dict:
    """A well-formed DAG covering the _brief() configs, pilot + full."""
    tasks = [
        # ---- pilot pass ----
        _serve("serve-32b-pilot", "qwen2.5-vl-32b", "pilot", [],
               ["eval-vlm32-pilot"]),
        _serve("serve-72b-pilot", "qwen2.5-vl-72b", "pilot",
               ["serve-32b-pilot"], ["eval-vlm72-pilot"]),
        _metric_eval("eval-mb-pilot", "pilot", []),
        _judge_eval("eval-vlm32-pilot", "vlm-32b", "qwen2.5-vl-32b", "pilot",
                    "serve-32b-pilot", "H1"),
        _judge_eval("eval-vlm72-pilot", "vlm-72b", "qwen2.5-vl-72b", "pilot",
                    "serve-72b-pilot", "H2"),
        _agg("agg-mb-pilot", "metric-baseline", "pilot", "eval-mb-pilot"),
        _agg("agg-vlm32-pilot", "vlm-32b", "pilot", "eval-vlm32-pilot"),
        _agg("agg-vlm72-pilot", "vlm-72b", "pilot", "eval-vlm72-pilot"),
        {"id": "gate-pilot", "type": "compare", "mode": "pilot", "gate": True,
         "depends_on": ["agg-mb-pilot", "agg-vlm32-pilot", "agg-vlm72-pilot"],
         "gpu_count": 0, "estimated_minutes": 2,
         "pilot": {"pass_criteria": "proceed if any tau gap is visible"},
         "expected_output": "experiments/pilot_decision.json"},
        # ---- full pass ----
        _serve("serve-32b-full", "qwen2.5-vl-32b", "full", ["gate-pilot"],
               ["eval-vlm32-full"]),
        _serve("serve-72b-full", "qwen2.5-vl-72b", "full",
               ["serve-32b-full"], ["eval-vlm72-full"]),
        _metric_eval("eval-mb-full", "full", ["gate-pilot"]),
        _judge_eval("eval-vlm32-full", "vlm-32b", "qwen2.5-vl-32b", "full",
                    "serve-32b-full", "H1"),
        _judge_eval("eval-vlm72-full", "vlm-72b", "qwen2.5-vl-72b", "full",
                    "serve-72b-full", "H2"),
        _agg("agg-mb-full", "metric-baseline", "full", "eval-mb-full"),
        _agg("agg-vlm32-full", "vlm-32b", "full", "eval-vlm32-full"),
        _agg("agg-vlm72-full", "vlm-72b", "full", "eval-vlm72-full"),
        {"id": "compare-final", "type": "compare", "mode": "full",
         "gate": False,
         "depends_on": ["agg-mb-full", "agg-vlm32-full", "agg-vlm72-full"],
         "gpu_count": 0, "estimated_minutes": 2,
         "expected_output": "experiments/results/summary.json"},
    ]
    return {
        "schema_version": 1, "iteration": 1, "debate_round": 1,
        "evaluation_goal": "retrieve the cheapest human-aligned evaluator",
        "gpu_pool": 4, "tasks": tasks,
    }


# ---- happy path ---------------------------------------------------------

def test_valid_plan_passes():
    assert validate_task_plan(_valid_plan(), _brief()) == []


# ---- structural rejections ---------------------------------------------

def test_rejects_non_dict_plan():
    assert validate_task_plan([], _brief()) != []


def test_rejects_empty_tasks():
    plan = _valid_plan()
    plan["tasks"] = []
    assert any("tasks must be a non-empty list" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_missing_top_field():
    plan = _valid_plan()
    del plan["gpu_pool"]
    assert any("gpu_pool" in p for p in validate_task_plan(plan, _brief()))


def test_rejects_missing_required_task_field():
    plan = _valid_plan()
    del plan["tasks"][2]["estimated_minutes"]
    assert any("estimated_minutes" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_duplicate_task_id():
    plan = _valid_plan()
    plan["tasks"][3]["id"] = plan["tasks"][2]["id"]
    assert any("duplicate task id" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_bad_task_type():
    plan = _valid_plan()
    plan["tasks"][2]["type"] = "train"
    assert any("is not one of" in p for p in validate_task_plan(plan, _brief()))


def test_rejects_gpu_count_over_pool():
    plan = _valid_plan()
    plan["tasks"][2]["gpu_count"] = 99
    assert any("exceeds gpu_pool" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_serve_not_full_pool():
    plan = _valid_plan()
    plan["tasks"][0]["gpu_count"] = 2  # serve must own the whole pool
    assert any("Rule 6" in p and "whole pool" in p
               for p in validate_task_plan(plan, _brief()))


# ---- parallel_packed serve packing (Phase D-6) --------------------------

def _parallel_plan() -> dict:
    """A parallel_packed-valid plan: right-sized (tp 2) independent judges."""
    plan = _valid_plan()
    serve_ids = {t["id"] for t in plan["tasks"] if t["type"] == "serve"}
    for t in plan["tasks"]:
        if t["type"] != "serve":
            continue
        t["gpu_count"] = 2                       # right-sized to tp, < gpu_pool
        t["serve"]["tensor_parallel"] = 2
        # Drop the Rule-6 serve chain so judges are independent / co-resident,
        # but keep full-mode serves gated behind the pilot compare.
        t["depends_on"] = [d for d in t["depends_on"] if d not in serve_ids]
        if t["mode"] == "full" and "gate-pilot" not in t["depends_on"]:
            t["depends_on"].append("gate-pilot")
    return plan


def test_parallel_plan_passes():
    assert validate_task_plan(
        _parallel_plan(), _brief(), "parallel_packed") == []


def test_serial_chain_plan_rejected_under_parallel():
    """The serial _valid_plan() chains its judges, which is invalid under
    parallel_packed (judges must be independent to co-reside)."""
    problems = validate_task_plan(_valid_plan(), _brief(), "parallel_packed")
    assert any("no serve chain" in p for p in problems)


def test_parallel_rejects_gpu_count_ne_tensor_parallel():
    plan = _parallel_plan()
    # find a serve task and break the right-sizing invariant
    serve = next(t for t in plan["tasks"] if t["type"] == "serve")
    serve["gpu_count"] = 3            # != tensor_parallel (2)
    assert any("must equal" in p and "tensor_parallel" in p
               for p in validate_task_plan(plan, _brief(), "parallel_packed"))


def test_parallel_rejects_serve_chain():
    plan = _parallel_plan()
    pilot_serves = [t for t in plan["tasks"]
                    if t["type"] == "serve" and t["mode"] == "pilot"]
    # re-introduce a chain link between the two pilot judges
    pilot_serves[1]["depends_on"].append(pilot_serves[0]["id"])
    assert any("no serve chain" in p
               for p in validate_task_plan(plan, _brief(), "parallel_packed"))


def test_parallel_rejects_gpu_count_over_pool():
    plan = _parallel_plan()
    serve = next(t for t in plan["tasks"] if t["type"] == "serve")
    serve["gpu_count"] = 8           # > gpu_pool (4)
    serve["serve"]["tensor_parallel"] = 8
    problems = validate_task_plan(plan, _brief(), "parallel_packed")
    assert any("exceeds gpu_pool" in p or "must be in" in p for p in problems)


# ---- dependency graph ---------------------------------------------------

def test_rejects_unresolved_dependency():
    plan = _valid_plan()
    plan["tasks"][2]["depends_on"] = ["ghost-task"]
    assert any("unknown task" in p for p in validate_task_plan(plan, _brief()))


def test_rejects_self_dependency():
    plan = _valid_plan()
    plan["tasks"][2]["depends_on"] = [plan["tasks"][2]["id"]]
    assert any("depends on itself" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_dependency_cycle():
    plan = _valid_plan()
    plan["tasks"][0]["depends_on"] = ["serve-72b-pilot"]  # 32b <-> 72b cycle
    assert any("cycle" in p for p in validate_task_plan(plan, _brief()))


# ---- per-type field contracts ------------------------------------------

def test_rejects_eval_bad_mode():
    plan = _valid_plan()
    plan["tasks"][2]["mode"] = "smoke"
    assert any("mode" in p for p in validate_task_plan(plan, _brief()))


def test_rejects_eval_missing_pilot_spec():
    plan = _valid_plan()
    del plan["tasks"][2]["pilot"]["pass_criteria"]
    assert any("pilot.pass_criteria" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_family_a_eval_with_gpus():
    plan = _valid_plan()
    plan["tasks"][3]["gpu_count"] = 2  # a Family-A eval consumes an endpoint
    assert any("gpu_count 0" in p for p in validate_task_plan(plan, _brief()))


def test_rejects_family_a_eval_without_serve():
    plan = _valid_plan()
    del plan["tasks"][3]["judge_task_id"]
    assert any("judge_task_id" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_family_a_serve_judge_mismatch():
    plan = _valid_plan()
    plan["tasks"][0]["judge"] = "some-other-judge"
    assert any("judge" in p for p in validate_task_plan(plan, _brief()))


def test_rejects_hybrid_judge_without_serve():
    """A hybrid eval that names a judge still needs a serve task (not just A)."""
    plan = _valid_plan()
    for t in plan["tasks"]:
        if t["id"] == "eval-mb-pilot":
            t["family"] = "hybrid"
            t["eval"] = {"judge": "qwen2.5-vl-72b",
                         "metric_subfamily": "clip", "prompt": "rubric",
                         "scope": "whole"}
    assert any("judge_task_id" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_bad_task_id_charset():
    plan = _valid_plan()
    plan["tasks"][2]["id"] = "eval mb pilot!"   # spaces + bang — shell-unsafe
    assert any("A-Za-z0-9" in p for p in validate_task_plan(plan, _brief()))


def test_rejects_serve_missing_spec():
    plan = _valid_plan()
    del plan["tasks"][0]["serve"]["model_path"]
    assert any("model_path" in p for p in validate_task_plan(plan, _brief()))


def test_rejects_eval_noncanonical_expected_output():
    """An eval whose scores file is not at the canonical mode-scoped path is
    rejected — that is exactly what produced a hollow summary in v0.1.4."""
    plan = _valid_plan()
    for t in plan["tasks"]:
        if t["id"] == "eval-mb-pilot":
            # the v0.1.4 bug shape: no mode segment, wrong extension
            t["expected_output"] = "experiments/results/metric-baseline/scores.json"
    problems = validate_task_plan(plan, _brief())
    assert any("expected_output" in p and "canonical" in p for p in problems)


# ---- Rule 6 serial chain -----------------------------------------------

def test_rejects_parallel_family_a_judges():
    plan = _valid_plan()
    plan["tasks"][1]["depends_on"] = []  # both pilot serves are now roots
    assert any("Rule 6" in p and "chain" in p
               for p in validate_task_plan(plan, _brief()))


# ---- coverage ----------------------------------------------------------

def test_rejects_uncovered_candidate_config():
    plan = _valid_plan()
    brief = _brief()
    brief["candidate_configs"].append({"combination_id": "vlm-235b",
                                       "family": "A"})
    assert any("vlm-235b" in p for p in validate_task_plan(plan, brief))


def test_rejects_config_missing_full_eval_task():
    plan = _valid_plan()
    plan["tasks"] = [t for t in plan["tasks"] if t["id"] != "eval-mb-full"]
    assert any("metric-baseline" in p and "full" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_eval_unknown_combination_id():
    plan = _valid_plan()
    plan["tasks"][2]["combination_id"] = "not-in-brief"
    assert any("not in the brief" in p
               for p in validate_task_plan(plan, _brief()))


# ---- pilot gate --------------------------------------------------------

def test_rejects_missing_gate_compare():
    plan = _valid_plan()
    for t in plan["tasks"]:
        if t["id"] == "gate-pilot":
            t["gate"] = False
    assert any("gating compare" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_full_eval_not_gated():
    plan = _valid_plan()
    for t in plan["tasks"]:
        if t["id"] == "eval-mb-full":
            t["depends_on"] = []  # no longer behind the pilot gate
    assert any("gated behind" in p
               for p in validate_task_plan(plan, _brief()))


def test_rejects_aggregate_missing_eval_dep():
    plan = _valid_plan()
    for t in plan["tasks"]:
        if t["id"] == "agg-mb-pilot":
            t["depends_on"] = ["gate-pilot"]  # wrong dependency
    assert any("aggregate" in p and "metric-baseline" in p
               for p in validate_task_plan(plan, _brief()))


# ---- CLI ----------------------------------------------------------------

def test_cli_check_task_plan_inline(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"plan": _valid_plan(), "brief": _brief()})))
    code = main(["check-task-plan"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is True
    assert result["problems"] == []


def test_cli_check_task_plan_from_file(tmp_path: Path, monkeypatch, capsys):
    plan = _valid_plan()
    plan["tasks"][0]["gpu_count"] = 1  # invalid serve gpu_count
    plan_path = tmp_path / "task_plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    brief_path = tmp_path / "experiment_brief.json"
    brief_path.write_text(json.dumps(_brief()), encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"plan_path": str(plan_path), "brief_path": str(brief_path)})))
    code = main(["check-task-plan"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is False
    assert result["problems"]


def test_cli_check_task_plan_missing_params(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"plan": _valid_plan()})))  # no brief
    code = main(["check-task-plan"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "missing_params"


def test_cli_check_task_plan_no_such_file(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"plan_path": "/nonexistent/task_plan.json", "brief": _brief()})))
    code = main(["check-task-plan"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "no_such_file"


def test_cli_check_task_plan_reads_mode_from_workspace(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys,
):
    """With workspace_path, the serve-packing rule comes from the workspace's
    experiment.family_a_execution (default parallel_packed) — so the serial
    chained _valid_plan() is reported invalid."""
    from conftest import valid_params
    from era.config import ERAConfig
    from era.workspace import Workspace

    params = valid_params(tmp_path, per_sample_root)
    cfg = ERAConfig.from_params(params)  # default family_a_execution=parallel_packed
    ws = Workspace(tmp_path, params["project_name"])
    ws.scaffold()
    ws.write_file("config.yaml", cfg.to_commented_yaml())

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"plan": _valid_plan(), "brief": _brief(),
         "workspace_path": str(ws.root)})))
    code = main(["check-task-plan"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["valid"] is False          # serial chain invalid under parallel
    assert any("parallel_packed" in p for p in result["problems"])

    # the parallel-shaped plan validates against the same workspace
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"plan": _parallel_plan(), "brief": _brief(),
         "workspace_path": str(ws.root)})))
    assert main(["check-task-plan"]) == 0
    assert json.loads(capsys.readouterr().out)["valid"] is True


# ---- Phase C-2: annotated mode + samples_subset ------------------------

def test_eval_modes_includes_annotated():
    """Sanity: the validator now accepts annotated as a valid mode."""
    from era.orchestration.task_plan import EVAL_MODES, REQUIRED_EVAL_MODES
    assert "annotated" in EVAL_MODES
    # The brief-coverage check still only requires pilot + full; annotated
    # is additive (omitted when there are too few annotations).
    assert set(REQUIRED_EVAL_MODES) == {"pilot", "full"}


def test_accepts_annotated_eval_with_samples_subset():
    """An annotated-mode eval task with a valid samples_subset is accepted."""
    plan = _valid_plan()
    annotated = _judge_eval(
        "eval-vlm32-annot", "vlm-32b", "qwen2.5-vl-32b", "annotated",
        "serve-32b-pilot",   # reuse pilot serve (any serve in chain works)
        "H1",
    )
    annotated["samples_subset"] = ["dress/s1", "dress/s2", "dress/s3"]
    plan["tasks"].append(annotated)
    problems = validate_task_plan(plan, _brief())
    # samples_subset itself should not be a problem.
    assert not any("samples_subset" in p for p in problems)


def test_rejects_annotated_eval_without_samples_subset():
    plan = _valid_plan()
    annotated = _judge_eval(
        "eval-vlm32-annot", "vlm-32b", "qwen2.5-vl-32b", "annotated",
        "serve-32b-pilot", "H1",
    )
    # No samples_subset
    plan["tasks"].append(annotated)
    problems = validate_task_plan(plan, _brief())
    assert any("samples_subset" in p for p in problems)


def test_accepts_full_eval_with_samples_subset():
    """Phase C-2.3: a full-mode eval with samples_subset is accepted —
    Stage 5's planner stamps a deterministically-random N-sample list
    via era.cli sample-window so all methods score the same shuffled
    subset. Was rejected pre-C-2.3."""
    plan = _valid_plan()
    full_task = next(t for t in plan["tasks"] if t["id"] == "eval-vlm32-full")
    full_task["samples_subset"] = ["s1", "s2"]
    problems = validate_task_plan(plan, _brief())
    assert not any("samples_subset" in p for p in problems), (
        f"unexpected samples_subset complaint: {problems}"
    )


def test_rejects_full_eval_with_malformed_samples_subset():
    """Well-formedness is still enforced: empty list or non-string entries
    fail even on full-mode."""
    plan = _valid_plan()
    full_task = next(t for t in plan["tasks"] if t["id"] == "eval-vlm32-full")
    # Empty list
    full_task["samples_subset"] = []
    problems = validate_task_plan(plan, _brief())
    assert any("samples_subset" in p and "non-empty list" in p
               for p in problems)
    # Non-string entries
    full_task["samples_subset"] = ["s1", 123]
    problems = validate_task_plan(plan, _brief())
    assert any("samples_subset" in p and "non-empty strings" in p
               for p in problems)


def test_accepts_brief_coverage_without_annotated_tasks():
    """Brief coverage only requires pilot + full — annotated is additive."""
    # The default _valid_plan() has no annotated tasks; it should still pass.
    assert validate_task_plan(_valid_plan(), _brief()) == []


def test_accepts_three_mode_plan_with_annotated_chain():
    """A plan with pilot + annotated + full eval tasks (each chained
    behind its mode's serve) passes validation. This is the shape Stage 5
    emits when annotations ≥ auto_validate_min_samples."""
    plan = _valid_plan()
    # Add annotated-mode serve + eval + aggregate for vlm-32b only (keep
    # the test focused on the new mode; vlm-72b stays pilot+full).
    annotated_serve = _serve(
        "serve-32b-annot", "qwen2.5-vl-32b", "annotated",
        ["serve-72b-pilot"],   # chain after pilot serves to honor Rule 6
        ["eval-vlm32-annot"],
    )
    annotated_eval = _judge_eval(
        "eval-vlm32-annot", "vlm-32b", "qwen2.5-vl-32b", "annotated",
        "serve-32b-annot", "H1",
    )
    annotated_eval["samples_subset"] = ["dress/s1", "dress/s2", "dress/s3"]
    annotated_agg = _agg(
        "agg-vlm32-annot", "vlm-32b", "annotated", "eval-vlm32-annot",
    )
    plan["tasks"].extend([annotated_serve, annotated_eval, annotated_agg])
    problems = validate_task_plan(plan, _brief())
    assert problems == [], f"unexpected problems: {problems}"


def test_stage5_prompt_advertises_phase_c21_annotated_mode():
    """Phase C-2.1 regression guard: Stage 5 prompt instructs the planner
    to call list-annotations, emit annotated-mode tasks, and stamp
    samples_subset. If any of these disappear from the prompt, the
    Phase C-2 gate becomes a no-op."""
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs" / "prompts" / "stage5_experiment_plan.md").read_text(
        encoding="utf-8")
    assert "list-annotations" in text
    assert "samples_subset" in text
    assert "annotated" in text
    assert "auto_validate_min_samples" in text


# ---- Phase C-2.2: pilot-block + judge-name normalization ----------------

def test_accepts_annotated_eval_without_pilot_block():
    """Phase C-2.2 fix #1: an annotated-mode eval task with NO pilot block
    must validate cleanly. The pilot block is pilot-mode-only metadata."""
    plan = _valid_plan()
    annotated = _judge_eval(
        "eval-vlm32-annot", "vlm-32b", "qwen2.5-vl-32b", "annotated",
        "serve-32b-pilot", "H1",
    )
    annotated["samples_subset"] = ["dress/s1", "dress/s2", "dress/s3"]
    # Remove the pilot block that _judge_eval injected by default.
    annotated.pop("pilot", None)
    plan["tasks"].append(annotated)
    problems = validate_task_plan(plan, _brief())
    assert not any("pilot" in p for p in problems), (
        f"unexpected pilot-block complaint: {problems}"
    )


def test_accepts_full_eval_without_pilot_block():
    """A full-mode eval task should also validate without a pilot block."""
    plan = _valid_plan()
    for t in plan["tasks"]:
        if t.get("type") == "eval" and t.get("mode") == "full":
            t.pop("pilot", None)
    problems = validate_task_plan(plan, _brief())
    assert not any("pilot.samples" in p or "pilot.seed" in p
                   or "pilot.timeout" in p or "pilot.pass_criteria" in p
                   for p in problems), f"unexpected pilot complaint: {problems}"


def test_pilot_eval_still_requires_pilot_block():
    """A pilot-mode eval task without a pilot block must STILL fail —
    the smoke pass needs samples / seed / timeout / pass_criteria."""
    plan = _valid_plan()
    for t in plan["tasks"]:
        if t.get("type") == "eval" and t.get("mode") == "pilot":
            t["pilot"] = None
            break
    problems = validate_task_plan(plan, _brief())
    assert any("pilot" in p for p in problems)


def test_judge_name_normalization_accepts_pointwise_suffix():
    """Phase C-2.2 fix #2: serve.judge='vlm-7b' + eval.judge='vlm-7b-pointwise'
    must validate — modality suffixes are decorative, not part of the
    canonical model name."""
    plan = _valid_plan()
    for t in plan["tasks"]:
        if t.get("id") == "eval-vlm32-pilot":
            t["eval"]["judge"] = "qwen2.5-vl-32b-pointwise"
            break
    problems = validate_task_plan(plan, _brief())
    assert not any("canonical names differ" in p for p in problems), (
        f"normalization did not strip suffix: {problems}"
    )


def test_judge_name_normalization_rejects_real_model_mismatch():
    """A genuine model mismatch (different size) must still be caught even
    after suffix stripping — guards against over-aggressive normalization."""
    plan = _valid_plan()
    for t in plan["tasks"]:
        if t.get("id") == "eval-vlm32-pilot":
            t["eval"]["judge"] = "qwen2.5-vl-72b"  # different SIZE
            break
    problems = validate_task_plan(plan, _brief())
    assert any("canonical names differ" in p for p in problems)


def test_judge_name_normalization_strips_decorations():
    """Direct unit test of _normalize_judge over the documented suffix set."""
    from era.orchestration.task_plan import _normalize_judge
    assert _normalize_judge("vlm-7b-pointwise") == "vlm-7b"
    assert _normalize_judge("vlm-7b-pairwise") == "vlm-7b"
    assert _normalize_judge("vlm-7b-flag") == "vlm-7b"
    assert _normalize_judge("vlm-7b-judge") == "vlm-7b"
    assert _normalize_judge("vlm-7b-rubric") == "vlm-7b"
    assert _normalize_judge("vlm-7b-v2") == "vlm-7b"
    assert _normalize_judge("vlm-7b-pointwise-v3") == "vlm-7b"
    # Substrings INSIDE the model name are preserved.
    assert _normalize_judge("qwen2.5-vl-7b") == "qwen2.5-vl-7b"
    # None / empty handled.
    assert _normalize_judge(None) is None
    assert _normalize_judge("") is None
