"""Tests for Stage 6 experiment state, recovery, and results aggregation."""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from pathlib import Path

from conftest import valid_params

from era.cli import main
from era.config import ERAConfig
from era.orchestration import experiment_results as er
from era.orchestration import experiment_state as es
from era.workspace import Workspace

_PLAN = {
    "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
    "gpu_pool": 4, "tasks": [
        {"id": "setup-x", "type": "setup", "mode": None, "depends_on": []},
        {"id": "eval-x", "type": "eval", "family": "B", "mode": "pilot",
         "combination_id": "cfg-x", "depends_on": ["setup-x"]},
        {"id": "eval-y", "type": "eval", "family": "B", "mode": "full",
         "combination_id": "cfg-y", "depends_on": []},
    ],
}


# ---- init_state ---------------------------------------------------------

def test_init_state_seeds_pilot_and_setup(tmp_path: Path):
    es.init_state(tmp_path, _PLAN, "pilot")
    state = es.load_state(tmp_path)
    assert set(state.tasks) == {"setup-x", "eval-x"}  # full task excluded
    assert all(t["status"] == "pending" for t in state.tasks.values())
    assert state.mode == "pilot"


def test_init_state_preserves_completed(tmp_path: Path):
    es.init_state(tmp_path, _PLAN, "pilot")
    es.complete_task(tmp_path, "setup-x")
    es.init_state(tmp_path, _PLAN, "full")  # second pass
    state = es.load_state(tmp_path)
    assert state.tasks["setup-x"]["status"] == "completed"  # preserved
    assert state.tasks["eval-y"]["status"] == "pending"     # newly seeded


def test_init_state_skip_marks_skipped(tmp_path: Path):
    res = es.init_state(
        tmp_path, _PLAN, "pilot",
        skip=({"task_id": "eval-x",
               "pivot_proof": "drop_configs:cfg-x",
               "skip_reason": "Stage 4 drop"},),
        valid_pivot_actions={"drop_configs:cfg-x"},
    )
    assert "error" not in res
    entry = es.load_state(tmp_path).tasks["eval-x"]
    assert entry["status"] == "skipped"
    assert entry["skip_proof"] == "drop_configs:cfg-x"
    assert entry["skip_reason"] == "Stage 4 drop"


def test_init_state_eval_skip_without_proof_rejected(tmp_path: Path):
    """An eval-task skip with no pivot_proof is the silent-scope back door."""
    res = es.init_state(
        tmp_path, _PLAN, "pilot",
        skip=({"task_id": "eval-x"},),
        valid_pivot_actions={"drop_configs:cfg-x"},
    )
    assert res["error"] == "unauthorized_skip"
    assert res["task_id"] == "eval-x"
    # state was left untouched
    assert es.load_state(tmp_path).tasks == {}


def test_init_state_eval_skip_unmatched_proof_rejected(tmp_path: Path):
    """A pivot_proof not in the brief's pivot_matrix is the same back door."""
    res = es.init_state(
        tmp_path, _PLAN, "pilot",
        skip=({"task_id": "eval-x",
               "pivot_proof": "pivot_matrix_init"},),
        valid_pivot_actions={"drop_configs:cfg-x"},
    )
    assert res["error"] == "unauthorized_skip"
    assert "valid_actions" in res
    assert es.load_state(tmp_path).tasks == {}


def test_init_state_bare_task_id_skip_rejected(tmp_path: Path):
    """Bare task-id strings are the legacy back door; reject them outright."""
    res = es.init_state(
        tmp_path, _PLAN, "pilot",
        skip=("eval-x",),
        valid_pivot_actions={"drop_configs:cfg-x"},
    )
    assert res["error"] == "bad_skip_entry"


def test_init_state_non_eval_skip_no_proof_ok(tmp_path: Path):
    """Non-eval tasks (serve / setup / aggregate) need no pivot proof."""
    res = es.init_state(
        tmp_path, _PLAN, "pilot",
        skip=({"task_id": "setup-x", "skip_reason": "no masks this run"},),
    )
    assert "error" not in res
    assert es.load_state(tmp_path).tasks["setup-x"]["status"] == "skipped"


def test_init_state_atomic(tmp_path: Path):
    es.init_state(tmp_path, _PLAN, "pilot")
    assert not (tmp_path / "experiments" / "experiment_state.json.tmp").exists()


# ---- task transitions ---------------------------------------------------

def test_register_running(tmp_path: Path):
    es.init_state(tmp_path, _PLAN, "pilot")
    es.register_running(tmp_path, [{"task_id": "eval-x", "gpu_ids": [4, 5],
                                    "kind": "eval"}])
    task = es.load_state(tmp_path).tasks["eval-x"]
    assert task["status"] == "running"
    assert task["gpu_ids"] == [4, 5]


def test_complete_and_fail(tmp_path: Path):
    es.init_state(tmp_path, _PLAN, "pilot")
    es.complete_task(tmp_path, "eval-x", result_dir="experiments/results/cfg-x")
    es.fail_task(tmp_path, "setup-x", error_summary="boom")
    state = es.load_state(tmp_path)
    assert state.tasks["eval-x"]["status"] == "completed"
    assert state.tasks["setup-x"]["status"] == "failed"
    assert state.tasks["setup-x"]["error_summary"] == "boom"


def test_complete_unknown_task(tmp_path: Path):
    es.init_state(tmp_path, _PLAN, "pilot")
    assert es.complete_task(tmp_path, "ghost")["error"] == "unknown_task"


def test_skip_task(tmp_path: Path):
    """A policy-dropped task is `skipped` (distinct from a genuine `failed`)."""
    es.init_state(tmp_path, _PLAN, "pilot")
    result = es.skip_task(tmp_path, "eval-x", reason="weights unavailable")
    assert result["status"] == "ok"
    task = es.load_state(tmp_path).tasks["eval-x"]
    assert task["status"] == "skipped"
    assert task["skip_reason"] == "weights unavailable"


def test_skip_unknown_task(tmp_path: Path):
    es.init_state(tmp_path, _PLAN, "pilot")
    assert es.skip_task(tmp_path, "ghost")["error"] == "unknown_task"


def test_status_snapshot_all_done(tmp_path: Path):
    es.init_state(tmp_path, _PLAN, "pilot")
    assert es.status_snapshot(tmp_path)["all_done"] is False
    es.complete_task(tmp_path, "setup-x")
    es.complete_task(tmp_path, "eval-x")
    snap = es.status_snapshot(tmp_path)
    assert snap["all_done"] is True
    assert snap["counts"]["completed"] == 2


# ---- detection + recovery ----------------------------------------------

def test_detection_script_mentions_tasks(tmp_path: Path):
    script = es.detection_script(tmp_path, ["eval-x", "eval-y"])
    assert "eval-x" in script and "eval-y" in script
    assert "DONE" in script and "RUNNING" in script


def test_parse_detection_output():
    out = ('DONE\teval-x\t{"status": "success"}\n'
           'RUNNING\teval-y\t\n'
           'DEAD\teval-z\t\n')
    parsed = es.parse_detection_output(out)
    assert parsed["eval-x"]["detected_status"] == "done"
    assert parsed["eval-x"]["done_info"]["status"] == "success"
    assert parsed["eval-y"]["detected_status"] == "running"
    assert parsed["eval-z"]["detected_status"] == "dead"


def test_recover_done_success_completes():
    state = es.ExperimentState(tasks={
        "t": {"status": "running", "retry_count": 0, "kind": "eval"}})
    detection = {"t": {"detected_status": "done",
                       "done_info": {"status": "success"}}}
    rec = es.recover_from_detection(state, detection, max_retries=1)
    assert rec.recovered_completed == ["t"]
    assert state.tasks["t"]["status"] == "completed"


def test_recover_dead_retries_then_fails():
    state = es.ExperimentState(tasks={
        "t": {"status": "running", "retry_count": 0, "kind": "eval"}})
    rec = es.recover_from_detection(
        state, {"t": {"detected_status": "dead"}}, max_retries=1)
    assert rec.retried == ["t"]
    assert state.tasks["t"]["status"] == "pending"
    assert state.tasks["t"]["retry_count"] == 1
    # a second dead detection — retries now exhausted
    rec2 = es.recover_from_detection(
        state, {"t": {"detected_status": "dead"}}, max_retries=1)
    assert rec2.recovered_failed == ["t"]
    assert state.tasks["t"]["status"] == "failed"


def test_recover_done_failure_retries():
    state = es.ExperimentState(tasks={
        "t": {"status": "running", "retry_count": 0, "kind": "eval"}})
    rec = es.recover_from_detection(
        state, {"t": {"detected_status": "done",
                      "done_info": {"status": "failure"}}}, max_retries=1)
    assert rec.retried == ["t"]


def test_recover_running_stays():
    state = es.ExperimentState(tasks={
        "t": {"status": "running", "retry_count": 0, "kind": "eval"}})
    rec = es.recover_from_detection(
        state, {"t": {"detected_status": "running"}}, max_retries=1)
    assert rec.still_running == ["t"]
    assert state.tasks["t"]["status"] == "running"


# ---- runner heartbeat ---------------------------------------------------

def test_detection_script_hung_vs_running(tmp_path: Path):
    """A live runner whose .progress.json went stale is detected HUNG."""
    logs = tmp_path / "experiments" / "logs"
    logs.mkdir(parents=True)
    for tid in ("fresh", "stale"):
        # both runners are "alive" — point at this live test process
        (logs / f"{tid}.pid").write_text(str(os.getpid()))
        (logs / f"{tid}.progress.json").write_text(
            json.dumps({"task_id": tid, "done": 1, "total": 9}))
    # age the stale runner's heartbeat well past the timeout
    old = time.time() - 9999
    os.utime(logs / "stale.progress.json", (old, old))

    script = es.detection_script(tmp_path, ["fresh", "stale"],
                                 heartbeat_timeout_s=60)
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    parsed = es.parse_detection_output(out.stdout)
    assert parsed["fresh"]["detected_status"] == "running"
    assert parsed["stale"]["detected_status"] == "hung"
    assert parsed["stale"]["pid"] == os.getpid()


def test_parse_detection_hung_carries_pid():
    parsed = es.parse_detection_output("HUNG\tt\t4242\n")
    assert parsed["t"]["detected_status"] == "hung"
    assert parsed["t"]["pid"] == 4242


def test_recover_hung_kills_runner_and_retries():
    """A hung task is killed and retried through the bounded path."""
    proc = subprocess.Popen(["sleep", "60"])
    try:
        state = es.ExperimentState(tasks={
            "t": {"status": "running", "retry_count": 0, "kind": "eval"}})
        rec = es.recover_from_detection(
            state, {"t": {"detected_status": "hung", "pid": proc.pid}},
            max_retries=1)
        assert rec.retried == ["t"]
        assert state.tasks["t"]["status"] == "pending"
        assert "hung" in state.tasks["t"]["error_summary"]
        # the hung runner's process was terminated
        assert proc.wait(timeout=5) != 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_recover_hung_exhausts_retries():
    state = es.ExperimentState(tasks={
        "t": {"status": "running", "retry_count": 1, "kind": "eval"}})
    rec = es.recover_from_detection(
        state, {"t": {"detected_status": "hung", "pid": 999999}},
        max_retries=1)
    assert rec.recovered_failed == ["t"]
    assert state.tasks["t"]["status"] == "failed"


def test_status_snapshot_supervisor_heartbeat(tmp_path: Path):
    es.init_state(tmp_path, _PLAN, "pilot")
    snap0 = es.status_snapshot(tmp_path)
    assert snap0["supervisor_heartbeat_at"] == ""
    assert snap0["supervisor_idle_seconds"] is None
    # a recovery pass stamps the supervisor heartbeat
    es.apply_detection(tmp_path, {}, max_retries=1)
    snap1 = es.status_snapshot(tmp_path)
    assert snap1["supervisor_heartbeat_at"]
    assert snap1["supervisor_idle_seconds"] is not None
    assert snap1["supervisor_idle_seconds"] >= 0


# ---- results aggregation ------------------------------------------------

def test_result_dir_is_mode_scoped(tmp_path: Path):
    """Both passes are mode-scoped — full lives under results/full/, not root."""
    assert er.result_dir(tmp_path, "cfg-x", "full") == (
        tmp_path / "experiments" / "results" / "full" / "cfg-x")
    assert er.result_dir(tmp_path, "cfg-x", "pilot") == (
        tmp_path / "experiments" / "results" / "pilot" / "cfg-x")


def test_scores_relpath_is_canonical():
    assert (er.scores_relpath("cfg-x", "full")
            == "experiments/results/full/cfg-x/scores.jsonl")
    assert (er.scores_relpath("cfg-x", "pilot")
            == "experiments/results/pilot/cfg-x/scores.jsonl")


def test_aggregate_config(tmp_path: Path):
    cdir = er.result_dir(tmp_path, "cfg-x", "pilot")
    cdir.mkdir(parents=True)
    (cdir / "scores.jsonl").write_text(
        '{"sample_key": "s1", "method_id": "m", "score": 0.8, "ok": true}\n'
        '{"sample_key": "s2", "method_id": "m", "score": 0.6, "ok": true}\n'
        '{"sample_key": "s3", "method_id": "m", "ok": false}\n',
        encoding="utf-8")
    result = er.aggregate_config(tmp_path, "cfg-x", mode="pilot", family="B")
    assert result["ok_count"] == 2
    assert result["failed_count"] == 1
    assert result["score_mean"] == 0.7
    assert result["mode"] == "pilot"
    assert (cdir / "config_result.json").is_file()


def test_write_summary(tmp_path: Path):
    for cid, score in (("cfg-a", 0.9), ("cfg-b", 0.5)):
        cdir = er.result_dir(tmp_path, cid)  # default mode "full"
        cdir.mkdir(parents=True)
        (cdir / "scores.jsonl").write_text(
            f'{{"sample_key": "s", "method_id": "m", "score": {score}, '
            f'"ok": true}}\n', encoding="utf-8")
        er.aggregate_config(tmp_path, cid, family="B")
    summary = er.write_summary(tmp_path)
    assert summary["config_count"] == 2
    assert er.read_summary(tmp_path)["config_count"] == 2


def test_write_summary_flags_missing_expected_config(tmp_path: Path):
    """An expected config that produced no result is listed incomplete."""
    cdir = er.result_dir(tmp_path, "cfg-ran")
    cdir.mkdir(parents=True)
    (cdir / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.7, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-ran")
    summary = er.write_summary(
        tmp_path, "full", expected_configs=["cfg-ran", "cfg-never-ran"])
    assert summary["config_count"] == 1
    assert "cfg-never-ran" in summary["totals"]["incomplete_configs"]


def test_write_summary_separates_pilot(tmp_path: Path):
    """The pilot sub-tree never leaks into the full summary."""
    full = er.result_dir(tmp_path, "cfg-a")
    full.mkdir(parents=True)
    (full / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.9, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-a")
    pilot = er.result_dir(tmp_path, "cfg-a", "pilot")
    pilot.mkdir(parents=True)
    (pilot / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.4, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-a", mode="pilot")
    assert er.write_summary(tmp_path, "full")["config_count"] == 1
    assert er.write_summary(tmp_path, "pilot")["config_count"] == 1


def test_write_summary_complete_flag(tmp_path: Path):
    """`complete` is true only when every listed config produced real scores."""
    # no configs at all -> not complete
    assert er.write_summary(tmp_path, "full")["complete"] is False

    # a config with real scores -> complete
    good = er.result_dir(tmp_path, "cfg-ok")
    good.mkdir(parents=True)
    (good / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.8, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-ok")
    assert er.write_summary(tmp_path, "full")["complete"] is True

    # add a hollow config (empty scores.jsonl, ok_count 0) -> not complete
    hollow = er.result_dir(tmp_path, "cfg-hollow")
    hollow.mkdir(parents=True)
    (hollow / "scores.jsonl").write_text("", encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-hollow")
    assert er.write_summary(tmp_path, "full")["complete"] is False


def test_write_summary_strict_against_brief_chosen_set(tmp_path: Path):
    """The tryon-eval shape: 1 of 9 scored → complete=false, 8 missing."""
    # one config actually scored
    good = er.result_dir(tmp_path, "cfg-1")
    good.mkdir(parents=True)
    (good / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.7, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-1")
    expected = [f"cfg-{i}" for i in range(1, 10)]  # 9 chosen configs
    summary = er.write_summary(tmp_path, "full", expected_configs=expected)
    assert summary["complete"] is False
    assert summary["scored_configs"] == ["cfg-1"]
    assert summary["missing_configs"] == [f"cfg-{i}" for i in range(2, 10)]
    assert summary["expected_configs"] == sorted(expected)


def test_write_summary_strict_complete_when_all_scored(tmp_path: Path):
    """All chosen configs scored → complete=true; missing_configs empty."""
    for cid in ("cfg-a", "cfg-b"):
        cdir = er.result_dir(tmp_path, cid)
        cdir.mkdir(parents=True)
        (cdir / "scores.jsonl").write_text(
            '{"sample_key": "s", "method_id": "m", "score": 0.9, "ok": true}\n',
            encoding="utf-8")
        er.aggregate_config(tmp_path, cid)
    summary = er.write_summary(
        tmp_path, "full", expected_configs=["cfg-a", "cfg-b"])
    assert summary["complete"] is True
    assert summary["missing_configs"] == []


# ---- check_experiment_completion ----------------------------------------

def _expand_iter_dir(iter_dir: Path, *, brief: dict, plan: dict) -> None:
    """Write a minimal design/experiment_brief + plans/task_plan into iter_dir."""
    (iter_dir / "design").mkdir(parents=True, exist_ok=True)
    (iter_dir / "design" / "experiment_brief.json").write_text(
        json.dumps(brief), encoding="utf-8")
    (iter_dir / "experiments" / "plans").mkdir(parents=True, exist_ok=True)
    (iter_dir / "experiments" / "plans" / "task_plan.json").write_text(
        json.dumps(plan), encoding="utf-8")


def test_check_experiment_completion_complete(tmp_path: Path):
    """Every chosen_config scored → complete=true, no missing."""
    brief = {"candidate_configs": [{"combination_id": "cfg-a"},
                                    {"combination_id": "cfg-b"}]}
    plan = {"tasks": [
        {"id": "eval-a", "type": "eval", "combination_id": "cfg-a"},
        {"id": "eval-b", "type": "eval", "combination_id": "cfg-b"},
    ]}
    _expand_iter_dir(tmp_path, brief=brief, plan=plan)
    for cid in ("cfg-a", "cfg-b"):
        cdir = er.result_dir(tmp_path, cid)
        cdir.mkdir(parents=True)
        (cdir / "scores.jsonl").write_text(
            '{"sample_key": "s", "method_id": "m", "score": 0.8, "ok": true}\n',
            encoding="utf-8")
        er.aggregate_config(tmp_path, cid)
    er.write_summary(tmp_path, "full",
                     expected_configs=["cfg-a", "cfg-b"])
    out = er.check_experiment_completion(tmp_path, "full")
    assert out["complete"] is True
    assert out["missing_configs"] == []
    assert out["scored_configs"] == ["cfg-a", "cfg-b"]


def test_check_experiment_completion_missing(tmp_path: Path):
    """A chosen config that never scored shows up in missing_configs."""
    brief = {"candidate_configs": [{"combination_id": "cfg-a"},
                                    {"combination_id": "cfg-b"}]}
    plan = {"tasks": [
        {"id": "eval-a", "type": "eval", "combination_id": "cfg-a"},
        {"id": "eval-b", "type": "eval", "combination_id": "cfg-b"},
    ]}
    _expand_iter_dir(tmp_path, brief=brief, plan=plan)
    # Seed state with eval-b failed (no scores written)
    es.init_state(tmp_path, plan, "full")
    es.fail_task(tmp_path, "eval-b", error_summary="OOM at minimal batch")
    # cfg-a scored
    cdir = er.result_dir(tmp_path, "cfg-a")
    cdir.mkdir(parents=True)
    (cdir / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.8, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-a")
    er.write_summary(tmp_path, "full",
                     expected_configs=["cfg-a", "cfg-b"])
    out = er.check_experiment_completion(tmp_path, "full")
    assert out["complete"] is False
    assert out["missing_configs"] == ["cfg-b"]
    assert out["failed_tasks"] == ["eval-b"]


def test_check_experiment_completion_excuses_skipped_with_proof(tmp_path: Path):
    """A Stage 4 pivot-matrix skip (init-level) counts as accounted-for."""
    brief = {
        "candidate_configs": [{"combination_id": "cfg-a"},
                              {"combination_id": "cfg-b"}],
        "pivot_matrix": [{"pilot_outcome": "x", "action": "drop_configs:cfg-b"}],
    }
    plan = {"tasks": [
        {"id": "eval-a", "type": "eval", "combination_id": "cfg-a"},
        {"id": "eval-b", "type": "eval", "combination_id": "cfg-b"},
    ]}
    _expand_iter_dir(tmp_path, brief=brief, plan=plan)
    # init seeds eval-b skipped with a real pivot_matrix action as its proof
    es.init_state(
        tmp_path, plan, "full",
        skip=({"task_id": "eval-b",
               "pivot_proof": "drop_configs:cfg-b",
               "skip_reason": "Stage 4 drop"},),
        valid_pivot_actions={"drop_configs:cfg-b"},
    )
    state = es.load_state(tmp_path)
    assert state.tasks["eval-b"]["skip_proof"] == "drop_configs:cfg-b"
    # eval-a ran successfully and cfg-a scored
    es.complete_task(tmp_path, "eval-a")
    cdir = er.result_dir(tmp_path, "cfg-a")
    cdir.mkdir(parents=True)
    (cdir / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.8, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-a")
    er.write_summary(tmp_path, "full",
                     expected_configs=["cfg-a", "cfg-b"])
    out = er.check_experiment_completion(tmp_path, "full")
    assert out["complete"] is True
    assert out["missing_configs"] == []
    assert out["skipped_with_proof"] == ["cfg-b"]


def test_check_experiment_completion_rejects_bogus_skip_proof(tmp_path: Path):
    """The gate re-validates skip_proof against the brief's pivot_matrix.

    Regression for the tryon-eval/iter_001 loophole: a state file with
    skip_proof "pivot_matrix_init" (a hardcoded label, not a real action)
    must NOT pass the gate when the brief's pivot_matrix lists different
    action strings. The skipped config must surface in missing_configs and
    unauthorized_skipped_tasks, and complete must be False.
    """
    brief = {
        "candidate_configs": [{"combination_id": "cfg-a"},
                              {"combination_id": "cfg-b"}],
        "pivot_matrix": [{"pilot_outcome": "x",
                          "action": "drop_configs:cfg-b-real-action"}],
    }
    plan = {"tasks": [
        {"id": "eval-a", "type": "eval", "combination_id": "cfg-a"},
        {"id": "eval-b", "type": "eval", "combination_id": "cfg-b"},
    ]}
    _expand_iter_dir(tmp_path, brief=brief, plan=plan)
    # Hand-write a state file with a bogus proof string — simulates the
    # legacy "pivot_matrix_init" stamp that init_state used to apply.
    state_path = tmp_path / es.STATE_REL
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "schema_version": 1, "mode": "full",
        "tasks": {
            "eval-a": {"kind": "eval", "status": "completed"},
            "eval-b": {"kind": "eval", "status": "skipped",
                       "skip_proof": "pivot_matrix_init",
                       "skip_reason": "pre-validated Stage 4 drop"},
        },
    }), encoding="utf-8")
    # cfg-a scored
    cdir = er.result_dir(tmp_path, "cfg-a")
    cdir.mkdir(parents=True)
    (cdir / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.8, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-a")
    er.write_summary(tmp_path, "full",
                     expected_configs=["cfg-a", "cfg-b"])
    out = er.check_experiment_completion(tmp_path, "full")
    assert out["complete"] is False
    assert out["missing_configs"] == ["cfg-b"]
    assert out["skipped_with_proof"] == []
    assert out["unauthorized_skipped_tasks"] == ["eval-b"]


def test_check_experiment_completion_blocks_on_in_progress(tmp_path: Path):
    """An eval task still pending/running blocks the gate even if others scored."""
    brief = {"candidate_configs": [{"combination_id": "cfg-a"},
                                    {"combination_id": "cfg-b"}]}
    plan = {"tasks": [
        {"id": "eval-a", "type": "eval", "combination_id": "cfg-a"},
        {"id": "eval-b", "type": "eval", "combination_id": "cfg-b"},
    ]}
    _expand_iter_dir(tmp_path, brief=brief, plan=plan)
    es.init_state(tmp_path, plan, "full")
    # cfg-a scored, eval-b still pending
    cdir = er.result_dir(tmp_path, "cfg-a")
    cdir.mkdir(parents=True)
    (cdir / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.8, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-a")
    er.write_summary(tmp_path, "full",
                     expected_configs=["cfg-a", "cfg-b"])
    out = er.check_experiment_completion(tmp_path, "full")
    assert out["complete"] is False
    assert "eval-b" in out["in_progress_tasks"]


def _hand_write_state(iter_dir: Path, tasks: dict) -> None:
    """Bypass init_state and write experiment_state.json directly — useful
    for completion-gate tests that need specific outcome fields the public
    transitions don't set at init time."""
    (iter_dir / "experiments").mkdir(parents=True, exist_ok=True)
    (iter_dir / "experiments" / "experiment_state.json").write_text(
        json.dumps({"schema_version": 1, "mode": "full", "tasks": tasks}),
        encoding="utf-8",
    )


def test_check_experiment_completion_runtime_failed_counts_as_resolved(
    tmp_path: Path,
):
    """A config whose only eval task is outcome=runtime_failed is resolved —
    not in missing_configs — so the gate advances."""
    brief = {"candidate_configs": [{"combination_id": "cfg-a"},
                                    {"combination_id": "cfg-b"}]}
    plan = {"tasks": [
        {"id": "eval-a", "type": "eval", "combination_id": "cfg-a",
         "mode": "full"},
        {"id": "eval-b", "type": "eval", "combination_id": "cfg-b",
         "mode": "full"},
    ]}
    _expand_iter_dir(tmp_path, brief=brief, plan=plan)
    _hand_write_state(tmp_path, {
        "eval-a": {"kind": "eval", "status": "completed"},
        "eval-b": {"kind": "eval", "status": "failed",
                   "outcome": "runtime_failed",
                   "failure_category": "oom"},
    })
    # cfg-a scored, cfg-b runtime_failed
    cdir = er.result_dir(tmp_path, "cfg-a")
    cdir.mkdir(parents=True)
    (cdir / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.8, "ok": true}\n',
        encoding="utf-8")
    er.aggregate_config(tmp_path, "cfg-a")
    er.write_summary(tmp_path, "full",
                     expected_configs=["cfg-a", "cfg-b"])
    out = er.check_experiment_completion(tmp_path, "full")
    assert out["complete"] is True
    assert out["runtime_failed_configs"] == ["cfg-b"]
    assert out["missing_configs"] == []
    assert out["runtime_failure_categories"]["cfg-b"] == ["oom"]


def test_check_experiment_completion_runtime_failure_cap_exceeded(
    tmp_path: Path,
):
    """When >30% of expected configs are runtime_failed, complete must be
    forced false and runtime_failure_cap_exceeded surfaced."""
    expected = [f"cfg-{i}" for i in range(1, 11)]   # 10 configs
    brief = {"candidate_configs": [{"combination_id": c} for c in expected]}
    plan = {"tasks": [
        {"id": f"eval-{c}", "type": "eval", "combination_id": c,
         "mode": "full"} for c in expected
    ]}
    _expand_iter_dir(tmp_path, brief=brief, plan=plan)
    # 4 of 10 runtime_failed (cap=ceil(0.3*10)=3)
    tasks: dict = {}
    for i, c in enumerate(expected):
        if i < 4:
            tasks[f"eval-{c}"] = {"kind": "eval", "status": "failed",
                                  "outcome": "runtime_failed",
                                  "failure_category": "oom"}
        else:
            tasks[f"eval-{c}"] = {"kind": "eval", "status": "completed"}
            cdir = er.result_dir(tmp_path, c)
            cdir.mkdir(parents=True)
            (cdir / "scores.jsonl").write_text(
                '{"sample_key": "s", "method_id": "m", "score": 0.7, '
                '"ok": true}\n', encoding="utf-8")
            er.aggregate_config(tmp_path, c)
    _hand_write_state(tmp_path, tasks)
    er.write_summary(tmp_path, "full", expected_configs=expected)
    out = er.check_experiment_completion(tmp_path, "full")
    assert out["complete"] is False
    assert "runtime_failure_cap_exceeded" in out
    assert out["runtime_failure_cap_exceeded"]["limit"] == 3
    assert out["runtime_failure_cap_exceeded"]["observed"] == 4


def test_check_experiment_completion_runtime_failed_with_pending_sibling(
    tmp_path: Path,
):
    """A config with BOTH a runtime_failed task AND a still-pending eval task
    is NOT counted as runtime_failed — it stays in in_progress / missing so
    the loop doesn't claim victory before all sample seeds finish."""
    brief = {"candidate_configs": [{"combination_id": "cfg-a"}]}
    plan = {"tasks": [
        {"id": "eval-a-1", "type": "eval", "combination_id": "cfg-a",
         "mode": "full"},
        {"id": "eval-a-2", "type": "eval", "combination_id": "cfg-a",
         "mode": "full"},
    ]}
    _expand_iter_dir(tmp_path, brief=brief, plan=plan)
    _hand_write_state(tmp_path, {
        "eval-a-1": {"kind": "eval", "status": "failed",
                     "outcome": "runtime_failed", "failure_category": "oom"},
        "eval-a-2": {"kind": "eval", "status": "pending"},
    })
    er.write_summary(tmp_path, "full", expected_configs=["cfg-a"])
    out = er.check_experiment_completion(tmp_path, "full")
    assert out["complete"] is False
    assert out["runtime_failed_configs"] == []
    assert "eval-a-2" in out["in_progress_tasks"]


# ---- runner-marker skip conversion --------------------------------------

def test_apply_detection_converts_eval_skipped_done_to_failure(tmp_path: Path):
    """A runner that writes done.json with status=skipped on an eval task
    must be converted to status=failure so the silent-skip path is closed."""
    plan = {"tasks": [
        {"id": "eval-x", "type": "eval", "family": "B", "mode": "pilot",
         "combination_id": "cfg-x", "depends_on": []},
    ]}
    es.init_state(tmp_path, plan, "pilot")
    es.register_running(tmp_path, [{"task_id": "eval-x",
                                     "gpu_ids": [0], "kind": "eval"}])
    detection = {
        "eval-x": {
            "detected_status": "done",
            "done_info": {
                "status": "skipped",
                "summary": "missing SAM/SCHP/mmpose deps",
            },
        },
    }
    es.apply_detection(tmp_path, detection)
    state = es.load_state(tmp_path)
    task = state.tasks["eval-x"]
    # the runner attempted "skipped" — recovery rejects it
    assert task["status"] in ("pending", "failed")  # retries are allowed
    assert "unauthorized skip" in (task.get("error_summary") or "")


def test_apply_detection_keeps_non_eval_skipped_path_open(tmp_path: Path):
    """A non-eval done.json status=skipped is also rejected as 'task reported
    failure' (the marker contract says success|failure); but it is NOT given
    the unauthorized-skip note — only eval tasks carry that conversion."""
    plan = {"tasks": [
        {"id": "setup-x", "type": "setup", "mode": None, "depends_on": []},
    ]}
    es.init_state(tmp_path, plan, "pilot")
    es.register_running(tmp_path, [{"task_id": "setup-x",
                                     "gpu_ids": [], "kind": "setup"}])
    detection = {
        "setup-x": {
            "detected_status": "done",
            "done_info": {"status": "skipped", "summary": "no masks needed"},
        },
    }
    es.apply_detection(tmp_path, detection)
    state = es.load_state(tmp_path)
    # setup-x is treated as a generic failure (not the unauthorized-skip path)
    assert "unauthorized skip" not in (state.tasks["setup-x"].get(
        "error_summary") or "")


# ---- CLI round-trips ----------------------------------------------------

def _cli_workspace(base: Path, per_sample_root: Path):
    params = valid_params(base, per_sample_root)
    cfg = ERAConfig.from_params(params)
    ws = Workspace(base, params["project_name"])
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({"project_name": params["project_name"],
                     "stage": "full_experiment", "stage_index": 6,
                     "iteration": 1, "run_state": "running"})
    ws.write_file("config.yaml", cfg.to_commented_yaml())
    ws.write_json("iter_001/experiments/plans/task_plan.json", _PLAN)
    return ws


def test_cli_init_experiment_skip_unmatched_proof_rejected(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """End-to-end: the CLI rejects an init skip whose pivot_proof does not
    match the brief's pivot_matrix actions. This is the workspace-level
    enforcement of the gate that closes the silent-scope back door."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    ws.write_json("iter_001/design/experiment_brief.json", {
        "candidate_configs": [{"combination_id": "cfg-x"}],
        "pivot_matrix": [{"pilot_outcome": "x",
                          "action": "drop_configs:real-action"}],
    })
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot",
         "skip": [{"task_id": "eval-x",
                   "pivot_proof": "pivot_matrix_init"}]})))
    code = main(["init-experiment"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "unauthorized_skip"
    # State was not seeded — the rejection happens before init writes anything
    assert not (ws.iter_path() / es.STATE_REL).exists()


def test_cli_init_experiment_and_status(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    assert main(["init-experiment"]) == 0
    capsys.readouterr()

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root)})))
    code = main(["experiment-status"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["counts"]["pending"] == 2
    assert "detection_script" in result


def test_cli_init_experiment_no_plan(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    params = valid_params(tmp_path, per_sample_root)
    ws = Workspace(tmp_path, params["project_name"])
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({"project_name": params["project_name"],
                     "stage": "full_experiment", "iteration": 1})
    ws.write_file("config.yaml",
                  ERAConfig.from_params(params).to_commented_yaml())
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    code = main(["init-experiment"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "no_task_plan"


def test_cli_record_task_aggregates_eval(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()

    # eval-x is a pilot-mode task -> its results live under results/pilot/
    cdir = er.result_dir(ws.iter_path(), "cfg-x", "pilot")
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "scores.jsonl").write_text(
        '{"sample_key": "s", "method_id": "m", "score": 0.5, "ok": true}\n',
        encoding="utf-8")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "success"})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["aggregate"]["ok_count"] == 1
    assert result["summary"]["mode"] == "pilot"


def test_cli_record_task_skipped_eval_without_proof_rejected(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """An eval skip without pivot_proof is rejected as unauthorized_skip.

    The Stage 6 completion gate forbids runtime silent scope-reduction: a skip
    on an eval task must carry a Stage 4 pivot-matrix proof. Without it, the
    request fails and the state is left untouched.
    """
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "skipped", "reason": "vton-iqa weights unavailable"})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "unauthorized_skip"
    # state untouched — the task is still pending, not skipped
    state = es.load_state(ws.iter_path())
    assert state.tasks["eval-x"]["status"] == "pending"


def test_cli_record_task_skipped_eval_with_proof(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """An eval skip with a matching pivot_proof is accepted."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    # Brief with a pivot_matrix entry that authorizes the drop.
    ws.write_json("iter_001/design/experiment_brief.json", {
        "candidate_configs": [{"combination_id": "cfg-x"}],
        "pivot_matrix": [
            {"pilot_outcome": "vton_iqa_weights_missing",
             "action": "drop_configs:cfg-x"},
        ],
    })
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "skipped", "reason": "vton-iqa weights unavailable",
         "pivot_proof": "drop_configs:cfg-x"})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["task"]["status"] == "skipped"
    assert result["task"]["skip_proof"] == "drop_configs:cfg-x"
    # a skipped eval must not aggregate — no hollow config_result.json
    assert "aggregate" not in result
    assert not (er.result_dir(ws.iter_path(), "cfg-x", "pilot")
                / "config_result.json").is_file()


def test_cli_record_task_skipped_eval_with_bad_proof_rejected(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """A pivot_proof that doesn't match any brief pivot_matrix action is rejected."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    ws.write_json("iter_001/design/experiment_brief.json", {
        "candidate_configs": [{"combination_id": "cfg-x"}],
        "pivot_matrix": [{"pilot_outcome": "x", "action": "drop_configs:other-cfg"}],
    })
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "skipped", "reason": "x",
         "pivot_proof": "this-is-not-in-the-pivot-matrix"})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "unauthorized_skip"
    assert "does not match" in result["message"]


def test_cli_record_task_skipped_eval_no_plan_rejected(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """A skip request without a task plan fails closed (the eval gate cannot
    be enforced without proof of task type)."""
    params = valid_params(tmp_path, per_sample_root)
    ws = Workspace(tmp_path, params["project_name"])
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({"project_name": params["project_name"],
                     "stage": "full_experiment", "stage_index": 6,
                     "iteration": 1, "run_state": "running"})
    ws.write_file("config.yaml",
                  ERAConfig.from_params(params).to_commented_yaml())
    # NOTE: no task_plan.json written

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "skipped", "reason": "deps unavailable"})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "no_task_plan_or_unknown_task"


def test_cli_record_task_skipped_non_eval_with_proof_kept(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """A non-eval skip that happens to carry pivot_proof keeps it recorded."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "setup-x",
         "outcome": "skipped", "reason": "no masks needed",
         "pivot_proof": "drop_setup:masks"})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["task"]["status"] == "skipped"
    assert result["task"]["skip_proof"] == "drop_setup:masks"


def test_cli_check_experiment_completion_excuses_init_skip(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """End-to-end: a Stage 4 init-skip is excused by the orchestration gate.

    Drives the public CLI on a workspace where one eval is init-skipped
    (with a `pivot_proof` matching a real `pivot_matrix.action`) and the
    other scored. The gate must report `complete: true` so Stage 7
    advances — this is the legitimate pivot-matrix escape valve."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    ws.write_json("iter_001/design/experiment_brief.json", {
        "candidate_configs": [{"combination_id": "cfg-x"}],
        "pivot_matrix": [{"pilot_outcome": "x",
                          "action": "drop_configs:cfg-x"}],
    })
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot",
         "skip": [{"task_id": "eval-x",
                   "pivot_proof": "drop_configs:cfg-x",
                   "skip_reason": "Stage 4 drop"}]})))
    main(["init-experiment"])
    capsys.readouterr()

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    code = main(["check-experiment-completion"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    # cfg-x has no scores but is skipped-with-proof → accounted-for, no missing
    assert result["complete"] is True
    assert result["missing_configs"] == []
    assert result["skipped_with_proof"] == ["cfg-x"]


def test_cli_record_task_skipped_setup_task_no_proof_needed(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """Non-eval task types keep unrestricted skip semantics (no proof needed)."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()

    # setup-x is a non-eval task in _PLAN — it can be skipped freely
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "setup-x",
         "outcome": "skipped", "reason": "no masks needed for this run"})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["task"]["status"] == "skipped"


def test_cli_check_experiment_completion(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """The CLI returns the orchestration-layer completion answer."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    ws.write_json("iter_001/design/experiment_brief.json", {
        "candidate_configs": [{"combination_id": "cfg-x"}],
    })
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    code = main(["check-experiment-completion"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    # cfg-x's eval is still pending; gate refuses to pass
    assert result["complete"] is False
    assert "cfg-x" in result["missing_configs"]


def test_cli_record_task_bad_outcome(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "bogus"})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "bad_outcome"


def test_cli_recover_experiment(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()
    es.register_running(ws.iter_path(), [{"task_id": "eval-x",
                                          "gpu_ids": [4], "kind": "eval"}])
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root),
         "detection_output": 'DONE\teval-x\t{"status": "success"}\n'})))
    code = main(["recover-experiment"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["recovered_completed"] == ["eval-x"]


# ---- runtime_fail_task --------------------------------------------------

def _exhaust_circuit_breaker(iter_dir: Path, task_id: str, error_text: str):
    """Drive heal_tick through enough attempts to open the breaker, then
    return the give_up envelope for use as ``heal_history``."""
    from era.orchestration import error_heal as eh
    cfg = ERAConfig()
    for _ in range(eh.CIRCUIT_BREAKER_MAX):
        eh.heal_tick(iter_dir, cfg, {"id": task_id}, error_text)
    give_up = eh.heal_tick(iter_dir, cfg, {"id": task_id}, error_text)
    assert give_up["action"] == "give_up", "fixture must open the breaker"
    return give_up


def test_runtime_fail_task_happy_path(tmp_path: Path):
    """Eligible category + open breaker + eval task → runtime_failed recorded."""
    es.init_state(tmp_path, _PLAN, "pilot")
    give_up = _exhaust_circuit_breaker(tmp_path, "eval-x", "CUDA out of memory")
    result = es.runtime_fail_task(
        tmp_path, "eval-x",
        failure_category="oom",
        heal_history=give_up,
    )
    assert result["status"] == "ok"
    task = es.load_state(tmp_path).tasks["eval-x"]
    assert task["status"] == "failed"
    assert task["outcome"] == "runtime_failed"
    assert task["failure_category"] == "oom"
    assert task["heal_history"]["circuit_broken"] is True


def test_runtime_fail_task_rejects_ineligible_category(tmp_path: Path):
    """import is not in RUNTIME_FAILURE_CATEGORIES — reject even with proof."""
    es.init_state(tmp_path, _PLAN, "pilot")
    give_up = _exhaust_circuit_breaker(
        tmp_path, "eval-x", "No module named 'cv2_broken'",
    )
    result = es.runtime_fail_task(
        tmp_path, "eval-x",
        failure_category="import",
        heal_history=give_up,
    )
    assert result["error"] == "runtime_failed_category_not_eligible"
    # state was NOT mutated
    assert es.load_state(tmp_path).tasks["eval-x"].get("outcome") != "runtime_failed"


def test_runtime_fail_task_rejects_missing_heal_history(tmp_path: Path):
    """No heal-tick envelope means the agent cannot prove the breaker opened."""
    es.init_state(tmp_path, _PLAN, "pilot")
    result = es.runtime_fail_task(
        tmp_path, "eval-x",
        failure_category="oom",
        heal_history={},  # empty — no error_id, no circuit_broken
    )
    assert result["error"] == "runtime_failed_no_heal_history"


def test_runtime_fail_task_rejects_no_open_circuit(tmp_path: Path):
    """heal-tick never fired for this (task, error_id) → no on-disk proof."""
    es.init_state(tmp_path, _PLAN, "pilot")
    # Hand-rolled envelope that LOOKS valid but has no matching disk record.
    fake_history = {"error_id": "deadbeefcafe", "circuit_broken": True,
                    "category": "oom", "attempts": 3}
    result = es.runtime_fail_task(
        tmp_path, "eval-x",
        failure_category="oom",
        heal_history=fake_history,
    )
    assert result["error"] == "runtime_failed_no_heal_history"


def test_runtime_fail_task_rejects_non_eval(tmp_path: Path):
    """Non-eval task (setup) cannot use the runtime_failed path."""
    es.init_state(tmp_path, _PLAN, "pilot")
    give_up = _exhaust_circuit_breaker(
        tmp_path, "setup-x", "CUDA out of memory",
    )
    result = es.runtime_fail_task(
        tmp_path, "setup-x",
        failure_category="oom",
        heal_history=give_up,
    )
    assert result["error"] == "runtime_failed_wrong_task_type"
    assert result["task_kind"] == "setup"


def test_cli_record_task_runtime_failed_happy_path(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """End-to-end CLI: record runtime_failed on an eval task with a real
    give_up envelope. State updates; no aggregate is written."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()
    give_up = _exhaust_circuit_breaker(
        ws.iter_path(), "eval-x", "CUDA out of memory",
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "runtime_failed",
         "failure_category": "oom",
         "heal_history": give_up})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["status"] == "ok"
    assert result["task"]["outcome"] == "runtime_failed"
    # No aggregate should be present (no hollow config_result)
    assert "aggregate" not in result


def test_cli_record_task_runtime_failed_ineligible_category(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """End-to-end CLI: import is not eligible — reject."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()
    give_up = _exhaust_circuit_breaker(
        ws.iter_path(), "eval-x", "No module named 'cv2_broken'",
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "runtime_failed",
         "failure_category": "import",
         "heal_history": give_up})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "runtime_failed_category_not_eligible"


def test_cli_record_task_runtime_failed_non_eval_rejected(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys
):
    """End-to-end CLI: a non-eval task with outcome=runtime_failed is rejected
    at the CLI gate (defense in depth — the orchestration layer also rejects)."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "setup-x",
         "outcome": "runtime_failed",
         "failure_category": "oom",
         "heal_history": {"error_id": "x", "circuit_broken": True}})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "runtime_failed_wrong_task_type"


# ---- wait_for_any_done (Phase D-1) --------------------------------------

def _touch_done_marker(iter_dir: Path, task_id: str) -> Path:
    """Write a minimal done.json marker for a task."""
    logs = iter_dir / es.LOGS_REL
    logs.mkdir(parents=True, exist_ok=True)
    path = logs / f"{task_id}.done.json"
    path.write_text(json.dumps({"task_id": task_id, "status": "success",
                                "exit_code": 0, "summary": ""}),
                    encoding="utf-8")
    return path


def test_wait_for_any_done_returns_when_a_marker_lands(tmp_path: Path):
    """The fast path: a runner finishes mid-wait, the call returns
    immediately with that task in `done`."""
    import threading

    def land_marker_after(delay_s: float, task_id: str):
        time.sleep(delay_s)
        _touch_done_marker(tmp_path, task_id)

    t = threading.Thread(target=land_marker_after, args=(0.4, "task_b"))
    t.start()
    try:
        result = es.wait_for_any_done(
            tmp_path, task_ids=["task_a", "task_b"], timeout_s=5.0,
        )
    finally:
        t.join()

    assert result["done"] == ["task_b"]
    assert result["still_running"] == ["task_a"]
    assert result["timed_out"] is False
    # Should return well within the 5s timeout — the marker landed at ~0.4s.
    assert result["elapsed_s"] < 2.0


def test_wait_for_any_done_returns_immediately_for_pre_existing_marker(
    tmp_path: Path,
):
    """A marker that already exists on entry is reported on the first poll —
    no busy-wait, no timeout."""
    _touch_done_marker(tmp_path, "task_a")
    result = es.wait_for_any_done(
        tmp_path, task_ids=["task_a", "task_b"], timeout_s=5.0,
    )
    assert "task_a" in result["done"]
    assert "task_b" in result["still_running"]
    assert result["timed_out"] is False
    assert result["elapsed_s"] < 0.5


def test_wait_for_any_done_times_out_with_no_marker(tmp_path: Path):
    """No marker lands within timeout_s → return cleanly with timed_out=true."""
    result = es.wait_for_any_done(
        tmp_path, task_ids=["task_a"], timeout_s=0.5,
    )
    assert result["done"] == []
    assert result["still_running"] == ["task_a"]
    assert result["timed_out"] is True
    # Bounded above the timeout but close to it.
    assert 0.4 <= result["elapsed_s"] < 2.0


def test_wait_for_any_done_handles_empty_task_ids(tmp_path: Path):
    """No tasks to wait on → return immediately, not blocked on timeout."""
    result = es.wait_for_any_done(tmp_path, task_ids=[], timeout_s=10.0)
    assert result["done"] == []
    assert result["still_running"] == []
    assert result["timed_out"] is False
    assert result["elapsed_s"] == 0.0


def test_cli_wait_for_any_done_returns_done_list(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys,
):
    """End-to-end CLI: marker present → CLI emits done list + timed_out:false."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    _touch_done_marker(ws.iter_path(), "eval-x")

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root),
         "task_ids": ["eval-x", "eval-y"],
         "timeout_s": 2.0})))
    code = main(["wait-for-any-done"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["done"] == ["eval-x"]
    assert result["still_running"] == ["eval-y"]
    assert result["timed_out"] is False


def test_cli_wait_for_any_done_rejects_bad_task_ids(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys,
):
    """task_ids must be a list of strings."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_ids": "not_a_list"})))
    code = main(["wait-for-any-done"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "bad_task_ids"


def test_cli_wait_for_any_done_default_timeout_from_config(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys,
):
    """When timeout_s is omitted, the CLI uses config.experiment.poll_interval_s
    so the Stage 6 skill can call this without specifying the cadence twice."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    _touch_done_marker(ws.iter_path(), "eval-x")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_ids": ["eval-x"]})))
    code = main(["wait-for-any-done"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["done"] == ["eval-x"]


# ---- Phase C-2: auto_validate_skips authorization ----------------------

_AV_PLAN = {
    "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
    "gpu_pool": 4, "tasks": [
        {"id": "eval-pass-full", "type": "eval", "family": "B",
         "mode": "full", "combination_id": "cfg-pass", "depends_on": []},
        {"id": "eval-fail-full", "type": "eval", "family": "B",
         "mode": "full", "combination_id": "cfg-fail", "depends_on": []},
        {"id": "agg-fail-full", "type": "aggregate", "mode": "full",
         "combination_id": "cfg-fail", "depends_on": ["eval-fail-full"]},
    ],
}


def _write_av_result(
    iter_dir: Path, *, failing: list[str], passing: list[str] | None = None,
) -> Path:
    """Stub a Phase C-2 auto_validate/result.json on disk."""
    path = iter_dir / "auto_validate" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1, "mode": "annotated",
        "thresholds": {"pass": 0.7, "recall": 0.6, "min_samples": 10},
        "annotated_sample_count": 24,
        "skipped_for_min_samples": False,
        "per_config": [], "any_passed": bool(passing),
        "passing_configs": passing or [], "failing_configs": failing,
        "at": "2026-05-27T00:00:00+00:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_init_state_auto_validate_skips_marks_failing_tasks(tmp_path: Path):
    """init_state with auto_validate_skips=['cfg-fail'] marks the
    matching full-mode eval + aggregate tasks `skipped`."""
    _write_av_result(tmp_path, failing=["cfg-fail"], passing=["cfg-pass"])
    result = es.init_state(
        tmp_path, _AV_PLAN, "full",
        auto_validate_skips=("cfg-fail",),
    )
    assert result["status"] == "ok"
    state = es.load_state(tmp_path)
    assert state.tasks["eval-pass-full"]["status"] == "pending"
    assert state.tasks["eval-fail-full"]["status"] == "skipped"
    assert state.tasks["eval-fail-full"]["skip_proof"] == "auto_validate_failed"
    assert state.tasks["eval-fail-full"]["skip_reason"] == "auto_validate_failed"
    # The aggregate task for the failing config also gets skipped.
    assert state.tasks["agg-fail-full"]["status"] == "skipped"


def test_init_state_auto_validate_skips_requires_result_file(tmp_path: Path):
    """No auto_validate/result.json on disk → reject the request."""
    # No file written
    result = es.init_state(
        tmp_path, _AV_PLAN, "full",
        auto_validate_skips=("cfg-fail",),
    )
    assert result["error"] == "missing_auto_validate_result"


def test_init_state_auto_validate_skips_rejects_unauthorized(tmp_path: Path):
    """A cid not in failing_configs is rejected — defense in depth."""
    _write_av_result(tmp_path, failing=["cfg-pass"], passing=["cfg-fail"])
    result = es.init_state(
        tmp_path, _AV_PLAN, "full",
        auto_validate_skips=("cfg-fail",),  # not in failing_configs!
    )
    assert result["error"] == "unauthorized_auto_validate_skip"
    assert "cfg-fail" in result["configs"]


def test_cli_record_task_auto_validate_skip_authorized(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys,
):
    """End-to-end CLI: record-task with auto_validate_skip:true is
    accepted iff the task's combination_id is in failing_configs."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    _write_av_result(ws.iter_path(), failing=["cfg-x"])
    # Seed eval-x as pending
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()
    # Now skip eval-x via auto_validate_skip
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "skipped",
         "auto_validate_skip": True,
         "reason": "phase C-2 gate failed"})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["task"]["status"] == "skipped"
    assert result["task"]["skip_proof"] == "auto_validate_failed"


def test_cli_record_task_auto_validate_skip_unauthorized(
    tmp_path: Path, per_sample_root: Path, monkeypatch, capsys,
):
    """A cid NOT in failing_configs gets rejected."""
    ws = _cli_workspace(tmp_path, per_sample_root)
    _write_av_result(ws.iter_path(), failing=["other-cfg"])
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot"})))
    main(["init-experiment"])
    capsys.readouterr()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "outcome": "skipped",
         "auto_validate_skip": True})))
    code = main(["record-task"])
    result = json.loads(capsys.readouterr().out)
    assert code == 1
    assert result["error"] == "unauthorized_skip"


def test_completion_gate_accepts_auto_validate_failed_proof(tmp_path: Path):
    """check_experiment_completion treats auto_validate_failed-skipped
    configs as resolved iff the result.json on disk lists them."""
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()
    es.init_state(iter_dir, _AV_PLAN, "full")
    _write_av_result(iter_dir, failing=["cfg-fail"], passing=["cfg-pass"])
    # Mark cfg-fail's eval task skipped with the auto_validate proof
    state = es.load_state(iter_dir)
    state.tasks["eval-fail-full"]["status"] = "skipped"
    state.tasks["eval-fail-full"]["skip_proof"] = "auto_validate_failed"
    es.save_state(iter_dir, state)
    # Write a minimal task_plan.json so completion gate can resolve combination_ids
    plan_path = iter_dir / "experiments" / "plans" / "task_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(_AV_PLAN), encoding="utf-8")
    # Empty summary (no scores yet)
    summary_path = iter_dir / "experiments" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "mode": "full", "scored_configs": ["cfg-pass"],
        "expected_configs": ["cfg-pass", "cfg-fail"], "complete": False,
    }), encoding="utf-8")
    result = er.check_experiment_completion(iter_dir, "full")
    assert "cfg-fail" in result["skipped_with_proof"]
    assert "cfg-fail" not in result["missing_configs"]
