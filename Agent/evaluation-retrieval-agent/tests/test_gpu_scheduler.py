"""Tests for the Stage 6 GPU scheduler (era/orchestration/gpu_scheduler.py)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from conftest import valid_params

from era.cli import main
from era.config import ERAConfig
from era.orchestration import experiment_state, gpu_scheduler as gs
from era.workspace import Workspace


@pytest.fixture(autouse=True)
def _stub_watchdog_subprocess(monkeypatch):
    """Stop tests from shelling out to ``sudo pkill`` / ``bash start.sh``.

    ``claim_batch`` now invokes :func:`suspend_watchdog`, which in turn calls
    ``subprocess.run(["sudo", "-n", "pkill", ...])``. Tests must not run
    that — stub both ``run`` and ``Popen`` on the gs module's subprocess.
    """
    class _FakeResult:
        returncode = 1  # pkill's "no process matched" — treated as ok
        stdout = ""
        stderr = ""

    def _fake_run(*_a, **_kw):
        return _FakeResult()

    def _fake_popen(*_a, **_kw):
        return None

    monkeypatch.setattr(gs.subprocess, "run", _fake_run)
    monkeypatch.setattr(gs.subprocess, "Popen", _fake_popen)

_SMI = """0, 120, 81920
1, 65000, 81920
2, 300, 81920
3, 79000, 81920
"""


# ---- free-GPU discovery -------------------------------------------------

def test_parse_gpu_snapshot():
    snap = gs.parse_gpu_snapshot(_SMI)
    assert len(snap) == 4
    assert snap[0] == {"gpu_id": 0, "memory_used_mb": 120,
                       "memory_total_mb": 81920, "memory_used_pct": 0.1}


def test_parse_gpu_snapshot_skips_garbage():
    assert gs.parse_gpu_snapshot("not,a,number\n\n5, 10, 100") == [
        {"gpu_id": 5, "memory_used_mb": 10, "memory_total_mb": 100,
         "memory_used_pct": 10.0},
    ]


def test_parse_free_gpus_threshold():
    assert gs.parse_free_gpus(_SMI, threshold_mb=2000) == [0, 2]


def test_parse_free_gpus_only_filter():
    assert gs.parse_free_gpus(_SMI, threshold_mb=2000, only_gpu_ids=[2, 3]) == [2]


def test_allowed_gpu_pool_respects_reserve_and_cap():
    cfg = ERAConfig()
    cfg.hardware.visible_gpu_ids = [4, 5, 6, 7]
    cfg.hardware.reserve_gpu_ids = [7]
    cfg.hardware.max_gpus_per_run = 2
    assert gs.allowed_gpu_pool(cfg) == [4, 5]


def test_allowed_gpu_pool_uncapped():
    cfg = ERAConfig()
    cfg.hardware.visible_gpu_ids = [0, 1, 2]
    cfg.hardware.max_gpus_per_run = 0
    assert gs.allowed_gpu_pool(cfg) == [0, 1, 2]


# ---- tensor-parallel snapping -------------------------------------------

def test_snap_tensor_parallel_power_of_two():
    assert gs.snap_tensor_parallel(4) == 4
    assert gs.snap_tensor_parallel(3) == 2
    assert gs.snap_tensor_parallel(1) == 1


def test_snap_tensor_parallel_divisor_of_heads():
    assert gs.snap_tensor_parallel(4, head_count=28) == 4   # 28 / 4 = 7
    assert gs.snap_tensor_parallel(3, head_count=28) == 2   # 3 -> 2 divides 28
    assert gs.snap_tensor_parallel(6, head_count=28) == 4


# ---- DAG topology -------------------------------------------------------

def _chain() -> list[dict]:
    return [
        {"id": "a", "depends_on": []},
        {"id": "b", "depends_on": ["a"]},
        {"id": "c", "depends_on": ["b"]},
    ]


def test_topo_sort_layers_linear():
    assert gs.topo_sort_layers(_chain()) == [["a"], ["b"], ["c"]]


def test_topo_sort_layers_diamond():
    tasks = [
        {"id": "root", "depends_on": []},
        {"id": "l", "depends_on": ["root"]},
        {"id": "r", "depends_on": ["root"]},
        {"id": "join", "depends_on": ["l", "r"]},
    ]
    layers = gs.topo_sort_layers(tasks)
    assert layers[0] == ["root"]
    assert sorted(layers[1]) == ["l", "r"]
    assert layers[2] == ["join"]


def test_compute_downstream_counts():
    counts = gs.compute_downstream_counts(_chain())
    assert counts["a"] == 2
    assert counts["b"] == 1
    assert counts["c"] == 0


def test_assign_gpus_greedy():
    tasks = [
        {"id": "t1", "gpu_count": 2, "estimated_minutes": 5},
        {"id": "t2", "gpu_count": 2, "estimated_minutes": 5},
    ]
    assigned, unfit = gs.assign_gpus(tasks, [0, 1, 2, 3])
    assert unfit == []
    assert {a["task_id"] for a in assigned} == {"t1", "t2"}
    assert sum(len(a["gpu_ids"]) for a in assigned) == 4


def test_assign_gpus_reports_unfit():
    tasks = [{"id": "big", "gpu_count": 4, "estimated_minutes": 5}]
    assigned, unfit = gs.assign_gpus(tasks, [0, 1])
    assert assigned == []
    assert unfit == ["big"]


def test_assign_gpus_serve_first():
    """A serve task claims the pool ahead of smaller metric tasks (Rule 6)."""
    tasks = [
        {"id": "metric", "type": "eval", "gpu_count": 1, "estimated_minutes": 5},
        {"id": "serve", "type": "serve", "gpu_count": 4, "estimated_minutes": 10},
    ]
    assigned, unfit = gs.assign_gpus(tasks, [0, 1, 2, 3])
    assert {a["task_id"] for a in assigned} == {"serve"}
    assert unfit == ["metric"]


# ---- the global GPU-lease file ------------------------------------------

@pytest.fixture
def _state_dir(tmp_path: Path, monkeypatch):
    """Redirect the cross-workspace lease file into a temp directory."""
    monkeypatch.setenv("ERA_STATE_DIR", str(tmp_path / "era_state"))
    return tmp_path


def test_leases_empty_by_default(_state_dir):
    assert gs.read_leases() == {}


def test_clean_stale_drops_old_leases(_state_dir):
    import time
    leases = {
        "0": {"claimed_at": time.time(), "task_ids": ["fresh"]},
        "1": {"claimed_at": time.time() - 9999, "task_ids": ["stale"]},
    }
    kept = gs._clean_stale(leases)
    assert "0" in kept and "1" not in kept


# ---- claim_batch --------------------------------------------------------

def _exp_workspace(base: Path, per_sample_root: Path, task_plan: dict):
    """Scaffold a workspace with a config.yaml + task_plan.json for Stage 6."""
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
    iter_dir = ws.iter_path()
    ws.write_json("iter_001/experiments/plans/task_plan.json", task_plan)
    return ws, iter_dir, cfg


def _metric_plan() -> dict:
    return {
        "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
        "gpu_pool": 4, "tasks": [
            {"id": "eval-x", "type": "eval", "family": "B", "mode": "pilot",
             "gpu_count": 2, "depends_on": [], "estimated_minutes": 5},
            {"id": "eval-y", "type": "eval", "family": "B", "mode": "pilot",
             "gpu_count": 2, "depends_on": [], "estimated_minutes": 5},
        ],
    }


def test_claim_batch_assigns_ready_metric_tasks(
    _state_dir, tmp_path: Path, per_sample_root: Path
):
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, _metric_plan())
    experiment_state.init_state(iter_dir, _metric_plan(), "pilot")
    result = gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    assert result["status"] == "ok"
    assert {b["task_id"] for b in result["batch"]} == {"eval-x", "eval-y"}
    assert result["total_count"] == 2


def test_claim_batch_blocks_on_scarce_gpus(
    _state_dir, tmp_path: Path, per_sample_root: Path
):
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, _metric_plan())
    experiment_state.init_state(iter_dir, _metric_plan(), "pilot")
    result = gs.claim_batch(ws.root, cfg, [0, 1], "pilot")  # room for one
    assert len(result["batch"]) == 1
    assert len(result["blocked"]) == 1


def test_claim_batch_writes_leases(
    _state_dir, tmp_path: Path, per_sample_root: Path
):
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, _metric_plan())
    experiment_state.init_state(iter_dir, _metric_plan(), "pilot")
    gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    leases = gs.read_leases()
    assert {int(g) for g in leases} == {0, 1, 2, 3}


def test_claim_batch_skips_gpus_leased_elsewhere(
    _state_dir, tmp_path: Path, per_sample_root: Path
):
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, _metric_plan())
    experiment_state.init_state(iter_dir, _metric_plan(), "pilot")
    # another workspace already holds GPUs 0 and 1
    import time
    gs._save_leases({
        "0": {"workspace_root": "/other/ws", "task_ids": ["x"],
              "claimed_at": time.time()},
        "1": {"workspace_root": "/other/ws", "task_ids": ["x"],
              "claimed_at": time.time()},
    })
    result = gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    assert len(result["batch"]) == 1  # only GPUs 2,3 free -> one 2-GPU task


def test_release_gpus_frees_leases(
    _state_dir, tmp_path: Path, per_sample_root: Path
):
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, _metric_plan())
    experiment_state.init_state(iter_dir, _metric_plan(), "pilot")
    gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    gs.release_gpus(ws.root, ["eval-x", "eval-y"])
    assert gs.read_leases() == {}


def test_claim_batch_releases_lease_for_completed_task(
    _state_dir, tmp_path: Path, per_sample_root: Path
):
    """A completed task's GPU lease is pruned so the next batch reuses the pool."""
    plan = {
        "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
        "gpu_pool": 4, "tasks": [
            {"id": "eval-a", "type": "eval", "family": "B", "mode": "pilot",
             "gpu_count": 4, "depends_on": [], "estimated_minutes": 5},
            {"id": "eval-b", "type": "eval", "family": "B", "mode": "pilot",
             "gpu_count": 4, "depends_on": ["eval-a"], "estimated_minutes": 5},
        ],
    }
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, plan)
    experiment_state.init_state(iter_dir, plan, "pilot")
    r1 = gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    assert {b["task_id"] for b in r1["batch"]} == {"eval-a"}
    # eval-a finishes — its lease must be released for the next claim
    experiment_state.complete_task(iter_dir, "eval-a")
    r2 = gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    assert {b["task_id"] for b in r2["batch"]} == {"eval-b"}
    assert {int(g) for g in gs.read_leases()} == {0, 1, 2, 3}  # now held by eval-b


def test_claim_batch_resident_judge_blocks_next_serve(
    _state_dir, tmp_path: Path, per_sample_root: Path
):
    """A completed serve task with a pending teardown eval is still resident."""
    plan = {
        "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
        "gpu_pool": 4, "tasks": [
            {"id": "serve-a", "type": "serve", "family": "A", "mode": "pilot",
             "gpu_count": 4, "depends_on": [], "estimated_minutes": 10,
             "teardown_after": ["eval-a"]},
            {"id": "serve-b", "type": "serve", "family": "A", "mode": "pilot",
             "gpu_count": 4, "depends_on": ["serve-a"], "estimated_minutes": 10,
             "teardown_after": ["eval-b"]},
            {"id": "eval-a", "type": "eval", "family": "A", "mode": "pilot",
             "gpu_count": 0, "depends_on": ["serve-a"], "estimated_minutes": 5},
            {"id": "eval-b", "type": "eval", "family": "A", "mode": "pilot",
             "gpu_count": 0, "depends_on": ["serve-b"], "estimated_minutes": 5},
        ],
    }
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, plan)
    cfg.experiment.family_a_execution = "serial_full_pool"  # Rule 6 regime
    experiment_state.init_state(iter_dir, plan, "pilot")
    # serve-a is up (endpoint validated) but its teardown eval has not run
    experiment_state.complete_task(iter_dir, "serve-a")
    result = gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    assert result["resident_judge"] == "serve-a"
    assert "serve-b" in result["blocked"]


def test_claim_batch_rule6_pauses_family_b_behind_judge(
    _state_dir, tmp_path: Path, per_sample_root: Path
):
    """serial_full_pool + before_after_family_a pauses Family-B behind the judge
    even when GPUs are free beside it (strict measurement isolation)."""
    plan = {
        "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
        "gpu_pool": 4, "tasks": [
            {"id": "serve-a", "type": "serve", "family": "A", "mode": "pilot",
             "gpu_count": 4, "depends_on": [], "estimated_minutes": 10,
             "teardown_after": ["eval-a"]},
            {"id": "eval-b", "type": "eval", "family": "B", "mode": "pilot",
             "gpu_count": 1, "depends_on": [], "estimated_minutes": 5},
        ],
    }
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, plan)
    cfg.experiment.family_a_execution = "serial_full_pool"
    cfg.experiment.family_b_schedule = "before_after_family_a"  # strict serial
    experiment_state.init_state(iter_dir, plan, "pilot")
    # the judge is already resident
    experiment_state.register_running(
        iter_dir, [{"task_id": "serve-a", "gpu_ids": [0, 1, 2, 3],
                    "kind": "serve"}])
    # GPUs 4,5 are free beside the judge — but fam_b_serial still pauses eval-b.
    result = gs.claim_batch(ws.root, cfg, [4, 5], "pilot")
    assert "eval-b" in result["blocked"]
    assert result["resident_judge"] == "serve-a"


# ---- parallel_packed co-resident judges (Phase D-6) ---------------------

def _parallel_serve_plan() -> dict:
    """An 8-GPU pool, two independent right-sized judges + one Family-B eval."""
    return {
        "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
        "gpu_pool": 8, "tasks": [
            {"id": "serve-a", "type": "serve", "family": "A", "mode": "pilot",
             "gpu_count": 4, "depends_on": [], "estimated_minutes": 10,
             "teardown_after": ["eval-a"]},
            {"id": "serve-b", "type": "serve", "family": "A", "mode": "pilot",
             "gpu_count": 2, "depends_on": [], "estimated_minutes": 10,
             "teardown_after": ["eval-b"]},
            {"id": "eval-a", "type": "eval", "family": "A", "mode": "pilot",
             "gpu_count": 0, "depends_on": ["serve-a"], "estimated_minutes": 5},
            {"id": "eval-b", "type": "eval", "family": "A", "mode": "pilot",
             "gpu_count": 0, "depends_on": ["serve-b"], "estimated_minutes": 5},
            {"id": "eval-c", "type": "eval", "family": "B", "mode": "pilot",
             "gpu_count": 1, "depends_on": [], "estimated_minutes": 5},
        ],
    }


def test_claim_batch_parallel_packs_judges_concurrently(
    _state_dir, tmp_path: Path, per_sample_root: Path,
):
    """Default parallel_packed claims both judges + the Family-B eval in one
    batch, on disjoint GPU subsets (pool saturated, not one judge holding all)."""
    ws, iter_dir, cfg = _exp_workspace(
        tmp_path, per_sample_root, _parallel_serve_plan())
    assert cfg.experiment.family_a_execution == "parallel_packed"  # the default
    experiment_state.init_state(iter_dir, _parallel_serve_plan(), "pilot")
    result = gs.claim_batch(ws.root, cfg, list(range(8)), "pilot")
    ids = {b["task_id"] for b in result["batch"]}
    assert {"serve-a", "serve-b", "eval-c"} <= ids
    # GPU assignments are disjoint and right-sized.
    by_id = {b["task_id"]: b["gpu_ids"] for b in result["batch"]}
    assert len(by_id["serve-a"]) == 4
    assert len(by_id["serve-b"]) == 2
    assert len(by_id["eval-c"]) == 1
    all_gpus = by_id["serve-a"] + by_id["serve-b"] + by_id["eval-c"]
    assert len(all_gpus) == len(set(all_gpus)) == 7  # 7/8 claimed, no overlap


def test_claim_batch_parallel_keeps_all_resident_judge_leases(
    _state_dir, tmp_path: Path, per_sample_root: Path,
):
    """A completed judge keeps its lease while its teardown eval is pending,
    even with a second judge co-resident — both leases survive the prune."""
    ws, iter_dir, cfg = _exp_workspace(
        tmp_path, per_sample_root, _parallel_serve_plan())
    experiment_state.init_state(iter_dir, _parallel_serve_plan(), "pilot")
    r1 = gs.claim_batch(ws.root, cfg, list(range(8)), "pilot")
    serve_a_gpus = {g for b in r1["batch"] if b["task_id"] == "serve-a"
                    for g in b["gpu_ids"]}
    assert len(serve_a_gpus) == 4
    # serve-a's endpoint comes up (task completed) but eval-a hasn't run yet.
    experiment_state.complete_task(iter_dir, "serve-a")
    r2 = gs.claim_batch(ws.root, cfg, list(range(8)), "pilot")
    assert sorted(r2["resident_judges"]) == ["serve-a", "serve-b"]
    # serve-a's GPUs stay leased despite its task being 'completed' — the
    # resident-judge lease survives the prune until its teardown eval resolves.
    leased = {int(g) for g in gs.read_leases()}
    assert serve_a_gpus <= leased


def test_claim_batch_parallel_caps_concurrent_judges(
    _state_dir, tmp_path: Path, per_sample_root: Path,
):
    """max_concurrent_judges=1 blocks the second judge even with GPUs free."""
    ws, iter_dir, cfg = _exp_workspace(
        tmp_path, per_sample_root, _parallel_serve_plan())
    cfg.experiment.max_concurrent_judges = 1
    experiment_state.init_state(iter_dir, _parallel_serve_plan(), "pilot")
    result = gs.claim_batch(ws.root, cfg, list(range(8)), "pilot")
    serves = {b["task_id"] for b in result["batch"] if b["type"] == "serve"}
    assert len(serves) == 1                       # only one judge launched
    blocked = set(result["blocked"])
    assert {"serve-a", "serve-b"} - serves == blocked & {"serve-a", "serve-b"}


def test_claim_batch_parallel_backfills_family_b_beside_judge(
    _state_dir, tmp_path: Path, per_sample_root: Path,
):
    """Under parallel_packed a Family-B eval runs on a GPU free beside a
    resident judge (no Rule-6 pause), regardless of family_b_schedule."""
    ws, iter_dir, cfg = _exp_workspace(
        tmp_path, per_sample_root, _parallel_serve_plan())
    experiment_state.init_state(iter_dir, _parallel_serve_plan(), "pilot")
    # serve-a is already resident on GPUs 0-3.
    experiment_state.register_running(
        iter_dir, [{"task_id": "serve-a", "gpu_ids": [0, 1, 2, 3],
                    "kind": "serve"}])
    # GPU 6 is free beside the judge; eval-c (Family-B, 1 GPU) should backfill it.
    result = gs.claim_batch(ws.root, cfg, [6], "pilot")
    batch_ids = {b["task_id"] for b in result["batch"]}
    assert "eval-c" in batch_ids
    assert result["resident_judge"] == "serve-a"


# ---- CLI round-trips ----------------------------------------------------

def test_cli_gpu_scan(_state_dir, tmp_path: Path, per_sample_root: Path,
                       monkeypatch, capsys):
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, _metric_plan())
    # the config's allowed pool is GPUs 4-7; free among them = 4 and 6
    smi = "4, 120, 81920\n5, 65000, 81920\n6, 300, 81920\n7, 79000, 81920\n"
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "nvidia_smi_output": smi})))
    code = main(["gpu-scan"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["free_gpus"] == [4, 6]
    assert result["allowed_pool"] == [4, 5, 6, 7]


def test_cli_claim_batch(_state_dir, tmp_path: Path, per_sample_root: Path,
                         monkeypatch, capsys):
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, _metric_plan())
    experiment_state.init_state(iter_dir, _metric_plan(), "pilot")
    free_smi = "0, 1, 1\n1, 1, 1\n2, 1, 1\n3, 1, 1\n"
    # config visible pool is [4,5,6,7]; remap the plan to that pool
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root), "mode": "pilot",
         "nvidia_smi_output": "4, 1, 1\n5, 1, 1\n6, 1, 1\n7, 1, 1\n"})))
    code = main(["claim-batch"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["status"] == "ok"
    assert len(result["batch"]) == 2


# ---- watchdog lifecycle -------------------------------------------------

def test_suspend_watchdog_creates_sentinel_and_calls_pkill(
    tmp_path: Path, monkeypatch,
):
    """First call shells out and stamps the sentinel."""
    calls: list[tuple] = []

    class _Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def _capture_run(argv, **kwargs):
        calls.append(("run", tuple(argv)))
        return _Result()

    monkeypatch.setattr(gs.subprocess, "run", _capture_run)
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()
    result = gs.suspend_watchdog(iter_dir)
    assert result["status"] == "ok"
    assert result["action"] == "suspended"  # rc=0 → matched a process
    assert (iter_dir / ".watchdog_suspended").is_file()
    assert len(calls) == 1
    assert "pkill" in calls[0][1] and gs.WATCHDOG_PROCESS_PATTERN in calls[0][1]


def test_suspend_watchdog_idempotent_with_sentinel(
    tmp_path: Path, monkeypatch,
):
    """Second call with the sentinel present must not re-invoke pkill."""
    calls: list[tuple] = []

    def _capture_run(argv, **kwargs):
        calls.append(argv)
        class _R:
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    monkeypatch.setattr(gs.subprocess, "run", _capture_run)
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()
    gs.suspend_watchdog(iter_dir)
    again = gs.suspend_watchdog(iter_dir)
    assert again["action"] == "already_suspended"
    assert len(calls) == 1  # only the first call hit pkill


def test_suspend_watchdog_no_matching_process(tmp_path: Path, monkeypatch):
    """pkill rc=1 means "no matching process" — still ok, sentinel still set."""
    class _R:
        returncode = 1
        stdout = ""
        stderr = ""

    monkeypatch.setattr(gs.subprocess, "run", lambda *_a, **_kw: _R())
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()
    result = gs.suspend_watchdog(iter_dir)
    assert result["action"] == "no_watchdog_running"
    assert (iter_dir / ".watchdog_suspended").is_file()


def test_suspend_watchdog_swallows_oserror(tmp_path: Path, monkeypatch):
    """A subprocess error (sudo missing, timeout) is reported but never raised."""
    def _raise(*_a, **_kw):
        raise FileNotFoundError("sudo")

    monkeypatch.setattr(gs.subprocess, "run", _raise)
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()
    result = gs.suspend_watchdog(iter_dir)
    assert result["action"] == "suspend_failed"
    assert "sudo" in result["reason"]
    assert (iter_dir / ".watchdog_suspended").is_file()


def test_resume_watchdog_no_sentinel_no_op(tmp_path: Path, monkeypatch):
    """No sentinel means no suspend ever happened — resume is a no-op."""
    popen_calls: list = []
    monkeypatch.setattr(
        gs.subprocess, "Popen",
        lambda *a, **kw: popen_calls.append((a, kw)) or None,
    )
    result = gs.resume_watchdog(tmp_path / "iter_001")
    assert result["action"] == "not_suspended"
    assert popen_calls == []


def test_resume_watchdog_with_sentinel_runs_start_script(
    tmp_path: Path, monkeypatch,
):
    """Sentinel present → bash start.sh is invoked, sentinel is removed."""
    popen_calls: list = []
    monkeypatch.setattr(
        gs.subprocess, "Popen",
        lambda argv, **kw: popen_calls.append((tuple(argv), kw)) or None,
    )
    monkeypatch.setattr(gs, "WATCHDOG_START_DIR", str(tmp_path))
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()
    (iter_dir / ".watchdog_suspended").write_text("2026-05-26\n", "utf-8")
    result = gs.resume_watchdog(iter_dir)
    assert result["action"] == "resumed"
    assert not (iter_dir / ".watchdog_suspended").exists()
    assert popen_calls and popen_calls[0][0] == ("bash", "start.sh")


def test_resume_watchdog_missing_start_dir_clears_sentinel(
    tmp_path: Path, monkeypatch,
):
    """Resume gracefully when the start dir doesn't exist (e.g. dev box)."""
    popen_calls: list = []
    monkeypatch.setattr(
        gs.subprocess, "Popen",
        lambda *a, **kw: popen_calls.append((a, kw)) or None,
    )
    monkeypatch.setattr(gs, "WATCHDOG_START_DIR", str(tmp_path / "not-here"))
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()
    (iter_dir / ".watchdog_suspended").write_text("ts\n", "utf-8")
    result = gs.resume_watchdog(iter_dir)
    assert result["action"] == "resume_skipped_no_dir"
    assert not (iter_dir / ".watchdog_suspended").exists()
    assert popen_calls == []


def test_claim_batch_invokes_suspend_watchdog(
    _state_dir, tmp_path: Path, per_sample_root: Path, monkeypatch,
):
    """The first claim of an iteration must fire suspend_watchdog."""
    suspended: list[Path] = []
    monkeypatch.setattr(
        gs, "suspend_watchdog",
        lambda iter_dir: suspended.append(iter_dir) or {
            "status": "ok", "action": "stub",
        },
    )
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, _metric_plan())
    experiment_state.init_state(iter_dir, _metric_plan(), "pilot")
    gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    assert suspended and suspended[0] == iter_dir


# ---- ensure_watchdog_alive (Stage 8) ------------------------------------

def _pgrep_stub(returncode: int):
    """A subprocess.run stub that emulates pgrep's exit code."""
    class _R:
        pass
    r = _R()
    r.returncode = returncode
    r.stdout = ""
    r.stderr = ""
    return lambda *_a, **_kw: r


def test_ensure_watchdog_alive_already_running(tmp_path: Path, monkeypatch):
    """pgrep finds a live watchdog → no start.sh, action already_alive."""
    popen_calls: list = []
    monkeypatch.setattr(gs.subprocess, "run", _pgrep_stub(0))  # pgrep: found
    monkeypatch.setattr(
        gs.subprocess, "Popen",
        lambda *a, **kw: popen_calls.append((a, kw)) or None)
    result = gs.ensure_watchdog_alive(tmp_path / "iter_001")
    assert result["action"] == "already_alive"
    assert popen_calls == []   # never spawns a duplicate watchdog


def test_ensure_watchdog_alive_starts_when_dead(tmp_path: Path, monkeypatch):
    """pgrep finds nothing → start.sh is launched from the watchdog dir."""
    popen_calls: list = []
    monkeypatch.setattr(gs.subprocess, "run", _pgrep_stub(1))  # pgrep: none
    monkeypatch.setattr(
        gs.subprocess, "Popen",
        lambda argv, **kw: popen_calls.append((tuple(argv), kw)) or None)
    monkeypatch.setattr(gs, "WATCHDOG_START_DIR", str(tmp_path))
    result = gs.ensure_watchdog_alive(tmp_path / "iter_001")
    assert result["action"] == "started"
    assert popen_calls and popen_calls[0][0] == ("bash", "start.sh")
    assert popen_calls[0][1]["cwd"] == str(tmp_path)


def test_ensure_watchdog_alive_no_start_dir(tmp_path: Path, monkeypatch):
    """Dead watchdog but no start dir (dev box) → start_skipped_no_dir."""
    popen_calls: list = []
    monkeypatch.setattr(gs.subprocess, "run", _pgrep_stub(1))
    monkeypatch.setattr(
        gs.subprocess, "Popen",
        lambda *a, **kw: popen_calls.append((a, kw)) or None)
    monkeypatch.setattr(gs, "WATCHDOG_START_DIR", str(tmp_path / "not-here"))
    result = gs.ensure_watchdog_alive(tmp_path / "iter_001")
    assert result["action"] == "start_skipped_no_dir"
    assert popen_calls == []


def test_ensure_watchdog_alive_clears_stale_sentinel(tmp_path: Path, monkeypatch):
    """A live watchdog contradicts a suspend sentinel — it gets cleared so the
    next iteration's suspend_watchdog fires correctly."""
    monkeypatch.setattr(gs.subprocess, "run", _pgrep_stub(0))  # already alive
    iter_dir = tmp_path / "iter_001"
    iter_dir.mkdir()
    sentinel = iter_dir / ".watchdog_suspended"
    sentinel.write_text("ts\n", encoding="utf-8")
    result = gs.ensure_watchdog_alive(iter_dir)
    assert result["action"] == "already_alive"
    assert not sentinel.exists()


def test_cli_ensure_watchdog(_state_dir, tmp_path: Path, per_sample_root: Path,
                             monkeypatch, capsys):
    """CLI round-trip: ensure-watchdog reports ok over a real workspace."""
    monkeypatch.setattr(gs.subprocess, "run", _pgrep_stub(0))  # already alive
    ws, iter_dir, cfg = _exp_workspace(tmp_path, per_sample_root, _metric_plan())
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"workspace_path": str(ws.root)})))
    code = main(["ensure-watchdog"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["status"] == "ok"
    assert result["action"] == "already_alive"


# ---- max_parallel_runners cap (Phase D-1) -------------------------------

def _three_task_plan() -> dict:
    return {
        "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
        "gpu_pool": 4, "tasks": [
            {"id": "eval-a", "type": "eval", "family": "B", "mode": "pilot",
             "gpu_count": 1, "depends_on": [], "estimated_minutes": 5},
            {"id": "eval-b", "type": "eval", "family": "B", "mode": "pilot",
             "gpu_count": 1, "depends_on": [], "estimated_minutes": 5},
            {"id": "eval-c", "type": "eval", "family": "B", "mode": "pilot",
             "gpu_count": 1, "depends_on": [], "estimated_minutes": 5},
        ],
    }


def test_claim_batch_respects_max_parallel_runners(
    _state_dir, tmp_path: Path, per_sample_root: Path,
):
    """When max_parallel_runners=2, claim_batch caps the batch to 2 even
    when 4 GPUs are free and 3 tasks are ready."""
    ws, iter_dir, cfg = _exp_workspace(
        tmp_path, per_sample_root, _three_task_plan(),
    )
    cfg.experiment.max_parallel_runners = 2
    experiment_state.init_state(iter_dir, _three_task_plan(), "pilot")
    result = gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    assert result["status"] == "ok"
    assert len(result["batch"]) == 2
    # The deferred task surfaces under blocked for visibility.
    assert len(result["blocked"]) == 1


def test_claim_batch_uncapped_when_max_parallel_runners_is_zero(
    _state_dir, tmp_path: Path, per_sample_root: Path,
):
    """Default (0) means uncapped — all 3 tasks should claim concurrently."""
    ws, iter_dir, cfg = _exp_workspace(
        tmp_path, per_sample_root, _three_task_plan(),
    )
    # Explicit assertion that the field defaults to 0 (no cap).
    assert cfg.experiment.max_parallel_runners == 0
    experiment_state.init_state(iter_dir, _three_task_plan(), "pilot")
    result = gs.claim_batch(ws.root, cfg, [0, 1, 2, 3], "pilot")
    assert {b["task_id"] for b in result["batch"]} == {"eval-a", "eval-b", "eval-c"}
