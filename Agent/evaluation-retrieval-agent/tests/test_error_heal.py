"""Tests for Stage 6 error healing (era/orchestration/error_heal.py)."""

from __future__ import annotations

import io
import json
from pathlib import Path

from conftest import valid_params

from era.cli import main
from era.config import ERAConfig
from era.orchestration import error_heal as eh
from era.workspace import Workspace

_CFG = ERAConfig()


# ---- categorization -----------------------------------------------------

def test_categorize_runtime_text():
    assert eh.categorize_runtime_text("CUDA out of memory") == "oom"
    assert eh.categorize_runtime_text("No module named 'cv2'") == "import"
    assert eh.categorize_runtime_text("Connection refused") == "serving"
    assert eh.categorize_runtime_text("No such file or directory") == "missing_dir"
    assert eh.categorize_runtime_text("json.decoder.JSONDecodeError") == "config"
    assert eh.categorize_runtime_text("runner hung — no heartbeat") == "hung"
    assert eh.categorize_runtime_text("something weird happened") == "runtime"


def test_error_id_stable_across_noise():
    a = eh.structured_error("OOM at 0x7fa12 batch 64 step 1200")
    b = eh.structured_error("OOM at 0x9bc44 batch 32 step 5000")
    assert a.error_id == b.error_id  # digits/hex normalized out


def test_error_id_differs_by_category():
    a = eh.structured_error("No module named 'cv2'")
    b = eh.structured_error("CUDA out of memory")
    assert a.error_id != b.error_id


# ---- attempt_auto_fix ---------------------------------------------------

def test_auto_fix_import_whitelisted():
    err = eh.structured_error("ModuleNotFoundError: No module named 'cv2'")
    fix = eh.attempt_auto_fix(err, iter_dir=Path("/tmp"), cfg=_CFG,
                              task={"id": "t"})
    assert fix["action"] == "pip_install"
    assert fix["patch"]["pip_install"] == "opencv-python-headless"


def test_auto_fix_import_unknown_escalates():
    err = eh.structured_error("No module named 'totally_obscure_pkg'")
    assert eh.attempt_auto_fix(err, iter_dir=Path("/tmp"), cfg=_CFG,
                               task={"id": "t"}) is None


def test_auto_fix_missing_dir_inside_experiments(tmp_path: Path):
    target = tmp_path / "experiments" / "results" / "new"
    err = eh.structured_error(
        f"FileNotFoundError: No such file or directory: '{target}'")
    fix = eh.attempt_auto_fix(err, iter_dir=tmp_path, cfg=_CFG,
                              task={"id": "t"})
    assert fix["action"] == "mkdir"
    assert target.is_dir()


def test_auto_fix_missing_dir_refuses_outside(tmp_path: Path):
    err = eh.structured_error(
        "FileNotFoundError: No such file or directory: '/etc/secret'")
    assert eh.attempt_auto_fix(err, iter_dir=tmp_path, cfg=_CFG,
                               task={"id": "t"}) is None


def test_auto_fix_oom_halves_batch(tmp_path: Path):
    err = eh.structured_error("torch.cuda.OutOfMemoryError: CUDA out of memory")
    fix = eh.attempt_auto_fix(err, iter_dir=tmp_path, cfg=_CFG,
                              task={"id": "t"})
    assert fix["action"] == "reduce_load"
    assert fix["patch"]["batch_size"] == 4   # default 8 -> 4
    assert fix["patch"]["gpu_memory_utilization"] == 0.85


def test_auto_fix_serving_rebinds_port(tmp_path: Path):
    err = eh.structured_error("OSError: address already in use")
    fix = eh.attempt_auto_fix(err, iter_dir=tmp_path, cfg=_CFG,
                              task={"id": "t"})
    assert fix["action"] == "rebind_port"
    assert fix["patch"]["port"] == _CFG.serving.port_range[0] + 1


def test_auto_fix_runtime_escalates(tmp_path: Path):
    err = eh.structured_error("AssertionError: tensor shape mismatch")
    assert eh.attempt_auto_fix(err, iter_dir=tmp_path, cfg=_CFG,
                               task={"id": "t"}) is None


# ---- heal_tick + circuit breaker ----------------------------------------

def test_heal_tick_retry_with_patch(tmp_path: Path):
    result = eh.heal_tick(tmp_path, _CFG, {"id": "t"}, "CUDA out of memory")
    assert result["action"] == "retry_with_patch"
    assert result["category"] == "oom"
    assert "batch_size" in result["patch"]


def test_heal_tick_escalates_runtime(tmp_path: Path):
    result = eh.heal_tick(tmp_path, _CFG, {"id": "t"}, "weird runtime crash")
    assert result["action"] == "escalate"


def test_heal_tick_circuit_breaker(tmp_path: Path):
    text = "CUDA out of memory"
    for _ in range(eh.CIRCUIT_BREAKER_MAX):
        assert eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)["action"] != "give_up"
    final = eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    assert final["action"] == "give_up"
    assert final["circuit_broken"] is True


# ---- runtime-failure taxonomy ------------------------------------------

def test_runtime_failure_categories_set():
    """The eligible-for-runtime_failed categories are oom / serving / hung —
    infra failures the agent has already tried to auto-fix and can't get past."""
    assert eh.RUNTIME_FAILURE_CATEGORIES == frozenset(
        {"oom", "serving", "hung"}
    )


def test_heal_tick_give_up_eligible_for_oom(tmp_path: Path):
    """oom is eligible — circuit-breaker open with `runtime_failure_eligible: true`."""
    text = "CUDA out of memory"
    for _ in range(eh.CIRCUIT_BREAKER_MAX):
        eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    final = eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    assert final["action"] == "give_up"
    assert final["category"] == "oom"
    assert final["runtime_failure_eligible"] is True


def test_heal_tick_give_up_not_eligible_for_import(tmp_path: Path):
    """import is NOT eligible — agent could fix this (pip install) so it stays
    in the planning-miss surface that blocks the loop."""
    text = "No module named 'totally_obscure_pkg'"
    for _ in range(eh.CIRCUIT_BREAKER_MAX):
        eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    final = eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    assert final["action"] == "give_up"
    assert final["category"] == "import"
    assert final["runtime_failure_eligible"] is False


def test_heal_tick_give_up_eligible_for_serving(tmp_path: Path):
    """serving (connection refused / port in use) is eligible — vLLM init
    fights and the auto-fix port rebind doesn't cure everything."""
    text = "ConnectionError: Connection refused"
    for _ in range(eh.CIRCUIT_BREAKER_MAX):
        eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    final = eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    assert final["action"] == "give_up"
    assert final["category"] == "serving"
    assert final["runtime_failure_eligible"] is True


def test_heal_tick_give_up_eligible_for_hung(tmp_path: Path):
    """hung (heartbeat timeout) is eligible — a deadlocked runner is infra."""
    text = "runner hung — no heartbeat for 1800s"
    for _ in range(eh.CIRCUIT_BREAKER_MAX):
        eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    final = eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    assert final["action"] == "give_up"
    assert final["category"] == "hung"
    assert final["runtime_failure_eligible"] is True


def test_heal_tick_give_up_not_eligible_for_config(tmp_path: Path):
    """config (JSONDecodeError, YAMLerror) is NOT eligible — it's an agent bug."""
    text = "json.decoder.JSONDecodeError: Expecting value"
    for _ in range(eh.CIRCUIT_BREAKER_MAX):
        eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    final = eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    assert final["action"] == "give_up"
    assert final["category"] == "config"
    assert final["runtime_failure_eligible"] is False


def test_check_circuit(tmp_path: Path):
    text = "CUDA out of memory"
    err = eh.structured_error(text)
    assert eh.check_circuit(tmp_path, "t", err.error_id) is False
    for _ in range(eh.CIRCUIT_BREAKER_MAX):
        eh.heal_tick(tmp_path, _CFG, {"id": "t"}, text)
    assert eh.check_circuit(tmp_path, "t", err.error_id) is True


def test_circuit_breaker_is_per_task(tmp_path: Path):
    """The same error signature on a different task keeps its own retry budget."""
    text = "CUDA out of memory"
    for _ in range(eh.CIRCUIT_BREAKER_MAX):
        eh.heal_tick(tmp_path, _CFG, {"id": "task-a"}, text)
    # task-a has exhausted its budget
    assert eh.heal_tick(tmp_path, _CFG, {"id": "task-a"}, text)["action"] == "give_up"
    # task-b — same error, fresh budget
    assert eh.heal_tick(tmp_path, _CFG, {"id": "task-b"}, text)["action"] != "give_up"


# ---- CLI ----------------------------------------------------------------

def test_cli_heal_tick(tmp_path: Path, per_sample_root: Path,
                        monkeypatch, capsys):
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
        {"workspace_path": str(ws.root), "task_id": "eval-x",
         "error_text": "CUDA out of memory"})))
    code = main(["heal-tick"])
    result = json.loads(capsys.readouterr().out)
    assert code == 0
    assert result["action"] == "retry_with_patch"
    assert result["category"] == "oom"
