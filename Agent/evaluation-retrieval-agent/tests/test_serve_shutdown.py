"""Tests for ERA Phase D-5 — safe vLLM teardown.

Each test spawns a *real* tiny subprocess instead of mocking the
signal+process machinery — that way the SIGTERM/SIGKILL plumbing is
actually exercised. `subprocess.run` / `nvidia-smi` / `sudo` calls
that we can't (or shouldn't) really fire are patched at the
module-attribute level via monkeypatch.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from era.cli import main
from era.orchestration import serve_shutdown
from era.orchestration.experiment_state import (
    LOGS_REL, init_state, marker_path,
)
from era.workspace import Workspace


# ---- fixtures -----------------------------------------------------------

_PLAN_WITH_SERVE = {
    "schema_version": 1, "iteration": 1, "evaluation_goal": "g",
    "gpu_pool": 4,
    "tasks": [
        {"id": "serve-q72", "type": "serve", "family": "A",
         "mode": "pilot", "depends_on": []},
    ],
}


def _make_workspace(base: Path, *, name: str = "sd-demo") -> Workspace:
    """Minimal workspace + config.yaml + seeded experiment_state."""
    ws = Workspace(base, name)
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({"project_name": name, "stage": "full_experiment",
                     "stage_index": 6, "iteration": 1,
                     "run_state": "running"})
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
        "experiment": {"free_gpu_threshold_mb": 2000},
    }), encoding="utf-8")
    init_state(ws.iter_path(), _PLAN_WITH_SERVE, "pilot")
    return ws


def _stamp_serve_entry(
    ws: Workspace, *, gpu_ids: list[int], served_model_name: str = "vlm-q72",
    endpoint: str = "http://127.0.0.1:18750/v1",
) -> None:
    """Update the seeded experiment_state with the serve task's runtime
    details so shutdown_judge can read them."""
    state_path = ws.iter_path() / "experiments" / "experiment_state.json"
    data = json.loads(state_path.read_text(encoding="utf-8"))
    entry = data["tasks"]["serve-q72"]
    entry.update({
        "gpu_ids": gpu_ids,
        "served_model_name": served_model_name,
        "endpoint": endpoint,
    })
    state_path.write_text(json.dumps(data), encoding="utf-8")


def _spawn_judge(handler: str) -> int:
    """Spawn a tiny child process that mimics a vLLM judge. ``handler``
    chooses how the child responds to SIGTERM:

    - ``"exit"``  — exit cleanly on SIGTERM (the graceful path).
    - ``"ignore"`` — ignore SIGTERM (forces SIGKILL escalation).

    Returns the child PID. Caller is responsible for cleanup.
    """
    if handler == "exit":
        body = (
            "import signal, sys, time\n"
            "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
            "while True: time.sleep(0.5)\n"
        )
    else:
        body = (
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "while True: time.sleep(0.5)\n"
        )
    proc = subprocess.Popen(
        [sys.executable, "-c", body],
        start_new_session=True,  # new pgid — matches the runner template
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Give the SIGTERM handler a moment to install.
    time.sleep(0.1)
    return proc.pid


def _kill_if_alive(pid: int) -> None:
    """Best-effort cleanup for spawned test judges."""
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGKILL)
    except OSError:
        pass


# ---- patch-friendly stubs for subprocess.run calls ---------------------

def _patch_nvidia_smi_used_mb(monkeypatch, sequence: list[int | None]) -> list:
    """Make ``_gpu_used_mb`` return values from ``sequence`` in order.
    Returns a list the test can inspect for call count."""
    calls: list[int] = []
    iterator = iter(sequence)

    def fake(gpu_id: int) -> int | None:
        calls.append(gpu_id)
        try:
            return next(iterator)
        except StopIteration:
            return sequence[-1] if sequence else None

    monkeypatch.setattr(serve_shutdown, "_gpu_used_mb", fake)
    return calls


# ---- 1. Graceful SIGTERM exit ------------------------------------------

def test_graceful_sigterm_exit(tmp_path: Path, monkeypatch):
    """Judge respects SIGTERM and exits within the graceful window.
    No SIGKILL, no GPU reset; status=ok."""
    ws = _make_workspace(tmp_path)
    pid = _spawn_judge("exit")
    try:
        marker_path(ws.iter_path(), "serve-q72", "pid").parent.mkdir(
            parents=True, exist_ok=True)
        marker_path(ws.iter_path(), "serve-q72", "pid").write_text(str(pid))
        _stamp_serve_entry(ws, gpu_ids=[0])
        # GPU "released" immediately after kill.
        _patch_nvidia_smi_used_mb(monkeypatch, [100])
        # Don't actually issue gpu-reset / pkill in tests.
        monkeypatch.setattr(serve_shutdown, "_gpu_reset",
                            lambda gpu_id: True)
        monkeypatch.setattr(serve_shutdown, "_orphan_sweep",
                            lambda name: None)
        # Endpoint not bound (no real server).
        result = serve_shutdown.shutdown_judge(
            ws.root, task_id="serve-q72", graceful_timeout_s=5.0,
        )
        assert result["status"] == "ok", result
        assert result["killed_pids"] == [pid]
        assert result["graceful_exit_s"] < 2.0
        assert "sigterm_pgid" in result["actions"]
        assert "sigkill_pgid" not in result["actions"]
    finally:
        _kill_if_alive(pid)


# ---- 2. SIGKILL escalation on ignored SIGTERM --------------------------

def test_escalates_to_sigkill_on_timeout(tmp_path: Path, monkeypatch):
    """Judge ignores SIGTERM. shutdown_judge escalates to SIGKILL;
    status=escalated_kill."""
    ws = _make_workspace(tmp_path)
    pid = _spawn_judge("ignore")
    try:
        marker_path(ws.iter_path(), "serve-q72", "pid").parent.mkdir(
            parents=True, exist_ok=True)
        marker_path(ws.iter_path(), "serve-q72", "pid").write_text(str(pid))
        _stamp_serve_entry(ws, gpu_ids=[0])
        _patch_nvidia_smi_used_mb(monkeypatch, [100])
        monkeypatch.setattr(serve_shutdown, "_gpu_reset",
                            lambda gpu_id: True)
        orphan_calls: list[str] = []
        monkeypatch.setattr(serve_shutdown, "_orphan_sweep",
                            lambda name: orphan_calls.append(name))
        result = serve_shutdown.shutdown_judge(
            ws.root, task_id="serve-q72",
            graceful_timeout_s=1.0,  # short → forces escalation
        )
        assert result["status"] == "escalated_kill", result
        assert "sigkill_pgid" in result["actions"]
        # Orphan sweep was triggered after SIGKILL.
        assert orphan_calls == ["vlm-q72"]
    finally:
        _kill_if_alive(pid)


# ---- 3. GPU-reset escalation -------------------------------------------

def test_gpu_reset_escalation(tmp_path: Path, monkeypatch):
    """Process exits cleanly but the GPU's memory stays allocated.
    shutdown_judge escalates to sudo nvidia-smi --gpu-reset."""
    ws = _make_workspace(tmp_path)
    pid = _spawn_judge("exit")
    try:
        marker_path(ws.iter_path(), "serve-q72", "pid").parent.mkdir(
            parents=True, exist_ok=True)
        marker_path(ws.iter_path(), "serve-q72", "pid").write_text(str(pid))
        _stamp_serve_entry(ws, gpu_ids=[3])
        # First several polls return >threshold (wedged), then after
        # reset returns 100 (released). free_gpu_threshold_mb=2000.
        _patch_nvidia_smi_used_mb(monkeypatch, [40000, 40000, 40000, 100])
        reset_calls: list[int] = []
        def fake_reset(gpu_id: int) -> bool:
            reset_calls.append(gpu_id)
            return True
        monkeypatch.setattr(serve_shutdown, "_gpu_reset", fake_reset)
        monkeypatch.setattr(serve_shutdown, "_orphan_sweep",
                            lambda name: None)
        result = serve_shutdown.shutdown_judge(
            ws.root, task_id="serve-q72",
            graceful_timeout_s=5.0, gpu_settle_timeout_s=1.5,
        )
        assert result["status"] == "escalated_reset", result
        assert reset_calls == [3]
        assert "gpu_reset_gpu3" in result["actions"]
    finally:
        _kill_if_alive(pid)


# ---- 4. Still-stuck after reset ----------------------------------------

def test_still_stuck_after_reset(tmp_path: Path, monkeypatch):
    """GPU stays wedged even after sudo nvidia-smi --gpu-reset.
    status=still_stuck — hardware issue."""
    ws = _make_workspace(tmp_path)
    pid = _spawn_judge("exit")
    try:
        marker_path(ws.iter_path(), "serve-q72", "pid").parent.mkdir(
            parents=True, exist_ok=True)
        marker_path(ws.iter_path(), "serve-q72", "pid").write_text(str(pid))
        _stamp_serve_entry(ws, gpu_ids=[2])
        # All polls return >threshold, including after reset.
        _patch_nvidia_smi_used_mb(monkeypatch,
                                  [40000, 40000, 40000, 40000])
        monkeypatch.setattr(serve_shutdown, "_gpu_reset",
                            lambda gpu_id: True)
        monkeypatch.setattr(serve_shutdown, "_orphan_sweep",
                            lambda name: None)
        result = serve_shutdown.shutdown_judge(
            ws.root, task_id="serve-q72",
            graceful_timeout_s=5.0, gpu_settle_timeout_s=1.0,
        )
        assert result["status"] == "still_stuck", result
    finally:
        _kill_if_alive(pid)


# ---- 5. Idempotent on missing PID --------------------------------------

def test_idempotent_when_pid_marker_absent(tmp_path: Path, monkeypatch):
    """No .pid marker on disk — task was never registered or already
    cleaned up. Don't crash; release the lease; return ok."""
    ws = _make_workspace(tmp_path)
    _stamp_serve_entry(ws, gpu_ids=[0])
    _patch_nvidia_smi_used_mb(monkeypatch, [100])
    monkeypatch.setattr(serve_shutdown, "_gpu_reset",
                        lambda gpu_id: True)
    monkeypatch.setattr(serve_shutdown, "_orphan_sweep",
                        lambda name: None)
    result = serve_shutdown.shutdown_judge(
        ws.root, task_id="serve-q72",
    )
    assert result["status"] == "ok"
    assert result["killed_pids"] == []
    assert "no_pid_marker" in result["actions"]
    # GPU lease release was attempted.
    assert "release_gpus" in result["actions"]


def test_idempotent_when_pid_already_gone(tmp_path: Path, monkeypatch):
    """PID marker points at a process that has already exited.
    shutdown_judge logs it and proceeds without trying to kill."""
    ws = _make_workspace(tmp_path)
    marker_path(ws.iter_path(), "serve-q72", "pid").parent.mkdir(
        parents=True, exist_ok=True)
    # Use a deliberately-bogus PID that's vanishingly unlikely to exist.
    marker_path(ws.iter_path(), "serve-q72", "pid").write_text("999999")
    _stamp_serve_entry(ws, gpu_ids=[0])
    _patch_nvidia_smi_used_mb(monkeypatch, [100])
    monkeypatch.setattr(serve_shutdown, "_gpu_reset",
                        lambda gpu_id: True)
    monkeypatch.setattr(serve_shutdown, "_orphan_sweep",
                        lambda name: None)
    result = serve_shutdown.shutdown_judge(
        ws.root, task_id="serve-q72",
    )
    assert result["status"] == "ok"
    assert "pid_already_gone" in result["actions"]


# ---- 6. Endpoint parser ------------------------------------------------

def test_parse_endpoint_variants():
    """The endpoint string carried in experiment_state can take a few
    shapes; the parser must handle them."""
    from era.orchestration.serve_shutdown import _parse_endpoint
    assert _parse_endpoint("http://127.0.0.1:8000/v1") == ("127.0.0.1", 8000)
    assert _parse_endpoint("https://localhost:18750/v1/chat/completions") == \
        ("localhost", 18750)
    assert _parse_endpoint("127.0.0.1:9000") == ("127.0.0.1", 9000)
    # Malformed → (host, None) at best.
    assert _parse_endpoint("not-a-url") == ("not-a-url", None)
    assert _parse_endpoint("") == (None, None)


# ---- 7. CLI round-trip -------------------------------------------------

def test_cli_shutdown_judge_round_trip(
    tmp_path: Path, monkeypatch, capsys,
):
    """End-to-end CLI: stdin JSON → exit code + stdout result."""
    ws = _make_workspace(tmp_path)
    pid = _spawn_judge("exit")
    try:
        marker_path(ws.iter_path(), "serve-q72", "pid").parent.mkdir(
            parents=True, exist_ok=True)
        marker_path(ws.iter_path(), "serve-q72", "pid").write_text(str(pid))
        _stamp_serve_entry(ws, gpu_ids=[0])
        _patch_nvidia_smi_used_mb(monkeypatch, [100])
        monkeypatch.setattr(serve_shutdown, "_gpu_reset",
                            lambda gpu_id: True)
        monkeypatch.setattr(serve_shutdown, "_orphan_sweep",
                            lambda name: None)
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
            "workspace_path": str(ws.root), "task_id": "serve-q72",
            "graceful_timeout_s": 5.0,
        })))
        rc = main(["shutdown-judge"])
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["status"] == "ok"
        assert payload["killed_pids"] == [pid]
    finally:
        _kill_if_alive(pid)


def test_cli_shutdown_judge_missing_task_id(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "workspace_path": "/tmp/whatever",
    })))
    rc = main(["shutdown-judge"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["error"] == "missing_task_id"


def test_cli_shutdown_judge_bad_timeout(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "workspace_path": "/tmp/whatever",
        "task_id": "serve-x",
        "graceful_timeout_s": "thirty",
    })))
    rc = main(["shutdown-judge"])
    payload = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert payload["error"] == "bad_graceful_timeout_s"
