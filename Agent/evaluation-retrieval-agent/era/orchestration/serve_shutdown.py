"""Phase D-5 — safe vLLM teardown for Stage 6 judges.

Stage 6 brings up VLM judges as detached background processes via
``ms-swift`` / vLLM. Raw ``kill <pid>`` (the pre-D-5 teardown) leaks
in three real failure modes:

- Tensor-parallel workers orphaned (parent SIGKILL doesn't reap the
  N spawned model-executor processes).
- GPU compute context wedged (the "phantom process" / "zombie GPU"
  pattern Phase D-4 added the manual reset for).
- Serving port stuck in TIME_WAIT.

This module ships a deterministic graceful shutdown sequence the
Stage 6 prompt calls via ``era.cli shutdown-judge`` whenever a judge's
``teardown_after`` evals all complete. The sequence:

1. SIGTERM the process group (pgid). The runner template launches
   vLLM with ``start_new_session=True`` so parent + tp-workers share
   one pgid — pgid signalling is the only way to reap orphaned
   tp-workers reliably.
2. Poll-exit at 250 ms cadence up to ``graceful_timeout_s``.
3. On timeout: SIGKILL the pgid; ``pkill -TERM -f <served_model_name>``
   as the orphan sweep.
4. Verify GPU memory release via ``nvidia-smi``; if still wedged after
   ``gpu_settle_timeout_s``, escalate to ``sudo nvidia-smi --gpu-reset
   -i <gpu_id>`` (Phase D-4 pre-approved this pattern).
5. Verify port unbinding (advisory only — TIME_WAIT is transient).
6. Drop the GPU lease via :func:`release_gpus`.

Idempotent: re-calling on a task whose PID is already gone returns
``status: "ok"`` with ``killed_pids: []``.
"""

from __future__ import annotations

import errno
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from ..config import ERAConfig
from ..workspace import Workspace, resolve_workspace_root
from .experiment_state import LOGS_REL, load_state, marker_path
from .gpu_scheduler import release_gpus


_POLL_INTERVAL_S = 0.25
_GPU_POLL_INTERVAL_S = 0.5


def _pid_alive(pid: int) -> bool:
    """True iff ``pid`` is a live (non-zombie) process.

    ``os.kill(pid, 0)`` alone returns success for **zombies** — processes
    that have exited but haven't been reaped by their parent yet. In
    production that's rare because vLLM is reparented to PID 1 (init)
    when its runner exits, and init reaps it promptly. But in tests
    (where pytest is the parent and doesn't auto-reap) the brief zombie
    window made graceful-exit checks look like timeouts. Read
    ``/proc/<pid>/status`` to detect the ``Z`` (zombie) state on Linux.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as exc:
        return exc.errno != errno.ESRCH
    # Process responds to signal-0 — verify it isn't a zombie via /proc.
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("State:"):
                    # "State:\tZ (zombie)" → take the letter.
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "Z":
                        return False
                    break
    except OSError:
        # /proc unavailable (non-Linux) or race condition — fall through
        # and treat signal-0 success as alive.
        pass
    return True


def _read_pid(iter_dir: Path, task_id: str) -> int | None:
    """Read the runner's ``.pid`` marker as an int; ``None`` if absent
    or malformed."""
    pid_path = marker_path(iter_dir, task_id, "pid")
    if not pid_path.is_file():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _killpg(pid: int, sig: int) -> bool:
    """Send ``sig`` to the process group containing ``pid``. Returns
    True on success, False if the pgid couldn't be resolved (process
    already gone) or the signal can't be sent."""
    try:
        pgid = os.getpgid(pid)
    except OSError:
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except OSError:
        return False


def _wait_for_exit(pid: int, timeout_s: float) -> float:
    """Poll ``_pid_alive(pid)`` until it's gone or ``timeout_s`` elapses.
    Returns the elapsed seconds (≤ timeout)."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if not _pid_alive(pid):
            return time.monotonic() - start
        time.sleep(_POLL_INTERVAL_S)
    return time.monotonic() - start


def _gpu_used_mb(gpu_id: int) -> int | None:
    """Query ``nvidia-smi`` for the used memory (MB) on one GPU.
    Returns ``None`` when nvidia-smi is absent or fails."""
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used",
             "--format=csv,noheader,nounits",
             "-i", str(gpu_id)],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def _port_bound(host: str, port: int) -> bool:
    """True iff ``(host, port)`` accepts a TCP connect — i.e. the
    serving port is still bound."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _orphan_sweep(served_model_name: str) -> None:
    """Send SIGTERM to any process whose cmdline matches the served
    model name. Best-effort: a missing ``pkill`` or zero matches is
    silently OK."""
    if not served_model_name:
        return
    try:
        subprocess.run(
            ["pkill", "-TERM", "-f", served_model_name],
            capture_output=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return


def _gpu_reset(gpu_id: int) -> bool:
    """Issue ``sudo nvidia-smi --gpu-reset -i <gpu_id>`` (Phase D-4
    pre-approved). Returns True on success."""
    try:
        out = subprocess.run(
            ["sudo", "nvidia-smi", "--gpu-reset", "-i", str(gpu_id)],
            capture_output=True, text=True, timeout=30,
        )
        return out.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def shutdown_judge(
    workspace_path: str | Path, *,
    task_id: str,
    graceful_timeout_s: float = 30.0,
    gpu_settle_timeout_s: float = 10.0,
) -> dict:
    """Safely tear down a Stage 6 serve task.

    Returns a structured outcome the Stage 6 prompt branches on:

    - ``status: "ok"`` — graceful SIGTERM exit (or task already gone).
    - ``status: "escalated_kill"`` — needed SIGKILL but cleaned up.
    - ``status: "escalated_reset"`` — GPU stayed allocated; triggered
      ``sudo nvidia-smi --gpu-reset``.
    - ``status: "still_stuck"`` — GPU stayed allocated even after
      the reset. Hardware-level issue; the Stage 6 prompt records
      this as ``runtime_failed`` and the heal loop takes over.

    Idempotent on a missing PID or a task whose PID has already
    exited.
    """
    root = resolve_workspace_root(workspace_path)
    ws = Workspace(root.parent, root.name)
    if not ws.exists():
        return {"error": "not_a_workspace",
                "message": f"{root} has no status.json"}

    cfg = ERAConfig.from_yaml(root / "config.yaml")
    free_threshold_mb = cfg.experiment.free_gpu_threshold_mb

    iter_dir = ws.iter_path()
    state = load_state(iter_dir)
    entry = state.tasks.get(task_id) or {}
    gpu_ids: list[int] = [int(g) for g in (entry.get("gpu_ids") or [])]
    endpoint = entry.get("endpoint") or ""
    served_model_name = entry.get("served_model_name") or ""

    actions: list[str] = []
    killed_pids: list[int] = []
    graceful_exit_s = 0.0
    pid = _read_pid(iter_dir, task_id)

    # ---- Step 1 + 2 + 3: SIGTERM the pgid, poll, escalate to SIGKILL.
    needed_kill = False
    if pid is None:
        actions.append("no_pid_marker")
    elif not _pid_alive(pid):
        actions.append("pid_already_gone")
    else:
        actions.append("sigterm_pgid")
        sent = _killpg(pid, signal.SIGTERM)
        if not sent:
            # Fall back to single-pid SIGTERM if pgid resolution failed.
            try:
                os.kill(pid, signal.SIGTERM)
                actions.append("sigterm_pid_fallback")
            except OSError:
                pass
        graceful_exit_s = _wait_for_exit(pid, graceful_timeout_s)
        if _pid_alive(pid):
            actions.append("sigkill_pgid")
            needed_kill = True
            if not _killpg(pid, signal.SIGKILL):
                try:
                    os.kill(pid, signal.SIGKILL)
                    actions.append("sigkill_pid_fallback")
                except OSError:
                    pass
            # Brief settle window after SIGKILL.
            _wait_for_exit(pid, 2.0)
            if served_model_name:
                actions.append("orphan_sweep_pkill")
                _orphan_sweep(served_model_name)
        killed_pids.append(pid)

    # ---- Step 4: Verify GPU memory release; escalate to gpu-reset.
    final_gpu_mb: dict[int, int | None] = {}
    needed_reset = False
    still_stuck = False
    for gpu_id in gpu_ids:
        # Poll until below threshold or timeout.
        start = time.monotonic()
        used: int | None = None
        while time.monotonic() - start < gpu_settle_timeout_s:
            used = _gpu_used_mb(gpu_id)
            if used is None or used < free_threshold_mb:
                break
            time.sleep(_GPU_POLL_INTERVAL_S)
        final_gpu_mb[gpu_id] = used
        if used is not None and used >= free_threshold_mb:
            # Stuck — escalate.
            actions.append(f"gpu_reset_gpu{gpu_id}")
            ok = _gpu_reset(gpu_id)
            needed_reset = True
            # Brief settle after reset.
            time.sleep(1.0)
            post = _gpu_used_mb(gpu_id)
            final_gpu_mb[gpu_id] = post
            if not ok or (post is not None and post >= free_threshold_mb):
                still_stuck = True

    # ---- Step 5: Port verify (advisory; TIME_WAIT is transient).
    port_released = True
    if endpoint:
        host, port = _parse_endpoint(endpoint)
        if host and port:
            actions.append("port_check")
            if _port_bound(host, port):
                # Tiny retry — sometimes the kernel needs a moment.
                time.sleep(1.0)
                port_released = not _port_bound(host, port)

    # ---- Step 6: Drop the GPU lease (idempotent at the scheduler).
    actions.append("release_gpus")
    release_gpus(str(root), [task_id])

    # Compose the final status.
    if still_stuck:
        status = "still_stuck"
    elif needed_reset:
        status = "escalated_reset"
    elif needed_kill:
        status = "escalated_kill"
    else:
        status = "ok"

    return {
        "status": status,
        "task_id": task_id,
        "killed_pids": killed_pids,
        "graceful_exit_s": round(graceful_exit_s, 3),
        "final_gpu_mb": {str(k): v for k, v in final_gpu_mb.items()},
        "port_released": port_released,
        "actions": actions,
    }


def _parse_endpoint(endpoint: str) -> tuple[str | None, int | None]:
    """Parse a serving endpoint like ``http://127.0.0.1:8000/v1`` into
    ``(host, port)``. Returns ``(None, None)`` on a malformed value."""
    # Strip scheme.
    rest = endpoint
    for prefix in ("http://", "https://"):
        if rest.startswith(prefix):
            rest = rest[len(prefix):]
            break
    # Strip path.
    rest = rest.split("/", 1)[0]
    if ":" not in rest:
        return rest or None, None
    host, _, port_str = rest.rpartition(":")
    try:
        return host or None, int(port_str)
    except ValueError:
        return host or None, None
