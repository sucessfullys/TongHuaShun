"""Lifecycle backbone for the ``/era:annotate`` image-annotation web app.

This is the deterministic counterpart to :mod:`era.orchestration.human_feedback`
but for a *standalone* dataset annotation flow — no ``ERAConfig``, no
``status.json``, no iteration. The operator points the app at a dataset on
disk; the server runs detached; per-sample annotations land at
``<dataset_root>/<sample_key>__annotation.json`` for later pickup by Stage 7
and Stage 9.

Pattern matches Stage 8's server lifecycle exactly: try a preferred port,
scan a range, launch ``python -m era.annotate.server`` with
``start_new_session=True``, poll ``/api/health`` for readiness, write a PID
file at ``<dataset_root>/.annotate_server.pid``. ``stop_annotate`` reads the
PID file and kills the process group.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..paths import repo_root
from .experiment_results import write_json_atomic

DEFAULT_PORT = 8801
_PORT_SCAN_END = 8900
PIDFILE_NAME = ".annotate_server.pid"
LOGFILE_NAME = ".annotate_server.log"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- process / port helpers --------------------------------------------

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _port_responds(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def _pick_port(host: str, preferred: int) -> int | None:
    for port in [preferred, *range(DEFAULT_PORT, _PORT_SCAN_END)]:
        if _port_free(host, port):
            return port
    return None


def _wait_responsive(host: str, port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _port_responds(host, port):
            return True
        time.sleep(0.3)
    return False


def _read_pidfile(pidfile: Path) -> dict | None:
    if not pidfile.is_file():
        return None
    try:
        rec = json.loads(pidfile.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return rec if isinstance(rec, dict) and rec.get("pid") else None


# ---- serve --------------------------------------------------------------

def _validate_dataset(dataset_root: Path) -> dict | None:
    """Return an error dict iff ``dataset_root`` is missing or empty, else None."""
    if not dataset_root.is_dir():
        return {
            "error": "no_dataset",
            "message": f"dataset root does not exist: {dataset_root}",
        }
    method_dirs = [
        p for p in dataset_root.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    ]
    if not method_dirs:
        return {
            "error": "empty_dataset",
            "message": (
                f"dataset root {dataset_root} has no method subdirectories — "
                f"expected layout <dataset>/<method>/<sample>/..."
            ),
        }
    return None


def serve_annotate(
    dataset_root: str | Path, *,
    host: str = "127.0.0.1", port: int | None = None,
    output_overrides: dict[str, str] | None = None,
    input_role_overrides: dict[str, str] | None = None,
) -> dict:
    """Launch the annotation web app for a dataset as a detached server.

    Idempotent: when a live server is already recorded for this dataset its
    record is returned with ``status: "already_running"``.

    ``output_overrides`` / ``input_role_overrides`` are operator-confirmed
    disambiguations from the slash command (when the Stage 0 probe
    returned ``needs_confirmation``). They are passed to the spawned
    server via JSON-encoded env vars so they survive the Popen boundary,
    and recorded on the pidfile for audit.
    """
    dataset = Path(dataset_root).resolve()
    err = _validate_dataset(dataset)
    if err is not None:
        return err

    pidfile = dataset / PIDFILE_NAME
    logfile = dataset / LOGFILE_NAME

    existing = _read_pidfile(pidfile)
    if existing and _pid_alive(existing["pid"]) and _port_responds(
            existing.get("host", host), existing["port"]):
        return {
            "status": "already_running",
            **existing,
            "dataset_root": str(dataset),
            "responsive": True,
            "pidfile": str(pidfile),
            "logfile": str(logfile),
        }
    if existing and not _pid_alive(existing["pid"]):
        pidfile.unlink(missing_ok=True)

    chosen = _pick_port(host, port or DEFAULT_PORT)
    if chosen is None:
        return {
            "error": "port_unavailable",
            "message": (
                f"no free port near {DEFAULT_PORT} on {host} "
                f"(scanned {DEFAULT_PORT}-{_PORT_SCAN_END - 1})"
            ),
        }

    env = {
        **os.environ,
        "ERA_ANNOTATE_DATASET_ROOT": str(dataset),
        "ERA_ANNOTATE_HOST": host,
        "ERA_ANNOTATE_PORT": str(chosen),
    }
    if output_overrides:
        env["ERA_ANNOTATE_OUTPUT_OVERRIDES"] = json.dumps(output_overrides)
    if input_role_overrides:
        env["ERA_ANNOTATE_INPUT_ROLE_OVERRIDES"] = json.dumps(
            input_role_overrides)
    # Detached: a new session (immune to the launcher's SIGHUP/SIGINT on
    # exit), stdin closed, output to a logfile — so the server outlives the
    # slash-command invocation.
    with open(logfile, "ab") as log:
        proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "era.annotate.server"],
            cwd=str(repo_root()), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    url = f"http://{host}:{chosen}/"
    record: dict = {
        "pid": proc.pid, "host": host, "port": chosen, "url": url,
        "dataset_root": str(dataset), "started_at": _now_iso(),
    }
    if output_overrides:
        record["output_overrides"] = dict(output_overrides)
    if input_role_overrides:
        record["input_role_overrides"] = dict(input_role_overrides)
    write_json_atomic(pidfile, record)

    responsive = _wait_responsive(host, chosen, timeout=20.0)
    if not responsive and not _pid_alive(proc.pid):
        return {
            "error": "server_failed",
            "message": (
                f"the annotation server exited on startup — see {logfile}"
            ),
            "logfile": str(logfile),
        }
    return {
        "status": "ok",
        "responsive": responsive,
        "pidfile": str(pidfile),
        "logfile": str(logfile),
        **record,
    }


def annotate_status(dataset_root: str | Path) -> dict:
    """Report whether the annotation server is up for this dataset."""
    dataset = Path(dataset_root).resolve()
    pidfile = dataset / PIDFILE_NAME
    rec = _read_pidfile(pidfile)
    if not rec:
        return {
            "status": "ok", "dataset_root": str(dataset),
            "server": {"running": False, "responsive": False,
                       "pid": None, "host": None, "port": None, "url": None},
        }
    alive = _pid_alive(rec["pid"])
    return {
        "status": "ok", "dataset_root": str(dataset),
        "server": {
            "running": alive, "pid": rec["pid"],
            "host": rec.get("host"), "port": rec.get("port"),
            "url": rec.get("url"),
            "responsive": alive and _port_responds(
                rec.get("host", "127.0.0.1"), rec.get("port", 0)),
            "pidfile_stale": (not alive) and bool(rec),
        },
    }


def stop_annotate(dataset_root: str | Path) -> dict:
    """Stop the detached annotation server for this dataset and clear its pidfile."""
    dataset = Path(dataset_root).resolve()
    pidfile = dataset / PIDFILE_NAME
    rec = _read_pidfile(pidfile)
    if not rec:
        return {"status": "not_running", "dataset_root": str(dataset)}

    killed = False
    pid = rec["pid"]
    if _pid_alive(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            killed = True
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGTERM)
                killed = True
            except ProcessLookupError:
                pass
    pidfile.unlink(missing_ok=True)
    return {"status": "ok", "dataset_root": str(dataset),
            "pid": pid, "killed": killed}
