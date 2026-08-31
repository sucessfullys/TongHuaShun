"""Stage 8 human feedback — serve the review web app, collect, and finalize.

This is the deterministic backbone of ERA Stage 8 (``human_feedback``). It:

- launches the FastAPI review web app as a **detached background process** so it
  outlives the loop iteration that started it (:func:`serve_feedback`);
- reports whether that server is up and whether feedback is finalized
  (:func:`feedback_status`);
- finalizes the operator's feedback — writing ``feedback.json`` and deriving the
  ``human_labels.json`` carry-forward contract (:func:`finalize_feedback`);
- tears the server down (:func:`stop_feedback`);
- resolves a prior iteration's labels for a later Stage 7 (:func:`find_prior_human_labels`).

It imports only ``era.webapp.data`` / ``era.webapp.store`` (both web-framework
free) — never FastAPI — so ``era.cli`` stays importable without the web stack.
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

from ..config import ERAConfig
from ..paths import repo_root
from ..webapp.data import build_review_model, derive_human_labels
from ..webapp.store import (
    feedback_path,
    human_labels_path,
    read_feedback,
    write_feedback,
)
from ..workspace import Workspace, resolve_workspace_root
from .experiment_results import write_json_atomic
from .review_adapter import derive_human_labels_from_model, load_review_model

DEFAULT_PORT = 8731
_PORT_SCAN_END = 8800


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- iteration resolution ----------------------------------------------

def _iter_num(iteration: object) -> int:
    """Coerce ``5`` / ``"5"`` / ``"iter_005"`` to the integer ``5``."""
    if isinstance(iteration, int):
        return iteration
    text = str(iteration).strip()
    if text.startswith("iter_"):
        text = text[len("iter_"):]
    return int(text)


def _resolve(workspace_path: str | Path,
             iteration: object = None) -> tuple[Path, Path, int]:
    """Return ``(workspace_root, iter_dir, iteration_number)``."""
    root = resolve_workspace_root(workspace_path)
    ws = Workspace(root.parent, root.name)
    if iteration in (None, ""):
        iter_dir = ws.iter_path()
    else:
        iter_dir = root / f"iter_{_iter_num(iteration):03d}"
    try:
        num = int(str(iter_dir.name).split("_")[-1])
    except (ValueError, IndexError):
        num = 1
    return root, iter_dir, num


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


def _has_results(iter_dir: Path) -> bool:
    results = iter_dir / "experiments" / "results"
    return (results / "summary.json").is_file() or (
        results / "pilot" / "summary.json").is_file()


# ---- serve --------------------------------------------------------------

def serve_feedback(
    workspace_path: str | Path, iteration: object = None,
    host: str = "127.0.0.1", port: int | None = None,
) -> dict:
    """Start the review web app for one iteration as a detached background server.

    Idempotent: when a live server is already recorded for this iteration its
    record is returned with ``status: "already_running"``.
    """
    root, iter_dir, num = _resolve(workspace_path, iteration)
    if not iter_dir.is_dir():
        return {"error": "no_iteration", "message": f"{iter_dir} does not exist"}
    if not _has_results(iter_dir):
        return {"error": "no_summary",
                "message": "no Stage 6 results in this iteration — "
                           "run the experiment before collecting feedback"}

    human_dir = iter_dir / "human"
    human_dir.mkdir(parents=True, exist_ok=True)
    pidfile = human_dir / "server.pid"
    logfile = human_dir / "server.log"

    existing = _read_pidfile(pidfile)
    if existing and _pid_alive(existing["pid"]) and _port_responds(
            existing.get("host", host), existing["port"]):
        return {"status": "already_running", **existing,
                "iteration_dir": iter_dir.name, "responsive": True,
                "pidfile": str(pidfile), "logfile": str(logfile)}
    if existing and not _pid_alive(existing["pid"]):
        pidfile.unlink(missing_ok=True)

    chosen = _pick_port(host, port or DEFAULT_PORT)
    if chosen is None:
        return {"error": "port_unavailable",
                "message": f"no free port near {DEFAULT_PORT} on {host}"}

    env = {
        **os.environ,
        "ERA_FEEDBACK_WORKSPACE": str(root),
        "ERA_FEEDBACK_ITERATION": str(num),
        "ERA_FEEDBACK_HOST": host,
        "ERA_FEEDBACK_PORT": str(chosen),
    }
    # Detached: a new session (immune to the loop's SIGHUP/SIGINT on exit),
    # stdin closed, output to a logfile — so the server outlives the loop.
    with open(logfile, "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "era.webapp.server"],
            cwd=str(repo_root()), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    url = f"http://{host}:{chosen}/"
    record = {
        "pid": proc.pid, "host": host, "port": chosen, "url": url,
        "workspace_path": str(root), "iteration": num,
        "started_at": _now_iso(),
    }
    write_json_atomic(pidfile, record)

    responsive = _wait_responsive(host, chosen, timeout=20.0)
    if not responsive and not _pid_alive(proc.pid):
        return {"error": "server_failed",
                "message": f"the feedback server exited on startup — see {logfile}",
                "logfile": str(logfile)}
    return {"status": "ok", "responsive": responsive,
            "iteration_dir": iter_dir.name,
            "pidfile": str(pidfile), "logfile": str(logfile), **record}


def feedback_status(workspace_path: str | Path,
                    iteration: object = None) -> dict:
    """Report whether the feedback server is up and whether feedback is final."""
    root, iter_dir, num = _resolve(workspace_path, iteration)
    pidfile = iter_dir / "human" / "server.pid"
    rec = _read_pidfile(pidfile)
    server: dict = {"running": False, "responsive": False,
                    "pid": None, "host": None, "port": None, "url": None}
    if rec:
        alive = _pid_alive(rec["pid"])
        server = {
            "running": alive, "pid": rec["pid"],
            "host": rec.get("host"), "port": rec.get("port"),
            "url": rec.get("url"),
            "responsive": alive and _port_responds(
                rec.get("host", "127.0.0.1"), rec.get("port", 0)),
        }

    fb_path = feedback_path(iter_dir)
    feedback: dict = {"present": fb_path.is_file(), "finalized": False,
                      "finalized_at": None,
                      "labels_present": human_labels_path(iter_dir).is_file()}
    if fb_path.is_file():
        try:
            fb = json.loads(fb_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            fb = {}
        feedback["finalized"] = fb.get("status") == "finalized"
        feedback["finalized_at"] = fb.get("finalized_at")

    return {"status": "ok", "iteration_dir": iter_dir.name,
            "server": server, "feedback": feedback}


# ---- finalize -----------------------------------------------------------

def finalize_feedback(
    workspace_path: str | Path, feedback: dict | None = None,
    iteration: object = None,
) -> dict:
    """Finalize feedback: write ``feedback.json`` + derive ``human_labels.json``.

    ``feedback`` may be supplied inline (the CLI / a script); when omitted the
    current ``feedback.json`` on disk is finalized (the web-app path, which has
    been saving marks incrementally). Idempotent — re-finalizing keeps the
    original ``finalized_at`` so ``human_labels.json`` is byte-stable.
    """
    root, iter_dir, num = _resolve(workspace_path, iteration)
    if not iter_dir.is_dir():
        return {"error": "no_iteration", "message": f"{iter_dir} does not exist"}

    cfg = ERAConfig.from_yaml(root / "config.yaml")
    # Prefer the Stage 8 adapter's normalized review_model.json — so the
    # carry-forward labels match exactly what the human reviewed (including any
    # sub-agent repairs). Fall back to a direct read when it is absent.
    review_model = load_review_model(iter_dir)
    mode = (review_model or {}).get("mode") if review_model else None
    model = None
    if mode is None:
        model = build_review_model(root, iter_dir, cfg)
        mode = model.mode

    existing = read_feedback(iter_dir, num, mode)
    fb = feedback if isinstance(feedback, dict) else existing
    fb.setdefault("schema_version", "1.0")
    fb.setdefault("iteration", num)
    fb.setdefault("iteration_dir", iter_dir.name)
    fb.setdefault("mode", mode)
    fb.setdefault("item_marks", [])
    fb.setdefault("comparison_marks", [])
    fb.setdefault("general_feedback", "")

    prior_final = (existing.get("finalized_at")
                   if existing.get("status") == "finalized" else None)
    fb["status"] = "finalized"
    fb["finalized_at"] = prior_final or fb.get("finalized_at") or _now_iso()

    write_feedback(iter_dir, fb)
    if review_model is not None:
        labels = derive_human_labels_from_model(review_model, fb)
    else:
        if model is None:
            model = build_review_model(root, iter_dir, cfg)
        labels = derive_human_labels(model, fb)
    write_json_atomic(human_labels_path(iter_dir), labels)

    return {
        "status": "ok",
        "iteration_dir": iter_dir.name,
        "feedback_json": str(feedback_path(iter_dir)),
        "human_labels_json": str(human_labels_path(iter_dir)),
        "labels": {
            "flagging_label_count": len(labels["labels"]),
            "comparison_label_count": len(labels["comparison_labels"]),
            "config_summary": labels["config_summary"],
        },
    }


def stop_feedback(workspace_path: str | Path,
                  iteration: object = None) -> dict:
    """Stop the detached feedback server for one iteration and clear its pidfile."""
    root, iter_dir, num = _resolve(workspace_path, iteration)
    pidfile = iter_dir / "human" / "server.pid"
    rec = _read_pidfile(pidfile)
    if not rec:
        return {"status": "not_running", "iteration_dir": iter_dir.name}

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
    return {"status": "ok", "iteration_dir": iter_dir.name,
            "pid": pid, "killed": killed}


# ---- carry-forward ------------------------------------------------------

def find_prior_human_labels(
    workspace_root: str | Path, current_iteration: object,
) -> Path | None:
    """The most recent prior iteration's ``human_labels.json``, or ``None``.

    Walks ``iter_{N-1}`` down to ``iter_001`` and returns the first file found —
    so a later Stage 7 (pre-human comparison) can score candidate protocols
    against the operator's last verdict without re-asking.
    """
    root = Path(workspace_root)
    for n in range(_iter_num(current_iteration) - 1, 0, -1):
        path = root / f"iter_{n:03d}" / "human" / "human_labels.json"
        if path.is_file():
            return path
    return None
