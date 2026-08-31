"""Stage 6 experiment state — task lifecycle, marker files, recovery.

``experiment_state.json`` (at ``iter_NNN/experiments/experiment_state.json``) is
the **single source of truth** for the Stage 6 task DAG: every task's status
(``pending`` / ``running`` / ``completed`` / ``failed`` / ``skipped``), the GPUs
it holds, and any served endpoint. The GPU scheduler (``gpu_scheduler.py``) and
the recovery loop both read and write it; all writes go through an fcntl lock so
two pollers never corrupt it.

Running tasks report progress out-of-band via **marker files** under
``iter_NNN/experiments/logs/``:

- ``<task_id>.pid``          — the runner's PID, written first.
- ``<task_id>.progress.json`` — periodic ``{done, total, updated_at}``.
- ``<task_id>.done.json``     — written last, on success **and** failure:
  ``{task_id, status, exit_code, summary, result_dir?, endpoint?, finished_at}``.

``detection_script`` generates a pure-bash scan of those markers; the Stage 6
loop runs it every poll and pipes the output through ``parse_detection_output``
→ ``recover_from_detection`` to advance the state.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fcntl
import json
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import error_heal

STATE_REL = "experiments/experiment_state.json"
_LOCK_REL = "experiments/.experiment_state.lock"
LOGS_REL = "experiments/logs"

TASK_STATUSES = ("pending", "running", "completed", "failed", "skipped")
# Statuses that count as resolved for dependency / completion purposes.
_RESOLVED = ("completed", "failed", "skipped")
_DEFAULT_MAX_RETRIES = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_age_seconds(iso_str: str) -> float | None:
    """Seconds elapsed since an ISO timestamp, or ``None`` if it is empty/bad."""
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return round((datetime.now(timezone.utc) - dt).total_seconds(), 1)


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON via a temp file + ``os.replace`` (readers never see a partial)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


@contextlib.contextmanager
def _state_lock(iter_dir: Path):
    """Exclusive fcntl lock over one iteration's experiment state."""
    lock_path = iter_dir / _LOCK_REL
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lock_path, "w", encoding="utf-8")  # noqa: SIM115
    try:
        fcntl.flock(fh, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


@dataclass
class ExperimentState:
    """The deserialized ``experiment_state.json``."""
    schema_version: int = 1
    mode: str = "pilot"
    tasks: dict = field(default_factory=dict)
    recovery_log: list = field(default_factory=list)
    last_recovery_at: str = ""
    created_at: str = ""
    updated_at: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "ExperimentState":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _state_path(iter_dir: Path) -> Path:
    return iter_dir / STATE_REL


def load_state(iter_dir: Path) -> ExperimentState:
    """Read ``experiment_state.json``; a missing file yields a fresh state."""
    path = _state_path(iter_dir)
    if path.is_file():
        return ExperimentState.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    return ExperimentState()


def save_state(iter_dir: Path, state: ExperimentState) -> None:
    """Persist a state object atomically (caller holds ``_state_lock``)."""
    state.updated_at = _now_iso()
    if not state.created_at:
        state.created_at = state.updated_at
    _write_json_atomic(_state_path(iter_dir), state.to_dict())


def marker_path(iter_dir: Path, task_id: str, suffix: str) -> Path:
    """Path of a runner marker file (``pid`` / ``progress.json`` / ``done.json``)."""
    return iter_dir / LOGS_REL / f"{task_id}.{suffix}"


_WAIT_POLL_INTERVAL_S = 0.25
_WAIT_MIN_TIMEOUT_S = 0.0
_WAIT_MAX_TIMEOUT_S = 3600.0


def wait_for_any_done(
    iter_dir: Path, *, task_ids: list[str], timeout_s: float,
) -> dict:
    """Block until any ``task_id``'s ``done.json`` marker appears, or timeout.

    Stats each task's marker file every ``_WAIT_POLL_INTERVAL_S`` seconds.
    Returns the moment ANY marker first exists — even if multiple appear in
    the same poll, all of them are reported. The Stage 6 loop uses this to
    re-claim freed GPUs immediately on the first task completion instead of
    sleeping for ``poll_interval_s`` between cadence ticks.

    A marker that already exists when ``wait_for_any_done`` is called is
    reported immediately on the first poll — this is intentional so the
    caller never blocks on a task that finished before the wait began.
    """
    if not task_ids:
        return {"done": [], "still_running": [], "elapsed_s": 0.0,
                "timed_out": False}

    timeout_s = max(_WAIT_MIN_TIMEOUT_S, min(_WAIT_MAX_TIMEOUT_S, float(timeout_s)))

    logs_dir = iter_dir / LOGS_REL
    marker_paths = {
        tid: logs_dir / f"{tid}.done.json" for tid in task_ids
    }

    start = time.monotonic()
    while True:
        done = [tid for tid, p in marker_paths.items() if p.is_file()]
        elapsed = time.monotonic() - start
        if done:
            still_running = [tid for tid in task_ids if tid not in set(done)]
            return {"done": done, "still_running": still_running,
                    "elapsed_s": round(elapsed, 3), "timed_out": False}
        if elapsed >= timeout_s:
            return {"done": [], "still_running": list(task_ids),
                    "elapsed_s": round(elapsed, 3), "timed_out": True}
        # Sleep a bounded amount that never overshoots the timeout deadline.
        remaining = timeout_s - elapsed
        time.sleep(min(_WAIT_POLL_INTERVAL_S, max(0.0, remaining)))


def init_state(
    iter_dir: Path,
    task_plan: dict,
    mode: str,
    skip: tuple[dict, ...] = (),
    *,
    valid_pivot_actions: set[str] | None = None,
    auto_validate_skips: tuple[str, ...] = (),
) -> dict:
    """Seed ``experiment_state.json`` for one experiment pass.

    Tasks of the requested ``mode`` (plus mode-agnostic ``setup`` tasks) are
    seeded ``pending``; any task already ``completed`` is preserved (so the
    pilot pass survives into the full pass).

    ``skip`` is a list of ``{"task_id", "pivot_proof", "skip_reason"?}``
    entries. Each entry's task is seeded ``skipped`` and stamped with the
    caller-supplied ``skip_proof`` / ``skip_reason``. For ``eval`` tasks the
    ``pivot_proof`` must match one of the Stage 4 brief's
    ``pivot_matrix[*].action`` strings (passed in via ``valid_pivot_actions``);
    a mismatch returns an error and leaves the state untouched, so the runtime
    cannot back-door a silent scope-reduction. Non-eval tasks (``serve`` /
    ``setup`` / ``aggregate`` / ``compare``) accept an empty or missing
    ``pivot_proof`` — those are legitimate orchestration decisions.

    ``auto_validate_skips`` (Phase C-2) is a list of ``combination_id``
    strings whose tasks in the requested ``mode`` should be seeded
    ``skipped`` with ``skip_reason: auto_validate_failed``. Each cid MUST
    appear in ``<iter>/auto_validate/result.json``'s ``failing_configs``
    field — the on-disk artifact is the authorization. This is the second
    authorized skip path alongside the Stage 4 pivot matrix; both routes
    leave a verifiable record on disk so the runtime cannot fabricate a
    silent scope-reduction.
    """
    plan_tasks = task_plan.get("tasks", []) or []
    plan_kind: dict[str, str | None] = {}
    for task in plan_tasks:
        tid = task.get("id")
        if tid:
            plan_kind[tid] = task.get("type")

    # Phase C-2: expand auto_validate_skips into per-task entries by
    # walking the plan for tasks whose (mode, combination_id) match a
    # listed cid. Validates against the on-disk
    # <iter>/auto_validate/result.json so the runtime cannot fabricate
    # this scope reduction.
    av_skip_task_ids: set[str] = set()
    if auto_validate_skips:
        av_path = iter_dir / "auto_validate" / "result.json"
        if not av_path.is_file():
            return {
                "error": "missing_auto_validate_result",
                "message": (
                    "auto_validate_skips requires "
                    "<iter>/auto_validate/result.json on disk"
                ),
                "expected_path": str(av_path),
            }
        try:
            av_data = json.loads(av_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return {
                "error": "bad_auto_validate_result",
                "message": str(exc),
            }
        failing = set(av_data.get("failing_configs") or [])
        unauthorized = [c for c in auto_validate_skips if c not in failing]
        if unauthorized:
            return {
                "error": "unauthorized_auto_validate_skip",
                "message": (
                    "auto_validate_skips combination_ids must appear in "
                    "auto_validate/result.json.failing_configs"
                ),
                "configs": unauthorized,
                "failing_configs": sorted(failing),
            }
        av_targets = set(auto_validate_skips)
        for task in plan_tasks:
            tid = task.get("id")
            if not tid:
                continue
            if (task.get("type") in ("eval", "aggregate")
                    and task.get("mode") == mode
                    and task.get("combination_id") in av_targets):
                av_skip_task_ids.add(tid)

    parsed_skip: dict[str, dict] = {}
    for tid in av_skip_task_ids:
        parsed_skip[tid] = {
            "skip_proof": "auto_validate_failed",
            "skip_reason": "auto_validate_failed",
        }

    for raw in skip:
        if isinstance(raw, str):
            return {
                "error": "bad_skip_entry",
                "message": (
                    "skip entries must be objects "
                    "{task_id, pivot_proof, skip_reason?} — bare task ids "
                    "are no longer accepted, so an unauthorized scope "
                    "reduction cannot slip through"
                ),
                "task_id": raw,
            }
        if not isinstance(raw, dict):
            return {"error": "bad_skip_entry",
                    "message": "skip entries must be objects"}
        tid = raw.get("task_id")
        if not tid:
            return {"error": "bad_skip_entry",
                    "message": "skip entry missing 'task_id'"}
        if tid not in plan_kind:
            return {"error": "unknown_skip_task", "task_id": tid,
                    "message": "skip task id not in task_plan"}
        proof = (raw.get("pivot_proof") or "").strip()
        kind = plan_kind.get(tid)
        if kind == "eval":
            if not proof:
                return {
                    "error": "unauthorized_skip", "task_id": tid,
                    "message": (
                        "an eval task may only be skipped at init with a "
                        "pivot_proof matching a Stage 4 "
                        "experiment_brief.pivot_matrix[*].action"
                    ),
                }
            if valid_pivot_actions and proof not in valid_pivot_actions:
                return {
                    "error": "unauthorized_skip", "task_id": tid,
                    "message": (
                        f"pivot_proof {proof!r} does not match any "
                        f"experiment_brief.pivot_matrix[*].action"
                    ),
                    "valid_actions": sorted(valid_pivot_actions),
                }
        parsed_skip[tid] = {
            "skip_proof": proof or None,
            "skip_reason": raw.get("skip_reason") or "",
        }

    with _state_lock(iter_dir):
        state = load_state(iter_dir)
        state.mode = mode
        for task in plan_tasks:
            tid = task.get("id")
            if not tid:
                continue
            tmode = task.get("mode")
            in_scope = tmode == mode or tmode is None
            if not in_scope:
                continue
            existing = state.tasks.get(tid)
            if tid in parsed_skip:
                status = "skipped"
            elif existing and existing.get("status") == "completed":
                status = "completed"
            else:
                status = "pending"
            entry = existing or {}
            entry.update({
                "kind": task.get("type"),
                "status": status,
                "gpu_ids": entry.get("gpu_ids", []),
                "retry_count": entry.get("retry_count", 0),
                "error_summary": entry.get("error_summary", ""),
            })
            if tid in parsed_skip:
                meta = parsed_skip[tid]
                if meta["skip_proof"]:
                    entry["skip_proof"] = meta["skip_proof"]
                entry["skip_reason"] = meta["skip_reason"]
            state.tasks[tid] = entry
        save_state(iter_dir, state)
        return {"status": "ok", "mode": mode, "task_count": len(state.tasks)}


def register_running(iter_dir: Path, assignments: list[dict]) -> dict:
    """Mark a batch of tasks ``running`` with their assigned GPUs.

    ``assignments`` — ``[{task_id, gpu_ids, kind?, tensor_parallel?}]``. Called
    by ``gpu_scheduler.claim_batch`` while it holds the global GPU-lease lock.
    """
    with _state_lock(iter_dir):
        state = load_state(iter_dir)
        registered: list[str] = []
        for a in assignments:
            tid = a.get("task_id")
            if not tid:
                continue
            entry = state.tasks.get(tid, {})
            entry.update({
                "kind": a.get("kind", entry.get("kind")),
                "status": "running",
                "gpu_ids": list(a.get("gpu_ids", [])),
                "pid_file": f"{LOGS_REL}/{tid}.pid",
                "registered_at": _now_iso(),
            })
            if a.get("tensor_parallel") is not None:
                entry["tensor_parallel"] = a["tensor_parallel"]
            state.tasks[tid] = entry
            registered.append(tid)
        save_state(iter_dir, state)
        return {"status": "ok", "registered": registered}


def complete_task(
    iter_dir: Path, task_id: str, *, endpoint: dict | None = None,
    result_dir: str | None = None,
) -> dict:
    """Mark a task ``completed`` and release its GPUs from the state."""
    with _state_lock(iter_dir):
        state = load_state(iter_dir)
        entry = state.tasks.get(task_id)
        if entry is None:
            return {"error": "unknown_task", "task_id": task_id}
        entry["status"] = "completed"
        entry["finished_at"] = _now_iso()
        entry["gpu_ids"] = []
        if endpoint is not None:
            entry["endpoint"] = endpoint
        if result_dir is not None:
            entry["result_dir"] = result_dir
        save_state(iter_dir, state)
        return {"status": "ok", "task_id": task_id, "task": entry}


def fail_task(iter_dir: Path, task_id: str, *, error_summary: str = "") -> dict:
    """Mark a task ``failed`` and release its GPUs from the state."""
    with _state_lock(iter_dir):
        state = load_state(iter_dir)
        entry = state.tasks.get(task_id)
        if entry is None:
            return {"error": "unknown_task", "task_id": task_id}
        entry["status"] = "failed"
        entry["finished_at"] = _now_iso()
        entry["gpu_ids"] = []
        entry["error_summary"] = error_summary
        save_state(iter_dir, state)
        return {"status": "ok", "task_id": task_id, "task": entry}


def runtime_fail_task(
    iter_dir: Path, task_id: str, *,
    failure_category: str, heal_history: dict,
) -> dict:
    """Mark an ``eval`` task ``runtime_failed`` — the deterministic forward path
    for infrastructure failures the heal-tick circuit breaker gave up on.

    Unlike ``fail_task``, the Stage 6 completion gate counts a
    ``runtime_failed`` config as **resolved** rather than as a missing
    config — so the loop advances instead of blocking. To prevent the agent
    from using this as a silent scope-reduction lever, every parameter is
    re-validated on the server side:

    - ``failure_category`` must be in :data:`error_heal.RUNTIME_FAILURE_CATEGORIES`.
    - ``heal_history`` must carry ``circuit_broken: true`` and an ``error_id``.
    - The on-disk heal-state must show the circuit-breaker open for
      ``(task_id, error_id)`` — i.e. heal-tick actually fired give_up.

    Caller (the Stage 6 record-task CLI path) verifies the task is type
    ``eval``; this function rejects with ``not_eval_task`` as a defense in
    depth.
    """
    if not failure_category:
        return {"error": "runtime_failed_missing_category",
                "task_id": task_id,
                "message": "runtime_fail_task requires 'failure_category'"}
    if failure_category not in error_heal.RUNTIME_FAILURE_CATEGORIES:
        return {
            "error": "runtime_failed_category_not_eligible",
            "task_id": task_id,
            "failure_category": failure_category,
            "eligible_categories": sorted(
                error_heal.RUNTIME_FAILURE_CATEGORIES
            ),
            "message": (
                f"failure_category {failure_category!r} is not in the "
                f"runtime-failure taxonomy — only infra failures the agent "
                f"cannot retry (oom / serving / hung) are runtime-failable; "
                f"others must be recorded as outcome=failure"
            ),
        }
    if not isinstance(heal_history, dict):
        return {"error": "runtime_failed_no_heal_history",
                "task_id": task_id,
                "message": "heal_history must be the heal-tick give_up envelope"}
    error_id = (heal_history.get("error_id") or "").strip()
    circuit_broken = bool(heal_history.get("circuit_broken"))
    if not error_id or not circuit_broken:
        return {
            "error": "runtime_failed_no_heal_history",
            "task_id": task_id,
            "message": (
                "heal_history must carry circuit_broken=true and a non-empty "
                "error_id from the heal-tick give_up envelope"
            ),
        }
    # Re-validate against the on-disk circuit breaker — trusting the caller's
    # envelope alone would let a runner fabricate a give-up record. The heal
    # subsystem persists the attempt counter under HEAL_STATE_REL.
    if not error_heal.check_circuit(iter_dir, task_id, error_id):
        return {
            "error": "runtime_failed_no_heal_history",
            "task_id": task_id,
            "error_id": error_id,
            "message": (
                f"no open circuit-breaker record for "
                f"({task_id!r}, {error_id!r}) — heal-tick must have returned "
                f"give_up for this task+error before record-task can promote "
                f"the task to runtime_failed"
            ),
        }

    with _state_lock(iter_dir):
        state = load_state(iter_dir)
        entry = state.tasks.get(task_id)
        if entry is None:
            return {"error": "unknown_task", "task_id": task_id}
        if entry.get("kind") != "eval":
            return {
                "error": "runtime_failed_wrong_task_type",
                "task_id": task_id,
                "task_kind": entry.get("kind"),
                "message": (
                    "runtime_failed is only valid for eval tasks — non-eval "
                    "tasks (serve / setup / aggregate / compare) keep "
                    "failure / skip semantics"
                ),
            }
        entry["status"] = "failed"
        entry["outcome"] = "runtime_failed"
        entry["finished_at"] = _now_iso()
        entry["gpu_ids"] = []
        entry["failure_category"] = failure_category
        entry["heal_history"] = heal_history
        entry["runtime_failed_at"] = _now_iso()
        save_state(iter_dir, state)
        return {"status": "ok", "task_id": task_id, "task": entry}


def skip_task(
    iter_dir: Path, task_id: str, *,
    reason: str = "", pivot_proof: str | None = None,
) -> dict:
    """Mark a task ``skipped`` and release its GPUs from the state.

    For a config dropped by a Stage 4 pivot-matrix decision (the legitimate
    force-majeure path) — distinct from ``failed`` (a genuine runtime error).
    When ``pivot_proof`` is given it is recorded on the state row as
    ``skip_proof`` (a string typically matching an ``experiment_brief.json``
    ``pivot_matrix[*].action``); the Stage 6 completion gate uses this to
    excuse the skip when deciding whether to advance.

    The CLI gate (``era.cli record-task``) refuses ``outcome=skipped`` for
    ``eval`` tasks that arrive without ``pivot_proof``; this function still
    accepts proof-less skips so that non-eval task types (``setup`` /
    ``serve`` / ``aggregate`` / ``compare``) keep their unrestricted skip
    semantics — those are legitimate orchestration decisions.
    """
    with _state_lock(iter_dir):
        state = load_state(iter_dir)
        entry = state.tasks.get(task_id)
        if entry is None:
            return {"error": "unknown_task", "task_id": task_id}
        entry["status"] = "skipped"
        entry["finished_at"] = _now_iso()
        entry["gpu_ids"] = []
        entry["skip_reason"] = reason
        if pivot_proof:
            entry["skip_proof"] = pivot_proof
        save_state(iter_dir, state)
        return {"status": "ok", "task_id": task_id, "task": entry}


def status_snapshot(iter_dir: Path) -> dict:
    """Bucket every task by status and report whether the pass is done."""
    state = load_state(iter_dir)
    buckets: dict[str, list[str]] = {s: [] for s in TASK_STATUSES}
    for tid, entry in state.tasks.items():
        buckets.setdefault(entry.get("status", "pending"), []).append(tid)
    counts = {s: len(v) for s, v in buckets.items()}
    all_done = not buckets["pending"] and not buckets["running"]
    return {
        "status": "ok",
        "mode": state.mode,
        "pending": sorted(buckets["pending"]),
        "running": sorted(buckets["running"]),
        "completed": sorted(buckets["completed"]),
        "failed": sorted(buckets["failed"]),
        "skipped": sorted(buckets["skipped"]),
        "counts": counts,
        "all_done": all_done,
        "running_task_ids": sorted(buckets["running"]),
        # The experiment-supervisor heartbeat: last_recovery_at is refreshed
        # every poll cycle, so a stale value means the Stage 6 supervisor loop
        # itself has stalled.
        "supervisor_heartbeat_at": state.last_recovery_at,
        "supervisor_idle_seconds": _iso_age_seconds(state.last_recovery_at),
    }


# ---- marker-file detection ----------------------------------------------

def detection_script(
    iter_dir: Path, task_ids: list[str], heartbeat_timeout_s: int = 1800,
) -> str:
    """Generate a pure-bash scan of the marker files for ``task_ids``.

    Emits one tab-separated line per task: ``<STATUS>\\t<task_id>\\t<payload>``
    where STATUS is DONE/RUNNING/HUNG/DEAD/UNKNOWN. For DONE the payload is the
    ``.done.json`` content with newlines/tabs stripped (still valid JSON); for
    HUNG it is the runner's pid. A task is **HUNG** when its process is alive
    but its heartbeat — the mtime of ``.progress.json`` (or, before its first
    write, ``.pid``) — has not advanced for ``heartbeat_timeout_s`` seconds.
    """
    logs = (iter_dir / LOGS_REL).resolve()
    ids = " ".join(_sh_quote(t) for t in task_ids)
    timeout = max(1, int(heartbeat_timeout_s))
    return f"""#!/bin/bash
# ERA Stage 6 task-marker detection — pure bash, no LLM tokens.
LOGS={_sh_quote(str(logs))}
TIMEOUT={timeout}
for tid in {ids}; do
  done_f="$LOGS/$tid.done.json"
  pid_f="$LOGS/$tid.pid"
  if [ -f "$done_f" ]; then
    payload=$(tr -d '\\n\\t' < "$done_f")
    printf 'DONE\\t%s\\t%s\\n' "$tid" "$payload"
  elif [ -f "$pid_f" ]; then
    pid=$(cat "$pid_f" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      prog_f="$LOGS/$tid.progress.json"
      hb_f="$pid_f"
      [ -f "$prog_f" ] && hb_f="$prog_f"
      mtime=$(stat -c %Y "$hb_f" 2>/dev/null || echo 0)
      age=$(( $(date +%s) - mtime ))
      if [ "$mtime" -gt 0 ] && [ "$age" -gt "$TIMEOUT" ]; then
        printf 'HUNG\\t%s\\t%s\\n' "$tid" "$pid"
      else
        printf 'RUNNING\\t%s\\t\\n' "$tid"
      fi
    else
      printf 'DEAD\\t%s\\t\\n' "$tid"
    fi
  else
    printf 'UNKNOWN\\t%s\\t\\n' "$tid"
  fi
done
"""


def _sh_quote(value: str) -> str:
    """Single-quote a string for safe POSIX-shell embedding."""
    return "'" + value.replace("'", "'\\''") + "'"


def parse_detection_output(output: str) -> dict:
    """Parse ``detection_script`` output into ``{task_id: {detected_status,…}}``."""
    result: dict[str, dict] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, tid = parts[0].strip(), parts[1].strip()
        if not tid:
            continue
        entry: dict = {"detected_status": status.lower()}
        payload = parts[2] if len(parts) > 2 else ""
        if status == "DONE" and payload.strip():
            try:
                entry["done_info"] = json.loads(payload)
            except json.JSONDecodeError:
                entry["done_info"] = {}
        elif status == "HUNG" and payload.strip():
            try:
                entry["pid"] = int(payload.strip())
            except ValueError:
                pass
        result[tid] = entry
    return result


# ---- bounded-retry recovery ---------------------------------------------

@dataclass
class RecoveryResult:
    """The outcome of applying a marker scan to the experiment state."""
    recovered_completed: list = field(default_factory=list)
    still_running: list = field(default_factory=list)
    recovered_failed: list = field(default_factory=list)
    retried: list = field(default_factory=list)
    progress: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _done_succeeded(done_info: dict) -> bool:
    """True when a ``.done.json`` payload reports a clean finish."""
    status = done_info.get("status")
    if status == "success":
        return True
    if status == "failure":
        return False
    return done_info.get("exit_code", 0) == 0


def _kill_pid(pid: object) -> None:
    """Best-effort terminate a hung runner — SIGTERM, a brief wait, then SIGKILL.

    All errors are swallowed: the process may already be gone, on another node,
    or owned by another user — recovery retries the task regardless.
    """
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        return
    if pid_int <= 1:
        return
    try:
        os.kill(pid_int, signal.SIGTERM)
    except OSError:
        return
    time.sleep(0.3)
    try:
        os.kill(pid_int, signal.SIGKILL)
    except OSError:
        pass


def recover_from_detection(
    state: ExperimentState, detection: dict, *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> RecoveryResult:
    """Apply a parsed marker scan to ``state`` in place; return what changed.

    A done+success task → ``completed``. A done+failure / dead / unknown task
    is reset to ``pending`` (re-schedulable) while its ``retry_count`` is below
    ``max_retries``, else marked ``failed``. A running task is left running.
    """
    result = RecoveryResult()
    for tid, info in detection.items():
        task = state.tasks.get(tid)
        if task is None:
            continue
        detected = info.get("detected_status")

        if detected == "done":
            done_info = info.get("done_info") or {}
            # A runner that writes status="skipped" for an eval task is
            # bypassing the Stage 6 completion gate — convert to failure so
            # the silent scope-reduction path is closed. Non-eval tasks
            # (setup/serve/aggregate/compare) keep the legitimate skip path.
            if (
                done_info.get("status") == "skipped"
                and task.get("kind") == "eval"
            ):
                reason = (
                    done_info.get("summary")
                    or done_info.get("reason")
                    or "no reason given"
                )
                _retry_or_fail(
                    task, tid,
                    f"runner attempted unauthorized skip; converted to failure "
                    f"(reason was: {reason})",
                    max_retries, result, state,
                )
            elif _done_succeeded(done_info):
                task["status"] = "completed"
                task["finished_at"] = _now_iso()
                task["gpu_ids"] = []
                if done_info.get("endpoint"):
                    task["endpoint"] = done_info["endpoint"]
                if done_info.get("result_dir"):
                    task["result_dir"] = done_info["result_dir"]
                result.recovered_completed.append(tid)
            else:
                _retry_or_fail(task, tid, "task reported failure",
                               max_retries, result, state)
        elif detected == "running":
            task["status"] = "running"
            result.still_running.append(tid)
            prog_f = info.get("progress")
            if prog_f:
                result.progress[tid] = prog_f
        elif detected == "hung":
            # Alive but its heartbeat went stale — kill it, then retry it
            # through the same bounded path as a dead task.
            _kill_pid(info.get("pid"))
            _retry_or_fail(task, tid, "runner hung — no heartbeat",
                           max_retries, result, state)
        elif detected in ("dead", "unknown"):
            _retry_or_fail(task, tid, f"process {detected}",
                           max_retries, result, state)

    state.last_recovery_at = _now_iso()
    return result


def apply_detection(
    iter_dir: Path, detection: dict, *,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> tuple[RecoveryResult, ExperimentState]:
    """Load the state, apply a parsed marker scan, and persist — under the lock."""
    with _state_lock(iter_dir):
        state = load_state(iter_dir)
        result = recover_from_detection(
            state, detection, max_retries=max_retries
        )
        save_state(iter_dir, state)
    return result, state


def _retry_or_fail(
    task: dict, tid: str, reason: str, max_retries: int,
    result: RecoveryResult, state: ExperimentState,
) -> None:
    """Reset a task to pending if retries remain, else mark it failed."""
    retry_count = int(task.get("retry_count", 0))
    if retry_count < max_retries:
        task["status"] = "pending"
        task["retry_count"] = retry_count + 1
        task["gpu_ids"] = []
        task["error_summary"] = reason
        result.retried.append(tid)
        state.recovery_log.append(
            f"[{_now_iso()}] {tid}: {reason} — retry #{retry_count + 1}"
        )
    else:
        task["status"] = "failed"
        task["finished_at"] = _now_iso()
        task["gpu_ids"] = []
        task["error_summary"] = reason
        result.recovered_failed.append(tid)
        state.recovery_log.append(
            f"[{_now_iso()}] {tid}: {reason} — retries exhausted, failed"
        )
