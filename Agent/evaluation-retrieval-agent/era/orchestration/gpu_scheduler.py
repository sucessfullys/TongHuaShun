"""Stage 6 GPU scheduler — free-GPU discovery, leasing, DAG-aware batching.

The scheduler walks the Stage 5 task DAG (``experiments/plans/task_plan.json``)
and decides which tasks may run right now on which GPUs. It is **deterministic
Python** — the Stage 6 skill runs ``nvidia-smi`` and pipes the output in; the
scheduler never shells out.

Three concerns:

- **Free-GPU discovery** — ``parse_free_gpus`` reads an ``nvidia-smi`` snapshot;
  ``allowed_gpu_pool`` applies ERA's pool rule (``visible \\ reserve`` capped by
  ``max_gpus_per_run``).
- **Cross-workspace leasing** — a global, fcntl-locked ``gpu_leases.json`` under
  ``.era/scheduler/`` so two ERA workspaces sharing a machine never claim the
  same GPU.
- **DAG batching** — ``claim_batch`` topologically resolves ready tasks, applies
  the configured Family-A serving policy, greedily assigns GPUs, writes the
  leases, and registers the batch ``running`` in ``experiment_state.json`` — all
  under one lock. Two policies (``experiment.family_a_execution``):
  ``"parallel_packed"`` (default) right-sizes each judge to its ``tensor_parallel``
  GPUs and lets multiple judges co-reside on disjoint subsets while Family-B
  evals backfill the rest; ``"serial_full_pool"`` runs Rule 6 — one judge at a
  time, owning the whole pool.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import ERAConfig
from ..paths import era_state_dir
from ..workspace import Workspace, resolve_workspace_root
from . import experiment_state

DEFAULT_FREE_THRESHOLD_MB = 2000
_LEASE_TTL_SEC = 3600
_RECENT_LEASE_SEC = 60          # leases this fresh are never reclaimed
_LOCK_TIMEOUT_SEC = 30.0
_RESOLVED = ("completed", "failed", "skipped")

# ---- GPU watchdog lifecycle ---------------------------------------------
#
# ``NoGPUAlarmNew.py`` (the cross-machine watchdog) keeps idle cards warm at
# ~35 GB / 100 % util and auto-releases when a "real" job acquires the card.
# That release fires *during* vLLM v1's memory profiling — initial_free grows
# mid-init and vLLM's profiler asserts, breaking every serve task. Stage 6
# owns the lifecycle instead: pkill before the iteration's first GPU claim,
# restart on terminal-state transition. The sentinel file
# ``<iter_dir>/.watchdog_suspended`` keeps both helpers idempotent across the
# many ralph-loop invocations the iteration spans.
WATCHDOG_PROCESS_PATTERN = "python3 -u NoGPUAlarmNew.py"
WATCHDOG_START_DIR = "/mnt/image-edit/datasets/xywang/code/GPU_OCU"
_WATCHDOG_SENTINEL_NAME = ".watchdog_suspended"


def _watchdog_sentinel(iter_dir: Path) -> Path:
    return iter_dir / _WATCHDOG_SENTINEL_NAME


def suspend_watchdog(iter_dir: Path) -> dict:
    """Stop ``NoGPUAlarmNew.py`` for the duration of this iteration (idempotent).

    Runs ``sudo -n pkill -9 -f "<pattern>"`` once per iteration, gated by the
    sentinel file. ``-n`` ensures sudo fails fast on no-credentials rather
    than blocking on a TTY prompt. Subprocess failures are swallowed — the
    watchdog might not be running on this machine, the user might not have
    sudo, or sudo might be unconfigured: any of those is fine, the sentinel
    still gets created so resume runs symmetrically.

    Returns a dict reporting the action (``suspended`` / ``already_suspended``
    / ``no_watchdog_running`` / ``suspend_failed``) and the pkill returncode.
    Always ``status: "ok"`` — this is best-effort plumbing, not a gate.
    """
    sentinel = _watchdog_sentinel(iter_dir)
    if sentinel.is_file():
        return {"status": "ok", "action": "already_suspended"}
    rc: int | None = None
    reason: str | None = None
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["sudo", "-n", "pkill", "-9", "-f", WATCHDOG_PROCESS_PATTERN],
            check=False, capture_output=True, text=True, timeout=10,
        )
        rc = proc.returncode
    except (subprocess.TimeoutExpired, OSError) as exc:
        reason = f"{type(exc).__name__}: {exc}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(_now_iso() + "\n", encoding="utf-8")
    if reason is not None:
        return {"status": "ok", "action": "suspend_failed", "reason": reason}
    # pkill returns 1 when no process matched — fine, treat as "nothing to do".
    action = "suspended" if rc == 0 else "no_watchdog_running"
    return {"status": "ok", "action": action, "returncode": rc}


def resume_watchdog(iter_dir: Path) -> dict:
    """Restart ``NoGPUAlarmNew.py`` after the iteration ends or blocks.

    Fires only when the sentinel is present, so concurrent / repeated calls
    are safe. The start command is fire-and-forget (the watchdog daemonizes
    itself via ``start.sh``); we never wait. Any failure to relaunch is
    reported but the sentinel still gets removed so a future iteration can
    suspend cleanly.
    """
    sentinel = _watchdog_sentinel(iter_dir)
    if not sentinel.is_file():
        return {"status": "ok", "action": "not_suspended"}
    start_dir = Path(WATCHDOG_START_DIR)
    if not start_dir.is_dir():
        sentinel.unlink(missing_ok=True)
        return {"status": "ok", "action": "resume_skipped_no_dir",
                "start_dir": str(start_dir)}
    reason: str | None = None
    try:
        subprocess.Popen(  # noqa: S603 - fixed argv, no shell, no untrusted input
            ["bash", "start.sh"],
            cwd=str(start_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        reason = f"{type(exc).__name__}: {exc}"
    sentinel.unlink(missing_ok=True)
    if reason is not None:
        return {"status": "ok", "action": "resume_failed", "reason": reason}
    return {"status": "ok", "action": "resumed"}


def _watchdog_running() -> bool | None:
    """Whether a ``NoGPUAlarmNew.py`` process is live (None if pgrep failed).

    Uses ``pgrep -f`` against the watchdog signature. Returns True when at
    least one process matches, False when none do, and None when pgrep itself
    could not run (missing binary, timeout) so the caller can treat the probe
    as inconclusive rather than as "dead".
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["pgrep", "-f", WATCHDOG_PROCESS_PATTERN],
            check=False, capture_output=True, text=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    # pgrep exits 0 when a process matched, 1 when none did, >1 on error.
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def ensure_watchdog_alive(iter_dir: Path | None = None) -> dict:
    """Make sure ``NoGPUAlarmNew.py`` is running; start it if it is not.

    Unlike :func:`resume_watchdog` (sentinel-gated — it only acts during the
    window an iteration explicitly suspended the watchdog), this checks the
    **actual process liveness** with ``pgrep`` and (re)starts the watchdog
    whenever none is running. Stage 8 calls it before the long human-feedback
    wait so idle experiment GPUs stay protected even if the watchdog died
    between Stage 6 and the hand-off.

    ``start.sh`` is *not* idempotent (it ``nohup``-launches a fresh process
    every time), so we only run it when the liveness probe says no watchdog is
    up — avoiding duplicate watchdogs. When ``iter_dir`` is given and a watchdog
    is confirmed/made alive, a stale ``.watchdog_suspended`` sentinel is cleared
    so the next iteration's :func:`suspend_watchdog` fires correctly ("alive"
    contradicts "suspended").

    Best-effort plumbing — never raises. Returns one of ``already_alive`` /
    ``started`` / ``start_skipped_no_dir`` / ``start_failed`` /
    ``probe_failed`` (pgrep could not run *and* no start dir to fall back on).
    """
    running = _watchdog_running()

    def _clear_sentinel() -> None:
        if iter_dir is not None:
            _watchdog_sentinel(iter_dir).unlink(missing_ok=True)

    if running is True:
        _clear_sentinel()
        return {"status": "ok", "action": "already_alive"}

    start_dir = Path(WATCHDOG_START_DIR)
    if not start_dir.is_dir():
        action = "start_skipped_no_dir" if running is False else "probe_failed"
        return {"status": "ok", "action": action, "start_dir": str(start_dir)}

    # running is False (definitely dead) or None (probe failed) — either way the
    # safest action is to (re)start, since a missing watchdog leaks idle GPUs.
    reason: str | None = None
    try:
        subprocess.Popen(  # noqa: S603 - fixed argv, no shell, no untrusted input
            ["bash", "start.sh"],
            cwd=str(start_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        reason = f"{type(exc).__name__}: {exc}"
    if reason is not None:
        return {"status": "ok", "action": "start_failed", "reason": reason}
    _clear_sentinel()
    return {"status": "ok", "action": "started",
            "probe_inconclusive": running is None}


def _now() -> float:
    return time.time()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- free-GPU discovery -------------------------------------------------

def nvidia_smi_query_cmd() -> str:
    """The ``nvidia-smi`` command the Stage 6 skill runs and pipes to the CLI."""
    return ("nvidia-smi --query-gpu=index,memory.used,memory.total "
            "--format=csv,noheader,nounits")


def parse_gpu_snapshot(text: str) -> list[dict]:
    """Parse ``nvidia-smi`` CSV output into per-GPU memory dicts."""
    snapshot: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            gpu_id = int(parts[0])
            used = int(float(parts[1]))
            total = int(float(parts[2])) if len(parts) > 2 and parts[2] else 0
        except (ValueError, IndexError):
            continue
        snapshot.append({
            "gpu_id": gpu_id,
            "memory_used_mb": used,
            "memory_total_mb": total,
            "memory_used_pct": round(used / total * 100, 1) if total else 0.0,
        })
    return snapshot


def parse_free_gpus(
    text: str, *, threshold_mb: int = DEFAULT_FREE_THRESHOLD_MB,
    only_gpu_ids: list[int] | None = None,
) -> list[int]:
    """Return the ids of GPUs whose used VRAM is below ``threshold_mb``."""
    allow = set(only_gpu_ids) if only_gpu_ids is not None else None
    free = [
        g["gpu_id"] for g in parse_gpu_snapshot(text)
        if g["memory_used_mb"] < threshold_mb
        and (allow is None or g["gpu_id"] in allow)
    ]
    return sorted(free)


def allowed_gpu_pool(cfg: ERAConfig) -> list[int]:
    """ERA's allowed GPU pool: ``visible \\ reserve`` capped by max_gpus_per_run."""
    hw = cfg.hardware
    pool = sorted(set(hw.visible_gpu_ids) - set(hw.reserve_gpu_ids))
    if hw.max_gpus_per_run and hw.max_gpus_per_run > 0:
        pool = pool[: hw.max_gpus_per_run]
    return pool


def snap_tensor_parallel(n: int, head_count: int | None = None) -> int:
    """Largest valid tensor-parallel size that is <= ``n``.

    A model's attention heads must divide evenly across the TP ranks, so when
    ``head_count`` is known the result is its largest divisor <= ``n``;
    otherwise it falls back to the largest power of two <= ``n``.
    """
    if n < 1:
        return 1
    if head_count and head_count > 0:
        for tp in range(min(n, head_count), 0, -1):
            if head_count % tp == 0:
                return tp
        return 1
    tp = 1
    while tp * 2 <= n:
        tp *= 2
    return tp


# ---- DAG topology -------------------------------------------------------

def topo_sort_layers(tasks: list[dict]) -> list[list[str]]:
    """Group task ids into dependency layers (Kahn BFS by in-degree)."""
    by_id = {t["id"]: t for t in tasks if t.get("id")}
    indegree = {
        tid: sum(1 for d in (t.get("depends_on") or []) if d in by_id)
        for tid, t in by_id.items()
    }
    layers: list[list[str]] = []
    placed: set[str] = set()
    while len(placed) < len(by_id):
        layer = sorted(
            tid for tid, deg in indegree.items()
            if deg == 0 and tid not in placed
        )
        if not layer:  # a cycle — stop rather than loop forever
            break
        layers.append(layer)
        placed.update(layer)
        for tid in layer:
            for other, t in by_id.items():
                if tid in (t.get("depends_on") or []):
                    indegree[other] -= 1
    return layers


def compute_downstream_counts(tasks: list[dict]) -> dict[str, int]:
    """Count each task's transitive descendants (its critical-path weight)."""
    by_id = {t["id"]: t for t in tasks if t.get("id")}
    children: dict[str, list[str]] = {tid: [] for tid in by_id}
    for tid, t in by_id.items():
        for dep in t.get("depends_on") or []:
            if dep in children:
                children[dep].append(tid)

    counts: dict[str, int] = {}

    def _count(tid: str, seen: set[str]) -> int:
        if tid in counts:
            return counts[tid]
        total = 0
        for child in children.get(tid, []):
            if child in seen:
                continue
            total += 1 + _count(child, seen | {child})
        counts[tid] = total
        return total

    for tid in by_id:
        _count(tid, {tid})
    return counts


def assign_gpus(
    tasks: list[dict], free_gpu_ids: list[int],
    downstream_counts: dict[str, int] | None = None,
) -> tuple[list[dict], list[str]]:
    """Greedily assign GPUs to ``tasks`` from ``free_gpu_ids``.

    Tasks are ordered ``(serve first, gpu_count ASC, downstream DESC, est_minutes
    DESC)`` — a ``serve`` task is placed ahead of metric tasks so Rule 6's
    whole-pool judge is never starved by a trickle of small Family-B evals;
    otherwise small tasks pack first and critical-path tasks win ties. Returns
    ``(assigned, unassigned)`` — ``assigned`` is ``[{task_id, gpu_ids}]``,
    ``unassigned`` is the ids that did not fit.
    """
    downstream = downstream_counts or {}
    ordered = sorted(
        tasks,
        key=lambda t: (
            0 if t.get("type") == "serve" else 1,
            t.get("gpu_count") or 0,
            -downstream.get(t.get("id"), 0),
            -(t.get("estimated_minutes") or 0),
        ),
    )
    pool = list(free_gpu_ids)
    assigned: list[dict] = []
    unassigned: list[str] = []
    for t in ordered:
        need = t.get("gpu_count") or 0
        if need <= len(pool):
            take, pool = pool[:need], pool[need:]
            assigned.append({"task_id": t.get("id"), "gpu_ids": take})
        else:
            unassigned.append(t.get("id"))
    return assigned, unassigned


# ---- the global GPU-lease file ------------------------------------------

def _scheduler_dir() -> Path:
    d = era_state_dir() / "scheduler"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _leases_path() -> Path:
    return _scheduler_dir() / "gpu_leases.json"


@contextlib.contextmanager
def _leases_lock(timeout: float = _LOCK_TIMEOUT_SEC):
    """Exclusive lock over the global lease file (force-acquire after timeout)."""
    lock_path = _scheduler_dir() / "gpu_leases.lock"
    fh = open(lock_path, "w", encoding="utf-8")  # noqa: SIM115
    try:
        deadline = _now() + timeout
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if _now() >= deadline:
                    fcntl.flock(fh, fcntl.LOCK_EX)  # force — wait it out
                    break
                time.sleep(0.1)
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def read_leases() -> dict:
    """Read the global GPU-lease map (``{gpu_id_str: lease}``); {} when absent."""
    path = _leases_path()
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_leases(leases: dict) -> None:
    path = _leases_path()
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(leases, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _clean_stale(leases: dict) -> dict:
    """Drop leases older than the TTL; keep very recent ones unconditionally."""
    now = _now()
    kept: dict = {}
    for gpu, lease in leases.items():
        age = now - float(lease.get("claimed_at", 0))
        if age < _RECENT_LEASE_SEC or age < _LEASE_TTL_SEC:
            kept[gpu] = lease
    return kept


def _prune_resolved_leases(
    leases: dict, workspace_root: Path,
    state: "experiment_state.ExperimentState", resident_ids: set[str],
) -> dict:
    """Release this workspace's leases for tasks that are no longer running.

    A lease should outlive its task only for a resident judge (which holds its
    GPUs until teardown). Under ``parallel_packed`` several judges can be
    resident at once, so ``resident_ids`` is a set. Every other same-workspace
    lease whose owning task is not ``running`` is stale — pruning it here makes
    the scheduler self-heal, so a completed / failed / retried metric or setup
    task frees its GPUs without depending on an explicit ``release-gpus`` call.
    Other workspaces' leases are left for their own scheduler / the TTL to reap.
    """
    ws = str(workspace_root)
    kept: dict = {}
    for gpu, lease in leases.items():
        if lease.get("workspace_root") != ws:
            kept[gpu] = lease            # another workspace — not ours to prune
            continue
        task_ids = lease.get("task_ids") or []
        alive = bool(task_ids) and all(
            tid in resident_ids
            or state.tasks.get(tid, {}).get("status") == "running"
            for tid in task_ids
        )
        if alive:
            kept[gpu] = lease
    return kept


# ---- batch claiming -----------------------------------------------------

def _iter_dir(workspace_root: str | Path) -> tuple[Path, Path]:
    """Resolve ``(workspace_root, active_iteration_dir)``."""
    root = resolve_workspace_root(workspace_root)
    ws = Workspace(root.parent, root.name)
    return root, ws.iter_path()


def _resident_judges(
    tasks_by_id: dict[str, dict], state: experiment_state.ExperimentState,
    resolved: set[str],
) -> list[str]:
    """The ids of every judge currently holding GPUs, in plan order.

    A judge is *resident* from when its ``serve`` task is dispatched until every
    task in its ``teardown_after`` list is resolved. Under ``serial_full_pool``
    at most one judge is ever resident (Rule 6); under ``parallel_packed``
    several co-reside on disjoint GPU subsets.
    """
    out: list[str] = []
    for tid, task in tasks_by_id.items():
        if task.get("type") != "serve":
            continue
        status = state.tasks.get(tid, {}).get("status")
        if status not in ("running", "completed"):
            continue
        teardown = task.get("teardown_after") or []
        if not all(e in resolved for e in teardown):
            out.append(tid)
    return out


def claim_batch(
    workspace_root: str | Path, cfg: ERAConfig, free_gpu_ids: list[int],
    mode: str,
) -> dict:
    """Resolve and claim the next runnable batch of Stage 6 tasks.

    Applies the configured Family-A serving policy
    (``experiment.family_a_execution``) — ``parallel_packed`` co-residents
    right-sized judges on disjoint GPU subsets, ``serial_full_pool`` runs Rule 6
    (one judge at a time, owning the whole pool) — assigns GPUs greedily from the
    unleased subset of ``free_gpu_ids``, writes the leases, and registers the
    batch ``running`` — all under the global lease lock.
    """
    root, iter_dir = _iter_dir(workspace_root)
    plan_path = iter_dir / "experiments" / "plans" / "task_plan.json"
    if not plan_path.is_file():
        return {"error": "no_task_plan", "message": f"{plan_path} not found"}
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {"error": "bad_task_plan", "message": str(exc)}

    # Stop NoGPUAlarmNew.py before the first GPU claim of the iteration so
    # vLLM v1's memory profiler doesn't trip on the watchdog's mid-init
    # auto-release. Idempotent — the sentinel keeps subsequent claims silent.
    suspend_watchdog(iter_dir)

    tasks = plan.get("tasks") or []
    tasks_by_id = {t["id"]: t for t in tasks if t.get("id")}
    gpu_pool = plan.get("gpu_pool") or 0
    parallel = cfg.experiment.family_a_execution == "parallel_packed"
    fam_b_serial = (
        cfg.experiment.family_b_schedule == "before_after_family_a"
    )

    with _leases_lock():
        leases = _clean_stale(read_leases())
        state = experiment_state.load_state(iter_dir)
        resolved = {
            tid for tid, e in state.tasks.items()
            if e.get("status") in _RESOLVED
        }
        residents = _resident_judges(tasks_by_id, state, resolved)
        resident_ids = set(residents)
        # Self-heal: release leases for tasks that are no longer running.
        leases = _prune_resolved_leases(leases, root, state, resident_ids)
        # parallel_packed: how many more judges may become resident this pass.
        # None = uncapped (serial mode, or max_concurrent_judges == 0).
        max_judges = cfg.experiment.max_concurrent_judges
        judge_headroom = (
            max(0, max_judges - len(residents))
            if (parallel and max_judges > 0) else None
        )

        # Tasks of this pass that are pending with every dependency resolved.
        ready = [
            t for t in tasks
            if t.get("mode") in (mode, None)
            and state.tasks.get(t.get("id"), {}).get("status") == "pending"
            and all(d in resolved for d in (t.get("depends_on") or []))
        ]

        leased_gpus = {int(g) for g in leases}
        available = sorted(set(free_gpu_ids) - leased_gpus)

        gpu_needing: list[dict] = []   # eligible tasks that need GPUs
        free_tasks: list[dict] = []    # eligible tasks holding no GPUs
        blocked: list[str] = []
        for t in ready:
            ttype = t.get("type")
            if ttype == "serve":
                if not parallel and residents:
                    blocked.append(t.get("id"))  # Rule 6: a judge is resident
                elif judge_headroom is not None and judge_headroom <= 0:
                    blocked.append(t.get("id"))  # max_concurrent_judges reached
                else:
                    gpu_needing.append(t)         # packed onto its tp GPUs
                    if judge_headroom is not None:
                        judge_headroom -= 1
            elif ttype == "eval" and t.get("family") == "A":
                free_tasks.append(t)             # consumes an endpoint, not GPUs
            elif (t.get("gpu_count") or 0) == 0:
                free_tasks.append(t)             # aggregate / compare
            else:                                # Family-B eval or setup
                # serial_full_pool may serialize Family-B behind the resident
                # judge; parallel_packed always backfills it onto free GPUs.
                if not parallel and residents and fam_b_serial:
                    blocked.append(t.get("id"))  # paused behind the judge
                else:
                    gpu_needing.append(t)

        downstream = compute_downstream_counts(tasks)

        # Phase D-1 cap: limit concurrent runners across the whole iter.
        # ``max_parallel_runners == 0`` means "no cap"; otherwise trim
        # ``gpu_needing`` (the heavy side) before assignment so we don't
        # take GPUs we won't use this pass. Untrimmed tasks remain pending
        # and are re-considered on the next claim_batch call.
        max_runners = cfg.experiment.max_parallel_runners
        if max_runners > 0:
            already_running = sum(
                1 for e in state.tasks.values()
                if e.get("status") == "running"
            )
            headroom = max(0, max_runners - already_running)
            # free_tasks are aggregate/compare-style — they run in the same
            # batch but don't compete for GPUs; we still count them.
            free_tasks = free_tasks[:headroom]
            headroom -= len(free_tasks)
            if headroom < len(gpu_needing):
                ordered = sorted(
                    gpu_needing,
                    key=lambda t: (
                        0 if t.get("type") == "serve" else 1,
                        t.get("gpu_count") or 0,
                        -downstream.get(t.get("id"), 0),
                        -(t.get("estimated_minutes") or 0),
                    ),
                )
                deferred = [t.get("id") for t in ordered[max(0, headroom):]]
                blocked.extend(deferred)
                gpu_needing = ordered[:max(0, headroom)]

        assigned, unfit = assign_gpus(gpu_needing, available, downstream)
        blocked.extend(unfit)                    # could not fit on free GPUs

        assigned_by_id = {a["task_id"]: a["gpu_ids"] for a in assigned}
        batch: list[dict] = []
        registrations: list[dict] = []
        new_leases = dict(leases)
        for t in free_tasks:
            tid = t.get("id")
            batch.append({"task_id": tid, "type": t.get("type"),
                          "family": t.get("family"), "gpu_ids": []})
            registrations.append({"task_id": tid, "gpu_ids": [],
                                  "kind": t.get("type")})
        for t in gpu_needing:
            tid = t.get("id")
            if tid not in assigned_by_id:
                continue
            gpu_ids = assigned_by_id[tid]
            tp = None
            if t.get("type") == "serve":
                tp = snap_tensor_parallel(len(gpu_ids))
            batch.append({"task_id": tid, "type": t.get("type"),
                          "family": t.get("family"), "gpu_ids": gpu_ids,
                          "tensor_parallel": tp})
            registrations.append({"task_id": tid, "gpu_ids": gpu_ids,
                                  "kind": t.get("type"), "tensor_parallel": tp})
            kind = "serve" if t.get("type") == "serve" else "metric"
            for g in gpu_ids:
                new_leases[str(g)] = {
                    "workspace_root": str(root),
                    "project_name": root.name,
                    "task_ids": [tid],
                    "gpu_ids": gpu_ids,
                    "claimed_at": _now(),
                    "kind": kind,
                }

        # Persist the lease file unconditionally — a prune-only claim (no new
        # registrations) must still free the GPUs it released.
        _save_leases(new_leases)
        if registrations:
            experiment_state.register_running(iter_dir, registrations)

        in_mode = [t for t in tasks if t.get("mode") in (mode, None)]
        done = sum(
            1 for t in in_mode
            if state.tasks.get(t.get("id"), {}).get("status") in _RESOLVED
        )
        batch_ids = {x["task_id"] for x in batch}
        est = max((
            t.get("estimated_minutes") or 0
            for t in gpu_needing + free_tasks
            if t.get("id") in batch_ids
        ), default=0)
        return {
            "status": "ok",
            "mode": mode,
            "batch": batch,
            "blocked": sorted(blocked),
            "resident_judge": residents[0] if residents else None,
            "resident_judges": sorted(residents),
            "completed_count": done,
            "remaining_count": len(in_mode) - done,
            "total_count": len(in_mode),
            "estimated_minutes": est,
        }


def release_gpus(workspace_root: str | Path, task_ids: list[str]) -> dict:
    """Drop every global GPU lease held by ``task_ids`` (e.g. on judge teardown)."""
    root = resolve_workspace_root(workspace_root)
    release = set(task_ids)
    with _leases_lock():
        leases = read_leases()
        freed: list[int] = []
        kept: dict = {}
        for gpu, lease in leases.items():
            lease_tasks = set(lease.get("task_ids") or [])
            same_ws = lease.get("workspace_root") == str(root)
            if same_ws and lease_tasks and lease_tasks <= release:
                freed.append(int(gpu))
            else:
                kept[gpu] = lease
        _save_leases(kept)
        return {"status": "ok", "freed_gpus": sorted(freed),
                "task_ids": sorted(release)}
