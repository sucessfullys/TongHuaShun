"""Stage 6 experiment results — per-config aggregation and the run summary.

Each evaluator runner appends one JSON line per sample to its config's canonical
``scores.jsonl`` — ``iter_NNN/experiments/results/<mode>/<combination_id>/scores.jsonl``
(``mode`` is ``pilot`` or ``full``):

    {"sample_key": "...", "method_id": "...", "score": 0.82,
     "sub_scores": {...}, "scope": "whole", "ok": true}

A per-sample failure is recorded ``"ok": false`` and never aborts the task.
``aggregate_config`` collapses one config's ``scores.jsonl`` into a
``config_result.json``; ``write_summary`` rolls every config up into
``results/summary.json`` (full) or ``results/pilot/summary.json`` (pilot).
``scores_relpath`` is the single source of truth for that path contract — the
Stage-5 ``check-task-plan`` guard pins each eval task's ``expected_output`` to
it so an agent-authored runner cannot write where the aggregator will not read.

``human_alignment`` is intentionally left ``null`` here — Stage 6 produces the
raw per-sample evaluator scores; the human-correlation metric is computed at
Stage 8 against the human-rating set.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from datetime import datetime, timezone
from numbers import Number
from pathlib import Path

from .experiment_brief import load_pivot_actions
from .experiment_state import STATE_REL
from .gpu_scheduler import resume_watchdog

RESULTS_REL = "experiments/results"

# Stage 6 completion gate — the maximum fraction of expected configs that may
# legitimately end up runtime_failed before the gate forces a block. Three
# heal-tick give-ups out of ten configs (the default round size) is the
# threshold past which the loop should not silently advance: a systemic infra
# failure must surface to the operator, not be absorbed.
MAX_RUNTIME_FAILED_FRACTION = 0.3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_number(value: object) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(tmp, path)


def results_root(iter_dir: Path) -> Path:
    return iter_dir / RESULTS_REL


def result_dir(iter_dir: Path, combination_id: str, mode: str = "full") -> Path:
    """The per-config results directory for one experiment pass.

    Both passes are mode-scoped — ``results/full/<id>/`` and
    ``results/pilot/<id>/`` — so a pilot run never contaminates the full pass and
    each mode's summary scans only its own subtree.
    """
    return results_root(iter_dir) / mode / combination_id


def scores_relpath(combination_id: str, mode: str = "full") -> str:
    """Canonical ``scores.jsonl`` path, relative to the iteration directory, for
    one config/pass.

    The single source of truth for the result-path contract: the Stage-6
    evaluator runners write here, ``aggregate_config`` reads here, and the
    Stage-5 ``check-task-plan`` guard pins each eval task's ``expected_output``
    to end with this — so a planner cannot silently diverge.
    """
    return f"{RESULTS_REL}/{mode}/{combination_id}/scores.jsonl"


def read_scores(path: Path) -> list[dict]:
    """Parse a ``scores.jsonl`` file, skipping blank / unparseable lines."""
    rows: list[dict] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def aggregate_config(
    iter_dir: Path, combination_id: str, *, mode: str = "full", **meta,
) -> dict:
    """Collapse one config's ``scores.jsonl`` into a ``config_result.json``.

    Idempotent — safe to call repeatedly. ``mode`` selects the pilot or full
    results directory; ``meta`` carries descriptive fields (``family``,
    ``judge``, ``scope``, ``cost_usd``, ``wallclock_minutes``) that the scores
    file itself does not hold.
    """
    cdir = result_dir(iter_dir, combination_id, mode)
    rows = read_scores(cdir / "scores.jsonl")

    ok_scores: list[float] = []
    failed = 0
    per_method_scores: dict[str, list[float]] = {}
    for row in rows:
        if row.get("ok") is False:
            failed += 1
            continue
        score = row.get("score")
        if not _is_number(score):
            failed += 1
            continue
        ok_scores.append(float(score))
        method = row.get("method_id") or "_unknown"
        per_method_scores.setdefault(method, []).append(float(score))

    per_method = {
        m: round(statistics.fmean(v), 6) for m, v in per_method_scores.items()
    }
    result = {
        "combination_id": combination_id,
        "family": meta.get("family"),
        "mode": mode,
        "judge": meta.get("judge"),
        "scope": meta.get("scope"),
        "sample_count": len(rows),
        "ok_count": len(ok_scores),
        "failed_count": failed,
        "score_mean": round(statistics.fmean(ok_scores), 6) if ok_scores else None,
        "score_std": (
            round(statistics.pstdev(ok_scores), 6) if len(ok_scores) > 1
            else 0.0 if ok_scores else None
        ),
        "per_method": per_method,
        "human_alignment": None,   # computed at Stage 8 vs the human-rating set
        "cost_usd": meta.get("cost_usd", 0.0),
        "wallclock_minutes": meta.get("wallclock_minutes", 0.0),
        "produced_at": _now_iso(),
    }
    write_json_atomic(cdir / "config_result.json", result)
    return result


def write_summary(
    iter_dir: Path, mode: str = "full",
    expected_configs: list[str] | None = None,
) -> dict:
    """Roll every per-config ``config_result.json`` of one pass into a summary.

    ``mode="full"`` summarizes ``results/full/*/`` into ``results/summary.json``;
    ``mode="pilot"`` summarizes ``results/pilot/*/`` into
    ``results/pilot/summary.json``. Each mode scans only its own subtree, so the
    sibling pass's directory can never be miscounted as a config.

    **The completion gate is strict against the brief's chosen set.** When
    ``expected_configs`` is given (the Stage 4 ``candidate_configs[*]``), the
    summary computes:

    - ``scored_configs`` — combination ids with ``ok_count > 0`` (real scores);
    - ``missing_configs`` — every expected id that did *not* score (whether
      it never ran, never wrote, crashed, or was marked skipped — every cause
      collapses to one list);
    - ``complete = (not missing_configs)`` AND ``len(scored_configs) > 0``.

    Pre-validated exclusions (Stage 4 pivot-matrix decisions) are NOT special-
    cased here — ``write_summary`` is a pure aggregator. The orchestration
    layer (``check_experiment_completion``) is what excuses
    skipped-with-proof tasks when deciding whether to advance past Stage 6.

    When ``expected_configs`` is omitted (legacy fallback), ``complete`` falls
    back to the pre-v0.1.6 rule: ``len(configs) > 0`` AND every listed config
    has ``ok_count > 0`` — this kept old call-sites and tests working.
    """
    root = results_root(iter_dir)
    search_root = root / mode
    summary_path = (root / "pilot" / "summary.json") if mode == "pilot" else (
        root / "summary.json")
    configs: list[dict] = []
    failed_configs: list[str] = []
    incomplete: list[str] = []
    total_cost = 0.0
    total_wallclock = 0.0

    candidates = (
        [p for p in search_root.glob("*") if p.is_dir()] if search_root.is_dir()
        else []
    )
    for cdir in sorted(candidates):
        cr_path = cdir / "config_result.json"
        if not cr_path.is_file():
            incomplete.append(cdir.name)
            continue
        try:
            cr = json.loads(cr_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            incomplete.append(cdir.name)
            continue
        configs.append({
            "combination_id": cr.get("combination_id", cdir.name),
            "family": cr.get("family"),
            "mode": cr.get("mode"),
            "score_mean": cr.get("score_mean"),
            "ok_count": cr.get("ok_count", 0),
            "failed_count": cr.get("failed_count", 0),
            "cost_usd": cr.get("cost_usd", 0.0),
            "wallclock_minutes": cr.get("wallclock_minutes", 0.0),
        })
        total_cost += float(cr.get("cost_usd") or 0.0)
        total_wallclock += float(cr.get("wallclock_minutes") or 0.0)
        if cr.get("ok_count", 0) == 0 and cr.get("failed_count", 0) > 0:
            failed_configs.append(cr.get("combination_id", cdir.name))

    scored_ids = {c["combination_id"] for c in configs if c.get("ok_count", 0) > 0}

    if expected_configs:
        expected_set = set(expected_configs)
        missing_configs = sorted(expected_set - scored_ids)
        complete = (not missing_configs) and bool(scored_ids)
        # Legacy `incomplete_configs` mirrors the union of incomplete dirs and
        # expected-but-absent configs so existing callers see no regression.
        legacy_incomplete = sorted(set(incomplete) | (expected_set - scored_ids))
    else:
        missing_configs = []
        complete = (
            len(configs) > 0
            and all(c.get("ok_count", 0) > 0 for c in configs)
        )
        legacy_incomplete = sorted(incomplete)

    summary = {
        "iteration_dir": iter_dir.name,
        "mode": mode,
        "config_count": len(configs),
        "complete": complete,
        "expected_configs": sorted(expected_configs) if expected_configs else [],
        "scored_configs": sorted(scored_ids),
        "missing_configs": missing_configs,
        "configs": configs,
        "totals": {
            "api_cost_usd": round(total_cost, 4),
            "wallclock_minutes": round(total_wallclock, 2),
            "failed_configs": sorted(failed_configs),
            "incomplete_configs": legacy_incomplete,
        },
        "produced_at": _now_iso(),
    }
    write_json_atomic(summary_path, summary)
    return summary


def read_summary(iter_dir: Path, mode: str = "full") -> dict:
    """Read a pass's ``summary.json``; an empty dict when it is not written yet."""
    root = results_root(iter_dir)
    path = (root / "pilot" / "summary.json") if mode == "pilot" else (
        root / "summary.json")
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
    return {}


# ---- Stage 6 completion gate -------------------------------------------

def check_experiment_completion(
    iter_dir: Path, mode: str = "full",
    expected_configs: list[str] | None = None,
) -> dict:
    """The authoritative "OK to advance past Stage 6" answer.

    Computes, for one experiment pass:

    - ``scored_configs``      — combination ids that produced ``ok_count > 0``;
    - ``skipped_with_proof``  — combination ids whose Stage 6 task carries a
      ``skip_proof`` in ``experiment_state.json`` (a legitimate Stage 4
      pivot-matrix decision, the only authorized force-majeure escape);
    - ``missing_configs``     — expected ids that are neither scored nor
      skipped-with-proof (the silent-skip path);
    - ``failed_tasks`` / ``in_progress_tasks`` — Stage 6 task ids in
      terminal-failure or still-pending/running state.

    ``complete`` is ``true`` iff ``missing_configs`` is empty AND no eval
    task is still pending/running. ``write_summary``'s own ``complete`` flag
    is strict (no skip-proof exception); this function is the *orchestration*
    layer that excuses pre-validated skips.

    The Stage 6 skill and the ralph loop use ``complete`` here as the
    advance gate; an incomplete result means the loop blocks
    (``run_state: blocked``) until the operator intervenes — either fixes
    the blocker and ``/era:resume``, or amends the Stage 4 brief to remove
    the failing config and re-plans.
    """
    summary = read_summary(iter_dir, mode)
    summary_expected = list(summary.get("expected_configs") or [])
    if expected_configs is None:
        expected_configs = summary_expected
    expected_set = set(expected_configs or [])
    scored_ids = set(summary.get("scored_configs") or [])

    # Read experiment_state to know which tasks were skipped with proof.
    state_path = iter_dir / STATE_REL
    state_tasks: dict = {}
    if state_path.is_file():
        try:
            state_tasks = (
                json.loads(state_path.read_text(encoding="utf-8")).get(
                    "tasks", {}
                )
                or {}
            )
        except json.JSONDecodeError:
            state_tasks = {}

    # Map Stage 6 task ids to their combination_id via the task plan.
    plan_path = iter_dir / "experiments" / "plans" / "task_plan.json"
    plan_tasks: list[dict] = []
    if plan_path.is_file():
        try:
            plan_tasks = (
                json.loads(plan_path.read_text(encoding="utf-8")).get(
                    "tasks", []
                )
                or []
            )
        except json.JSONDecodeError:
            plan_tasks = []
    tid_to_cid: dict[str, str | None] = {}
    tid_to_kind: dict[str, str | None] = {}
    tid_to_mode: dict[str, str | None] = {}
    for task in plan_tasks:
        tid = task.get("id")
        if not tid:
            continue
        tid_to_cid[tid] = task.get("combination_id")
        tid_to_kind[tid] = task.get("type")
        tid_to_mode[tid] = task.get("mode")

    # The gate re-validates ``skip_proof`` against the brief's pivot_matrix
    # actions on every call. Trusting whatever string the marker carries
    # would let a runner back-door a silent scope-reduction by writing a
    # bogus proof at init time; the gate cross-checks instead.
    valid_actions = load_pivot_actions(iter_dir)

    # Phase C-2: the second authorized skip path. Tasks marked
    # ``skip_proof: "auto_validate_failed"`` are valid iff the task's
    # combination_id appears in <iter>/auto_validate/result.json's
    # failing_configs. Read the file once here so we don't re-parse it
    # per task.
    av_result_path = iter_dir / "auto_validate" / "result.json"
    av_failing_configs: set[str] = set()
    if av_result_path.is_file():
        try:
            av_data = json.loads(av_result_path.read_text(encoding="utf-8"))
            av_failing_configs = set(av_data.get("failing_configs") or [])
        except json.JSONDecodeError:
            av_failing_configs = set()

    skipped_with_proof: set[str] = set()
    bogus_skipped: list[str] = []
    failed_tasks: list[str] = []
    in_progress: list[str] = []
    # combination_id -> {"runtime_failed": [...categories...],
    #                    "other_eval": int} — used to compute which configs
    # are *fully* runtime-failed (no scored task, no remaining work).
    eval_by_config: dict[str, dict] = {}
    for tid, entry in state_tasks.items():
        status = entry.get("status")
        kind = tid_to_kind.get(tid) or entry.get("kind")
        cid = tid_to_cid.get(tid)
        task_mode = tid_to_mode.get(tid)
        if status == "skipped" and entry.get("skip_proof") and cid:
            # Skip a task counts as authorized only for the pass it belongs
            # to. A pilot task's valid proof must not paper over the same
            # config's full-pass task carrying a bogus proof — that exact
            # asymmetry was the silent scope-reduction in tryon-eval/iter_001.
            if task_mode is not None and task_mode != mode:
                continue
            proof = entry.get("skip_proof")
            # Phase C-2: auto_validate_failed is the second authorized
            # path. Valid iff the cid appears in the on-disk
            # auto_validate/result.json.failing_configs.
            if proof == "auto_validate_failed":
                if cid in av_failing_configs:
                    skipped_with_proof.add(cid)
                else:
                    bogus_skipped.append(tid)
            # Empty valid_actions => brief missing/unreadable; fall back to
            # "any non-empty proof is accepted" so legacy briefs without a
            # pivot_matrix still work (matches the cli.py:_brief_pivot_actions
            # contract).
            elif not valid_actions or proof in valid_actions:
                skipped_with_proof.add(cid)
            else:
                bogus_skipped.append(tid)
        elif status == "failed" and kind == "eval":
            failed_tasks.append(tid)
            if cid:
                bucket = eval_by_config.setdefault(
                    cid, {"runtime_failed": [], "other_eval": 0}
                )
                if entry.get("outcome") == "runtime_failed":
                    bucket["runtime_failed"].append(
                        entry.get("failure_category") or "unknown"
                    )
                else:
                    bucket["other_eval"] += 1
        elif status in ("pending", "running") and kind == "eval":
            in_progress.append(tid)
            if cid:
                bucket = eval_by_config.setdefault(
                    cid, {"runtime_failed": [], "other_eval": 0}
                )
                bucket["other_eval"] += 1
        elif kind == "eval" and cid:
            bucket = eval_by_config.setdefault(
                cid, {"runtime_failed": [], "other_eval": 0}
            )
            bucket["other_eval"] += 1

    # A config is *fully* runtime-failed only when every eval task we know
    # about for it is outcome=runtime_failed AND it produced no scores. If
    # any sibling eval task is success / failure-non-runtime / still pending,
    # the config is NOT runtime-failed — it remains missing (must finish or
    # surface as a true failure to the operator).
    runtime_failed_configs: set[str] = set()
    runtime_failure_categories: dict[str, list[str]] = {}
    for cid, bucket in eval_by_config.items():
        if cid in scored_ids or cid in skipped_with_proof:
            continue
        if bucket["runtime_failed"] and bucket["other_eval"] == 0:
            runtime_failed_configs.add(cid)
            runtime_failure_categories[cid] = sorted(set(bucket["runtime_failed"]))

    missing_configs = sorted(
        expected_set - scored_ids - skipped_with_proof - runtime_failed_configs
    )
    cap = max(1, math.ceil(MAX_RUNTIME_FAILED_FRACTION * len(expected_set))) \
        if expected_set else 0
    cap_exceeded = (
        bool(expected_set) and len(runtime_failed_configs) > cap
    )
    complete = (
        bool(expected_set)
        and not missing_configs
        and not in_progress
        and not cap_exceeded
    )

    # The completion gate is the iteration's natural checkpoint — every Stage 6
    # invocation calls it before deciding to advance or block. Resume the GPU
    # watchdog here regardless of complete: a happy path advances to Stage 7
    # and the watchdog should reclaim idle cards; a block leaves the iteration
    # stalled for the operator and the watchdog still protects what's free.
    # ``resume_watchdog`` is sentinel-gated, so duplicate calls are silent.
    resume_watchdog(iter_dir)

    result = {
        "status": "ok",
        "iteration_dir": iter_dir.name,
        "mode": mode,
        "complete": complete,
        "expected_configs": sorted(expected_set),
        "scored_configs": sorted(scored_ids),
        "skipped_with_proof": sorted(skipped_with_proof),
        "missing_configs": missing_configs,
        "failed_tasks": sorted(failed_tasks),
        "in_progress_tasks": sorted(in_progress),
        # Skipped eval tasks whose skip_proof did not match the brief's
        # pivot_matrix — they fall through to missing_configs so the gate
        # blocks the loop. Surfaced separately so the operator can see why
        # the gate refused to advance.
        "unauthorized_skipped_tasks": sorted(bogus_skipped),
        # Configs whose only eval outcome was heal-tick give_up in a
        # runtime-failure category — counted as resolved (not missing) so the
        # loop advances, but surfaced for Stage 9 React's evolution_state.
        "runtime_failed_configs": sorted(runtime_failed_configs),
        "runtime_failure_categories": runtime_failure_categories,
    }
    if cap_exceeded:
        # Too many infra failures — block the loop and surface the limit so
        # the operator can triage (fix infra, amend brief, or both).
        result["runtime_failure_cap_exceeded"] = {
            "limit": cap,
            "observed": len(runtime_failed_configs),
            "fraction": MAX_RUNTIME_FAILED_FRACTION,
            "configs": sorted(runtime_failed_configs),
        }
    return result
