"""Stage 9 — the ReAct iteration gate.

Stage 9 reads every iteration's finalized human feedback, decides whether the
recommended evaluation protocol is good enough, and either terminates the run
(``ADVANCE`` -> Stage 10 final report) or spawns a new ``iter_NNN+1/`` that
carries the cumulative learning forward. The new iter may optionally re-run
Stage 1 to refresh the literature.

This module is the **deterministic backbone** the ``era-react`` skill drives:

- :func:`aggregate_cumulative_feedback` — walks ``iter_*/human/{feedback,
  human_labels}.json`` and returns the ``cumulative_feedback`` block of
  ``evolution_state.json`` (per-config endorsement trajectories + recurring
  ``general_feedback_themes``).
- :func:`react_state` / :func:`react_tick` — the bounded-loop counter, the
  exact shape of :mod:`era.orchestration.debate`. ``react.max_iterations``
  (from ``config.yaml``) is the hard cap; once ``status.iteration`` reaches it,
  any ``REVISE_*`` verdict is forced to ``ADVANCE`` so the loop terminates.
- :func:`create_next_iteration` — atomically scaffolds ``iter_{N+1}/``, swaps
  the ``current`` pointer, populates ``iteration.json.parent_feedback`` with
  the prior iter's carry-forward paths, and updates ``status.json``
  (``stage_index = 1`` for a literature refresh, ``2`` otherwise).
- :func:`check_evolution_state` — schema guard for the JSON the
  ``era-react-advisor`` sub-agent writes.

The semantic synthesis (per-config trajectories -> ``evolution_proposals``,
the ADVANCE/REVISE call, the literature_update_brief text) lives in the
sub-agent. This module owns *structure*, *atomicity*, and the *bounded loop*.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ERAConfig
from ..workspace import Workspace, resolve_workspace_root
from .human_feedback import _iter_num
from .lifecycle import STAGES

# The Stage 9 verdicts. ``ADVANCE`` terminates the run. ``REVISE_SKIP_STAGE1``
# spawns a new iter that starts at Stage 2. ``REVISE_RERUN_STAGE1`` spawns a
# new iter that starts at Stage 1 (the advisor's literature_update_brief.md
# tells Stage 1 which sections to refresh).
VERDICTS = ("ADVANCE", "REVISE_SKIP_STAGE1", "REVISE_RERUN_STAGE1")

# Typed evolution proposals. The advisor picks one type per proposal so Stage 2
# brainstorm knows how to integrate it.
PROPOSAL_TYPES = (
    "prompt_rewrite", "model_upsize", "hybrid_compose", "prompt_restructure",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _workspace(workspace_path: str | Path) -> Workspace:
    """Build a ``Workspace`` handle from a workspace root or ``current`` pointer."""
    root = resolve_workspace_root(workspace_path)
    return Workspace(root.parent, root.name)


def _max_iterations(ws: Workspace) -> int:
    config_path = ws.root / "config.yaml"
    if config_path.is_file():
        return ERAConfig.from_yaml(config_path).react.max_iterations
    return 5


def _iter_dir(ws: Workspace, n: int) -> Path:
    return ws.root / f"iter_{n:03d}"


# ---- cumulative feedback ------------------------------------------------

def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def aggregate_cumulative_feedback(
    workspace_path: str | Path, current_iteration: object,
) -> dict:
    """Walk every iter's finalized feedback and build the cumulative block.

    Returns:

        {
          "iters_analyzed": [int, ...],
          "per_config_trajectory": [
            {"combination_id", "modality",
             "endorsement_rate_by_iter": [(iter, rate, n_samples), ...],
             "wrong_themes": [str, ...]},
            ...
          ],
          "general_feedback_themes": [str, ...],
          "runtime_failure_signal": [
            {"iteration": int, "combination_id": str,
             "failure_category": str},
            ...
          ],
          "runtime_failure_category_counts": {"oom": int, "serving": int, ...}
        }

    Iterations whose ``human_labels.json`` does not exist (Stage 8 not yet
    finalized) are skipped. ``runtime_failure_signal`` is computed across
    *every* prior iter (not just feedback-finalized ones), because an infra
    failure that prevented an iteration from ever reaching human feedback is
    still a re-plan signal the advisor needs to see. The shape is what
    ``evolution_state.json``'s ``cumulative_feedback`` block expects; the
    advisor sub-agent layers ``evolution_proposals`` / ``exclude_list`` /
    ``general_failure_modes`` on top.
    """
    ws = _workspace(workspace_path)
    current = _iter_num(current_iteration)

    iters_analyzed: list[int] = []
    trajectory: dict[str, dict] = {}
    general_themes: list[str] = []
    runtime_failures: list[dict] = []
    runtime_category_counts: dict[str, int] = {}

    for n in range(1, current + 1):
        iter_dir = _iter_dir(ws, n)
        # Scan Stage 6 experiment_state.json for runtime_failed eval tasks
        # regardless of whether human feedback finalized — an infra failure
        # that blocked an iteration from reaching Stage 8 is still a re-plan
        # signal the advisor needs to see for the next iter's brief.
        state = _read_json(iter_dir / "experiments" / "experiment_state.json")
        for tid, entry in (state or {}).get("tasks", {}).items():
            if not isinstance(entry, dict):
                continue
            if entry.get("outcome") != "runtime_failed":
                continue
            cat = entry.get("failure_category") or "unknown"
            # Pull combination_id from the task id pattern "eval-<mode>-<cid>"
            # if the state entry doesn't carry it directly.
            cid = entry.get("combination_id") or ""
            if not cid and isinstance(tid, str):
                parts = tid.split("-", 2)
                cid = parts[2] if len(parts) == 3 else tid
            runtime_failures.append({
                "iteration": n, "combination_id": cid,
                "task_id": tid, "failure_category": cat,
            })
            runtime_category_counts[cat] = runtime_category_counts.get(cat, 0) + 1

        labels = _read_json(iter_dir / "human" / "human_labels.json")
        feedback = _read_json(iter_dir / "human" / "feedback.json")
        if labels is None:
            continue
        iters_analyzed.append(n)

        # Per-config endorsement / correct rate trajectory.
        for entry in labels.get("config_summary", []) or []:
            cid = entry.get("combination_id")
            if not cid:
                continue
            slot = trajectory.setdefault(cid, {
                "combination_id": cid,
                "modality": entry.get("modality"),
                "family": entry.get("family"),
                "endorsement_rate_by_iter": [],
                "wrong_themes": [],
            })
            rate = (
                entry.get("endorsement_rate")
                if "endorsement_rate" in entry
                else entry.get("correct_rate")
            )
            total = entry.get("total")
            slot["endorsement_rate_by_iter"].append({
                "iteration": n, "rate": rate, "n_samples": total,
            })

        # Wrong-themes: collect non-empty comments from item_marks /
        # comparison_marks, attributed to the config they targeted.
        if feedback:
            for mark in feedback.get("item_marks", []) or []:
                cid = mark.get("combination_id")
                comment = (mark.get("comment") or "").strip()
                if cid and comment and cid in trajectory:
                    trajectory[cid]["wrong_themes"].append(
                        f"iter_{n:03d}: {comment}"
                    )
            for mark in feedback.get("comparison_marks", []) or []:
                cid = mark.get("combination_id")
                comment = (mark.get("comment") or "").strip()
                if cid and comment and cid in trajectory:
                    trajectory[cid]["wrong_themes"].append(
                        f"iter_{n:03d}: {comment}"
                    )

        gf = (
            (feedback.get("general_feedback") if feedback else None)
            or labels.get("general_feedback")
            or ""
        ).strip()
        if gf:
            general_themes.append(f"iter_{n:03d}: {gf}")

    return {
        "iters_analyzed": iters_analyzed,
        "per_config_trajectory": list(trajectory.values()),
        "general_feedback_themes": general_themes,
        "runtime_failure_signal": runtime_failures,
        "runtime_failure_category_counts": runtime_category_counts,
    }


# ---- bounded loop -------------------------------------------------------

def _decision_path(ws: Workspace, n: int) -> Path:
    return _iter_dir(ws, n) / "react" / "decision.json"


def _history_path(ws: Workspace, n: int) -> Path:
    return _iter_dir(ws, n) / "react" / "history.jsonl"


def react_state(workspace_path: str | Path) -> dict:
    """Snapshot Stage 9 state for the current iteration."""
    ws = _workspace(workspace_path)
    if not ws.exists():
        return {
            "error": "not_a_workspace",
            "message": f"{ws.root} has no status.json — run /era:init first",
        }
    status = ws.read_status()
    current = _iter_num(status.get("iteration", 1))
    cap = _max_iterations(ws)
    prior_decisions: list[dict] = []
    for n in range(1, current + 1):
        decision = _read_json(_decision_path(ws, n))
        if decision is not None:
            prior_decisions.append(decision)
    return {
        "status": "ok",
        "iteration": current,
        "max_iterations": cap,
        "at_cap": current >= cap,
        "prior_decisions": prior_decisions,
    }


def react_tick(
    workspace_path: str | Path, verdict: str,
    rationale: str | None = None,
) -> dict:
    """Record a Stage 9 verdict and return the next action.

    The verdict is one of ``VERDICTS``. ``ADVANCE`` always passes through;
    ``REVISE_SKIP_STAGE1`` / ``REVISE_RERUN_STAGE1`` pass through *unless* the
    workspace is already at ``react.max_iterations``, in which case the
    verdict is **forced to ADVANCE** so the loop terminates.

    The decision is written to ``iter_NNN/react/decision.json`` (overwritten on
    re-tick — the latest call wins) and appended to
    ``iter_NNN/react/history.jsonl`` (every call recorded, so the audit trail
    is preserved).
    """
    ws = _workspace(workspace_path)
    if not ws.exists():
        return {
            "error": "not_a_workspace",
            "message": f"{ws.root} has no status.json — run /era:init first",
        }
    verdict = str(verdict).upper()
    if verdict not in VERDICTS:
        return {
            "error": "bad_verdict",
            "message": f"verdict must be one of: {', '.join(VERDICTS)}",
        }

    status = ws.read_status()
    current = _iter_num(status.get("iteration", 1))
    cap = _max_iterations(ws)

    forced = False
    decision = verdict
    if verdict != "ADVANCE" and current >= cap:
        decision = "ADVANCE"
        forced = True

    entry = {
        "iteration": current,
        "verdict": verdict,
        "decision": decision,
        "forced": forced,
        "at": _now_iso(),
    }
    if rationale:
        entry["rationale"] = str(rationale)

    iter_dir = _iter_dir(ws, current)
    (iter_dir / "react").mkdir(parents=True, exist_ok=True)
    _write_json_atomic(_decision_path(ws, current), entry)

    hist = _history_path(ws, current)
    with hist.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return {
        "status": "ok",
        "decision": decision,
        "forced": forced,
        "iteration": current,
        "max_iterations": cap,
        "decision_path": str(_decision_path(ws, current)),
    }


# ---- next iteration -----------------------------------------------------

def create_next_iteration(
    workspace_path: str | Path, *, rerun_stage1: bool,
) -> dict:
    """Atomically advance to ``iter_{N+1}``.

    Scaffolds ``iter_{N+1}/`` via :meth:`Workspace.create_iteration`,
    populating its ``iteration.json.parent_feedback`` with workspace-relative
    pointers to the prior iter's carry-forward artifacts (``human_labels.json``,
    ``react/evolution_state.json``, and ``react/literature_update_brief.md`` if
    it exists). Swaps the ``current`` symlink to the new iter and updates
    ``status.json``: ``iteration`` += 1, ``run_state = running``, and
    ``stage_index`` set to the *last completed* stage so the next ralph pass
    dispatches the right stage (``ralph_loop.md`` §80): ``0`` for
    ``rerun_stage1`` (so Stage 1 / research runs next) or ``1`` otherwise
    (so Stage 2 / plan_brainstorm runs next, skipping the literature
    refresh).

    Idempotent: re-calling when ``iter_{N+1}`` already exists merges any
    newly-resolved ``parent_feedback`` pointer without overwriting prior
    state.
    """
    ws = _workspace(workspace_path)
    if not ws.exists():
        return {
            "error": "not_a_workspace",
            "message": f"{ws.root} has no status.json — run /era:init first",
        }

    status = ws.read_status()
    current = _iter_num(status.get("iteration", 1))
    next_n = current + 1
    cap = _max_iterations(ws)
    if current >= cap:
        return {
            "error": "at_cap",
            "message": (
                f"iteration {current} is at react.max_iterations={cap}; "
                f"react_tick should have forced ADVANCE"
            ),
            "iteration": current,
            "max_iterations": cap,
        }

    # Build parent_feedback from the prior iter's well-known paths. Only
    # include pointers whose file exists — Stage 2 / Stage 7 / Stage 9
    # use a missing pointer to fall back to single-iter behavior.
    prior_dir = _iter_dir(ws, current)
    pf: dict[str, str] = {}
    candidates = {
        "human_labels": prior_dir / "human" / "human_labels.json",
        "evolution_state": prior_dir / "react" / "evolution_state.json",
        "literature_update_brief": prior_dir / "react" / "literature_update_brief.md",
        # Auto-revise marker — present when the prior iter blocked at a
        # pre-Stage-8 stage and routed through era.orchestration.auto_revise.
        # Stage 9's advisor reads this to learn which stage failed and
        # adjust the next iter's brief accordingly (e.g. drop the failing
        # configs from candidate_configs).
        "auto_revise_trigger": prior_dir / "auto_revise" / "trigger.json",
    }
    for key, path in candidates.items():
        if path.is_file():
            pf[key] = str(path.relative_to(ws.root))
    parent_feedback = pf or None

    iter_dir = ws.create_iteration(next_n, parent_feedback=parent_feedback)
    ws.set_current(next_n)

    # ``stage_index`` stores the *last completed* stage (ralph_loop.md §80):
    # the next ralph pass dispatches ``STAGES[stage_index + 1]``. For a fresh
    # REVISE_RERUN_STAGE1 iter we want Stage 1 (``research``) to run, so the
    # new iter's last-completed marker is Stage 0 (``task_init`` —
    # workspace-global, ran once at /era:init). For REVISE_SKIP_STAGE1 we
    # want Stage 2 (``plan_brainstorm``) to run, so the marker is Stage 1
    # (``research`` — carried forward from the prior iter).
    last_completed_index = 0 if rerun_stage1 else 1
    last_completed_stage = STAGES[last_completed_index]
    new_status = dict(status)
    new_status["iteration"] = next_n
    new_status["stage"] = last_completed_stage
    new_status["stage_index"] = last_completed_index
    new_status["run_state"] = "running"
    new_status["updated_at"] = _now_iso()
    ws.write_status(new_status)

    return {
        "status": "ok",
        "iteration": next_n,
        "stage": last_completed_stage,
        "stage_index": last_completed_index,
        "rerun_stage1": rerun_stage1,
        "iter_path": str(iter_dir),
        "parent_feedback": parent_feedback,
    }


# ---- schema guard -------------------------------------------------------

REQUIRED_TOP_KEYS = (
    "schema_version", "iteration", "cumulative_feedback", "exclude_list",
    "evolution_proposals", "general_failure_modes", "decision",
    "literature_update_requested",
)


def check_evolution_state(payload: Any) -> list[str]:
    """Return a list of schema problems for an ``evolution_state.json`` payload.

    Empty list == valid. Each problem is a human-readable string naming the
    bad field — Stage 9's skill surfaces these to the operator log so a
    malformed sub-agent run is visible.
    """
    problems: list[str] = []
    if not isinstance(payload, dict):
        return ["evolution_state must be a JSON object"]

    for key in REQUIRED_TOP_KEYS:
        if key not in payload:
            problems.append(f"missing required key: {key!r}")

    decision = payload.get("decision")
    if decision is not None and decision not in VERDICTS:
        problems.append(
            f"decision must be one of {list(VERDICTS)}, got {decision!r}"
        )

    lit_req = payload.get("literature_update_requested")
    if lit_req is not None and not isinstance(lit_req, bool):
        problems.append("literature_update_requested must be a boolean")
    if (decision == "REVISE_RERUN_STAGE1" and lit_req is False):
        problems.append(
            "decision REVISE_RERUN_STAGE1 requires literature_update_requested=true"
        )
    if (decision in {"ADVANCE", "REVISE_SKIP_STAGE1"} and lit_req is True):
        problems.append(
            f"decision {decision!r} must have literature_update_requested=false"
        )

    cumulative = payload.get("cumulative_feedback")
    if cumulative is not None and not isinstance(cumulative, dict):
        problems.append("cumulative_feedback must be an object")
    elif isinstance(cumulative, dict):
        for k in ("iters_analyzed", "per_config_trajectory",
                  "general_feedback_themes"):
            if k not in cumulative:
                problems.append(f"cumulative_feedback.{k} is missing")

    excludes = payload.get("exclude_list")
    if excludes is not None:
        if not isinstance(excludes, list):
            problems.append("exclude_list must be a list")
        else:
            for i, item in enumerate(excludes):
                if not isinstance(item, dict):
                    problems.append(f"exclude_list[{i}] must be an object")
                    continue
                for req in ("scope", "match", "reason"):
                    if req not in item:
                        problems.append(
                            f"exclude_list[{i}] is missing {req!r}"
                        )

    proposals = payload.get("evolution_proposals")
    if proposals is not None:
        if not isinstance(proposals, list):
            problems.append("evolution_proposals must be a list")
        else:
            seen_ids: set[str] = set()
            for i, prop in enumerate(proposals):
                if not isinstance(prop, dict):
                    problems.append(f"evolution_proposals[{i}] must be an object")
                    continue
                for req in ("id", "type", "for_combination", "proposal"):
                    if req not in prop:
                        problems.append(
                            f"evolution_proposals[{i}] is missing {req!r}"
                        )
                ptype = prop.get("type")
                if ptype is not None and ptype not in PROPOSAL_TYPES:
                    problems.append(
                        f"evolution_proposals[{i}].type must be one of "
                        f"{list(PROPOSAL_TYPES)}, got {ptype!r}"
                    )
                pid = prop.get("id")
                if pid:
                    if pid in seen_ids:
                        problems.append(
                            f"evolution_proposals[{i}].id {pid!r} is not unique"
                        )
                    seen_ids.add(pid)

    failure_modes = payload.get("general_failure_modes")
    if failure_modes is not None and not isinstance(failure_modes, list):
        problems.append("general_failure_modes must be a list")

    # Phase C-2.5 (Step 2a) — must_include_configs: optional list of
    # combination_id strings the next iter MUST re-evaluate. Stage 9's
    # advisor populates this from the prior iter's partial-pass diagnostic
    # (passing_configs that cleared the C-2 gate but were fewer than
    # min_passing). Stage 4's brief gate enforces
    # chosen_configs ⊇ must_include_configs. Empty/absent means "no
    # carry-forward elitism" (today's behavior).
    must_include = payload.get("must_include_configs")
    if must_include is not None:
        if not isinstance(must_include, list):
            problems.append("must_include_configs must be a list of strings")
        else:
            seen: set[str] = set()
            for i, cid in enumerate(must_include):
                if not isinstance(cid, str) or not cid:
                    problems.append(
                        f"must_include_configs[{i}] must be a non-empty string"
                    )
                    continue
                if cid in seen:
                    problems.append(
                        f"must_include_configs[{i}] {cid!r} is duplicated"
                    )
                seen.add(cid)

    # Phase C-2.5 (Step 2b) — lessons_learned: structured EA-style
    # reflection. Optional object with three lists of pattern records.
    # Each pattern record has {pattern: str, evidence: list[str],
    # confidence: "low"|"medium"|"high", since_iter: int, plus
    # optional remedy_proposed: str}. open_questions is a flat list
    # of strings.
    lessons = payload.get("lessons_learned")
    if lessons is not None:
        if not isinstance(lessons, dict):
            problems.append("lessons_learned must be an object")
        else:
            for key in ("success_patterns", "failure_patterns"):
                entries = lessons.get(key)
                if entries is None:
                    continue
                if not isinstance(entries, list):
                    problems.append(f"lessons_learned.{key} must be a list")
                    continue
                for i, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        problems.append(
                            f"lessons_learned.{key}[{i}] must be an object"
                        )
                        continue
                    if not entry.get("pattern") or not isinstance(
                        entry.get("pattern"), str
                    ):
                        problems.append(
                            f"lessons_learned.{key}[{i}].pattern must be "
                            f"a non-empty string"
                        )
                    ev = entry.get("evidence")
                    if ev is not None and (
                        not isinstance(ev, list)
                        or not all(isinstance(e, str) for e in ev)
                    ):
                        problems.append(
                            f"lessons_learned.{key}[{i}].evidence must be "
                            f"a list of strings"
                        )
                    conf = entry.get("confidence")
                    if conf is not None and conf not in {
                        "low", "medium", "high"
                    }:
                        problems.append(
                            f"lessons_learned.{key}[{i}].confidence must "
                            f"be one of low/medium/high, got {conf!r}"
                        )
                    since = entry.get("since_iter")
                    if since is not None and not (
                        isinstance(since, int) and since >= 1
                    ):
                        problems.append(
                            f"lessons_learned.{key}[{i}].since_iter must "
                            f"be a positive int"
                        )
            opens = lessons.get("open_questions")
            if opens is not None and (
                not isinstance(opens, list)
                or not all(isinstance(q, str) for q in opens)
            ):
                problems.append(
                    "lessons_learned.open_questions must be a list of strings"
                )

    # Phase C-2.5 (Step 2b) — hall_of_fame: top-K configs across all
    # iters, ranked by fitness_composite descending. Each entry is
    # {hypothesis_id: str, combination_id: str, fitness_composite: float
    # in [0,1], pass_rate: float, recall_rate: float,
    # human_endorsement_rate: float, iter_introduced: int,
    # last_iter_seen: int, lineage: list[str]}. ``must_include_configs``
    # must be a subset of hall_of_fame's combination_ids (the elite
    # carry-forward picks from the population memory).
    hof = payload.get("hall_of_fame")
    if hof is not None:
        if not isinstance(hof, list):
            problems.append("hall_of_fame must be a list")
        else:
            hof_cids: set[str] = set()
            for i, entry in enumerate(hof):
                if not isinstance(entry, dict):
                    problems.append(f"hall_of_fame[{i}] must be an object")
                    continue
                cid = entry.get("combination_id")
                if not isinstance(cid, str) or not cid:
                    problems.append(
                        f"hall_of_fame[{i}].combination_id must be a "
                        f"non-empty string"
                    )
                else:
                    if cid in hof_cids:
                        problems.append(
                            f"hall_of_fame[{i}].combination_id {cid!r} is "
                            f"duplicated"
                        )
                    hof_cids.add(cid)
                fit = entry.get("fitness_composite")
                if fit is not None and not (
                    isinstance(fit, (int, float)) and 0.0 <= fit <= 1.0
                ):
                    problems.append(
                        f"hall_of_fame[{i}].fitness_composite must be a "
                        f"number in [0,1]"
                    )
                for fkey in ("pass_rate", "recall_rate",
                             "human_endorsement_rate"):
                    fval = entry.get(fkey)
                    if fval is not None and not (
                        isinstance(fval, (int, float)) and 0.0 <= fval <= 1.0
                    ):
                        problems.append(
                            f"hall_of_fame[{i}].{fkey} must be a number "
                            f"in [0,1]"
                        )
                for ikey in ("iter_introduced", "last_iter_seen"):
                    ival = entry.get(ikey)
                    if ival is not None and not (
                        isinstance(ival, int) and ival >= 1
                    ):
                        problems.append(
                            f"hall_of_fame[{i}].{ikey} must be a positive int"
                        )
                lineage = entry.get("lineage")
                if lineage is not None and (
                    not isinstance(lineage, list)
                    or not all(isinstance(p, str) for p in lineage)
                ):
                    problems.append(
                        f"hall_of_fame[{i}].lineage must be a list of strings"
                    )
            # must_include ⊆ hall_of_fame: elitism picks from population
            # memory. Skip the check if either is empty.
            if isinstance(must_include, list) and hof_cids:
                stray = [
                    cid for cid in must_include
                    if isinstance(cid, str) and cid and cid not in hof_cids
                ]
                if stray:
                    problems.append(
                        f"must_include_configs {sorted(stray)} are not in "
                        f"hall_of_fame — every elite carry-forward must "
                        f"come from the population memory"
                    )

    return problems


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


__all__ = [
    "VERDICTS",
    "PROPOSAL_TYPES",
    "aggregate_cumulative_feedback",
    "react_state",
    "react_tick",
    "create_next_iteration",
    "check_evolution_state",
]
