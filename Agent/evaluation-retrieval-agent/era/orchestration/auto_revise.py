"""Auto-revise: replace pre-Stage-8 blocks with a Stage 9 REVISE.

The ralph-loop used to halt at ``run_state: blocked`` whenever a
pre-Stage-8 stage couldn't produce its expected output (Stage 4 brief
invalid, Stage 6 incomplete, Stage 7 comparison missing, etc.). This
module ships the alternative: any pre-Stage-8 failure routes through
:func:`auto_revise`, which records the failure context and triggers
:func:`era.orchestration.react.react_tick` with
``REVISE_SKIP_STAGE1`` so the next iter re-plans with the failure as
evidence.

The cap is unchanged: ``react.max_iterations`` (default 5) — when hit,
``react_tick`` forces ``decision: ADVANCE`` and ``auto_revise`` returns
``forced_advance: True`` without scaffolding a new iter. The caller
(ralph-loop) then advances ``stage_index`` normally, which lands the
loop at Stage 10's terminal block.

Idempotent: re-calling ``auto_revise`` on an iter that already wrote
``<iter>/auto_revise/trigger.json`` returns the prior trigger without
re-firing REVISE. Protects against two different ralph-loop stages
both deciding to auto-revise on the same iter.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..workspace import Workspace, resolve_workspace_root
from .experiment_results import write_json_atomic
from .react import (
    create_next_iteration,
    react_tick,
)

TRIGGER_REL = "auto_revise/trigger.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iter_num(value: object) -> int:
    """Coerce ``5`` / ``"5"`` / ``"iter_005"`` to the int ``5``."""
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.startswith("iter_"):
        text = text[len("iter_"):]
    return int(text)


def _trigger_path(workspace_root: Path, iteration: int) -> Path:
    return workspace_root / f"iter_{iteration:03d}" / TRIGGER_REL


def auto_revise(
    workspace_path: str | Path, *,
    reason: str,
    source_stage: int,
    blocker_summary: str,
    diagnostic: dict | None = None,
) -> dict:
    """Replace a pre-Stage-8 block with a Stage 9 REVISE_SKIP_STAGE1.

    Returns a dict with one of three shapes:

    - ``{"status": "ok", "decision": "REVISE_SKIP_STAGE1",
       "next_iter": int, "iter_path": str, "trigger_path": str,
       "forced_advance": False}`` — under-cap, new iter scaffolded.
    - ``{"status": "ok", "decision": "ADVANCE",
       "forced_advance": True, "trigger_path": str}`` — at cap,
       no new iter; ralph-loop should advance ``stage_index`` and let
       Stage 10 terminate the loop.
    - ``{"status": "ok", "already_triggered": True,
       "trigger_path": str, ...}`` — idempotent re-call.
    - ``{"error": ...}`` — workspace doesn't exist or react_tick failed.
    """
    if not reason:
        return {"error": "missing_reason",
                "message": "auto_revise requires a non-empty 'reason'"}
    if not isinstance(source_stage, int) or source_stage < 1 or source_stage > 7:
        return {"error": "bad_source_stage",
                "message": f"source_stage must be int in [1,7], got {source_stage!r}"}

    root = resolve_workspace_root(workspace_path)
    ws = Workspace(root.parent, root.name)
    if not ws.exists():
        return {"error": "not_a_workspace",
                "message": f"{root} has no status.json — run /era:init first"}

    status = ws.read_status()
    current = _iter_num(status.get("iteration", 1))
    trigger_path = _trigger_path(root, current)

    # Idempotency: if we already auto-revised this iter, return the prior
    # record without re-firing REVISE.
    if trigger_path.is_file():
        try:
            existing = json.loads(trigger_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = None
        if isinstance(existing, dict):
            return {
                "status": "ok",
                "already_triggered": True,
                "trigger_path": str(trigger_path),
                **existing,
            }

    # Record the trigger before firing react_tick so the next iter's
    # advisor (and the ralph-loop tests) can see it even if react_tick
    # itself fails.
    trigger: dict[str, Any] = {
        "iteration": current,
        "reason": reason,
        "source_stage": source_stage,
        "blocker_summary": blocker_summary,
        "diagnostic": diagnostic or {},
        "at": _now_iso(),
    }
    write_json_atomic(trigger_path, trigger)

    # Fire REVISE. react_tick handles the iter-cap force-ADVANCE itself,
    # so we just inspect its returned decision.
    rationale = f"auto: {reason}: {(blocker_summary or '')[:120]}"
    tick = react_tick(root, "REVISE_SKIP_STAGE1", rationale=rationale)
    if tick.get("error"):
        return {
            "error": "react_tick_failed",
            "message": tick.get("message", ""),
            "trigger_path": str(trigger_path),
        }

    decision = tick.get("decision")
    if decision == "ADVANCE":
        # Cap reached — ralph-loop advances stage_index normally, the loop
        # then hits Stage 10's terminal block on the next pass.
        return {
            "status": "ok",
            "decision": "ADVANCE",
            "forced_advance": True,
            "iteration": current,
            "max_iterations": tick.get("max_iterations"),
            "trigger_path": str(trigger_path),
        }

    # Under-cap REVISE — scaffold the next iter.
    advance = create_next_iteration(root, rerun_stage1=False)
    if advance.get("error"):
        return {
            "error": "create_next_iteration_failed",
            "message": advance.get("message", ""),
            "trigger_path": str(trigger_path),
        }
    return {
        "status": "ok",
        "decision": "REVISE_SKIP_STAGE1",
        "forced_advance": False,
        "iteration": current,
        "next_iter": advance["iteration"],
        "iter_path": advance.get("iter_path"),
        "stage_index": advance.get("stage_index"),
        "trigger_path": str(trigger_path),
    }


def read_trigger(iter_dir: Path) -> dict | None:
    """Read an iter's auto-revise trigger (if any) for Stage 9's advisor.

    Returns ``None`` when the iter wasn't auto-revised.
    """
    path = Path(iter_dir) / TRIGGER_REL
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
