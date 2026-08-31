"""The Stage 2-4 ADVANCE/REVISE debate-loop counter.

Stage 4 (``era-plan-decision``) runs a bounded refinement loop: the synthesizer
returns ADVANCE or REVISE, and Stage 4 re-runs the brainstorm + review until it
converges or hits ``debate.max_rounds``. The round counter is **deterministic
Python** — owned here, persisted atomically in
``iter_NNN/design/debate_state.json`` — so the cap is enforced even in the
no-Stop-hook manual fallback, and a fresh ralph context window recovers it.

``debate_tick`` is the single mutation point: it appends to ``history`` and,
crucially, forces ``advance`` once the cap is reached so the loop always
terminates.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..config import ERAConfig
from ..workspace import Workspace, resolve_workspace_root

VERDICTS = ("ADVANCE", "REVISE")

# The transient per-persona dirs the Stage 2/3 fan-out writes into and the merge
# step globs. They must be emptied at the start of each REVISE round so a stale
# round-N file cannot contaminate the round-(N+1) merge — mirrors SibylSystem's
# ``prepare_idea_refinement_round``.
_REVISE_TRANSIENT_DIRS = ("design/candidates", "design/reviews")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _workspace(workspace_path: str | Path) -> Workspace:
    """Build a ``Workspace`` handle from a workspace root or ``current`` pointer."""
    root = resolve_workspace_root(workspace_path)
    return Workspace(root.parent, root.name)


def _state_rel(ws: Workspace) -> str:
    """The debate-state path, relative to the workspace root."""
    return str((ws.iter_path() / "design" / "debate_state.json").relative_to(ws.root))


def _clear_revise_artifacts(ws: Workspace) -> list[str]:
    """Empty the transient per-persona debate dirs before a REVISE round.

    Mirrors SibylSystem's ``prepare_idea_refinement_round``: ``rmtree`` the
    directory the fan-out globs, then recreate it empty so the next round writes
    into clean state. A cleanup failure is swallowed — it must never break the
    bounded loop. Returns the (iteration-relative) dirs that are now clean.
    """
    iter_dir = ws.iter_path()
    cleared: list[str] = []
    for rel in _REVISE_TRANSIENT_DIRS:
        target = iter_dir / rel
        try:
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            cleared.append(rel)
        except OSError:
            pass
    return cleared


def _max_rounds(ws: Workspace) -> int:
    config_path = ws.root / "config.yaml"
    if config_path.is_file():
        return ERAConfig.from_yaml(config_path).debate.max_rounds
    return 4


def _load_or_init(ws: Workspace) -> dict:
    """Return the debate state, initializing + persisting it on first use."""
    path = ws.iter_path() / "design" / "debate_state.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    state = {
        "round": 1,
        "max_rounds": _max_rounds(ws),
        "history": [],
        "created_at": _now_iso(),
    }
    ws.write_json(_state_rel(ws), state)
    return state


def debate_state(workspace_path: str | Path) -> dict:
    """Read (initializing on first use) the Stage 2-4 debate state."""
    ws = _workspace(workspace_path)
    if not ws.exists():
        return {
            "error": "not_a_workspace",
            "message": f"{ws.root} has no status.json — run /era:init first",
        }
    state = _load_or_init(ws)
    return {"status": "ok", "state": state, "state_path": str(ws.root / _state_rel(ws))}


def debate_tick(workspace_path: str | Path, verdict: str,
                reason: str | None = None) -> dict:
    """Record a synthesizer verdict and return the next loop action.

    ``ADVANCE`` -> ``action: advance``. ``REVISE`` below the cap -> increment
    the round, ``action: revise``. ``REVISE`` at the cap -> ``action: advance``
    with ``forced: true`` (the loop is bounded — it must terminate).

    On ``action: revise`` the transient per-persona debate dirs
    (``design/candidates/`` and ``design/reviews/``) are emptied so the next
    round's fan-out starts clean and no stale round-N file leaks into the
    round-(N+1) merge. ``reason`` (the synthesizer's revision brief, when given)
    is recorded on the history entry so ``debate_state.json`` says *why* each
    round happened.
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

    state = _load_or_init(ws)
    current = int(state.get("round", 1))
    cap = int(state.get("max_rounds", 4))

    if verdict == "ADVANCE":
        action, forced = "advance", False
    elif current < cap:
        action, forced = "revise", False
    else:
        action, forced = "advance", True

    entry = {
        "round": current,
        "verdict": verdict,
        "action": action,
        "forced": forced,
        "at": _now_iso(),
    }
    if reason:
        entry["reason"] = str(reason)
    state.setdefault("history", []).append(entry)

    cleared: list[str] = []
    if action == "revise":
        state["round"] = current + 1
        cleared = _clear_revise_artifacts(ws)
    ws.write_json(_state_rel(ws), state)

    return {
        "status": "ok",
        "action": action,
        "forced": forced,
        "round": state["round"],
        "max_rounds": cap,
        "history_len": len(state["history"]),
        "cleared": cleared,
        "state_path": str(ws.root / _state_rel(ws)),
    }
