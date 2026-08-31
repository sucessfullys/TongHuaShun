"""Lifecycle helpers for the ERA runtime — inspect and patch ``status.json``.

``/era:status`` / `/era:stop` / `/era:resume` and the Ralph loop use these to
read a workspace's progress and to flip its ``run_state`` / advance its
``stage`` without re-scaffolding. All lifecycle state lives in
``<workspace>/status.json`` — there is no separate daemon or breadcrumb file.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from ..workspace import Workspace, resolve_workspace_root

# The 11 ERA pipeline stages, in order — the single source of truth for stage
# ids and their indices. `stage_index` is the position in this tuple. There is
# no standalone deployment stage: ERA retrieves an evaluation protocol, never
# trains/hosts a model that needs deploying, and Stage 6 already proves every
# evaluator runs. There is no `dataset_ship` stage either — ERA does not ship
# datasets; v0.1.6 removed the slot when the ReAct iteration gate landed at
# index 9.
STAGES = (
    "task_init",             # 0  Stage 0 — /era:init
    "research",              # 1  Stage 1 — literature research (may be re-run
                             #            by Stage 9's REVISE_RERUN_STAGE1)
    "plan_brainstorm",       # 2
    "multi_review",          # 3
    "plan_decision",         # 4
    "experiment_plan",       # 5
    "full_experiment",       # 6
    "pre_human_comparison",  # 7
    "human_feedback",        # 8  the loop pauses here for human review
    "react",                 # 9  Stage 9 — ReAct iteration gate
    "final_report",          # 10
)

# Fields a caller may patch into status.json via update_status().
# `iteration` is deliberately excluded — changing it must go through a future
# start_new_iteration() helper that also creates iter_NNN/ and repoints current.
PATCHABLE_FIELDS = ("stage", "stage_index", "run_state")

# Allowed values for the run_state field:
#   idle           — initialized (/era:init done), never started
#   running        — the Ralph loop is active
#   awaiting_human — paused at the human-feedback stage, needs operator input
#   blocked        — stopped because the next stage has no runner (v0.1.x stub)
#   done           — the pipeline completed through the end
#   stopped        — halted by the operator via /era:stop
RUN_STATES = ("idle", "running", "awaiting_human", "blocked", "done", "stopped")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _workspace(workspace_path: str | Path) -> Workspace:
    """Build a ``Workspace`` handle from a workspace root or ``current`` pointer."""
    root = resolve_workspace_root(workspace_path)
    return Workspace(root.parent, root.name)


def _resolve_stage_fields(fields: dict) -> dict | None:
    """Validate and cross-fill ``stage`` / ``stage_index`` against ``STAGES``.

    Returns an ``error`` dict on a bad value, or ``None`` when ``fields`` is
    valid (mutated in place so the missing one of the pair is derived).
    """
    stage = fields.get("stage")
    index = fields.get("stage_index")

    if stage is not None and stage not in STAGES:
        return {
            "error": "bad_stage",
            "message": f"stage must be one of: {', '.join(STAGES)}",
        }
    if index is not None and (
        isinstance(index, bool) or not isinstance(index, int)
        or not 0 <= index < len(STAGES)
    ):
        return {
            "error": "bad_stage",
            "message": f"stage_index must be an integer in [0, {len(STAGES)})",
        }
    if stage is not None and index is not None and STAGES[index] != stage:
        return {
            "error": "bad_stage",
            "message": (
                f"stage {stage!r} does not match stage_index {index} "
                f"({STAGES[index]!r})"
            ),
        }
    if stage is not None and index is None:
        fields["stage_index"] = STAGES.index(stage)
    elif index is not None and stage is None:
        fields["stage"] = STAGES[index]
    return None


def update_status(workspace_path: str | Path, **fields: object) -> dict:
    """Patch a whitelisted set of fields into ``<workspace>/status.json``.

    Accepts ``stage``, ``stage_index``, ``run_state``. ``stage`` and
    ``stage_index`` are validated against ``STAGES`` and the missing one of the
    pair is derived automatically. ``run_state`` is validated against
    ``RUN_STATES``. ``updated_at`` is always refreshed. Returns the updated
    status dict, or an ``error`` dict on failure.
    """
    ws = _workspace(workspace_path)
    if not ws.exists():
        return {
            "error": "not_a_workspace",
            "message": f"{ws.root} has no status.json — run /era:init first",
            "workspace_path": str(ws.root),
        }

    unknown = [k for k in fields if k not in PATCHABLE_FIELDS]
    if unknown:
        return {
            "error": "unknown_fields",
            "message": f"not patchable: {', '.join(sorted(unknown))}",
            "patchable": list(PATCHABLE_FIELDS),
        }
    if "run_state" in fields and fields["run_state"] not in RUN_STATES:
        return {
            "error": "bad_run_state",
            "message": f"run_state must be one of: {', '.join(RUN_STATES)}",
        }
    stage_error = _resolve_stage_fields(fields)
    if stage_error is not None:
        return stage_error

    status = ws.read_status()
    status.update(fields)
    status["updated_at"] = _now_iso()
    ws.write_status(status)
    return {"status": "ok", "workspace_path": str(ws.root), "status_json": status}


def _config_warnings(root: Path) -> list[str]:
    """Surface backward-incompat warnings for a workspace's ``config.yaml``.

    Phase C-2.5 introduced ``experiment.auto_validate_min_passing`` with
    default 3. Workspaces predating this phase that don't pin the value
    will silently inherit the stricter rule (iters that previously
    advanced on 1 passing config now auto-revise). Surface a warning so
    operators upgrading from <v0.1.7 are not surprised.
    """
    cfg_path = root / "config.yaml"
    if not cfg_path.is_file():
        return []
    try:
        raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(raw, dict):
        return []
    exp = raw.get("experiment") or {}
    if not isinstance(exp, dict):
        return []
    warnings: list[str] = []
    if "auto_validate_min_passing" not in exp:
        warnings.append(
            "config.yaml does not pin experiment.auto_validate_min_passing"
            " — inheriting the v0.1.7 Phase C-2.5 default of 3. Iters"
            " that previously advanced with 1-2 passing configs will"
            " now auto-revise to gather more. Pin"
            " experiment.auto_validate_min_passing: 1 in config.yaml to"
            " restore legacy behavior, or accept the new default."
        )
    return warnings


def status_summary(
    workspaces_dir: str | Path,
    workspace_path: str | Path | None = None,
) -> dict:
    """Summarize one workspace, or every workspace under ``workspaces_dir``."""
    if workspace_path:
        roots = [_workspace(workspace_path).root]
    else:
        base = Path(workspaces_dir).expanduser()
        roots = sorted(
            p for p in (base.iterdir() if base.is_dir() else [])
            if (p / "status.json").exists()
        )

    projects = []
    for root in roots:
        status = Workspace(root.parent, root.name).read_status()
        if not status:
            continue
        summary = {
            "project_name": status.get("project_name", root.name),
            "stage": status.get("stage"),
            "stage_index": status.get("stage_index"),
            "iteration": status.get("iteration"),
            "run_state": status.get("run_state", "unknown"),
            "updated_at": status.get("updated_at"),
            "has_literature": (root / "research" / "literature.md").is_file(),
            "workspace_path": str(root),
        }
        warnings = _config_warnings(root)
        if warnings:
            summary["warnings"] = warnings
        projects.append(summary)
    return {"status": "ok", "count": len(projects), "projects": projects}
