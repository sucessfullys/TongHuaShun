"""Resolve which sub-agent tier runs a tiered pipeline stage.

The Stage 2-6 orchestrator skills call ``era.cli agent-tier`` to learn which
``era-<tier>`` sub-agent to dispatch their persona fan-out to. The tier is read
from ``config.yaml``'s ``agent_modes`` block — a deterministic lookup kept in
the ``era`` package so the skills never have to parse YAML themselves.
"""

from __future__ import annotations

from pathlib import Path

from ..config import AGENT_TIERS, ERAConfig
from ..workspace import resolve_workspace_root

# Pipeline stages that resolve a sub-agent tier — each is an ``agent_modes``
# field naming its tier (Stages 2-4 debate + Stage 5 plan + Stage 6 run).
TIERED_STAGES = (
    "plan_brainstorm", "multi_review", "plan_decision",
    "experiment_plan", "full_experiment",
)


def resolve_agent_tier(workspace_path: str | Path, stage: str) -> dict:
    """Return ``{tier, agent}`` for a tiered stage, read from ``config.yaml``.

    Returns an ``error`` dict for an unknown stage, a missing config, or a tier
    value outside ``AGENT_TIERS``.
    """
    if stage not in TIERED_STAGES:
        return {
            "error": "bad_stage",
            "message": f"stage must be one of: {', '.join(TIERED_STAGES)}",
        }
    root = resolve_workspace_root(workspace_path)
    config_path = root / "config.yaml"
    if not config_path.is_file():
        return {
            "error": "no_config",
            "message": f"{root} has no config.yaml — run /era:init first",
            "workspace_path": str(root),
        }

    cfg = ERAConfig.from_yaml(config_path)
    tier = getattr(cfg.agent_modes, stage)
    if tier not in AGENT_TIERS:
        return {
            "error": "bad_tier",
            "message": (
                f"agent_modes.{stage}={tier!r} is not one of {AGENT_TIERS}"
            ),
        }
    return {
        "status": "ok",
        "stage": stage,
        "tier": tier,
        "agent": f"era-{tier}",
    }
