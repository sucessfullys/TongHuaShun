"""``cli_init_project`` — the single Python entrypoint for ERA Stage 0.

Given the operator-confirmed facts (a plain dict), it builds an ``ERAConfig``,
validates it, scaffolds an iteration-aware workspace, writes every Stage 0
artifact, and returns a JSON-serializable result containing the operator guide.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..config import ERAConfig
from ..probe.annotations import probe_annotations
from ..prompt_loader import render_prompt
from ..workspace import Workspace
from .guide import build_post_init_guide
from .mcp import ensure_mcp_config
from .ralph import ensure_claude_settings

_WORKSPACE_GITIGNORE = """\
# ERA workspace — local runtime state
*.tmp
.current.tmp
__pycache__/
"""


def cli_init_project(
    params: dict,
    *,
    repo_root: Path,
    workspaces_dir: Path,
) -> dict:
    """Scaffold an ERA project workspace from confirmed init parameters.

    ``params`` carries the config groups (``hardware``, ``data``, …) plus an
    optional ``probe`` sub-dict of raw probe artifacts. Returns a dict with
    either an ``error`` key or ``{project_name, workspace_path, spec_path,
    config_path, guide}``.
    """
    project_name = _slugify(params.get("project_name") or _derive_name(params))
    ws = Workspace(workspaces_dir, project_name)
    if ws.exists():
        return {
            "error": "workspace_exists",
            "message": (
                f"a workspace named {project_name!r} already exists at "
                f"{ws.root} — choose a different project name"
            ),
            "workspace_path": str(ws.root),
        }

    cfg_params = dict(params)
    cfg_params.pop("probe", None)
    cfg_params["project_name"] = project_name
    cfg_params["workspaces_dir"] = str(workspaces_dir)
    cfg_params["era_version"] = __version__
    cfg_params["created_at"] = cfg_params.get("created_at") or _now_iso()
    cfg = ERAConfig.from_params(cfg_params)

    problems = cfg.validate()
    if problems:
        return {"error": "invalid_config", "problems": problems}

    # ---- scaffold ----
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)

    now = _now_iso()
    ws.write_status({
        "project_name": project_name,
        "era_version": __version__,
        "stage": "task_init",
        "stage_index": 0,
        "run_state": "idle",
        "iteration": 1,
        "iteration_dirs": True,
        "created_at": now,
        "updated_at": now,
    })

    ws.write_file("config.yaml", cfg.to_commented_yaml())
    ws.write_file("spec.md", _render_spec(cfg))
    ws.write_file("CLAUDE.md", _render_claude_md(cfg))
    ws.write_file(".gitignore", _WORKSPACE_GITIGNORE)

    # Wire the workspace for Claude Code, *before* the operator launches a
    # session here: .claude/settings.json enables the ralph-loop loop engine
    # and auto-trusts MCP; .mcp.json registers the arXiv MCP server so Stage 1
    # literature research reaches arXiv from inside the workspace.
    # Pass the caller's ``repo_root`` so the workspace inherits any
    # operator pins from ``<repo_root>/.claude/settings.json`` (Phase
    # D-2 extension). Tests inject a fake repo_root for isolation.
    ensure_claude_settings(ws.root, repo_root_path=repo_root)
    ensure_mcp_config(ws.root)

    probe = params.get("probe") or {}
    # Annotation probe: if the agent-driven init flow didn't pre-run it,
    # probe inline so the post-init guide can report any pre-existing
    # operator annotations from a prior /era:annotate run.
    annotation_probe = probe.get("annotations")
    if annotation_probe is None and cfg.data.data_root:
        method_paths = {
            m.get("method_id"): m.get("path")
            for m in cfg.data.methods
            if m.get("method_id") and m.get("path")
        }
        try:
            annotation_probe = probe_annotations(
                cfg.data.data_root, method_paths or None,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            # Probe never blocks init — a missing/unreadable annotations
            # tree, bad JSON, or bad sample_key just degrades to "no
            # pre-existing annotations" in the post-init guide.
            annotation_probe = None
    for key, filename in (
        ("gpu", "gpu_inventory.json"),
        ("data", "data_layout.json"),
        ("checkpoints", "checkpoints.json"),
        ("credentials", "credentials.json"),
        ("annotations", "annotations.json"),
    ):
        payload = probe.get(key)
        if key == "annotations" and payload is None:
            payload = annotation_probe
        ws.write_json(f"probe/{filename}", payload or {})

    guide = build_post_init_guide(
        ws=ws,
        cfg=cfg,
        repo_root=repo_root,
        warnings=_collect_warnings(cfg),
        annotations=annotation_probe or {},
    )

    return {
        "project_name": project_name,
        "workspace_path": str(ws.root),
        "spec_path": str(ws.root / "spec.md"),
        "config_path": str(ws.root / "config.yaml"),
        "guide": guide,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "era-project"


def _derive_name(params: dict) -> str:
    adapter = (params.get("task_adapter") or "").strip()
    if adapter and adapter != "generic":
        return f"{adapter}-eval"
    mission = params.get("mission") or "era project"
    return "-".join(mission.split()[:4])


def _collect_warnings(cfg: ERAConfig) -> list[str]:
    warnings: list[str] = []
    if not cfg.hardware.probe_ok:
        warnings.append(
            "GPU probe failed — hardware values were entered manually; "
            "verify config.yaml hardware section"
        )
    if not (cfg.credentials.openai or cfg.credentials.anthropic
            or cfg.credentials.google):
        warnings.append(
            "no API credentials detected — ERA will rely on locally-served "
            "VLM judges + metrics"
        )
    if not cfg.checkpoints.detected:
        warnings.append(
            "no local checkpoints detected under the model root"
        )
    return warnings


def _render_spec(cfg: ERAConfig) -> str:
    data = cfg.data
    creds = cfg.credentials
    return render_prompt(
        "spec_template",
        PROJECT_NAME=cfg.project_name,
        MISSION=cfg.mission or "(not provided)",
        TASK_FAMILY=cfg.task_family,
        TASK_ADAPTER=cfg.task_adapter,
        DATA_ROOT=data.data_root or "(not set)",
        LAYOUT=data.layout,
        SAMPLE_COUNT=data.sample_count,
        ITER_SAMPLE_COUNT=cfg.effective_iter_sample_count(),
        SAMPLE_GLOB=data.sample_glob or "(samples are direct children)",
        SAMPLE_KEY=data.sample_key,
        INPUT_ROOT=data.input_root or "(inputs co-located in each sample dir)",
        METHODS_TABLE=_methods_table(data.methods),
        INPUT_ROLES_TABLE=_input_roles_table(data.input_roles),
        GPU_LIST=_fmt_list(cfg.hardware.visible_gpu_ids),
        GPU_MODEL=cfg.hardware.gpu_model or "unknown",
        PER_GPU_MEM=cfg.hardware.per_gpu_memory_gb,
        RESERVE_GPUS=_fmt_list(cfg.hardware.reserve_gpu_ids) or "none",
        MAX_GPUS=cfg.hardware.max_gpus_per_run,
        SERVING_BACKEND=cfg.serving.backend,
        SERVING_FALLBACKS=", ".join(cfg.serving.fallbacks),
        MODEL_ROOT=cfg.checkpoints.local_model_root or "(none)",
        DETECTED_CKPTS=_fmt_list(cfg.checkpoints.detected) or "none",
        CRED_OPENAI=_yn(creds.openai),
        CRED_ANTHROPIC=_yn(creds.anthropic),
        CRED_GOOGLE=_yn(creds.google),
        BUDGET_API=cfg.budget.api_cost_cap_usd,
        BUDGET_WALL=cfg.budget.wallclock_cap_hours,
    )


def _render_claude_md(cfg: ERAConfig) -> str:
    return render_prompt(
        "workspace_claude_template",
        PROJECT_NAME=cfg.project_name,
        TASK_FAMILY=cfg.task_family,
        TASK_ADAPTER=cfg.task_adapter,
        ERA_VERSION=cfg.era_version,
    )


def _methods_table(methods: list[dict]) -> str:
    if not methods:
        return "_(no generation methods detected)_"
    rows = ["| Method | Path | Output file |", "| --- | --- | --- |"]
    for m in methods:
        rows.append(
            f"| `{m.get('method_id', '?')}` | `{m.get('path', '?')}` "
            f"| `{m.get('output_file', '?')}` |"
        )
    return "\n".join(rows)


def _input_roles_table(input_roles: dict) -> str:
    if not input_roles:
        return "_(no input roles — generation task, or inputs in a separate root)_"
    rows = ["| Role | Filename glob |", "| --- | --- |"]
    for role, glob in input_roles.items():
        rows.append(f"| `{role}` | `{glob}` |")
    return "\n".join(rows)


def _fmt_list(values) -> str:
    return ", ".join(str(v) for v in values) if values else ""


def _yn(value: bool) -> str:
    return "available" if value else "not available"
