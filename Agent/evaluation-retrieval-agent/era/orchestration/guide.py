"""Build the post-init operator guide printed verbatim by ``/era:init``."""

from __future__ import annotations

from pathlib import Path

from ..config import ERAConfig
from ..prompt_loader import render_prompt
from ..workspace import Workspace


def build_post_init_guide(
    *,
    ws: Workspace,
    cfg: ERAConfig,
    repo_root: Path,
    warnings: list[str],
    annotations: dict | None = None,
) -> str:
    """Render the boxed 'Init done' guide from the post_init_guide template."""
    warn_block = (
        "\n".join(f"  ! {w}" for w in warnings) if warnings else "  (none)"
    )
    return render_prompt(
        "post_init_guide",
        PROJECT_NAME=cfg.project_name,
        WORKSPACE_PATH=str(ws.root),
        PLUGIN_DIR=str(repo_root / "plugin"),
        SUMMARY=_summary_block(cfg, annotations or {}),
        WARNINGS=warn_block,
    )


def _summary_block(cfg: ERAConfig, annotations: dict | None = None) -> str:
    data = cfg.data
    hw = cfg.hardware
    methods = ", ".join(
        f"{m.get('method_id', '?')}→{m.get('output_file', '?')}"
        for m in data.methods
    ) or "(none)"
    inputs = ", ".join(
        f"{role}={glob}" for role, glob in data.input_roles.items()
    ) or "(none)"
    creds = [
        name for name, on in (
            ("openai", cfg.credentials.openai),
            ("anthropic", cfg.credentials.anthropic),
            ("google", cfg.credentials.google),
        ) if on
    ]
    api_cap = cfg.budget.api_cost_cap_usd
    budget_api = "no paid API" if api_cap == 0 else f"${api_cap} API"
    effective_cap = cfg.effective_iter_sample_count()
    cap_note = (
        f"{effective_cap} (≤ {data.sample_count} available)"
        if effective_cap < data.iter_sample_count
        else f"{effective_cap}"
    )
    # Annotation summary line — only shown when at least one annotation
    # was found by the probe; otherwise the line is skipped so the guide
    # stays tight for fresh datasets.
    annotation_line = None
    central_count = (annotations or {}).get("central_count", 0)
    if central_count:
        per_method_count = annotations.get("per_method_count") or {}
        per_method_total = annotations.get("per_method_count_total", 0)
        sync_note = "" if annotations.get("annotators_in_sync") else " (out of sync — re-run era.cli annotate-mirror)"
        annotation_line = (
            f"  Annotations   : {central_count} pre-annotated "
            f"(per-method copies: {per_method_total}){sync_note}"
        )
        # Surface per-method coverage when methods are uneven.
        coverage = annotations.get("method_coverage") or {}
        if coverage and len(set(coverage.values())) > 1:
            cov_summary = ", ".join(
                f"{mid}={n}" for mid, n in sorted(coverage.items())
            )
            annotation_line += f" · per-method: {cov_summary}"
    # Auto-validate threshold line — always shown so the operator sees
    # the gate the iteration loop is going to apply.
    exp = cfg.experiment
    auto_validate_line = (
        f"  Auto-validate : PASS ≥ {exp.auto_validate_pass_threshold:.2f}, "
        f"RECALL ≥ {exp.auto_validate_recall_threshold:.2f}, "
        f"MIN-PASSING ≥ {exp.auto_validate_min_passing} configs "
        f"(min {exp.auto_validate_min_samples} annotations to gate)"
    )
    use_evidence = "yes" if data.use_annotation_evidence else "no (Stage 2 ignores annotations)"
    annotation_use_line = f"  Use evidence  : Stage 2 reads annotations → {use_evidence}"

    lines = [
        f"  Mission       : {_truncate(cfg.mission, 70)}",
        f"  Task          : {cfg.task_family} / {cfg.task_adapter}",
        f"  Data          : {data.layout} · {data.sample_count} "
        f"samples/method · glob {data.sample_glob or '*'}",
        f"  Per-iter cap  : {cap_note} samples/method evaluated each iter",
        f"  Methods       : {methods}",
        f"  Input roles   : {inputs}",
    ]
    if annotation_line:
        lines.append(annotation_line)
        lines.append(annotation_use_line)
    lines.append(auto_validate_line)
    lines.extend([
        f"  GPUs          : {hw.visible_gpu_ids or '(none)'}"
        f"  ({hw.gpu_model or 'unknown'}, {hw.per_gpu_memory_gb} GB each)",
        f"  Serving       : {cfg.serving.backend}"
        f"  (fallbacks: {', '.join(cfg.serving.fallbacks)})",
        f"  Checkpoints   : {cfg.checkpoints.local_model_root or '(none)'}"
        f"  ({len(cfg.checkpoints.detected)} detected)",
        f"  Credentials   : {', '.join(creds) if creds else '(none)'}",
        f"  Budget        : {budget_api} / {cfg.budget.wallclock_cap_hours} h",
    ])
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
