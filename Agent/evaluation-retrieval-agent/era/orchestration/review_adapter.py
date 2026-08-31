"""Stage 8 review-model adapter — normalize stage 2-6 artifacts for the web app.

The Stage 8 review web app must render *any* iteration of *any* AIGC task
correctly, not just the hand-built demo. Stages 2-6 emit partly agent-authored
artifacts — the per-config ``scores.jsonl`` especially — whose exact shape
varies (a runner may emit a string score, omit ``sub_scores``, name a key
``sample_id``, or use an unfamiliar ``scope``). This module reads whatever those
stages produced and writes one normalized, fully-specified artifact,
``iter_NNN/human/review_model.json``, that the web app renders directly.

It is deterministic and runs first (``era.cli build-review-model``); the
``era-review-adapter`` sub-agent runs afterwards to repair anything recorded in
``warnings``. The web app loads ``review_model.json`` when present and falls
back to a direct read otherwise, so older iterations and the standalone demo
keep working. This module never changes a stage 1-7 artifact — it only reads
them and writes its own file under ``iter_NNN/human/``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..config import ERAConfig
from ..webapp.data import (
    COMPARISON_FAMILY,
    FLAGGING_FAMILIES,
    build_review_model,
    classify_score,
    coerce_number,
    load_brief,
    load_hypotheses,
    review_payload,
)
from ..webapp.images import resolve_image
from .experiment_results import result_dir, write_json_atomic

SCHEMA_VERSION = "1.0"


def review_model_path(iter_dir: Path) -> Path:
    """Canonical path of the normalized review artifact for one iteration."""
    return iter_dir / "human" / "review_model.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---- the deterministic normalizer ---------------------------------------

def build_review_model_json(
    workspace_root: Path, iter_dir: Path, cfg: ERAConfig, mode: str | None = None,
) -> dict:
    """Normalize one iteration's stage 2-6 artifacts into a review_model dict.

    Reuses :func:`era.webapp.data.build_review_model` / ``review_payload`` for
    the base structure, then enriches it: per-config ``score_display`` render
    hints, per-image ``available`` flags, and a ``warnings`` list naming every
    artifact that was missing or had to be coerced — the sub-agent's worklist.
    """
    model = build_review_model(workspace_root, iter_dir, cfg, mode)
    warnings: list[str] = []

    brief = load_brief(iter_dir)
    if not brief:
        warnings.append("design/experiment_brief.json is missing or unreadable")
    elif not brief.get("candidate_configs"):
        warnings.append("experiment_brief.json carries no candidate_configs")
    if not load_hypotheses(iter_dir) and any(
        c.get("hypothesis_id") for c in model.configs
    ):
        warnings.append(
            "design/hypotheses.md is missing — config descriptions omit hypotheses"
        )
    if not model.configs:
        warnings.append("no evaluator configs found — the review would be empty")
    if not model.sample_keys:
        warnings.append("no scored samples found in any config's scores.jsonl")

    # Per-config: scores.jsonl presence, score-shape coercions, render hints.
    for cfg_entry in model.configs:
        cid = cfg_entry["combination_id"]
        rows = model.rows.get(cid, {})
        scores_path = result_dir(iter_dir, cid, model.mode) / "scores.jsonl"
        if not scores_path.is_file():
            warnings.append(f"config {cid!r}: scores.jsonl not found ({scores_path})")
        elif not rows:
            warnings.append(f"config {cid!r}: scores.jsonl has no usable rows")
        bad = sum(1 for r in rows.values() if coerce_number(r.get("score")) is None)
        if bad:
            warnings.append(
                f"config {cid!r}: {bad} row(s) have a non-numeric score"
            )
        cfg_entry["score_display"] = _score_display(rows.values())

    payload = review_payload(model, {})
    payload.pop("feedback", None)

    missing = _annotate_image_availability(payload, cfg)
    if missing:
        warnings.append(
            f"{missing} image file(s) referenced by the review are missing on disk"
        )

    payload["schema_version"] = SCHEMA_VERSION
    payload["generated_at"] = _now_iso()
    payload["task_adapter"] = getattr(cfg, "task_adapter", "") or ""
    payload["warnings"] = warnings
    payload["adapter"] = {
        "deterministic_ok": True,
        "subagent_ran": False,
        "subagent_notes": "",
    }
    return payload


def _score_display(rows) -> dict:
    """A task-agnostic render hint — the score kind + the observed factor keys.

    ``factors`` is the union of numeric ``sub_scores`` keys actually seen across
    a config's rows — never a hardcoded set — so any task's rubric is named.
    """
    kind = ""
    factors: dict[str, None] = {}
    for row in rows:
        sub = row.get("sub_scores") or {}
        k, _ = classify_score(row.get("scope", ""), row.get("score"), sub)
        kind = kind or k
        for key, val in sub.items():
            if coerce_number(val) is not None:
                factors.setdefault(key, None)
    return {
        "kind": kind or "score",
        "factors": [{"key": k, "label": k.replace("_", " ")} for k in factors],
    }


def _annotate_image_availability(payload: dict, cfg: ERAConfig) -> int:
    """Mark every referenced image available/missing on disk; return the count missing."""
    missing = 0
    for sample in payload.get("samples", []):
        sk = sample.get("sample_key", "")
        for img in sample.get("input", {}).get("images", []):
            ok = resolve_image(
                cfg, img.get("method_id", ""), sk, img.get("role", "")
            ) is not None
            img["available"] = ok
            missing += 0 if ok else 1
        for cell in sample.get("cells", []):
            ok = resolve_image(
                cfg, cell.get("method_id", ""), sk, "output"
            ) is not None
            cell["output_available"] = ok
            missing += 0 if ok else 1
    return missing


# ---- persistence --------------------------------------------------------

def write_review_model(iter_dir: Path, model: dict) -> Path:
    """Persist ``review_model.json`` atomically; return its path."""
    path = review_model_path(iter_dir)
    write_json_atomic(path, model)
    return path


def load_review_model(iter_dir: Path) -> dict | None:
    """Load a normalized ``review_model.json``, or ``None`` if absent/invalid."""
    path = review_model_path(iter_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and data.get("schema_version"):
        return data
    return None


def build_and_write_review_model(
    workspace_path: str | Path, iteration: object = None,
) -> dict:
    """Resolve a workspace + iteration, normalize it, and write review_model.json.

    The deterministic entry point behind ``era.cli build-review-model``. Returns
    a JSON-serializable result (or an ``error`` object) — it never raises for an
    expected condition, so the Stage 8 autonomous loop never halts here.
    """
    from ..workspace import Workspace, resolve_workspace_root

    root = resolve_workspace_root(workspace_path)
    ws = Workspace(root.parent, root.name)
    if iteration in (None, ""):
        iter_dir = ws.iter_path()
    else:
        text = str(iteration).strip()
        if text.startswith("iter_"):
            text = text[len("iter_"):]
        iter_dir = root / f"iter_{int(text):03d}"

    if not iter_dir.is_dir():
        return {"error": "no_iteration", "message": f"{iter_dir} does not exist"}

    cfg = ERAConfig.from_yaml(root / "config.yaml")
    model = build_review_model_json(root, iter_dir, cfg)
    path = write_review_model(iter_dir, model)
    return {
        "status": "ok",
        "iteration_dir": iter_dir.name,
        "review_model_json": str(path),
        "config_count": len(model.get("configs", [])),
        "sample_count": len(model.get("samples", [])),
        "warning_count": len(model.get("warnings", [])),
        "warnings": model.get("warnings", []),
    }


# ---- human_labels.json derivation from the normalized model -------------

def derive_human_labels_from_model(rm: dict, feedback: dict) -> dict:
    """Derive ``human_labels.json`` from a normalized ``review_model.json``.

    Same default-endorse rule as :func:`era.webapp.data.derive_human_labels`,
    but reading the normalized (and possibly sub-agent-repaired) artifact — so
    the carry-forward labels match exactly what the human reviewed.
    """
    wrong_items = {
        (m.get("sample_key"), m.get("method_id"), m.get("combination_id")):
            m.get("comment", "")
        for m in feedback.get("item_marks", [])
        if m.get("label") == "wrong"
    }
    wrong_cmps = {
        (m.get("sample_key"), m.get("combination_id")): m.get("comment", "")
        for m in feedback.get("comparison_marks", [])
        if m.get("label") == "wrong"
    }

    configs = rm.get("configs", []) or []
    samples = rm.get("samples", []) or []
    labels: list[dict] = []
    comparison_labels: list[dict] = []
    config_summary: list[dict] = []

    for cfg in configs:
        cid = cfg.get("combination_id")
        family = cfg.get("family", "")
        if family in FLAGGING_FAMILIES:
            total = endorsed = flagged = 0
            for s in samples:
                sk = s.get("sample_key")
                for cell in s.get("cells", []) or []:
                    mid = cell.get("method_id")
                    judge = next(
                        (j for j in cell.get("judges", []) or []
                         if j.get("combination_id") == cid), None,
                    )
                    if judge is None:
                        continue
                    key = (sk, mid, cid)
                    is_wrong = key in wrong_items
                    labels.append({
                        "sample_key": sk, "method_id": mid,
                        "combination_id": cid, "family": family,
                        "judge_score": judge.get("score"),
                        "human_verdict": "wrong" if is_wrong else "correct",
                        "comment": wrong_items.get(key, ""),
                    })
                    total += 1
                    flagged += 1 if is_wrong else 0
                    endorsed += 0 if is_wrong else 1
            config_summary.append({
                "combination_id": cid, "family": family, "modality": "flag",
                "total": total, "endorsed": endorsed, "flagged_wrong": flagged,
                "endorsement_rate": round(endorsed / total, 4) if total else None,
            })
        elif family == COMPARISON_FAMILY:
            total = correct = wrong = 0
            for s in samples:
                sk = s.get("sample_key")
                ranking = next(
                    (r for r in s.get("family_b_rankings", []) or []
                     if r.get("combination_id") == cid), None,
                )
                if ranking is None:
                    continue
                key = (sk, cid)
                is_wrong = key in wrong_cmps
                comparison_labels.append({
                    "sample_key": sk, "combination_id": cid, "family": family,
                    "ranked_method_ids": ranking.get("ranked_method_ids", []),
                    "human_verdict": "wrong" if is_wrong else "correct",
                    "comment": wrong_cmps.get(key, ""),
                })
                total += 1
                wrong += 1 if is_wrong else 0
                correct += 0 if is_wrong else 1
            config_summary.append({
                "combination_id": cid, "family": family,
                "modality": "comparison",
                "total": total, "correct": correct, "wrong": wrong,
                "correct_rate": round(correct / total, 4) if total else None,
            })

    return {
        "schema_version": "1.0",
        "iteration": rm.get("iteration"),
        "iteration_dir": rm.get("iteration_dir"),
        "finalized_at": feedback.get("finalized_at"),
        "task_family": rm.get("task_family"),
        "mode": rm.get("mode"),
        "default_rule": "unmarked_is_correct",
        "flagging_configs": [c.get("combination_id") for c in configs
                             if c.get("family") in FLAGGING_FAMILIES],
        "comparison_configs": [c.get("combination_id") for c in configs
                               if c.get("family") == COMPARISON_FAMILY],
        "labels": labels,
        "comparison_labels": comparison_labels,
        "config_summary": config_summary,
        "general_feedback": feedback.get("general_feedback", ""),
    }
