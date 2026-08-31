"""Phase C-2 pass/recall auto-validation gate.

After Stage 6's annotated-mode round scores every evaluation config
on the operator's annotated subset, this module:

1. (``build_batches``) Joins each config's per-sample scores against
   the operator's free-text annotations from ``<dataset>/annotations/``
   and writes one input batch per ``(combination_id, method_id)`` to
   ``<iter>/auto_validate/inputs/``. The Stage 6 skill dispatches the
   ``era-auto-validator`` sub-agent on each batch (one Task call per
   batch, parallel-dispatched). Each sub-agent reads its input,
   decides agree/disagree per sample by semantic comparison, and
   writes its judgments to
   ``<iter>/auto_validate/judgments/<cid>__<method_id>.json``.

2. (``aggregate_judgments``) Reads every judgments file, computes
   pass_rate / recall_rate per config from the agree booleans, and
   writes ``<iter>/auto_validate/result.json``. Configs are tagged
   ``passed: bool`` based on the operator's pinned thresholds; the
   final ``any_passed`` flag drives Stage 6's branch:

       any_passed: true   → run the full N=50 round on the PASSING
                            configs only (failing configs are
                            authorised-skipped via auto_validate_proof)
       any_passed: false  → call ``auto_revise`` with
                            reason=stage7_auto_validate_failed; the
                            Phase C-1 machinery scaffolds the next iter

   When fewer than ``auto_validate_min_samples`` annotated samples
   exist (default 10), the gate is **skipped**: any_passed=true with
   passing_configs covering every chosen config, identical to today's
   behaviour on un-annotated datasets.

The deterministic backbone owns the join, batching, persistence, and
the arithmetic. The semantic agree/disagree decision lives in the
sub-agent — a threshold over the scalar score cannot judge whether
a method missed a defect the operator described in prose.
"""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import ERAConfig
from ..webapp.data import load_hypotheses
from ..workspace import Workspace, resolve_workspace_root
from .experiment_results import (
    read_scores,
    result_dir,
    write_json_atomic,
)


SCHEMA_VERSION = 1
AUTO_VALIDATE_REL = "auto_validate"
INPUTS_REL = f"{AUTO_VALIDATE_REL}/inputs"
JUDGMENTS_REL = f"{AUTO_VALIDATE_REL}/judgments"
RESULT_REL = f"{AUTO_VALIDATE_REL}/result.json"
ANNOTATIONS_DIR = "annotations"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_slug(value: str) -> str:
    """Coerce arbitrary text into a filename-safe slug.

    Combination ids and method ids land in input/output filenames as
    ``<cid>__<method_id>.json``. They are already constrained by the
    task-plan validator (``_VALID_TASK_ID``) and config loader, but
    re-coerce defensively here so a bad value can never escape the
    `<iter>/auto_validate/` subtree.
    """
    out = []
    for ch in str(value):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out)
    return slug or "_"


def _batch_filename(combination_id: str, method_id: str) -> str:
    return f"{_safe_slug(combination_id)}__{_safe_slug(method_id)}.json"


def load_full_annotations(data_root: str | Path) -> list[dict]:
    """Walk ``<data_root>/annotations/**/*.json`` and return EVERY
    operator annotation record, no cap.

    Differs from :func:`era.probe.annotations.probe_annotations`: the
    probe is for the post-init guide (truncated preview, sync drift
    check). This function is the runtime reader the gate uses — it
    needs the full set so it can build the join.

    Each returned dict has the shape::

        {"sample_key": "dress/.../sample_001",
         "per_method": {"method_a": "note text", ...}}

    Samples whose per_method dict is empty after the annotate UI
    deleted the file are simply not on disk (the store cleans them
    up) so they don't appear here.
    """
    root = (Path(data_root) / ANNOTATIONS_DIR).resolve()
    out: list[dict] = []
    if not root.is_dir():
        return out
    for path in sorted(root.rglob("*.json")):
        if not path.is_file() or path.name.startswith("__"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        sample_key = data.get("sample_key")
        per_method = data.get("per_method")
        if not sample_key or not isinstance(per_method, dict):
            continue
        # Keep all string slots even if blank — Phase C-2 needs the
        # "operator looked, didn't flag" signal which is represented
        # by an empty string. The store only deletes the FILE when
        # ALL slots are empty, so non-blank values dominate.
        per_method_clean = {
            str(k): v for k, v in per_method.items()
            if isinstance(v, str)
        }
        out.append({"sample_key": sample_key, "per_method": per_method_clean})
    return out


# Annotate-v2.0 writes an explicit GOOD verdict (e.g. "Good") rather than
# an empty slot when the operator looked and found no defect. For the
# pass/recall gate that is semantically identical to "operator did not
# flag this method" — collapse it to "" so ``operator_flagged`` is False.
_GOOD_TOKENS = {"", "good", "good.", "ok", "okay", "fine",
                "no issue", "no issues", "none", "n/a", "na"}


def _normalize_note(note: Any) -> str:
    """Map an operator's GOOD-style verdict to '' (not-flagged); pass any
    real complaint through unchanged. Applied only where a note is consumed
    as a flag/complaint — NOT where samples are counted, so a 'Good' slot
    still counts as "operator looked at this sample"."""
    if not isinstance(note, str):
        return ""
    return "" if note.strip().lower() in _GOOD_TOKENS else note


def _remap_per_method(annotations: list[dict], cfg: ERAConfig) -> list[dict]:
    """Re-key each record's ``per_method`` dict from the annotation file's
    method labels to the config's ``method_id`` values.

    The ``/era:annotate`` store labels ``per_method`` by the dataset method
    name (typically the lowercased result-dir basename, e.g.
    ``tryon_results``), while the scores + experiment_brief use the config
    ``method_id`` (e.g. ``flux2klein``). Without this remap the
    ``(sample_key, method_id)`` join silently misses and every operator
    note reads as empty — making recall vacuously 1.0 and the gate a no-op.

    Backward-compatible: ``method_id`` is tried as a candidate key first,
    so annotations already labelled by ``method_id`` keep working. Note
    VALUES are preserved verbatim here (incl. 'Good') so
    ``_annotated_samples_count`` keeps its "operator looked at" semantics;
    GOOD→'' normalization happens at the consumption sites via
    ``_normalize_note``."""
    methods = getattr(cfg.data, "methods", None) or []
    candidates: dict[str, list[str]] = {}
    for m in methods:
        mid = m.get("method_id") if isinstance(m, dict) else None
        if not mid:
            continue
        keys = [mid]
        path = (m.get("path") or "") if isinstance(m, dict) else ""
        base = os.path.basename(path.rstrip("/")).lower()
        if base and base not in keys:
            keys.append(base)
        low = str(mid).lower()
        if low not in keys:
            keys.append(low)
        candidates[mid] = keys
    if not candidates:
        return annotations
    out: list[dict] = []
    for rec in annotations:
        pm = rec.get("per_method") or {}
        # Skip remap when keys already match config method_ids exactly
        # (avoids clobbering a correctly-labelled store).
        new_pm: dict[str, str] = {}
        for mid, keys in candidates.items():
            note = ""
            for k in keys:
                if k in pm and isinstance(pm[k], str):
                    note = pm[k]
                    break
            new_pm[mid] = note
        out.append({"sample_key": rec.get("sample_key"), "per_method": new_pm})
    return out


def _chosen_configs(iter_dir: Path) -> list[dict]:
    """Load the Stage 4 brief's chosen_configs as a list of dicts.

    Each carries at least ``combination_id`` + ``family`` (per the
    brief schema). Returns ``[]`` when the brief is missing or
    malformed — the caller treats that as "no configs to validate".
    """
    path = iter_dir / "design" / "experiment_brief.json"
    if not path.is_file():
        return []
    try:
        brief = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    configs = brief.get("candidate_configs") if isinstance(brief, dict) else None
    if not isinstance(configs, list):
        return []
    return [c for c in configs if isinstance(c, dict) and c.get("combination_id")]


def _load_task_plan(iter_dir: Path) -> dict | None:
    """Read ``iter_dir/experiments/plans/task_plan.json`` if present.

    Mirrors ``era.cli._load_task_plan``; duplicated here so
    ``auto_validate`` doesn't reach into the CLI module. Returns
    ``None`` when the file is missing or malformed.
    """
    path = iter_dir / "experiments" / "plans" / "task_plan.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _plan_has_annotated_tasks(plan: dict | None) -> bool:
    """True iff ``plan`` contains at least one annotated-mode eval task.

    Used by Phase C-2.4's fail-loud diagnostic to distinguish "Stage 5
    omitted annotated tasks" (legacy plan or LLM error) from "Stage 6's
    annotated round ran but produced no scores" (runtime failure).
    """
    if not isinstance(plan, dict):
        return False
    for task in plan.get("tasks") or []:
        if (isinstance(task, dict)
                and task.get("type") == "eval"
                and task.get("mode") == "annotated"):
            return True
    return False


def _annotated_samples_count(annotations: list[dict]) -> int:
    """Number of distinct sample_keys with at least one non-empty slot."""
    n = 0
    for entry in annotations:
        if any(
            isinstance(v, str) and v.strip()
            for v in entry.get("per_method", {}).values()
        ):
            n += 1
    return n


def build_batches(
    workspace_path: str | Path, *,
    mode: str = "annotated",
) -> dict:
    """Build the per-(cid, method_id) sub-agent input batches.

    Returns:
        {"status": "ok", "batches": [...], "thresholds": {...},
         "annotated_sample_count": int, "skipped_for_min_samples": bool}

    When fewer than ``experiment.auto_validate_min_samples`` annotated
    samples exist, ``batches`` is empty and ``skipped_for_min_samples``
    is true — the caller skips the sub-agent dispatch and treats every
    config as passing.

    Each batch dict shape::

        {"combination_id": "...", "method_id": "...",
         "input_path": "<abs>", "output_path": "<abs>",
         "sample_count": int}

    The input JSON written to ``input_path`` carries::

        {"combination_id": "...", "method_id": "...",
         "thresholds": {...},
         "samples": [
            {"sample_key": "...", "score": float, "sub_scores": {...},
             "ok": bool, "operator_annotation": "free text"},
            ...
         ]}
    """
    root = resolve_workspace_root(workspace_path)
    ws = Workspace(root.parent, root.name)
    if not ws.exists():
        return {"error": "not_a_workspace",
                "message": f"{root} has no status.json"}

    cfg = ERAConfig.from_yaml(root / "config.yaml")
    exp = cfg.experiment
    thresholds = {
        "pass": exp.auto_validate_pass_threshold,
        "recall": exp.auto_validate_recall_threshold,
        "min_samples": exp.auto_validate_min_samples,
        "min_passing": exp.auto_validate_min_passing,
    }

    iter_dir = ws.iter_path()
    data_root = cfg.data.data_root
    annotations = _remap_per_method(load_full_annotations(data_root), cfg)
    annotated_count = _annotated_samples_count(annotations)

    av_dir = iter_dir / AUTO_VALIDATE_REL
    av_dir.mkdir(parents=True, exist_ok=True)

    if annotated_count < thresholds["min_samples"]:
        return {
            "status": "ok",
            "batches": [],
            "thresholds": thresholds,
            "annotated_sample_count": annotated_count,
            "skipped_for_min_samples": True,
            "mode": mode,
        }

    # Index annotations by sample_key for O(1) lookup.
    anno_by_key: dict[str, dict[str, str]] = {
        e["sample_key"]: e["per_method"] for e in annotations
    }
    annotated_keys = set(anno_by_key)

    configs = _chosen_configs(iter_dir)
    if not configs:
        return {
            "status": "ok",
            "batches": [],
            "thresholds": thresholds,
            "annotated_sample_count": annotated_count,
            "skipped_for_min_samples": False,
            "mode": mode,
            "note": "no chosen_configs in experiment_brief.json",
        }

    inputs_dir = iter_dir / INPUTS_REL
    inputs_dir.mkdir(parents=True, exist_ok=True)
    judgments_dir = iter_dir / JUDGMENTS_REL
    judgments_dir.mkdir(parents=True, exist_ok=True)

    # Resolve each config's measured-dimension description so the
    # sub-agent can scope-gate Family-B verdicts (Phase C-2 scope
    # awareness). hypothesis_text is the richest intent signal; the
    # metric_subfamily + scope are the concrete Family-B descriptors.
    hyp_map = load_hypotheses(iter_dir)

    batches: list[dict] = []
    for cfg_entry in configs:
        cid = cfg_entry["combination_id"]
        scores_path = result_dir(iter_dir, cid, mode) / "scores.jsonl"
        rows = read_scores(scores_path)
        if not rows:
            # The annotated round produced no scores for this config —
            # the join will treat it as zero-annotated. We still emit
            # a placeholder batch so finalize can mark it as "no data".
            continue

        # Group score rows by method_id, then filter to annotated samples.
        by_method: dict[str, list[dict]] = {}
        for row in rows:
            mid = row.get("method_id")
            sk = row.get("sample_key")
            if not mid or not sk or sk not in annotated_keys:
                continue
            by_method.setdefault(str(mid), []).append(row)

        for method_id, method_rows in sorted(by_method.items()):
            samples: list[dict] = []
            for row in sorted(method_rows, key=lambda r: r.get("sample_key", "")):
                sk = row["sample_key"]
                note = _normalize_note(anno_by_key.get(sk, {}).get(method_id, ""))
                samples.append({
                    "sample_key": sk,
                    "score": row.get("score"),
                    "sub_scores": row.get("sub_scores") or {},
                    "scope": row.get("scope"),
                    "ok": row.get("ok", True),
                    "operator_annotation": note if isinstance(note, str) else "",
                })
            if not samples:
                continue

            fname = _batch_filename(cid, method_id)
            input_path = inputs_dir / fname
            output_path = judgments_dir / fname
            payload = {
                "schema_version": SCHEMA_VERSION,
                "combination_id": cid,
                "method_id": method_id,
                "thresholds": thresholds,
                "scope_gating_enabled": (cfg_entry.get("family") == "B"),
                "evaluation_target": {
                    "family": cfg_entry.get("family"),
                    "scope": cfg_entry.get("scope"),
                    "metric_subfamily": cfg_entry.get("metric_subfamily"),
                    "judge": cfg_entry.get("judge"),
                    "prompt": cfg_entry.get("prompt"),
                    "hypothesis_id": cfg_entry.get("hypothesis_id"),
                    "hypothesis_text": hyp_map.get(
                        cfg_entry.get("hypothesis_id") or "", ""),
                },
                "samples": samples,
            }
            write_json_atomic(input_path, payload)
            batches.append({
                "combination_id": cid,
                "method_id": method_id,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "sample_count": len(samples),
            })

    if not batches:
        # Phase C-2.4 fail-loud: annotations exist (count >= min_samples)
        # but no annotated-mode scores landed on disk. Distinguish the
        # two real causes so Stage 6 can route correctly:
        # - no_annotated_tasks_in_plan: Stage 5 didn't emit annotated
        #   tasks (legacy v0.1.6 plan, or Stage 5 LLM error). Stage 6
        #   should auto-revise with stage5_missing_annotated_tasks so
        #   Phase C-1 scaffolds a fresh iter where Stage 5 re-plans.
        # - annotated_round_didnt_run: annotated tasks were planned but
        #   produced no scores (runtime failure of every annotated
        #   eval). Stage 6 should skip the gate and proceed to the full
        #   round with all configs (no auto_validate_skips).
        plan = _load_task_plan(iter_dir)
        plan_has_annotated = _plan_has_annotated_tasks(plan)
        missing_reason = (
            "no_annotated_tasks_in_plan" if not plan_has_annotated
            else "annotated_round_didnt_run"
        )
        return {
            "error": "no_annotated_scores",
            "missing_reason": missing_reason,
            "annotated_sample_count": annotated_count,
            "thresholds": thresholds,
            "plan_has_annotated": plan_has_annotated,
            "expected_results_dir": str(
                iter_dir / "experiments" / "results" / mode
            ),
            "message": (
                "Annotations exist but no annotated-mode scores.jsonl "
                "files were produced; the Phase C-2 gate cannot run on "
                "this iter."
            ),
            "mode": mode,
        }

    return {
        "status": "ok",
        "batches": batches,
        "thresholds": thresholds,
        "annotated_sample_count": annotated_count,
        "skipped_for_min_samples": False,
        "mode": mode,
    }


def _read_judgments(output_path: Path) -> list[dict] | None:
    """Read a sub-agent's judgments file; return ``None`` if missing
    or malformed."""
    if not output_path.is_file():
        return None
    try:
        data = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    judgments = data.get("judgments")
    if not isinstance(judgments, list):
        return None
    out: list[dict] = []
    for row in judgments:
        if not isinstance(row, dict):
            continue
        sk = row.get("sample_key")
        agree = row.get("agree")
        if not isinstance(sk, str) or not isinstance(agree, bool):
            continue
        # ``applicable`` is the Phase C-2 scope-gating flag: false means
        # the operator's note for this sample is about a dimension this
        # config does not measure (out of scope). Optional — defaults to
        # true so pre-scope-gating judgments files aggregate unchanged.
        applicable = row.get("applicable")
        if not isinstance(applicable, bool):
            applicable = True
        out.append({
            "sample_key": sk,
            "agree": agree,
            "applicable": applicable,
            "rationale": str(row.get("rationale") or "")[:200],
        })
    return out


def aggregate_judgments(
    workspace_path: str | Path, *,
    mode: str = "annotated",
) -> dict:
    """Read every batch's judgments file, compute pass/recall per
    config, write ``<iter>/auto_validate/result.json``.

    Returns the result JSON, or ``{"error": "missing_judgments",
    "missing": [(cid, method_id), ...]}`` when one or more batches
    weren't produced (the Stage 6 skill retries the dispatch once
    before falling through to auto-revise).
    """
    root = resolve_workspace_root(workspace_path)
    ws = Workspace(root.parent, root.name)
    if not ws.exists():
        return {"error": "not_a_workspace",
                "message": f"{root} has no status.json"}

    cfg = ERAConfig.from_yaml(root / "config.yaml")
    exp = cfg.experiment
    thresholds = {
        "pass": exp.auto_validate_pass_threshold,
        "recall": exp.auto_validate_recall_threshold,
        "min_samples": exp.auto_validate_min_samples,
        "min_passing": exp.auto_validate_min_passing,
    }

    iter_dir = ws.iter_path()
    data_root = cfg.data.data_root
    annotations = _remap_per_method(load_full_annotations(data_root), cfg)
    annotated_count = _annotated_samples_count(annotations)

    # Min-samples fast path — same semantics as build_batches.
    if annotated_count < thresholds["min_samples"]:
        configs = _chosen_configs(iter_dir)
        result = {
            "schema_version": SCHEMA_VERSION,
            "mode": mode,
            "thresholds": thresholds,
            "annotated_sample_count": annotated_count,
            "skipped_for_min_samples": True,
            "per_config": [],
            "passing_configs": [c["combination_id"] for c in configs],
            "failing_configs": [],
            "any_passed": True,
            "at": _now_iso(),
        }
        write_json_atomic(iter_dir / RESULT_REL, result)
        return {"status": "ok", **result}

    anno_by_key: dict[str, dict[str, str]] = {
        e["sample_key"]: e["per_method"] for e in annotations
    }
    annotated_keys = set(anno_by_key)
    configs = _chosen_configs(iter_dir)
    inputs_dir = iter_dir / INPUTS_REL
    judgments_dir = iter_dir / JUDGMENTS_REL

    # Rebuild the (cid, method_id) batch list from inputs_dir so we can
    # validate every expected judgments file is present.
    expected_batches: list[tuple[str, str]] = []
    if inputs_dir.is_dir():
        for input_path in sorted(inputs_dir.glob("*.json")):
            try:
                data = json.loads(input_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            cid = data.get("combination_id")
            mid = data.get("method_id")
            if isinstance(cid, str) and isinstance(mid, str):
                expected_batches.append((cid, mid))

    if not expected_batches:
        # Phase C-2.4 fail-loud: annotations exist (count >= min_samples
        # — we already returned otherwise above) but no batches were
        # ever written by build_batches. Surface the distinct cause so
        # Stage 6 can route correctly — the same error shape that
        # build_batches returns.
        plan = _load_task_plan(iter_dir)
        plan_has_annotated = _plan_has_annotated_tasks(plan)
        missing_reason = (
            "no_annotated_tasks_in_plan" if not plan_has_annotated
            else "annotated_round_didnt_run"
        )
        return {
            "error": "no_annotated_scores",
            "missing_reason": missing_reason,
            "annotated_sample_count": annotated_count,
            "thresholds": thresholds,
            "plan_has_annotated": plan_has_annotated,
            "expected_results_dir": str(
                iter_dir / "experiments" / "results" / mode
            ),
            "message": (
                "Annotations exist but no annotated-mode scores.jsonl "
                "files were produced; the Phase C-2 gate cannot run on "
                "this iter."
            ),
            "mode": mode,
        }

    missing: list[dict] = []
    per_config: dict[str, dict] = {
        c["combination_id"]: {
            "combination_id": c["combination_id"],
            "family": c.get("family"),
            "annotated_samples": 0,
            "operator_flagged": 0,
            "in_scope_flagged": 0,
            "out_of_scope_flagged": 0,
            "agreed_samples": 0,
            "method_caught_flagged": 0,
            "per_method": [],
            "judgments_paths": [],
        } for c in configs
    }
    # Configs without a row in the brief are still reported if they have
    # judgments on disk — defensive against brief edits between rounds.
    for cid, _mid in expected_batches:
        per_config.setdefault(cid, {
            "combination_id": cid, "family": None,
            "annotated_samples": 0, "operator_flagged": 0,
            "in_scope_flagged": 0, "out_of_scope_flagged": 0,
            "agreed_samples": 0, "method_caught_flagged": 0,
            "per_method": [], "judgments_paths": [],
        })

    for cid, method_id in expected_batches:
        output_path = judgments_dir / _batch_filename(cid, method_id)
        judgments = _read_judgments(output_path)
        if judgments is None:
            missing.append({"combination_id": cid, "method_id": method_id,
                            "output_path": str(output_path)})
            continue

        agreed = 0
        operator_flagged_n = 0
        in_scope_flagged_n = 0
        out_of_scope_flagged_n = 0
        method_caught = 0
        for row in judgments:
            sk = row["sample_key"]
            agree = row["agree"]
            applicable = row.get("applicable", True)
            operator_note = _normalize_note(anno_by_key.get(sk, {}).get(method_id, ""))
            operator_flagged = bool(operator_note.strip())
            if agree:
                agreed += 1
            if operator_flagged:
                operator_flagged_n += 1
                if applicable:
                    # In-scope flagged sample — counts toward recall.
                    in_scope_flagged_n += 1
                    if agree:
                        method_caught += 1
                else:
                    # Out of scope (Phase C-2 scope-gating): the note is
                    # about a dimension this config does not measure, so it
                    # is excluded from the recall denominator.
                    out_of_scope_flagged_n += 1

        agg = per_config[cid]
        agg["annotated_samples"] += len(judgments)
        agg["operator_flagged"] += operator_flagged_n
        agg["in_scope_flagged"] += in_scope_flagged_n
        agg["out_of_scope_flagged"] += out_of_scope_flagged_n
        agg["agreed_samples"] += agreed
        agg["method_caught_flagged"] += method_caught
        agg["per_method"].append({
            "method_id": method_id,
            "samples": len(judgments),
            "agreed": agreed,
            "operator_flagged": operator_flagged_n,
            "in_scope_flagged": in_scope_flagged_n,
            "out_of_scope_flagged": out_of_scope_flagged_n,
            "method_caught_flagged": method_caught,
        })
        agg["judgments_paths"].append(
            str(output_path.relative_to(iter_dir))
            if output_path.is_relative_to(iter_dir) else str(output_path)
        )

    if missing:
        return {
            "error": "missing_judgments",
            "message": f"{len(missing)} batch(es) have no judgments file yet",
            "missing": missing,
        }

    # Roll per-config tallies into pass_rate / recall_rate.
    per_config_rows: list[dict] = []
    passing: list[str] = []
    failing: list[str] = []
    for cid, agg in per_config.items():
        n = agg["annotated_samples"]
        op_flagged = agg["operator_flagged"]
        in_scope = agg["in_scope_flagged"]
        pass_rate = (agg["agreed_samples"] / n) if n else 0.0
        # Recall is over IN-SCOPE operator-flagged samples — out-of-scope
        # flags (Phase C-2 scope-gating) are excluded from the denominator.
        # Vacuously 1.0 when the operator flagged nothing in this config's
        # measured dimension.
        recall_rate = (agg["method_caught_flagged"] / in_scope) if in_scope else 1.0
        passed = (
            n > 0
            and pass_rate >= thresholds["pass"]
            and recall_rate >= thresholds["recall"]
        )
        row = {
            "combination_id": cid,
            "family": agg["family"],
            "annotated_samples": n,
            "operator_flagged": op_flagged,
            "in_scope_flagged": in_scope,
            "out_of_scope_flagged": agg["out_of_scope_flagged"],
            "agreed_samples": agg["agreed_samples"],
            "method_caught_flagged": agg["method_caught_flagged"],
            "pass_rate": round(pass_rate, 6),
            "recall_rate": round(recall_rate, 6),
            "passed": passed,
            "per_method": agg["per_method"],
            "judgments_paths": agg["judgments_paths"],
        }
        per_config_rows.append(row)
        (passing if passed else failing).append(cid)

    result = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "thresholds": thresholds,
        "annotated_sample_count": annotated_count,
        "skipped_for_min_samples": False,
        "per_config": per_config_rows,
        "passing_configs": sorted(passing),
        "failing_configs": sorted(failing),
        "passing_count": len(passing),
        "min_passing": int(thresholds.get("min_passing", 1)),
        "any_passed": len(passing) >= int(thresholds.get("min_passing", 1)),
        "at": _now_iso(),
    }
    write_json_atomic(iter_dir / RESULT_REL, result)
    return {"status": "ok", **result}


def read_result(iter_dir: Path) -> dict | None:
    """Read an iter's auto_validate result (if present). Returns ``None``
    when the gate hasn't run yet."""
    path = Path(iter_dir) / RESULT_REL
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None
