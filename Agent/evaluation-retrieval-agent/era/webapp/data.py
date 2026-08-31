"""Review model — assemble Stage 6 results into a human-review payload.

This module is the read side of the Stage 8 human-feedback web app. It joins,
for one iteration, the experiment brief (what each evaluator *is*) with every
config's ``scores.jsonl`` (what each evaluator *said*) into a per-sample view:
each try-on / generated result alongside what every judge scored it.

It is deliberately free of any FastAPI / web dependency so the orchestration
layer (``era.orchestration.human_feedback``) can import it to derive the
``human_labels.json`` carry-forward contract without pulling in the web stack.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ERAConfig
from ..orchestration.experiment_results import read_scores, read_summary, result_dir

# Families whose results the human flags per-result (default-endorse applies).
FLAGGING_FAMILIES = ("A", "hybrid")
# Family whose results the human reviews as a relative-ranking comparison.
COMPARISON_FAMILY = "B"


# ---- brief + hypotheses -------------------------------------------------

def load_brief(iter_dir: Path) -> dict:
    """Read ``design/experiment_brief.json`` for an iteration (``{}`` if absent)."""
    path = iter_dir / "design" / "experiment_brief.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def load_hypotheses(iter_dir: Path) -> dict[str, str]:
    """Map ``H<n>`` ids to their heading text from ``design/hypotheses.md``.

    Best-effort: any markdown heading that starts with an ``H<digits>`` token
    contributes ``{id: rest-of-heading}``. Missing file → empty mapping.
    """
    path = iter_dir / "design" / "hypotheses.md"
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    pattern = re.compile(r"^#{1,6}\s*(H\d+)\b[\s:.—-]*(.*)$")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pattern.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def _compose_description(cfg: dict, hypothesis: str = "") -> str:
    """Build a plain-language description of one evaluator config."""
    family = cfg.get("family", "")
    fam_label = {
        "A": "VLM-judge evaluation (Family A) — a multimodal model scores the result.",
        "B": "Metric / feature evaluation (Family B) — computer-vision metrics score the result.",
        "hybrid": "Hybrid evaluation — combines a VLM judge and metric scores.",
    }.get(family, "Evaluation method.")
    parts = [fam_label]

    def add(label: str, key: str) -> None:
        val = (cfg.get(key) or "").strip()
        if val and val.lower() != "n/a":
            parts.append(f"{label}: {val}.")

    add("Judge", "judge")
    add("Metrics", "metric_subfamily")
    add("Prompt protocol", "prompt")
    add("Scope", "scope")
    if hypothesis:
        parts.append(f"Tests the hypothesis: {hypothesis}")
    return " ".join(parts)


# ---- score classification ----------------------------------------------

def coerce_number(value: object) -> float | None:
    """Best-effort numeric coercion — an int/float or a numeric string, else None.

    Agent-authored runners occasionally emit a score as a string; this keeps the
    review readable instead of silently dropping the cell.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def classify_score(scope: str, score: object, sub_scores: dict) -> tuple[str, str]:
    """Return ``(score_kind, display)`` for one heterogeneous scores.jsonl row.

    ``score_kind`` is one of pointwise / pairwise / metric / vote / fusion /
    score; ``display`` is a compact human-readable rendering. The pointwise
    rubric is **task-agnostic** — it names whatever numeric factors the runner
    put in ``sub_scores``, with no hardcoded factor set.
    """
    sub_scores = sub_scores or {}
    num = coerce_number(score)
    scope = (scope or "").lower()

    if "pointwise" in scope:
        # Name every numeric factor the runner emitted, in its own order — so a
        # try-on rubric and a generation rubric both read correctly.
        bits = [
            f"{k.replace('_', ' ')}: {n:g}"
            for k, v in sub_scores.items()
            if (n := coerce_number(v)) is not None
        ]
        rubric = " / ".join(bits)
        disp = rubric
        if num is not None:
            disp = f"{rubric}  ·  {num:.2f}" if rubric else f"{num:.2f}"
        return "pointwise", disp or "—"
    if "pairwise" in scope and "arbiter" not in scope:
        return "pairwise", f"win-rate {num:.2f}" if num is not None else "—"
    if "arbiter" in scope or scope == "vote":
        return "vote", f"vote {num:.2f}" if num is not None else "—"
    if "fusion" in scope or "posthoc" in scope:
        return "fusion", f"fusion {num:.3f}" if num is not None else "—"
    if "region" in scope or "metric" in scope or "whole" in scope:
        return "metric", f"z {num:+.3f}" if num is not None else "—"
    return "score", f"{num:.3f}" if num is not None else "—"


# ---- review model -------------------------------------------------------

@dataclass
class ReviewModel:
    """Everything the Stage 8 web app needs to render one iteration's review."""

    workspace_root: Path
    iter_dir: Path
    iteration: int
    mode: str
    task_family: str
    input_roles: dict[str, str]
    methods: list[dict]                 # [{method_id, path, output_file}]
    configs: list[dict]                 # [{combination_id, family, judge, slot,
    #                                       description, hypothesis_id}]
    sample_keys: list[str]
    # rows[combination_id][(sample_key, method_id)] -> scores.jsonl row
    rows: dict[str, dict[tuple[str, str], dict]] = field(default_factory=dict)

    # -- derived views ----------------------------------------------------

    def _config(self, combination_id: str) -> dict:
        for c in self.configs:
            if c["combination_id"] == combination_id:
                return c
        return {}

    def cells_for(self, combination_id: str) -> list[dict]:
        """Every scored (sample_key, method_id) cell of one config."""
        out: list[dict] = []
        for (sk, mid), row in sorted(self.rows.get(combination_id, {}).items()):
            out.append({"sample_key": sk, "method_id": mid, "row": row})
        return out

    def family_b_ranking(self, combination_id: str, sample_key: str) -> dict | None:
        """The Family-B relative method ranking for one (config, sample)."""
        cfg = self._config(combination_id)
        if cfg.get("family") != COMPARISON_FAMILY:
            return None
        scored: list[tuple[str, float, str]] = []
        for method in self.methods:
            mid = method["method_id"]
            row = self.rows.get(combination_id, {}).get((sample_key, mid))
            num = coerce_number(row.get("score")) if row else None
            if num is None:
                continue
            _, disp = classify_score(
                row.get("scope", ""), row.get("score"), row.get("sub_scores") or {}
            )
            scored.append((mid, num, disp))
        if not scored:
            return None
        scored.sort(key=lambda t: t[1], reverse=True)
        return {
            "combination_id": combination_id,
            "ranked_method_ids": [mid for mid, _, _ in scored],
            "per_method_display": {mid: disp for mid, _, disp in scored},
        }

    def flagging_config_ids(self) -> list[str]:
        return [c["combination_id"] for c in self.configs
                if c.get("family") in FLAGGING_FAMILIES]

    def comparison_config_ids(self) -> list[str]:
        return [c["combination_id"] for c in self.configs
                if c.get("family") == COMPARISON_FAMILY]


def build_review_model(
    workspace_root: Path, iter_dir: Path, cfg: ERAConfig, mode: str | None = None,
) -> ReviewModel:
    """Assemble a :class:`ReviewModel` for one iteration's experiment results.

    ``mode`` defaults to ``full`` when the full-pass summary exists, else
    ``pilot`` — so the standalone demo and the loop both review real results.
    """
    if mode is None:
        mode = "full" if read_summary(iter_dir, "full") else "pilot"

    brief = load_brief(iter_dir)
    hypotheses = load_hypotheses(iter_dir)
    iteration = 1
    try:
        iteration = int(str(iter_dir.name).split("_")[-1])
    except (ValueError, IndexError):
        pass

    methods = [
        {"method_id": m.get("method_id", ""), "path": m.get("path", ""),
         "output_file": m.get("output_file", "")}
        for m in cfg.data.methods
    ]

    configs: list[dict] = []
    rows: dict[str, dict[tuple[str, str], dict]] = {}
    sample_keys: set[str] = set()

    for cand in brief.get("candidate_configs", []):
        cid = cand.get("combination_id")
        if not cid:
            continue
        hyp = hypotheses.get(cand.get("hypothesis_id", ""), "")
        configs.append({
            "combination_id": cid,
            "family": cand.get("family", ""),
            "judge": cand.get("judge", ""),
            "slot": cand.get("slot", ""),
            "hypothesis_id": cand.get("hypothesis_id", ""),
            "description": _compose_description(cand, hyp),
        })
        cfg_rows: dict[tuple[str, str], dict] = {}
        for row in read_scores(result_dir(iter_dir, cid, mode) / "scores.jsonl"):
            if row.get("ok") is False:
                continue
            # Tolerate an agent-authored runner that named the keys slightly
            # differently (sample_id / method).
            sk = row.get("sample_key") or row.get("sample_id")
            mid = row.get("method_id") or row.get("method")
            if not sk or not mid:
                continue
            sk = str(sk)
            mid = str(mid)
            # Tolerate runners that prefixed sample_key with the method
            # directory ("<method_id>/sample_001"). The image resolver appends
            # sample_key under the method's base path, so a method-prefixed key
            # double-counts the directory and breaks both image resolution and
            # per-sample cell merging across methods.
            prefix = f"{mid}/"
            if sk.startswith(prefix):
                sk = sk[len(prefix):]
            cfg_rows[(sk, mid)] = row
            sample_keys.add(sk)
        rows[cid] = cfg_rows

    return ReviewModel(
        workspace_root=workspace_root,
        iter_dir=iter_dir,
        iteration=iteration,
        mode=mode,
        task_family=cfg.task_family,
        input_roles=dict(cfg.data.input_roles),
        methods=methods,
        configs=configs,
        sample_keys=sorted(sample_keys),
        rows=rows,
    )


# ---- /api/review payload ------------------------------------------------

def review_payload(model: ReviewModel, feedback: dict) -> dict:
    """The full JSON payload served by ``GET /api/review``."""
    samples: list[dict] = []
    for sk in model.sample_keys:
        # Input — content-adaptive. A sample may carry input-role images, an
        # input text (a generation prompt or an edit instruction), or both;
        # each key is present only when that modality exists.
        input_block: dict = {}
        if model.input_roles:
            input_block["images"] = [
                {"role": role, "method_id": _first_method_with(model, sk)}
                for role in model.input_roles
            ]
        prompt = _read_prompt(model, sk)
        if prompt:
            input_block["prompt"] = prompt

        cells: list[dict] = []
        for method in model.methods:
            mid = method["method_id"]
            judges: list[dict] = []
            for cfg in model.configs:
                if cfg["family"] == COMPARISON_FAMILY:
                    continue  # Family B is rendered as a ranking, not a cell
                row = model.rows.get(cfg["combination_id"], {}).get((sk, mid))
                if not row:
                    continue
                kind, disp = classify_score(
                    row.get("scope", ""), row.get("score"),
                    row.get("sub_scores") or {},
                )
                judges.append({
                    "combination_id": cfg["combination_id"],
                    "family": cfg["family"],
                    "score": row.get("score"),
                    "score_kind": kind,
                    "display": disp,
                    "sub_scores": row.get("sub_scores") or {},
                })
            cells.append({
                "method_id": mid,
                "output_image": {"method_id": mid, "sample_key": sk},
                "judges": judges,
            })

        family_b = [
            r for r in (
                model.family_b_ranking(cid, sk)
                for cid in model.comparison_config_ids()
            ) if r is not None
        ]
        samples.append({
            "sample_key": sk,
            "input": input_block,
            "cells": cells,
            "family_b_rankings": family_b,
        })

    return {
        "iteration": model.iteration,
        "iteration_dir": model.iter_dir.name,
        "mode": model.mode,
        "task_family": model.task_family,
        "input_roles": model.input_roles,
        "methods": [{"method_id": m["method_id"],
                     "output_file": m["output_file"]} for m in model.methods],
        "configs": model.configs,
        "samples": samples,
        "feedback": feedback,
    }


def _first_method_with(model: ReviewModel, sample_key: str) -> str:
    """A method_id whose sample dir should hold the shared input images."""
    return model.methods[0]["method_id"] if model.methods else ""


def _read_prompt(model: ReviewModel, sample_key: str) -> str:
    """Best-effort input text — a generation prompt or an edit instruction.

    Looked up per sample, in order: a ``prompt`` / ``input_prompt`` /
    ``caption`` / ``instruction`` string key in ``metadata.json``, then a
    plain ``prompt.txt``. Empty string when no method dir carries one.
    """
    for method in model.methods:
        sample_dir = Path(method["path"]) / sample_key
        meta = sample_dir / "metadata.json"
        if meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}
            for key in ("prompt", "input_prompt", "caption", "instruction"):
                if isinstance(data.get(key), str) and data[key].strip():
                    return data[key].strip()
        txt = sample_dir / "prompt.txt"
        if txt.is_file():
            try:
                text = txt.read_text(encoding="utf-8").strip()
            except OSError:
                text = ""
            if text:
                return text
    return ""


# ---- human_labels.json derivation (the carry-forward contract) ----------

def derive_human_labels(model: ReviewModel, feedback: dict) -> dict:
    """Derive the ``human_labels.json`` ground-truth set from raw feedback.

    Two modalities, both default-endorse:

    - Flagging (Family A + hybrid): every scored cell not flagged ``wrong`` is
      ``correct`` — the operator's "unmarked = right" rule.
    - Comparison (Family B): every per-sample relative ranking not marked
      ``wrong`` is ``correct``.
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

    labels: list[dict] = []
    comparison_labels: list[dict] = []
    config_summary: list[dict] = []

    for cfg in model.configs:
        cid = cfg["combination_id"]
        family = cfg.get("family", "")
        if family in FLAGGING_FAMILIES:
            total = endorsed = flagged = 0
            for cell in model.cells_for(cid):
                sk, mid = cell["sample_key"], cell["method_id"]
                key = (sk, mid, cid)
                is_wrong = key in wrong_items
                labels.append({
                    "sample_key": sk,
                    "method_id": mid,
                    "combination_id": cid,
                    "family": family,
                    "judge_score": cell["row"].get("score"),
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
            for sk in model.sample_keys:
                ranking = model.family_b_ranking(cid, sk)
                if ranking is None:
                    continue
                key = (sk, cid)
                is_wrong = key in wrong_cmps
                comparison_labels.append({
                    "sample_key": sk,
                    "combination_id": cid,
                    "family": family,
                    "ranked_method_ids": ranking["ranked_method_ids"],
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
        "iteration": model.iteration,
        "iteration_dir": model.iter_dir.name,
        "finalized_at": feedback.get("finalized_at"),
        "task_family": model.task_family,
        "mode": model.mode,
        "default_rule": "unmarked_is_correct",
        "flagging_configs": model.flagging_config_ids(),
        "comparison_configs": model.comparison_config_ids(),
        "labels": labels,
        "comparison_labels": comparison_labels,
        "config_summary": config_summary,
        "general_feedback": feedback.get("general_feedback", ""),
    }
