"""Human-labels derivation — the default-endorse carry-forward contract.

``derive_human_labels_from_model`` is a verbatim port of ERA's review-adapter
function: every Family-A/hybrid cell the operator did not flag ``wrong`` is
``correct`` (the "unmarked = right" rule), and per-config ``endorsement_rate``
is summarized from that.
"""
from __future__ import annotations

# Families whose results the human flags per-result (default-endorse applies).
FLAGGING_FAMILIES = ("A", "hybrid")
# Family whose results the human reviews as a relative-ranking comparison.
COMPARISON_FAMILY = "B"


def derive_human_labels_from_model(rm: dict, feedback: dict) -> dict:
    """Derive ``human_labels.json`` from a normalized ``review_model.json``."""
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
