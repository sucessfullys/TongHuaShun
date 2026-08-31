"""Ranking, CSV/JSON artifact generation, and next-round context building."""
from __future__ import annotations

import copy
import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from system_prompt_retrieval_agent.schemas import PromptPair, PromptPairScoreContext

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "0.2"

# V0.2.1 user_prompt_scores.csv columns (plan §10.6 / S07.04d)
_USER_PROMPT_SCORES_COLUMNS = [
    "pair_id",
    "user_prompt_id",
    "language",
    "mean",
    "n_samples",
]

# CSV column order matching long_memory.csv schema (plan §9.3)
_WEIGHTED_CSV_COLUMNS = [
    "schema_version",
    "prompt_pair_id",
    "system_prompt_id",
    "negative_prompt_id",
    "round",
    "overall_score",
    "total_score",
    "qwen_pass_rate",
    "edit_correctness",
    "garment_transfer_correctness",
    "preservation",
    "artifact_penalty",
    "dress_weighted",
    "lower_weighted",
    "upper_weighted",
    "fallback",
    "timestamp",
]

_CATEGORY_CSV_COLUMNS = [
    "schema_version",
    "prompt_pair_id",
    "round",
    "category",
    "weighted_score",
    "total_score",
    "qwen_pass_rate",
    "edit_correctness",
    "garment_transfer_correctness",
    "preservation",
    "artifact_penalty",
    "missing_score_reason",
]


def rank_pairs(
    pairs: list[PromptPair],
    *,
    tie_break_field: str = "prompt_pair_id",
) -> list[PromptPair]:
    """
    Sort pairs descending by pair.scores.overall_score.
    None scores sort last.
    Stable tie-break by tie_break_field ascending.
    """

    def _sort_key(pair: PromptPair) -> tuple:
        score = None
        if pair.scores is not None:
            score = pair.scores.overall_score
        # None scores sort last (use -inf for negation, then None check)
        if score is None:
            primary = (1, 0.0)  # sort after real scores
        else:
            primary = (0, -score)  # negate for descending order
        secondary = getattr(pair, tie_break_field, "")
        return (primary, secondary)

    return sorted(pairs, key=_sort_key)


def pair_row_for_csv(pair: PromptPair, round_id: int) -> dict[str, Any]:
    """
    Flatten a PromptPair to CSV columns matching long_memory.csv schema (plan §9.3).
    """
    scores: PromptPairScoreContext | None = pair.scores
    sub = scores.sub_scores if scores else None

    def _ss(field: str) -> str:
        if sub is None:
            return ""
        val = getattr(sub, field, None)
        return "" if val is None else str(val)

    def _cat_weighted(cat: str) -> str:
        if scores is None or not scores.category_scores:
            return ""
        ctx = scores.category_scores.get(cat)
        if ctx is None:
            return ""
        val = ctx.weighted_score
        return "" if val is None else str(val)

    return {
        "schema_version": _SCHEMA_VERSION,
        "prompt_pair_id": pair.prompt_pair_id,
        "system_prompt_id": pair.system_prompt_id,
        "negative_prompt_id": pair.negative_prompt_id,
        "round": round_id,
        "overall_score": "" if (scores is None or scores.overall_score is None) else scores.overall_score,
        "total_score": "" if (scores is None or scores.total_score is None) else scores.total_score,
        "qwen_pass_rate": _ss("qwen_pass_rate"),
        "edit_correctness": _ss("edit_correctness"),
        "garment_transfer_correctness": _ss("garment_transfer_correctness"),
        "preservation": _ss("preservation"),
        "artifact_penalty": _ss("artifact_penalty"),
        "dress_weighted": _cat_weighted("dress"),
        "lower_weighted": _cat_weighted("lower"),
        "upper_weighted": _cat_weighted("upper"),
        "fallback": pair.fallback,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


def write_ranking_artifacts(
    round_dir: Path,
    pairs: list[PromptPair],
    *,
    round_id: int,
) -> None:
    """
    Writes scoring artifacts to {round_dir}/scoring/:
      - weighted_scores.csv
      - category_scores.csv
      - ranking.json

    Uses pandas if available, falls back to csv stdlib.
    """
    scoring_dir = round_dir / "scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)

    ranked = rank_pairs(pairs)

    # --- weighted_scores.csv ---
    rows = [pair_row_for_csv(p, round_id) for p in ranked]
    _write_csv(scoring_dir / "weighted_scores.csv", _WEIGHTED_CSV_COLUMNS, rows)

    # --- category_scores.csv ---
    cat_rows: list[dict[str, Any]] = []
    for pair in ranked:
        scores = pair.scores
        if scores is None or not scores.category_scores:
            continue
        for cat, ctx in scores.category_scores.items():
            if ctx is None:
                cat_rows.append(
                    {
                        "schema_version": _SCHEMA_VERSION,
                        "prompt_pair_id": pair.prompt_pair_id,
                        "round": round_id,
                        "category": cat,
                        "weighted_score": "",
                        "total_score": "",
                        "qwen_pass_rate": "",
                        "edit_correctness": "",
                        "garment_transfer_correctness": "",
                        "preservation": "",
                        "artifact_penalty": "",
                        "missing_score_reason": "no_samples",
                    }
                )
                continue
            sub = ctx.sub_scores
            cat_rows.append(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "prompt_pair_id": pair.prompt_pair_id,
                    "round": round_id,
                    "category": cat,
                    "weighted_score": "" if ctx.weighted_score is None else ctx.weighted_score,
                    "total_score": "" if ctx.total_score is None else ctx.total_score,
                    "qwen_pass_rate": "" if sub.qwen_pass_rate is None else sub.qwen_pass_rate,
                    "edit_correctness": "" if sub.edit_correctness is None else sub.edit_correctness,
                    "garment_transfer_correctness": "" if sub.garment_transfer_correctness is None else sub.garment_transfer_correctness,
                    "preservation": "" if sub.preservation is None else sub.preservation,
                    "artifact_penalty": "" if sub.artifact_penalty is None else sub.artifact_penalty,
                    "missing_score_reason": ctx.missing_score_reason or "",
                }
            )
    _write_csv(scoring_dir / "category_scores.csv", _CATEGORY_CSV_COLUMNS, cat_rows)

    # --- ranking.json ---
    ranking_entries = []
    for rank_idx, pair in enumerate(ranked, start=1):
        overall = None
        if pair.scores is not None:
            overall = pair.scores.overall_score
        ranking_entries.append(
            {
                "prompt_pair_id": pair.prompt_pair_id,
                "overall_score": overall,
                "rank": rank_idx,
            }
        )
    (scoring_dir / "ranking.json").write_text(
        json.dumps(ranking_entries, indent=2), encoding="utf-8"
    )

    logger.debug(
        "Wrote ranking artifacts to %s: %d pairs, %d category rows",
        scoring_dir,
        len(ranked),
        len(cat_rows),
    )


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV, attempting pandas first then falling back to csv stdlib."""
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(path, index=False)
    except ImportError:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def build_next_round_context(
    ranked_pairs: list[PromptPair],
    *,
    top_n: int = 3,
) -> list[PromptPair]:
    """
    Return deep-copied PromptPair objects (top N by rank) with their scores
    attached, suitable for memory injection into the next round.
    """
    selected = ranked_pairs[:top_n]
    return [copy.deepcopy(p) for p in selected]


# ---------------------------------------------------------------------------
# V0.2.1 — hard gate, sensitivity penalty, language-brittle (plan §10.6)
# ---------------------------------------------------------------------------

def rank_pairs_with_gate(
    pairs: list[PromptPair],
    scoring_config,
    *,
    tie_break_field: str = "prompt_pair_id",
) -> tuple[list[PromptPair], list[dict]]:
    """
    Apply the V0.2.1 hard gate and sensitivity penalty, then rank.

    Hard gate (plan §10.6): any per_user_prompt mean < scoring_config.min_per_user_prompt_score
    moves the pair to rank_rejected_pairs[] with rejection_reason="per_user_prompt_below_min".
    Rejected pairs are NOT included in the ranked output but are returned separately.

    Sensitivity penalty:
        pair_overall = base_mean - w * max(0, sensitivity - tolerance)
    where base_mean = language_balanced_base_mean (already encoded in pair.scores.total_score
    by build_score_context_for_pair).  If the context lacks the V0.2.1 fields the pair is
    ranked by its existing overall_score without modification.

    language_brittle pairs ARE ranked (gate does not reject them); only flag is set on context.

    Returns:
        (ranked_accepted, rank_rejected_pairs)
    where rank_rejected_pairs is a list of dicts:
        {pair_id, rejection_reason, worst_user_prompt_id, worst_user_prompt_score}
    """
    min_score = getattr(scoring_config, "min_per_user_prompt_score", 0.30)
    penalty_weight = getattr(scoring_config, "prompt_sensitivity_penalty_weight", 0.5)
    tolerance = getattr(scoring_config, "prompt_sensitivity_tolerance", 0.10)

    accepted: list[PromptPair] = []
    rejected: list[dict] = []

    for pair in pairs:
        scores = pair.scores

        # --- Hard gate ---
        if scores is not None and scores.mean_score_by_user_prompt:
            worst_id = scores.worst_user_prompt_id
            worst_score = scores.worst_user_prompt_score
            # fall back to min of mean_score_by_user_prompt when worst fields absent
            if worst_score is None and scores.mean_score_by_user_prompt:
                worst_id = min(scores.mean_score_by_user_prompt,
                               key=lambda k: scores.mean_score_by_user_prompt[k])
                worst_score = scores.mean_score_by_user_prompt[worst_id]

            if worst_score is not None and worst_score < min_score:
                rejected.append({
                    "pair_id": pair.prompt_pair_id,
                    "rejection_reason": "per_user_prompt_below_min",
                    "worst_user_prompt_id": worst_id,
                    "worst_user_prompt_score": worst_score,
                })
                logger.debug(
                    "Pair %s hard-gated: worst_user_prompt_score=%.4f < %.4f",
                    pair.prompt_pair_id, worst_score, min_score,
                )
                continue

        # --- Sensitivity penalty (applied to a copy so original context is not mutated) ---
        if scores is not None and scores.prompt_sensitivity is not None:
            sensitivity = scores.prompt_sensitivity
            penalty = float(penalty_weight) * max(0.0, sensitivity - float(tolerance))
            base = scores.total_score if scores.total_score is not None else (scores.overall_score or 0.0)
            new_overall = float(max(0.0, min(1.0, base - penalty)))
            # Return modified pair via deep-copy to avoid mutating source
            pair = copy.deepcopy(pair)
            pair.scores.overall_score = new_overall  # type: ignore[union-attr]

        accepted.append(pair)

    ranked = rank_pairs(accepted, tie_break_field=tie_break_field)
    return ranked, rejected


def write_user_prompt_scores_csv(
    path: Path,
    pair_id: str,
    per_up_means: dict[str, float],
    library,
    cell_counts: dict[str, int] | None = None,
) -> None:
    """
    Write user_prompt_scores.csv (plan §10.6 / S07.04d).

    Columns: pair_id, user_prompt_id, language, mean, n_samples.
    library: list of GemmaUserPrompt (provides language per user_prompt_id).
    cell_counts: optional dict[user_prompt_id -> n_samples]; defaults to 1 if absent.
    """
    lang_map: dict[str, str] = {}
    for p in library:
        if hasattr(p, "user_prompt_id") and hasattr(p, "language"):
            lang_map[p.user_prompt_id] = p.language

    rows: list[dict[str, Any]] = []
    for uid, mean_score in per_up_means.items():
        rows.append({
            "pair_id": pair_id,
            "user_prompt_id": uid,
            "language": lang_map.get(uid, ""),
            "mean": mean_score,
            "n_samples": (cell_counts or {}).get(uid, 1),
        })

    _write_csv(path, _USER_PROMPT_SCORES_COLUMNS, rows)
