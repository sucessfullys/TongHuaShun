"""Per-category score computation."""
from __future__ import annotations

import logging
from typing import Sequence

from system_prompt_retrieval_agent.schemas import CategoryScoreContext, PromptPairSubScores
from system_prompt_retrieval_agent.scoring.aggregate import compute_overall_score

logger = logging.getLogger(__name__)

_SUB_SCORE_FIELDS = (
    "edit_correctness",
    "garment_transfer_correctness",
    "preservation",
    "artifact_penalty",
)


def compute_category_scores(
    per_sample_scores: list[dict],
    weights: dict,
    categories: Sequence[str] = ("dress", "lower", "upper"),
) -> dict[str, CategoryScoreContext | None]:
    """
    Compute per-category scores from per-sample score dicts.

    per_sample_scores: list of {
        sample_id, category, qwen_status (yes|no|parse_fail_*),
        edit_correctness, garment_transfer_correctness,
        preservation, artifact_penalty
    }

    For each category:
      - qwen_pass_rate = yes / (yes + no); parse_fail excluded from denominator
      - average each sub_score across samples in that category
      - weighted_score = compute_overall_score(category_sub_scores, weights, cat_balance_bonus=None)
      - total_score = simple unweighted mean of sub_scores (excluding artifact_penalty in mean)

    Returns dict[category] = CategoryScoreContext.
    Missing category -> None entry.
    """
    # Bucket samples by category
    buckets: dict[str, list[dict]] = {}
    for sample in per_sample_scores:
        cat = sample.get("category")
        if cat is not None:
            buckets.setdefault(cat, []).append(sample)

    result: dict[str, CategoryScoreContext | None] = {}

    for cat in categories:
        samples = buckets.get(cat)
        if not samples:
            result[cat] = None
            continue

        # Compute qwen_pass_rate excluding parse_fail samples
        yes_count = 0
        no_count = 0
        for s in samples:
            status = s.get("qwen_status", "")
            if status == "yes":
                yes_count += 1
            elif status == "no":
                no_count += 1
            # parse_fail_* statuses are excluded from denominator

        denom = yes_count + no_count
        qwen_pass_rate = (yes_count / denom) if denom > 0 else 0.0

        # Average numeric sub-scores
        sub_score_sums: dict[str, float] = {k: 0.0 for k in _SUB_SCORE_FIELDS}
        sub_score_counts: dict[str, int] = {k: 0 for k in _SUB_SCORE_FIELDS}

        for s in samples:
            for field in _SUB_SCORE_FIELDS:
                val = s.get(field)
                if val is not None:
                    sub_score_sums[field] += float(val)
                    sub_score_counts[field] += 1

        averaged: dict[str, float | None] = {}
        for field in _SUB_SCORE_FIELDS:
            cnt = sub_score_counts[field]
            averaged[field] = sub_score_sums[field] / cnt if cnt > 0 else None

        sub_scores_obj = PromptPairSubScores(
            qwen_pass_rate=qwen_pass_rate,
            edit_correctness=averaged.get("edit_correctness"),
            garment_transfer_correctness=averaged.get("garment_transfer_correctness"),
            preservation=averaged.get("preservation"),
            artifact_penalty=averaged.get("artifact_penalty"),
        )

        weighted_score = compute_overall_score(
            sub_scores_obj,
            weights,
            category_balance_bonus=None,
        )

        # total_score: unweighted mean of positive sub-scores (qwen + edit + gt + preservation)
        positive_fields = ["qwen_pass_rate", "edit_correctness", "garment_transfer_correctness", "preservation"]
        pos_vals = []
        for field in positive_fields:
            val = getattr(sub_scores_obj, field, None)
            if val is not None:
                pos_vals.append(val)

        total_score = sum(pos_vals) / len(pos_vals) if pos_vals else 0.0

        result[cat] = CategoryScoreContext(
            weighted_score=weighted_score,
            total_score=total_score,
            sub_scores=sub_scores_obj,
        )

    return result


# ---------------------------------------------------------------------------
# V0.2.1 — per-category × user_prompt brittleness analysis (S07.06-.07)
# ---------------------------------------------------------------------------

class CategoryAggregator:
    """
    Bucket per-sample cells by (category, user_prompt_id) and compute per-category
    brittleness metrics across phrasings (plan §10.6 / S07.06-.07).

    Usage:
        agg = CategoryAggregator(weights, categories=["dress", "lower", "upper"])
        agg.add_cells(per_sample_scores)  # list of dicts with category + user_prompt_id
        breakdown = agg.category_user_prompt_breakdown()
        brittle = agg.brittle_categories(flag_gap=0.10)
    """

    def __init__(
        self,
        weights: dict,
        categories: Sequence[str] = ("dress", "lower", "upper"),
    ) -> None:
        self._weights = weights
        self._categories = list(categories)
        # _cells[cat][user_prompt_id] -> list of sample dicts
        self._cells: dict[str, dict[str, list[dict]]] = {}

    def add_cells(self, per_sample_scores: list[dict]) -> None:
        """Add a batch of per-sample score dicts to the internal buckets."""
        for sample in per_sample_scores:
            cat = sample.get("category")
            uid = sample.get("user_prompt_id")
            if cat is None:
                continue
            cat_bucket = self._cells.setdefault(cat, {})
            uid_key = uid if uid is not None else "__no_uid__"
            cat_bucket.setdefault(uid_key, []).append(sample)

    def category_user_prompt_breakdown(
        self,
    ) -> dict[str, dict[str, CategoryScoreContext | None]]:
        """
        Return dict[category -> dict[user_prompt_id -> CategoryScoreContext]].

        For each (category, user_prompt_id) bucket, computes per-bucket scores
        using the same logic as compute_category_scores().
        """
        result: dict[str, dict[str, CategoryScoreContext | None]] = {}
        for cat in self._categories:
            uid_map: dict[str, CategoryScoreContext | None] = {}
            if cat in self._cells:
                for uid, samples in self._cells[cat].items():
                    contexts = compute_category_scores(samples, self._weights, categories=[cat])
                    uid_map[uid] = contexts.get(cat)
            result[cat] = uid_map
        return result

    def brittle_categories(
        self,
        *,
        flag_gap: float = 0.10,
    ) -> dict[str, bool]:
        """
        For each category, compute cross-phrasing brittleness.

        Brittleness = max(weighted_score across phrasings) - min(weighted_score across phrasings)
        > flag_gap.

        Returns dict[category -> is_brittle].
        Categories with fewer than 2 phrasings or any None score return False.
        """
        breakdown = self.category_user_prompt_breakdown()
        brittle: dict[str, bool] = {}
        for cat, uid_map in breakdown.items():
            scores_by_uid = [
                ctx.weighted_score
                for ctx in uid_map.values()
                if ctx is not None and ctx.weighted_score is not None
            ]
            if len(scores_by_uid) < 2:
                brittle[cat] = False
            else:
                gap = max(scores_by_uid) - min(scores_by_uid)
                brittle[cat] = gap > flag_gap
        return brittle
