"""Scoring aggregation: overall score and category balance bonus."""
from __future__ import annotations

import logging
from typing import Union

from system_prompt_retrieval_agent.schemas import (
    GemmaUserPrompt,
    PromptPairScoreContext,
    PromptPairSubScores,
)

logger = logging.getLogger(__name__)

# Default weight keys
_WEIGHT_KEYS = (
    "qwen_pass_rate",
    "edit_correctness",
    "garment_transfer_correctness",
    "preservation",
    "artifact_penalty",
    "category_balance_bonus",
)

_DEFAULT_WEIGHTS = {
    "qwen_pass_rate": 0.40,
    "edit_correctness": 0.20,
    "garment_transfer_correctness": 0.15,
    "preservation": 0.15,
    "artifact_penalty": -0.10,
    "category_balance_bonus": 0.10,
}


def _resolve_weights(weights: dict) -> dict:
    """Merge caller-supplied weights with defaults; accept EvaluationWeights via model_dump."""
    if hasattr(weights, "model_dump"):
        weights = weights.model_dump()
    merged = dict(_DEFAULT_WEIGHTS)
    merged.update(weights)
    return merged


def _get_sub_score(sub_scores: Union[dict, PromptPairSubScores], key: str) -> float:
    """Safely retrieve a sub-score, returning 0.0 and logging DEBUG if missing/None."""
    if isinstance(sub_scores, PromptPairSubScores):
        val = getattr(sub_scores, key, None)
    else:
        val = sub_scores.get(key, None)
    if val is None:
        logger.debug("Sub-score '%s' missing or None; treating as 0.0", key)
        return 0.0
    return float(val)


def compute_overall_score(
    sub_scores: Union[dict, PromptPairSubScores],
    weights: dict,
    *,
    category_balance_bonus: float | None = None,
    hard_gate_threshold: float = 0.5,
    hard_gate_multiplier: float = 0.3,
) -> float:
    """
    Compute weighted overall score from sub-scores.

    overall = (w_qwen * qwen_pass_rate)
            + (w_edit * edit_correctness)
            + (w_gt   * garment_transfer_correctness)
            + (w_pres * preservation)
            + (w_art  * artifact_penalty)
            + (w_cat  * category_balance_bonus or 0.0)

    If qwen_pass_rate < hard_gate_threshold: multiply result by hard_gate_multiplier.
    Missing sub-score -> treat as 0.0 and log at DEBUG (do not raise).
    Clamp output to [0.0, 1.0] at the end.
    """
    w = _resolve_weights(weights)

    qwen_pass_rate = _get_sub_score(sub_scores, "qwen_pass_rate")
    edit_correctness = _get_sub_score(sub_scores, "edit_correctness")
    garment_transfer_correctness = _get_sub_score(sub_scores, "garment_transfer_correctness")
    preservation = _get_sub_score(sub_scores, "preservation")
    artifact_penalty = _get_sub_score(sub_scores, "artifact_penalty")

    cat_bonus = category_balance_bonus if category_balance_bonus is not None else 0.0

    score = (
        w.get("qwen_pass_rate", 0.0) * qwen_pass_rate
        + w.get("edit_correctness", 0.0) * edit_correctness
        + w.get("garment_transfer_correctness", 0.0) * garment_transfer_correctness
        + w.get("preservation", 0.0) * preservation
        + w.get("artifact_penalty", 0.0) * artifact_penalty
        + w.get("category_balance_bonus", 0.0) * cat_bonus
    )

    if qwen_pass_rate < hard_gate_threshold:
        logger.debug(
            "qwen_pass_rate=%.3f below hard_gate_threshold=%.3f; applying multiplier %.3f",
            qwen_pass_rate,
            hard_gate_threshold,
            hard_gate_multiplier,
        )
        score *= hard_gate_multiplier

    # Clamp to [0.0, 1.0]
    return float(max(0.0, min(1.0, score)))


def compute_category_balance_bonus(
    category_scores: dict[str, float],
    *,
    min_ratio: float = 0.70,
) -> float:
    """
    Compute a balance bonus based on uniformity across category scores.

    If min(category_scores) / max(category_scores) >= min_ratio -> return 1.0 else ratio.
    Returns 0.0 when fewer than 2 categories or any score is 0.
    """
    if len(category_scores) < 2:
        return 0.0

    values = list(category_scores.values())
    min_val = min(values)
    max_val = max(values)

    if min_val <= 0.0 or max_val <= 0.0:
        return 0.0

    ratio = min_val / max_val
    return 1.0 if ratio >= min_ratio else ratio


# ---------------------------------------------------------------------------
# V0.2.1 user-prompt-aware aggregation (plan §10.6)
# ---------------------------------------------------------------------------

def compute_per_user_prompt_means(
    cell_results: list[dict],
) -> dict[str, float]:
    """
    Aggregate per-cell scores by user_prompt_id to produce a mean score per phrasing.

    cell_results: list of dicts with at minimum:
        {user_prompt_id: str, overall_score: float}
      (any cell missing user_prompt_id or overall_score is silently skipped)

    Returns dict[user_prompt_id -> mean_overall_score].

    Strategy: plain mean over all cells belonging to each user_prompt_id.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for cell in cell_results:
        uid = cell.get("user_prompt_id")
        score = cell.get("overall_score")
        if uid is None or score is None:
            logger.debug("Skipping cell with missing user_prompt_id or overall_score: %s", cell)
            continue
        sums[uid] = sums.get(uid, 0.0) + float(score)
        counts[uid] = counts.get(uid, 0) + 1
    return {uid: sums[uid] / counts[uid] for uid in sums}


def compute_zh_en_means(
    per_up_means: dict[str, float],
    library: list[GemmaUserPrompt],
) -> tuple[float | None, float | None]:
    """
    Separate per-user-prompt means by language and compute per-language averages.

    per_up_means: dict[user_prompt_id -> mean_score]
    library: list of GemmaUserPrompt (provides language for each user_prompt_id)

    Returns (zh_mean, en_mean).
    If no enabled prompts exist for a language, that language mean is None.
    Only enabled prompts in library contribute (enabled=True).
    """
    lang_map: dict[str, str] = {p.user_prompt_id: p.language for p in library if p.enabled}

    zh_vals: list[float] = []
    en_vals: list[float] = []
    for uid, mean_score in per_up_means.items():
        lang = lang_map.get(uid)
        if lang == "zh":
            zh_vals.append(mean_score)
        elif lang == "en":
            en_vals.append(mean_score)

    zh_mean = sum(zh_vals) / len(zh_vals) if zh_vals else None
    en_mean = sum(en_vals) / len(en_vals) if en_vals else None
    return zh_mean, en_mean


def language_balanced_base_mean(
    zh_mean: float | None,
    en_mean: float | None,
) -> float:
    """
    Compute the language-balanced base mean per plan §10.6.

    If both zh_mean and en_mean are available:
        base_mean = (zh_mean + en_mean) / 2   [equal-weighted by language, not by phrasing count]
    Else if only one language is available:
        base_mean = that language's mean
    If neither is available: returns 0.0.
    """
    if zh_mean is not None and en_mean is not None:
        return (zh_mean + en_mean) / 2.0
    if zh_mean is not None:
        return zh_mean
    if en_mean is not None:
        return en_mean
    return 0.0


def compute_prompt_sensitivity(
    per_up_means: dict[str, float],
) -> tuple[float, float | None, float | None]:
    """
    Compute prompt_sensitivity = max - min of per-user-prompt means.

    Uses max - min (range) rather than stddev so it is easily interpretable as
    the worst-case gap any single phrasing can impose. This also matches the
    plan §10.6 formula: `prompt_sensitivity = max − min`.

    Returns (sensitivity, worst_user_prompt_id, worst_user_prompt_score).
    If per_up_means is empty: returns (0.0, None, None).
    """
    if not per_up_means:
        return 0.0, None, None

    worst_uid = min(per_up_means, key=lambda k: per_up_means[k])
    worst_score = per_up_means[worst_uid]
    best_score = max(per_up_means.values())
    sensitivity = best_score - worst_score
    return sensitivity, worst_uid, worst_score


def build_score_context_for_pair(
    pair_id: str,
    cell_results: list[dict],
    library: list[GemmaUserPrompt],
    scoring_config,
) -> PromptPairScoreContext:
    """
    Produce a fully-populated PromptPairScoreContext for the given pair.

    Populates all 7 V0.2.1 user-prompt-aware fields:
      - per_user_prompt      (dict[user_prompt_id, PromptPairSubScores] — empty stubs; caller
                              may enrich with full sub-scores if available)
      - mean_score_by_user_prompt
      - zh_mean_score
      - en_mean_score
      - prompt_sensitivity
      - cross_lingual_gap
      - worst_user_prompt_id / worst_user_prompt_score

    Sets overall_score using the language-balanced formula with sensitivity penalty
    (plan §10.6). Clamps to [0.0, 1.0]. Does NOT write to memory/long.py.

    scoring_config: ScoringConfig or any object with the four V0.2.1 penalty fields.
    """
    per_up_means = compute_per_user_prompt_means(cell_results)

    zh_mean, en_mean = compute_zh_en_means(per_up_means, library)
    base_mean = language_balanced_base_mean(zh_mean, en_mean)

    sensitivity, worst_uid, worst_score = compute_prompt_sensitivity(per_up_means)

    # cross_lingual_gap only meaningful when both languages present
    if zh_mean is not None and en_mean is not None:
        cross_lingual_gap = abs(zh_mean - en_mean)
    else:
        cross_lingual_gap = None

    # Sensitivity penalty
    penalty_weight = getattr(scoring_config, "prompt_sensitivity_penalty_weight", 0.5)
    tolerance = getattr(scoring_config, "prompt_sensitivity_tolerance", 0.10)
    penalty = float(penalty_weight) * max(0.0, sensitivity - float(tolerance))

    overall_score = float(max(0.0, min(1.0, base_mean - penalty)))

    # language_brittle flag
    flag_gap = getattr(scoring_config, "flag_language_brittle_gap", 0.10)
    language_brittle = (cross_lingual_gap is not None and cross_lingual_gap > float(flag_gap))

    if language_brittle:
        logger.debug(
            "pair %s flagged language_brittle: cross_lingual_gap=%.4f > %.4f",
            pair_id, cross_lingual_gap, flag_gap,
        )

    # Build mean_score_by_user_prompt (flat mean over all phrasings — for compatibility)
    mean_score_by_user_prompt = dict(per_up_means)

    # per_user_prompt: stub PromptPairSubScores keyed by user_prompt_id
    # (caller/parent can enrich these with full sub-score breakdowns)
    per_user_prompt: dict[str, PromptPairSubScores] = {
        uid: PromptPairSubScores() for uid in per_up_means
    }

    return PromptPairScoreContext(
        overall_score=overall_score,
        total_score=base_mean,
        per_user_prompt=per_user_prompt,
        mean_score_by_user_prompt=mean_score_by_user_prompt,
        zh_mean_score=zh_mean,
        en_mean_score=en_mean,
        prompt_sensitivity=sensitivity if per_up_means else None,
        cross_lingual_gap=cross_lingual_gap,
        worst_user_prompt_id=worst_uid,
        worst_user_prompt_score=worst_score,
    )
