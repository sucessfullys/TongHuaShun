"""
evaluation/invariance.py — V0.2.1 user-prompt-aware metrics (plan §10.6, S06.10).

METRICS-ONLY: this module computes per-cell, per-pair, and per-pair-per-user-prompt
counters. It does NOT place pairs in failure buckets. The ``min_per_user_prompt_score``
hard gate and ``rank_rejected_pairs[]`` are owned by S07.04c (scoring).

The ``language_brittle`` flag is set on the returned result dict when
``cross_lingual_gap > scoring.flag_language_brittle_gap`` (default 0.10); the
caller is responsible for storing it on the ``RoundState``.

``failed_pairs[]`` is reserved exclusively for stage-execution failures (manifest
validator). This module never writes to it.
"""
from __future__ import annotations

from typing import Optional


def compute_invariance_metrics(
    score_grid: dict[tuple[str, str], float],
    *,
    user_prompt_language: dict[str, str] | None = None,
    flag_language_brittle_gap: float = 0.10,
) -> dict:
    """
    Compute user-prompt-aware metrics for a single prompt pair.

    Parameters
    ----------
    score_grid:
        Mapping ``{(user_prompt_id, sample_id): score}`` where ``score`` is
        a float in [0, 1].  All entries must belong to the same
        ``prompt_pair_id``; the pair identity is not checked here.
    user_prompt_language:
        Optional mapping ``{user_prompt_id: "zh" | "en"}`` used to split
        ``zh_mean_score`` and ``en_mean_score``.  When ``None`` or when a
        ``user_prompt_id`` is absent, that prompt is counted in neither
        language group.
    flag_language_brittle_gap:
        Threshold above which the ``language_brittle`` flag is set.
        Default 0.10 (plan §10.6).

    Returns
    -------
    dict with keys:
        per_user_prompt_score    dict[str, float]   mean score per phrasing
        mean_score_by_user_prompt dict[str, float]  alias for above (same object)
        zh_mean_score            float | None
        en_mean_score            float | None
        prompt_sensitivity       float   max − min of per_user_prompt_score values
        cross_lingual_gap        float | None   abs(zh_mean − en_mean)
        worst_user_prompt_id     str | None
        worst_user_prompt_score  float | None
        language_brittle         bool
        n_cells                  int   total cells in score_grid
    """
    if not score_grid:
        return {
            "per_user_prompt_score": {},
            "mean_score_by_user_prompt": {},
            "zh_mean_score": None,
            "en_mean_score": None,
            "prompt_sensitivity": 0.0,
            "cross_lingual_gap": None,
            "worst_user_prompt_id": None,
            "worst_user_prompt_score": None,
            "language_brittle": False,
            "n_cells": 0,
        }

    # Group scores by user_prompt_id
    by_up: dict[str, list[float]] = {}
    for (up_id, _sample_id), score in score_grid.items():
        by_up.setdefault(up_id, []).append(float(score))

    # Per-phrasing mean
    per_user_prompt_score: dict[str, float] = {
        up_id: sum(scores) / len(scores)
        for up_id, scores in by_up.items()
    }

    # Prompt sensitivity: max − min across phrasings
    values = list(per_user_prompt_score.values())
    if len(values) >= 2:
        prompt_sensitivity = max(values) - min(values)
    else:
        prompt_sensitivity = 0.0

    # Worst phrasing
    if per_user_prompt_score:
        worst_user_prompt_id = min(per_user_prompt_score, key=lambda k: per_user_prompt_score[k])
        worst_user_prompt_score: Optional[float] = per_user_prompt_score[worst_user_prompt_id]
    else:
        worst_user_prompt_id = None
        worst_user_prompt_score = None

    # Language-split means
    zh_mean_score: Optional[float] = None
    en_mean_score: Optional[float] = None
    cross_lingual_gap: Optional[float] = None

    if user_prompt_language:
        zh_scores = [
            per_user_prompt_score[up_id]
            for up_id in per_user_prompt_score
            if user_prompt_language.get(up_id) == "zh"
        ]
        en_scores = [
            per_user_prompt_score[up_id]
            for up_id in per_user_prompt_score
            if user_prompt_language.get(up_id) == "en"
        ]
        if zh_scores:
            zh_mean_score = sum(zh_scores) / len(zh_scores)
        if en_scores:
            en_mean_score = sum(en_scores) / len(en_scores)
        if zh_mean_score is not None and en_mean_score is not None:
            cross_lingual_gap = abs(zh_mean_score - en_mean_score)

    language_brittle = (
        cross_lingual_gap is not None and cross_lingual_gap > flag_language_brittle_gap
    )

    return {
        "per_user_prompt_score": per_user_prompt_score,
        "mean_score_by_user_prompt": per_user_prompt_score,
        "zh_mean_score": zh_mean_score,
        "en_mean_score": en_mean_score,
        "prompt_sensitivity": prompt_sensitivity,
        "cross_lingual_gap": cross_lingual_gap,
        "worst_user_prompt_id": worst_user_prompt_id,
        "worst_user_prompt_score": worst_user_prompt_score,
        "language_brittle": language_brittle,
        "n_cells": len(score_grid),
    }
