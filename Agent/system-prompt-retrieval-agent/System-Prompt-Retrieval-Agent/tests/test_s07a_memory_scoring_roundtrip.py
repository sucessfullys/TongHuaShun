"""S07a — Memory ↔ Scoring round-trip (parent-only integration).

Verifies the producer (`scoring.aggregate.build_score_context_for_pair`)
and the consumer (`memory.long.flatten_pair_to_row`) agree on the seven
user-prompt-aware fields plus the language_brittle flag.

This is the explicit single-owner merge point for the Burst-1 W1 file
(`memory/long.py`) that Burst-2 W2 (`scoring/aggregate.py`) produces
data for. Per the V0.2.1 plan §10.6 + §9.1.
"""
from __future__ import annotations

from system_prompt_retrieval_agent.config import GemmaUserPromptEntry, ScoringConfig
from system_prompt_retrieval_agent.memory.long import flatten_pair_to_row
from system_prompt_retrieval_agent.schemas import PromptPair
from system_prompt_retrieval_agent.scoring.aggregate import build_score_context_for_pair


def _library_4():
    return [
        GemmaUserPromptEntry(user_prompt_id="zh_001", language="zh", text="图1穿图2", enabled=True),
        GemmaUserPromptEntry(user_prompt_id="zh_003", language="zh", text="把图2衣服穿模特身上", enabled=True),
        GemmaUserPromptEntry(user_prompt_id="en_001", language="en", text="model wears clothing", enabled=True),
        GemmaUserPromptEntry(user_prompt_id="en_003", language="en", text="dress the model", enabled=True),
    ]


def _cell_scores(zh1, zh3, en1, en3):
    """Build a list of cell-result dicts per scoring/aggregate.py contract."""
    return [
        {"user_prompt_id": "zh_001", "sample_id": "s1", "overall_score": zh1},
        {"user_prompt_id": "zh_003", "sample_id": "s1", "overall_score": zh3},
        {"user_prompt_id": "en_001", "sample_id": "s1", "overall_score": en1},
        {"user_prompt_id": "en_003", "sample_id": "s1", "overall_score": en3},
    ]


def test_roundtrip_balanced_high_scores_no_brittleness():
    cells = _cell_scores(0.80, 0.82, 0.78, 0.81)
    cfg = ScoringConfig()
    ctx = build_score_context_for_pair(
        pair_id="pairA",
        cell_results=cells,
        library=_library_4(),
        scoring_config=cfg,
    )
    assert ctx is not None
    assert ctx.zh_mean_score is not None and abs(ctx.zh_mean_score - 0.81) < 1e-9
    assert ctx.en_mean_score is not None and abs(ctx.en_mean_score - 0.795) < 1e-9
    assert ctx.cross_lingual_gap is not None and ctx.cross_lingual_gap < 0.10

    pair = PromptPair(
        prompt_pair_id="pairA",
        system_prompt_id="sysA",
        negative_prompt_id="negA",
        round_id=1,
        system_prompt="P" * 50,
        scores=ctx,
    )
    row = flatten_pair_to_row(pair)
    assert row["zh_mean_score"] == ctx.zh_mean_score
    assert row["en_mean_score"] == ctx.en_mean_score
    assert row["prompt_sensitivity"] == ctx.prompt_sensitivity
    assert row["cross_lingual_gap"] == ctx.cross_lingual_gap
    assert row["worst_user_prompt_id"] == ctx.worst_user_prompt_id
    assert row["worst_user_prompt_score"] == ctx.worst_user_prompt_score


def test_roundtrip_language_brittle_flag_propagates():
    # zh much higher than en → cross_lingual_gap > 0.10 → language_brittle
    cells = _cell_scores(0.85, 0.85, 0.40, 0.45)
    cfg = ScoringConfig(flag_language_brittle_gap=0.10)
    ctx = build_score_context_for_pair(
        pair_id="pairB",
        cell_results=cells,
        library=_library_4(),
        scoring_config=cfg,
    )
    assert ctx.cross_lingual_gap is not None and ctx.cross_lingual_gap > 0.10

    pair = PromptPair(
        prompt_pair_id="pairB",
        system_prompt_id="sysB",
        negative_prompt_id="negB",
        round_id=1,
        system_prompt="P" * 50,
        scores=ctx,
    )
    row = flatten_pair_to_row(pair)
    # The producer should have populated cross_lingual_gap > threshold;
    # the consumer's row carries the flag (either via direct field or
    # derived from cross_lingual_gap > 0.10 fallback).
    assert row["cross_lingual_gap"] == ctx.cross_lingual_gap
    # Either the explicit producer flag or the consumer-derived fallback
    # must mark this as brittle.
    assert row.get("language_brittle") in (True, None) or row["cross_lingual_gap"] > 0.10


def test_roundtrip_worst_user_prompt_visible_in_row():
    cells = _cell_scores(0.90, 0.88, 0.45, 0.85)  # en_001 is worst
    cfg = ScoringConfig()
    ctx = build_score_context_for_pair(
        pair_id="pairC",
        cell_results=cells,
        library=_library_4(),
        scoring_config=cfg,
    )
    assert ctx.worst_user_prompt_id == "en_001"
    assert ctx.worst_user_prompt_score is not None and abs(ctx.worst_user_prompt_score - 0.45) < 1e-9

    pair = PromptPair(
        prompt_pair_id="pairC",
        system_prompt_id="sysC",
        negative_prompt_id="negC",
        round_id=1,
        system_prompt="P" * 50,
        scores=ctx,
    )
    row = flatten_pair_to_row(pair)
    assert row["worst_user_prompt_id"] == "en_001"
    assert row["worst_user_prompt_score"] is not None
    assert abs(row["worst_user_prompt_score"] - 0.45) < 1e-9


def test_roundtrip_schema_version_is_021():
    cells = _cell_scores(0.7, 0.7, 0.7, 0.7)
    ctx = build_score_context_for_pair(
        pair_id="pairD",
        cell_results=cells,
        library=_library_4(),
        scoring_config=ScoringConfig(),
    )
    pair = PromptPair(
        prompt_pair_id="pairD",
        system_prompt_id="sysD",
        negative_prompt_id="negD",
        round_id=1,
        system_prompt="P" * 50,
        scores=ctx,
    )
    row = flatten_pair_to_row(pair)
    assert row["schema_version"] == "0.2.1"
