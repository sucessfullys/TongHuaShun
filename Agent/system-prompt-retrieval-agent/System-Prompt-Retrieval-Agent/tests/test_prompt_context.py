"""Tests for PromptPairHistoryContext building and validation (S03.10)."""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.prompt_generation.context_builder import (
    assert_rich_context,
    build_history_context,
)
from system_prompt_retrieval_agent.schemas import (
    PromptPair,
    PromptPairHistoryContext,
    PromptPairScoreContext,
    PromptPairSubScores,
)


def _make_pair(
    pair_id: str,
    scores: PromptPairScoreContext | None = None,
    round_id: int = 1,
) -> PromptPair:
    return PromptPair(
        prompt_pair_id=pair_id,
        system_prompt_id=f"sys_{pair_id}",
        negative_prompt_id=f"neg_{pair_id}",
        round_id=round_id,
        system_prompt="A" * 30,
        scores=scores,
    )


def _make_score(overall: float | None = 0.75, reason: str | None = None) -> PromptPairScoreContext:
    return PromptPairScoreContext(
        overall_score=overall,
        missing_score_reason=reason,
    )


# ---------------------------------------------------------------------------
# Fixture: fully populated context with all 5 blocks
# ---------------------------------------------------------------------------

@pytest.fixture
def rich_context() -> PromptPairHistoryContext:
    scored = _make_score(0.80)
    pairs_long = [_make_pair("lm_01", scored), _make_pair("lm_02", _make_score(0.70))]
    pairs_proj = [_make_pair("pm_01", _make_score(0.65))]
    pairs_top = [_make_pair("top_01", _make_score(0.85))]
    pairs_prev = [_make_pair("pr_01", _make_score(0.55)), _make_pair("pr_02", _make_score(0.60))]
    pairs_rand = [_make_pair("rh_01", _make_score(0.50))]
    return PromptPairHistoryContext(
        round_id=2,
        long_memory_prompt_pairs=pairs_long,
        project_memory_prompt_pairs=pairs_proj,
        previous_top_prompt_pairs=pairs_top,
        previous_round_prompt_pairs=pairs_prev,
        random_history_prompt_pairs=pairs_rand,
    )


# ---------------------------------------------------------------------------
# Test 1: assert_rich_context passes for a valid context
# ---------------------------------------------------------------------------

def test_assert_rich_context_passes(rich_context: PromptPairHistoryContext) -> None:
    # Should not raise
    assert_rich_context(rich_context)


# ---------------------------------------------------------------------------
# Test 2: scores=None raises AssertionError
# ---------------------------------------------------------------------------

def test_assert_rich_context_raises_on_none_scores() -> None:
    bad_pair = _make_pair("bad_01", scores=None)  # scores=None, no missing_score_reason
    ctx = PromptPairHistoryContext(
        round_id=2,
        long_memory_prompt_pairs=[bad_pair],
    )
    with pytest.raises(AssertionError, match="detached-table"):
        assert_rich_context(ctx)


def test_assert_rich_context_raises_in_previous_round_block() -> None:
    bad_pair = _make_pair("bad_02", scores=None)
    ctx = PromptPairHistoryContext(
        round_id=3,
        previous_round_prompt_pairs=[bad_pair],
    )
    with pytest.raises(AssertionError, match="detached-table"):
        assert_rich_context(ctx)


def test_assert_rich_context_raises_in_previous_top_block() -> None:
    bad_pair = _make_pair("bad_03", scores=None)
    ctx = PromptPairHistoryContext(
        round_id=3,
        previous_top_prompt_pairs=[bad_pair],
    )
    with pytest.raises(AssertionError, match="detached-table"):
        assert_rich_context(ctx)


# ---------------------------------------------------------------------------
# Test 3: missing_score_reason on PromptPairScoreContext is preserved after
#         serialization (attached-to-pair shape)
# ---------------------------------------------------------------------------

def test_attached_pair_shape_preserved_in_serialization() -> None:
    score_with_reason = PromptPairScoreContext(
        overall_score=None,
        missing_score_reason="Not yet evaluated",
    )
    pair = _make_pair("shape_01", scores=score_with_reason)
    ctx = PromptPairHistoryContext(
        round_id=1,
        long_memory_prompt_pairs=[pair],
    )

    # Verify assert_rich_context passes (score is attached as PromptPairScoreContext)
    assert_rich_context(ctx)

    # Verify round-trip serialization preserves attached shape
    json_str = ctx.model_dump_json()
    restored = PromptPairHistoryContext.model_validate_json(json_str)
    restored_pair = restored.long_memory_prompt_pairs[0]
    assert restored_pair.scores is not None
    assert isinstance(restored_pair.scores, PromptPairScoreContext)
    assert restored_pair.scores.missing_score_reason == "Not yet evaluated"


def test_mixed_pairs_shape_preserved() -> None:
    """All five blocks with mixed scored/reason pairs preserve attached shape."""
    real_score = _make_score(0.72)
    reason_score = PromptPairScoreContext(
        overall_score=None, missing_score_reason="seed pair"
    )
    ctx = PromptPairHistoryContext(
        round_id=5,
        long_memory_prompt_pairs=[_make_pair("m1", real_score)],
        project_memory_prompt_pairs=[_make_pair("m2", reason_score)],
        previous_top_prompt_pairs=[_make_pair("m3", real_score)],
        previous_round_prompt_pairs=[_make_pair("m4", real_score)],
        random_history_prompt_pairs=[_make_pair("m5", reason_score)],
    )
    assert_rich_context(ctx)
    json_str = ctx.model_dump_json()
    restored = PromptPairHistoryContext.model_validate_json(json_str)
    for block_name in [
        "long_memory_prompt_pairs",
        "project_memory_prompt_pairs",
        "previous_top_prompt_pairs",
        "previous_round_prompt_pairs",
        "random_history_prompt_pairs",
    ]:
        pairs = getattr(restored, block_name)
        assert pairs is not None
        for p in pairs:
            assert isinstance(p.scores, PromptPairScoreContext)


# ---------------------------------------------------------------------------
# Test 4: build_history_context delegates to memory_manager.load_for_generation
# ---------------------------------------------------------------------------

def test_build_history_context_delegates_to_memory_manager() -> None:
    expected_ctx = PromptPairHistoryContext(round_id=3)

    class MockMemoryManager:
        def __init__(self):
            self.calls = []

        def load_for_generation(self, round_id: int, N: int) -> PromptPairHistoryContext:
            self.calls.append((round_id, N))
            return expected_ctx

    mgr = MockMemoryManager()
    result = build_history_context(mgr, round_id=3, N=5)

    assert result is expected_ctx
    assert mgr.calls == [(3, 5)]


def test_build_history_context_passes_correct_args() -> None:
    class MockMemoryManager:
        def __init__(self):
            self.round_id = None
            self.N = None

        def load_for_generation(self, round_id: int, N: int) -> PromptPairHistoryContext:
            self.round_id = round_id
            self.N = N
            return PromptPairHistoryContext(round_id=round_id)

    mgr = MockMemoryManager()
    build_history_context(mgr, round_id=7, N=3)
    assert mgr.round_id == 7
    assert mgr.N == 3


# ---------------------------------------------------------------------------
# Test 5: assert_rich_context passes on empty/None blocks
# ---------------------------------------------------------------------------

def test_assert_rich_context_empty_context() -> None:
    ctx = PromptPairHistoryContext(round_id=1)
    # All blocks None — should pass silently
    assert_rich_context(ctx)


def test_assert_rich_context_partial_blocks() -> None:
    ctx = PromptPairHistoryContext(
        round_id=1,
        long_memory_prompt_pairs=[_make_pair("ok1", _make_score(0.9))],
    )
    assert_rich_context(ctx)
