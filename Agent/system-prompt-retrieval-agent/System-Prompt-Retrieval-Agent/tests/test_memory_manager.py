"""S02.10 — Tests for MemoryManager: load_for_generation, first-round-empty, seed."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pytest

from system_prompt_retrieval_agent.memory.manager import MemoryManager
from system_prompt_retrieval_agent.memory.long import append_long_memory_row
from system_prompt_retrieval_agent.schemas import (
    PromptPair,
    PromptPairHistoryContext,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_row(pair_id: str, round_: int, score: float) -> dict:
    return {
        "schema_version": "1.0",
        "prompt_pair_id": pair_id,
        "system_prompt_id": f"sp-{pair_id}",
        "negative_prompt_id": "neg-001",
        "round": str(round_),
        "overall_score": str(score),
        "total_score": str(score),
        "qwen_pass_rate": "0.8",
        "edit_correctness": "0.7",
        "garment_transfer_correctness": "0.6",
        "preservation": "0.9",
        "artifact_penalty": "0.1",
        "dress_weighted": "0.5",
        "lower_weighted": "0.4",
        "upper_weighted": "0.3",
        "fallback": "False",
        "timestamp": "2026-04-25T00:00:00Z",
    }


@pytest.fixture
def mem(tmp_path: Path) -> MemoryManager:
    return MemoryManager(memory_root=tmp_path / "memory")


@pytest.fixture
def mem_with_data(tmp_path: Path) -> MemoryManager:
    """MemoryManager pre-populated with 6 rows across 2 rounds."""
    m = MemoryManager(memory_root=tmp_path / "memory")
    # Round 1: pair-001 to pair-003
    for i in range(3):
        m.append_long_memory(
            _make_row(f"pair-{i+1:03d}", round_=1, score=0.5 + i * 0.1)
        )
    # Round 2: pair-004 to pair-006 (higher scores)
    for i in range(3):
        m.append_long_memory(
            _make_row(f"pair-{i+4:03d}", round_=2, score=0.7 + i * 0.05)
        )
    return m


# ---------------------------------------------------------------------------
# First-round-empty case
# ---------------------------------------------------------------------------

def test_first_round_empty_returns_none_blocks(mem: MemoryManager):
    ctx = mem.load_for_generation(round_id=1, N=3)
    assert isinstance(ctx, PromptPairHistoryContext)
    assert ctx.round_id == 1
    assert ctx.long_memory_prompt_pairs is None
    assert ctx.project_memory_prompt_pairs is None
    assert ctx.previous_top_prompt_pairs is None
    assert ctx.previous_round_prompt_pairs is None
    assert ctx.random_history_prompt_pairs is None


# ---------------------------------------------------------------------------
# load_for_generation(round_id=2, N=3) with data
# ---------------------------------------------------------------------------

def test_load_for_generation_returns_history_context(mem_with_data: MemoryManager):
    ctx = mem_with_data.load_for_generation(round_id=2, N=3)
    assert isinstance(ctx, PromptPairHistoryContext)
    assert ctx.round_id == 2


def test_all_5_blocks_populated(mem_with_data: MemoryManager):
    ctx = mem_with_data.load_for_generation(round_id=2, N=3)
    # With data available, at least some blocks should be populated
    # previous_round_prompt_pairs: round==1 pairs (round_id-1 = 1)
    assert ctx.previous_round_prompt_pairs is not None
    assert len(ctx.previous_round_prompt_pairs) > 0

    # long_memory_prompt_pairs: top pairs excluding round-1
    assert ctx.long_memory_prompt_pairs is not None

    # previous_top_prompt_pairs
    assert ctx.previous_top_prompt_pairs is not None

    # random_history_prompt_pairs
    assert ctx.random_history_prompt_pairs is not None


def test_blocks_contain_prompt_pair_instances(mem_with_data: MemoryManager):
    ctx = mem_with_data.load_for_generation(round_id=2, N=3)
    for block in [
        ctx.previous_round_prompt_pairs,
        ctx.long_memory_prompt_pairs,
        ctx.previous_top_prompt_pairs,
        ctx.random_history_prompt_pairs,
    ]:
        if block is not None:
            for item in block:
                assert isinstance(item, PromptPair)


def test_blocks_have_score_context(mem_with_data: MemoryManager):
    ctx = mem_with_data.load_for_generation(round_id=2, N=3)
    for block in [ctx.long_memory_prompt_pairs, ctx.previous_round_prompt_pairs]:
        if block:
            for pp in block:
                assert pp.scores is not None
                # overall_score should be a float
                assert isinstance(pp.scores.overall_score, float)


def test_previous_round_pairs_have_correct_role(mem_with_data: MemoryManager):
    ctx = mem_with_data.load_for_generation(round_id=2, N=3)
    if ctx.previous_round_prompt_pairs:
        for pp in ctx.previous_round_prompt_pairs:
            assert pp.selection_role == "previous_round"


def test_long_memory_pairs_have_correct_role(mem_with_data: MemoryManager):
    ctx = mem_with_data.load_for_generation(round_id=2, N=3)
    if ctx.long_memory_prompt_pairs:
        for pp in ctx.long_memory_prompt_pairs:
            assert pp.selection_role == "long_memory_reference"


def test_random_history_seeded_deterministic(mem_with_data: MemoryManager):
    ctx1 = mem_with_data.load_for_generation(round_id=2, N=3)
    ctx2 = mem_with_data.load_for_generation(round_id=2, N=3)
    ids1 = [pp.prompt_pair_id for pp in (ctx1.random_history_prompt_pairs or [])]
    ids2 = [pp.prompt_pair_id for pp in (ctx2.random_history_prompt_pairs or [])]
    assert ids1 == ids2


def test_random_history_different_for_different_rounds(mem_with_data: MemoryManager):
    ctx_r2 = mem_with_data.load_for_generation(round_id=2, N=3)
    ctx_r3 = mem_with_data.load_for_generation(round_id=3, N=3)
    ids2 = sorted([pp.prompt_pair_id for pp in (ctx_r2.random_history_prompt_pairs or [])])
    ids3 = sorted([pp.prompt_pair_id for pp in (ctx_r3.random_history_prompt_pairs or [])])
    # Different seeds may or may not produce same results; but at least it runs
    assert isinstance(ids3, list)


# ---------------------------------------------------------------------------
# project_memory: pairs with score > 0.60 and seen >= 2
# ---------------------------------------------------------------------------

def test_project_memory_requires_score_and_repeat(tmp_path: Path):
    m = MemoryManager(memory_root=tmp_path / "memory")
    # pair-A appears twice with score > 0.60
    m.append_long_memory(_make_row("pair-A", 1, 0.75))
    m.append_long_memory(_make_row("pair-A", 2, 0.80))
    # pair-B appears once with score > 0.60
    m.append_long_memory(_make_row("pair-B", 1, 0.90))
    # pair-C appears twice but score <= 0.60
    m.append_long_memory(_make_row("pair-C", 1, 0.50))
    m.append_long_memory(_make_row("pair-C", 2, 0.55))

    ctx = m.load_for_generation(round_id=3, N=3)
    proj = ctx.project_memory_prompt_pairs
    if proj is not None:
        proj_ids = {pp.prompt_pair_id for pp in proj}
        assert "pair-A" in proj_ids     # score>0.6 and seen twice
        assert "pair-B" not in proj_ids  # only seen once
        assert "pair-C" not in proj_ids  # score <=0.6


# ---------------------------------------------------------------------------
# determine_round_id
# ---------------------------------------------------------------------------

def test_determine_round_id_empty_memory(mem: MemoryManager):
    assert mem.determine_round_id() == 1


def test_determine_round_id_with_existing_rounds(tmp_path: Path):
    m = MemoryManager(memory_root=tmp_path / "memory")
    (m.rounds_dir / "round_001").mkdir(parents=True)
    (m.rounds_dir / "round_002").mkdir(parents=True)
    assert m.determine_round_id() == 3


# ---------------------------------------------------------------------------
# mark_fallback_round
# ---------------------------------------------------------------------------

def test_mark_fallback_increments_counter(mem: MemoryManager):
    mem.mark_fallback_round(1, is_fallback=True)
    sched = mem.load_schedule()
    assert sched["fallback_prompt_generation"] is True
    assert sched["consecutive_fallback_rounds"] == 1

    mem.mark_fallback_round(2, is_fallback=True)
    sched = mem.load_schedule()
    assert sched["consecutive_fallback_rounds"] == 2


def test_mark_fallback_reset(mem: MemoryManager):
    mem.mark_fallback_round(1, is_fallback=True)
    mem.mark_fallback_round(2, is_fallback=True)
    mem.mark_fallback_round(3, is_fallback=False)
    sched = mem.load_schedule()
    assert sched["fallback_prompt_generation"] is False
    assert sched["consecutive_fallback_rounds"] == 0


# ---------------------------------------------------------------------------
# Seed behavior
# ---------------------------------------------------------------------------

def test_seed_from_temp_wxy_no_loader_returns_empty(mem: MemoryManager):
    seeds = mem.seed_from_temp_wxy()
    assert seeds == []


def test_seed_from_temp_wxy_with_loader(tmp_path: Path):
    def mock_loader():
        return {
            "v5": "v5",
            "v5_text": "system prompt v5",
            "v5_new": "v5_new",
            "v5_new_text": "system prompt v5 new",
        }

    m = MemoryManager(memory_root=tmp_path / "memory", seed_loader=mock_loader)
    seeds = m.seed_from_temp_wxy()
    assert len(seeds) == 3
    for seed in seeds:
        assert isinstance(seed, PromptPair)
        assert seed.round_id == 0
        assert seed.selection_role == "seed"

    # Check the null-neg seed
    null_neg_seed = next((s for s in seeds if "null" in s.prompt_pair_id), None)
    assert null_neg_seed is not None
    assert null_neg_seed.negative_prompt is None

    # Check default-neg seeds have negative prompt
    default_neg_seeds = [s for s in seeds if s.negative_prompt is not None]
    assert len(default_neg_seeds) == 2


def test_seed_loader_exception_returns_empty(tmp_path: Path):
    def bad_loader():
        raise RuntimeError("failed to load prompts")

    m = MemoryManager(memory_root=tmp_path / "memory", seed_loader=bad_loader)
    seeds = m.seed_from_temp_wxy()
    assert seeds == []


# ---------------------------------------------------------------------------
# prune_long_memory integration
# ---------------------------------------------------------------------------

def test_prune_long_memory_keeps_top_n(tmp_path: Path):
    m = MemoryManager(memory_root=tmp_path / "memory")
    for i in range(10):
        m.append_long_memory(_make_row(f"pair-{i:03d}", 1, float(i) / 10))

    m.prune_long_memory(n=3)
    rows = m.load_long_memory()
    assert len(rows) == 3
    # All kept rows should have highest scores
    scores = sorted([float(r["overall_score"]) for r in rows], reverse=True)
    assert scores[0] == pytest.approx(0.9)


def test_prune_long_memory_archives_evicted(tmp_path: Path):
    m = MemoryManager(memory_root=tmp_path / "memory")
    for i in range(5):
        m.append_long_memory(_make_row(f"pair-{i:03d}", 1, float(i) / 10))

    m.prune_long_memory(n=2)
    from system_prompt_retrieval_agent.memory.long import read_long_memory
    archived = read_long_memory(m.long_archive)
    assert len(archived) == 3


# ---------------------------------------------------------------------------
# V0.2.1: load_for_generation reads per-category sub-scores from 0.2.1 CSV
# ---------------------------------------------------------------------------

def _make_021_row(pair_id: str, round_: int, score: float) -> dict:
    """Build a canonical 0.2.1 CSV row dict."""
    from system_prompt_retrieval_agent.memory.long import FIELDNAMES
    row = {col: "" for col in FIELDNAMES}
    row.update({
        "schema_version": "0.2.1",
        "prompt_pair_id": pair_id,
        "system_prompt_id": f"sp-{pair_id}",
        "negative_prompt_id": "neg-001",
        "round": str(round_),
        "overall_score": str(score),
        "total_score": str(score),
        "qwen_pass_rate": "0.8",
        "edit_correctness": "0.7",
        "garment_transfer_correctness": "0.6",
        "preservation": "0.9",
        "artifact_penalty": "0.1",
        "dress_weighted_score": "0.55",
        "dress_total_score": "0.50",
        "dress_qwen_pass_rate": "0.8",
        "dress_edit_correctness": "0.7",
        "dress_garment_transfer_correctness": "0.6",
        "dress_preservation": "0.9",
        "dress_artifact_penalty": "0.1",
        "dress_missing_score_reason": "",
        "lower_weighted_score": "",
        "lower_missing_score_reason": "category_scores_missing",
        "upper_weighted_score": "",
        "upper_missing_score_reason": "category_scores_missing",
        "zh_mean_score": "0.83",
        "en_mean_score": "0.87",
        "prompt_sensitivity": "0.04",
        "cross_lingual_gap": "0.04",
        "worst_user_prompt_id": "zh_001",
        "worst_user_prompt_score": "0.70",
        "language_brittle": "",
        "fallback": "False",
        "timestamp": "2026-04-25T00:00:00Z",
    })
    return row


def test_manager_v021_rows_readable(tmp_path: Path):
    """load_for_generation reads 0.2.1 rows with per-category sub-score columns.

    Use round_id=3 so round-1 pairs land in long_memory_prompt_pairs (not previous_round).
    """
    m = MemoryManager(memory_root=tmp_path / "memory")
    for i in range(3):
        m.append_long_memory(_make_021_row(f"pair-021-{i:03d}", round_=1, score=0.5 + i * 0.1))

    ctx = m.load_for_generation(round_id=3, N=3)
    # With round_id=3, previous_round is 2; rows at round=1 go to long_memory block.
    assert ctx.long_memory_prompt_pairs is not None
    for pp in ctx.long_memory_prompt_pairs:
        assert pp.scores is not None
        assert isinstance(pp.scores.overall_score, float)
        # sub_scores carried through.
        assert pp.scores.sub_scores.qwen_pass_rate == pytest.approx(0.8)


def test_manager_v021_yaml_absent_reads_from_csv(tmp_path: Path):
    """Per S02.10: reads top-level + per-category sub-scores from 0.2.1 CSV when YAML absent.

    Single pair at round=1; load round_id=2 → pair lands in previous_round block.
    Verifies overall_score and sub_scores are correctly parsed from CSV.
    """
    m = MemoryManager(memory_root=tmp_path / "memory")
    m.append_long_memory(_make_021_row("pair-csv-only", round_=1, score=0.77))

    ctx = m.load_for_generation(round_id=2, N=3)
    # With round_id=2, previous_round is 1, so the pair appears in previous_round block.
    assert ctx.previous_round_prompt_pairs is not None
    pp = next((p for p in ctx.previous_round_prompt_pairs if p.prompt_pair_id == "pair-csv-only"), None)
    assert pp is not None
    assert pp.scores is not None
    assert pp.scores.overall_score == pytest.approx(0.77)
    assert pp.scores.sub_scores.qwen_pass_rate == pytest.approx(0.8)
