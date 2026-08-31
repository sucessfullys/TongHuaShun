"""Tests for remote/partition.py — partition_stage_pairs.

Plan reference: §6.4, §11.5, S04.06.
"""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.remote.partition import partition_stage_pairs
from system_prompt_retrieval_agent.schemas import (
    PerPairManifest,
    PerUserPromptManifest,
    StageManifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(
    pair_id: str,
    ok: int = 1,
    errors: int = 0,
    total: int | None = None,
    user_prompt_ids: list[str] | None = None,
    per_ok: int = 1,
    per_err: int = 0,
) -> PerPairManifest:
    """Build a PerPairManifest with per_user_prompt rollups."""
    if user_prompt_ids is None:
        user_prompt_ids = ["zh_001", "en_001"]
    if total is None:
        total = ok + errors
    per_up = {
        up_id: PerUserPromptManifest(ok=per_ok, errors=per_err, total=per_ok + per_err)
        for up_id in user_prompt_ids
    }
    return PerPairManifest(
        prompt_pair_id=pair_id,
        ok=ok,
        errors=errors,
        total=total,
        per_user_prompt=per_up,
    )


def _make_manifest(pairs: dict[str, PerPairManifest]) -> StageManifest:
    return StageManifest(
        stage="gemma",
        run_id="r1",
        pairs=pairs,
        surviving_pairs=[],
        failed_pairs=[],
        user_prompt_corpus_hash="a" * 64,
    )


# ---------------------------------------------------------------------------
# Happy path: all pairs survive (strict mode)
# ---------------------------------------------------------------------------


def test_all_pairs_survive_strict():
    pairs = {
        "pair_A": _make_pair("pair_A", ok=2, errors=0, total=2),
        "pair_B": _make_pair("pair_B", ok=2, errors=0, total=2),
    }
    manifest = _make_manifest(pairs)

    surviving, failed = partition_stage_pairs(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={"pair_A": ["s1"], "pair_B": ["s1"]},
    )

    assert set(surviving) == {"pair_A", "pair_B"}
    assert failed == []


# ---------------------------------------------------------------------------
# Strict mode: one pair with errors → moves to failed_pairs
# ---------------------------------------------------------------------------


def test_strict_mode_pair_with_errors_fails():
    pairs = {
        "pair_A": _make_pair("pair_A", ok=2, errors=0, total=2),
        "pair_B": _make_pair("pair_B", ok=1, errors=1, total=2, per_ok=0, per_err=1),
    }
    # Adjust pair_B totals to reflect mixed per_user_prompt.
    pairs["pair_B"] = PerPairManifest(
        prompt_pair_id="pair_B",
        ok=1,
        errors=1,
        total=2,
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
            "en_001": PerUserPromptManifest(ok=0, errors=1, total=1),
        },
    )
    manifest = _make_manifest(pairs)

    surviving, failed = partition_stage_pairs(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={"pair_A": ["s1"], "pair_B": ["s1"]},
    )

    assert surviving == ["pair_A"]
    assert len(failed) == 1
    assert failed[0]["prompt_pair_id"] == "pair_B"
    assert failed[0]["failure_reason"] is not None


# ---------------------------------------------------------------------------
# Strict mode: count mismatch → pair fails
# ---------------------------------------------------------------------------


def test_strict_mode_count_mismatch():
    """ok + errors != total causes the pair to fail."""
    bad_pair = PerPairManifest(
        prompt_pair_id="pair_X",
        ok=1,
        errors=0,
        total=2,  # mismatch: ok + errors = 1 != 2
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
        },
    )
    manifest = _make_manifest({"pair_X": bad_pair})

    surviving, failed = partition_stage_pairs(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001"],
        expected_samples_for_pair={"pair_X": ["s1"]},
    )

    assert surviving == []
    assert len(failed) == 1
    assert failed[0]["prompt_pair_id"] == "pair_X"


# ---------------------------------------------------------------------------
# Strict mode: incomplete total (expected > got) → pair fails
# ---------------------------------------------------------------------------


def test_strict_mode_incomplete_total():
    """When expected total > got total, pair fails in strict mode."""
    pair = _make_pair("pair_A", ok=2, errors=0, total=2)
    manifest = _make_manifest({"pair_A": pair})

    surviving, failed = partition_stage_pairs(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={"pair_A": ["s1", "s2"]},  # expects 4 cells total
    )

    # 2 user_prompts × 2 samples = 4 expected, but only total=2 → fails
    assert surviving == []
    assert len(failed) == 1


# ---------------------------------------------------------------------------
# Partial mode: pair with one cell error survives if any (pair, UP) succeeded
# ---------------------------------------------------------------------------


def test_partial_mode_pair_with_one_cell_error_survives():
    """In partial mode a pair survives if at least one (pair, UP) cell ok > 0."""
    pair = PerPairManifest(
        prompt_pair_id="pair_A",
        ok=1,
        errors=1,
        total=2,
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
            "en_001": PerUserPromptManifest(ok=0, errors=1, total=1),
        },
    )
    manifest = _make_manifest({"pair_A": pair})

    surviving, failed = partition_stage_pairs(
        stage_manifest=manifest,
        allow_partial=True,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={"pair_A": ["s1"]},
    )

    assert "pair_A" in surviving
    assert failed == []


# ---------------------------------------------------------------------------
# Partial mode: pair with ALL cells failed → pair still fails
# ---------------------------------------------------------------------------


def test_partial_mode_all_cells_failed():
    """Partial mode still fails a pair when zero (pair, UP) cells succeeded."""
    pair = PerPairManifest(
        prompt_pair_id="pair_A",
        ok=0,
        errors=2,
        total=2,
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=0, errors=1, total=1),
            "en_001": PerUserPromptManifest(ok=0, errors=1, total=1),
        },
    )
    manifest = _make_manifest({"pair_A": pair})

    surviving, failed = partition_stage_pairs(
        stage_manifest=manifest,
        allow_partial=True,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={"pair_A": ["s1"]},
    )

    assert surviving == []
    assert len(failed) == 1


# ---------------------------------------------------------------------------
# Empty pairs dict → empty results
# ---------------------------------------------------------------------------


def test_empty_pairs():
    manifest = _make_manifest({})
    surviving, failed = partition_stage_pairs(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001"],
        expected_samples_for_pair={},
    )
    assert surviving == []
    assert failed == []


# ---------------------------------------------------------------------------
# Multiple pairs, mixed: some survive, some fail
# ---------------------------------------------------------------------------


def test_mixed_survival():
    pairs = {
        "pair_A": _make_pair("pair_A", ok=2, errors=0, total=2),
        "pair_B": PerPairManifest(
            prompt_pair_id="pair_B",
            ok=0,
            errors=2,
            total=2,
            per_user_prompt={
                "zh_001": PerUserPromptManifest(ok=0, errors=1, total=1),
                "en_001": PerUserPromptManifest(ok=0, errors=1, total=1),
            },
        ),
        "pair_C": _make_pair("pair_C", ok=2, errors=0, total=2),
    }
    manifest = _make_manifest(pairs)

    surviving, failed = partition_stage_pairs(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={
            "pair_A": ["s1"],
            "pair_B": ["s1"],
            "pair_C": ["s1"],
        },
    )

    assert set(surviving) == {"pair_A", "pair_C"}
    assert len(failed) == 1
    assert failed[0]["prompt_pair_id"] == "pair_B"


# ---------------------------------------------------------------------------
# per_user_prompt rollup mismatch → pair fails (duplicate_sample_id path)
# ---------------------------------------------------------------------------


def test_per_user_prompt_rollup_mismatch():
    """When per_user_prompt sums don't match pair totals, pair should fail."""
    pair = PerPairManifest(
        prompt_pair_id="pair_A",
        ok=2,
        errors=0,
        total=2,
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
            "en_001": PerUserPromptManifest(ok=2, errors=0, total=2),
            # sums: ok=3, errors=0, total=3 != pair ok=2, total=2
        },
    )
    manifest = _make_manifest({"pair_A": pair})

    surviving, failed = partition_stage_pairs(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={"pair_A": ["s1"]},
    )

    assert surviving == []
    assert len(failed) == 1
    assert failed[0]["failure_reason"] == "duplicate_sample_id"
