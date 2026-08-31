"""Tests for barrier_cell_keyed (V0.2.1) — plan §11.5, S04.08.

Tests cover:
  - Happy path (all pairs survive, no violations).
  - Strict mode: surviving pair with errors > 0 raises BarrierViolation.
  - Partial mode: surviving pair with errors allowed.
  - manifest_purpose="resume_missing_cells" — omitted cells not penalised.
  - manifest_purpose="prior_stage_survivor_cells" — any errors on survivor raise.
  - Count mismatch (ok + errors != total) raises BarrierViolation.
  - Per-user-prompt rollup mismatch raises BarrierViolation.
  - Pair in surviving_pairs but absent from pairs dict raises BarrierViolation.
  - Full Cartesian dispatch: expected total > got total raises BarrierViolation.
"""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.remote.barrier import (
    BarrierViolation,
    barrier_cell_keyed,
)
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


def _make_manifest(
    pairs: dict[str, PerPairManifest],
    surviving_pairs: list[str] | None = None,
    failed_pairs: list[dict] | None = None,
    corpus_hash: str = "a" * 64,
) -> StageManifest:
    if surviving_pairs is None:
        surviving_pairs = list(pairs.keys())
    if failed_pairs is None:
        failed_pairs = []
    return StageManifest(
        stage="gemma",
        run_id="r1",
        pairs=pairs,
        surviving_pairs=surviving_pairs,
        failed_pairs=failed_pairs,
        user_prompt_corpus_hash=corpus_hash,
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_barrier_happy_all_ok():
    """All pairs ok, no violations → returns expected surviving/failed lists."""
    pairs = {
        "pair_A": _make_pair("pair_A", ok=2, errors=0, total=2),
        "pair_B": _make_pair("pair_B", ok=2, errors=0, total=2),
    }
    manifest = _make_manifest(pairs, surviving_pairs=["pair_A", "pair_B"])

    surviving, failed = barrier_cell_keyed(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        sample_universe_for_pair={"pair_A": ["s1"], "pair_B": ["s1"]},
        manifest_purpose=None,
    )

    assert set(surviving) == {"pair_A", "pair_B"}
    assert failed == []


# ---------------------------------------------------------------------------
# Strict mode: errors > 0 on surviving pair raises BarrierViolation
# ---------------------------------------------------------------------------


def test_barrier_strict_surviving_pair_with_errors_raises():
    """Strict mode: surviving pair with errors > 0 → BarrierViolation."""
    bad_pair = PerPairManifest(
        prompt_pair_id="pair_A",
        ok=1,
        errors=1,
        total=2,
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
            "en_001": PerUserPromptManifest(ok=0, errors=1, total=1),
        },
    )
    manifest = _make_manifest(
        {"pair_A": bad_pair},
        surviving_pairs=["pair_A"],
    )

    with pytest.raises(BarrierViolation) as exc_info:
        barrier_cell_keyed(
            stage_manifest=manifest,
            allow_partial=False,
            enabled_user_prompt_ids=["zh_001", "en_001"],
            sample_universe_for_pair={"pair_A": ["s1"]},
            manifest_purpose=None,
        )
    assert exc_info.value.code == "strict_has_errors"


# ---------------------------------------------------------------------------
# Failed pair does NOT raise (only surviving-pair violations raise)
# ---------------------------------------------------------------------------


def test_barrier_failed_pair_does_not_raise():
    """Non-empty failed_pairs is recorded but does not raise BarrierViolation."""
    pairs = {
        "pair_A": _make_pair("pair_A", ok=2, errors=0, total=2),
        "pair_B": PerPairManifest(
            prompt_pair_id="pair_B",
            ok=0,
            errors=2,
            total=2,
            failure_reason="worker_crash",
            per_user_prompt={
                "zh_001": PerUserPromptManifest(ok=0, errors=1, total=1),
                "en_001": PerUserPromptManifest(ok=0, errors=1, total=1),
            },
        ),
    }
    manifest = _make_manifest(
        pairs,
        surviving_pairs=["pair_A"],
        failed_pairs=[{"prompt_pair_id": "pair_B", "failure_reason": "worker_crash"}],
    )

    # Should not raise.
    surviving, failed = barrier_cell_keyed(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        sample_universe_for_pair={"pair_A": ["s1"], "pair_B": ["s1"]},
        manifest_purpose=None,
    )

    assert "pair_A" in surviving
    assert any(f.get("prompt_pair_id") == "pair_B" for f in failed)


# ---------------------------------------------------------------------------
# Count mismatch on surviving pair raises BarrierViolation
# ---------------------------------------------------------------------------


def test_barrier_count_mismatch_raises():
    """ok + errors != total raises BarrierViolation('count_mismatch')."""
    bad_pair = PerPairManifest(
        prompt_pair_id="pair_A",
        ok=1,
        errors=0,
        total=3,  # mismatch
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
        },
    )
    manifest = _make_manifest({"pair_A": bad_pair}, surviving_pairs=["pair_A"])

    with pytest.raises(BarrierViolation) as exc_info:
        barrier_cell_keyed(
            stage_manifest=manifest,
            allow_partial=False,
            enabled_user_prompt_ids=["zh_001"],
            sample_universe_for_pair={"pair_A": ["s1"]},
            manifest_purpose=None,
        )
    assert exc_info.value.code == "count_mismatch"


# ---------------------------------------------------------------------------
# Per-user-prompt rollup mismatch raises BarrierViolation
# ---------------------------------------------------------------------------


def test_barrier_per_up_rollup_mismatch_raises():
    """Per-user-prompt sums != pair rollup raises BarrierViolation."""
    bad_pair = PerPairManifest(
        prompt_pair_id="pair_A",
        ok=3,
        errors=0,
        total=3,
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
            "en_001": PerUserPromptManifest(ok=1, errors=0, total=1),
            # sums: ok=2, total=2 != pair ok=3, total=3
        },
    )
    manifest = _make_manifest({"pair_A": bad_pair}, surviving_pairs=["pair_A"])

    with pytest.raises(BarrierViolation) as exc_info:
        barrier_cell_keyed(
            stage_manifest=manifest,
            allow_partial=False,
            enabled_user_prompt_ids=["zh_001", "en_001"],
            sample_universe_for_pair={"pair_A": ["s1"]},
            manifest_purpose=None,
        )
    assert exc_info.value.code == "per_up_rollup_mismatch"


# ---------------------------------------------------------------------------
# Surviving pair absent from pairs dict raises BarrierViolation
# ---------------------------------------------------------------------------


def test_barrier_surviving_pair_absent_from_pairs():
    """surviving_pairs references a pair_id not in pairs dict → raises."""
    manifest = StageManifest(
        stage="gemma",
        run_id="r1",
        pairs={},
        surviving_pairs=["ghost_pair"],
        failed_pairs=[],
        user_prompt_corpus_hash="a" * 64,
    )

    with pytest.raises(BarrierViolation) as exc_info:
        barrier_cell_keyed(
            stage_manifest=manifest,
            allow_partial=False,
            enabled_user_prompt_ids=["zh_001"],
            sample_universe_for_pair={"ghost_pair": ["s1"]},
            manifest_purpose=None,
        )
    assert exc_info.value.code == "missing_pair_manifest"


# ---------------------------------------------------------------------------
# Full Cartesian: expected total > got total raises BarrierViolation
# ---------------------------------------------------------------------------


def test_barrier_cartesian_underdispatch_raises():
    """When expected cells > received total, BarrierViolation('count_mismatch')."""
    pair = _make_pair("pair_A", ok=2, errors=0, total=2)
    manifest = _make_manifest({"pair_A": pair}, surviving_pairs=["pair_A"])

    # 2 user_prompts × 3 samples = 6 expected, but total=2 → mismatch
    with pytest.raises(BarrierViolation) as exc_info:
        barrier_cell_keyed(
            stage_manifest=manifest,
            allow_partial=False,
            enabled_user_prompt_ids=["zh_001", "en_001"],
            sample_universe_for_pair={"pair_A": ["s1", "s2", "s3"]},
            manifest_purpose=None,
        )
    assert exc_info.value.code == "count_mismatch"


# ---------------------------------------------------------------------------
# manifest_purpose="resume_missing_cells" — omitted cells not penalised
# ---------------------------------------------------------------------------


def test_barrier_resume_missing_cells_purpose_ok():
    """resume_missing_cells: omitted cells assumed on disk; no strict check."""
    # Pair has only 1 cell dispatched (the missing one), errors=0.
    pair = PerPairManifest(
        prompt_pair_id="pair_A",
        ok=1,
        errors=0,
        total=1,
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
        },
    )
    manifest = _make_manifest({"pair_A": pair}, surviving_pairs=["pair_A"])

    # Should not raise even though total (1) < expected Cartesian (2×2=4)
    surviving, failed = barrier_cell_keyed(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        sample_universe_for_pair={"pair_A": ["s1", "s2"]},
        manifest_purpose="resume_missing_cells",
    )
    assert "pair_A" in surviving
    assert failed == []


# ---------------------------------------------------------------------------
# manifest_purpose="prior_stage_survivor_cells" — errors on surviving raise
# ---------------------------------------------------------------------------


def test_barrier_prior_stage_survivor_cells_errors_raise():
    """prior_stage_survivor_cells: errors in surviving pair raises."""
    bad_pair = PerPairManifest(
        prompt_pair_id="pair_A",
        ok=1,
        errors=1,
        total=2,
        per_user_prompt={
            "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
            "en_001": PerUserPromptManifest(ok=0, errors=1, total=1),
        },
    )
    manifest = _make_manifest({"pair_A": bad_pair}, surviving_pairs=["pair_A"])

    with pytest.raises(BarrierViolation) as exc_info:
        barrier_cell_keyed(
            stage_manifest=manifest,
            allow_partial=False,
            enabled_user_prompt_ids=["zh_001", "en_001"],
            sample_universe_for_pair={"pair_A": ["s1"]},
            manifest_purpose="prior_stage_survivor_cells",
        )
    assert exc_info.value.code == "strict_has_errors"


# ---------------------------------------------------------------------------
# Partial mode: surviving pair with errors does NOT raise
# ---------------------------------------------------------------------------


def test_barrier_partial_mode_errors_not_raised():
    """allow_partial=True: errors on surviving pair do not raise."""
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
    manifest = _make_manifest({"pair_A": pair}, surviving_pairs=["pair_A"])

    # Should not raise in partial mode.
    surviving, failed = barrier_cell_keyed(
        stage_manifest=manifest,
        allow_partial=True,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        sample_universe_for_pair={"pair_A": ["s1"]},
        manifest_purpose=None,
    )
    assert "pair_A" in surviving


# ---------------------------------------------------------------------------
# Empty surviving_pairs → empty results
# ---------------------------------------------------------------------------


def test_barrier_empty_surviving_pairs():
    manifest = StageManifest(
        stage="gemma",
        run_id="r1",
        pairs={},
        surviving_pairs=[],
        failed_pairs=[],
        user_prompt_corpus_hash="a" * 64,
    )
    surviving, failed = barrier_cell_keyed(
        stage_manifest=manifest,
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001"],
        sample_universe_for_pair={},
        manifest_purpose=None,
    )
    assert surviving == []
    assert failed == []
