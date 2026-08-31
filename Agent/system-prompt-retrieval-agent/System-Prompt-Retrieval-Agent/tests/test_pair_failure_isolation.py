"""S09 — Pair-failure isolation under strict mode (plan §6.4).

When one pair fails (any enabled cell errors), it goes to failed_pairs[]
without contaminating surviving_pairs[]. Strict mode (allow_partial_samples=False)
demands all enabled cells succeed for surviving pairs.
"""
from __future__ import annotations

from system_prompt_retrieval_agent.remote.partition import partition_stage_pairs
from system_prompt_retrieval_agent.schemas import (
    PerPairManifest,
    PerUserPromptManifest,
    StageManifest,
)


def _mf(pair_id, per_up):
    return PerPairManifest(
        prompt_pair_id=pair_id,
        ok=sum(u.ok for u in per_up.values()),
        errors=sum(u.errors for u in per_up.values()),
        total=sum(u.total for u in per_up.values()),
        per_user_prompt=per_up,
    )


def _up(ok, errors, total):
    return PerUserPromptManifest(ok=ok, errors=errors, total=total)


def _stage_manifest(pairs):
    return StageManifest(
        stage="gemma",
        run_id="r1",
        pairs=pairs,
        user_prompt_corpus_hash="a" * 64,
    )


def test_one_pair_failed_others_survive_strict():
    pairs = {
        "good": _mf("good", {"zh_001": _up(2, 0, 2), "en_001": _up(2, 0, 2)}),
        "bad": _mf("bad", {"zh_001": _up(1, 1, 2), "en_001": _up(2, 0, 2)}),
    }
    surv, failed = partition_stage_pairs(
        _stage_manifest(pairs),
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={"good": ["s1", "s2"], "bad": ["s1", "s2"]},
    )
    assert surv == ["good"]
    assert {f["prompt_pair_id"] for f in failed} == {"bad"}


def test_two_pairs_fail_independently_strict():
    pairs = {
        "A": _mf("A", {"zh_001": _up(1, 1, 2), "en_001": _up(2, 0, 2)}),
        "B": _mf("B", {"zh_001": _up(0, 2, 2), "en_001": _up(2, 0, 2)}),
        "C": _mf("C", {"zh_001": _up(2, 0, 2), "en_001": _up(2, 0, 2)}),
    }
    surv, failed = partition_stage_pairs(
        _stage_manifest(pairs),
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={k: ["s1", "s2"] for k in pairs},
    )
    assert surv == ["C"]
    assert {f["prompt_pair_id"] for f in failed} == {"A", "B"}


def test_failure_reason_recorded_per_pair():
    pairs = {
        "B": _mf("B", {"zh_001": _up(0, 2, 2), "en_001": _up(2, 0, 2)}),
    }
    _, failed = partition_stage_pairs(
        _stage_manifest(pairs),
        allow_partial=False,
        enabled_user_prompt_ids=["zh_001", "en_001"],
        expected_samples_for_pair={"B": ["s1", "s2"]},
    )
    assert failed[0]["failure_reason"]
