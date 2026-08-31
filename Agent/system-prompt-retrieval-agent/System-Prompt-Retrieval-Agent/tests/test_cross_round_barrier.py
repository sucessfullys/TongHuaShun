"""S09.09 — Cross-round barrier check.

Under workflow.cross_round_pipelining=False (default), round K+1 must
NOT begin before round K is fully finalized. Violations raise
CrossRoundBarrierViolation.
"""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.agent_loop_helpers import (
    CrossRoundBarrierViolation,
    assert_cross_round_barrier,
)


def test_round_1_always_passes():
    """First round has no prior round to wait on."""
    assert_cross_round_barrier(
        current_round=1,
        prior_round_finalized=False,
        cross_round_pipelining_enabled=False,
    )


def test_round_2_blocks_if_round_1_not_finalized():
    with pytest.raises(CrossRoundBarrierViolation):
        assert_cross_round_barrier(
            current_round=2,
            prior_round_finalized=False,
            cross_round_pipelining_enabled=False,
        )


def test_round_2_passes_if_round_1_finalized():
    assert_cross_round_barrier(
        current_round=2,
        prior_round_finalized=True,
        cross_round_pipelining_enabled=False,
    )


def test_pipelining_disables_barrier():
    """When cross_round_pipelining=True (debug only), the barrier is bypassed."""
    assert_cross_round_barrier(
        current_round=2,
        prior_round_finalized=False,
        cross_round_pipelining_enabled=True,
    )


def test_violation_message_includes_round_numbers():
    try:
        assert_cross_round_barrier(
            current_round=3,
            prior_round_finalized=False,
            cross_round_pipelining_enabled=False,
        )
    except CrossRoundBarrierViolation as e:
        assert "round 3" in str(e)
        assert "round 2" in str(e)
    else:
        raise AssertionError("Expected CrossRoundBarrierViolation")
