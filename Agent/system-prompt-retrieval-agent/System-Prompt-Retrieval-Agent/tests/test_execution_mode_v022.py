"""S09 tests for V0.2.2 execution-mode isolation."""

from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.execution_mode_v022 import (
    ExecutionModeError,
    PRODUCTION_MODE,
    assert_legacy_modes_are_distinct,
    assert_production_mode_does_not_call_stage_all,
    resolve_execution_mode,
    trace_for_mode,
)
from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp


# ---------------------------------------------------------------------------
# S09.04 — default mode
# ---------------------------------------------------------------------------


def test_default_mode_is_v022_stage_major():
    assert resolve_execution_mode({}) == PRODUCTION_MODE
    assert PRODUCTION_MODE == "v022_stage_major"


def test_explicit_legacy_modes_resolved():
    assert resolve_execution_mode({"execution": {"mode": "legacy_stage_all_compat"}}) == "legacy_stage_all_compat"
    assert resolve_execution_mode({"execution": {"mode": "legacy_cartesian_compat"}}) == "legacy_cartesian_compat"


def test_unknown_mode_rejected():
    with pytest.raises(ExecutionModeError, match="unknown execution.mode"):
        resolve_execution_mode({"execution": {"mode": "stage_all"}})


# ---------------------------------------------------------------------------
# S09.05 — production mode hard-fails on /stage/all
# ---------------------------------------------------------------------------


def test_production_calling_stage_all_raises():
    with pytest.raises(ExecutionModeError, match="/stage/all"):
        assert_production_mode_does_not_call_stage_all(PRODUCTION_MODE, "/stage/all")


def test_legacy_compat_can_call_stage_all():
    # Should not raise.
    assert_production_mode_does_not_call_stage_all(
        "legacy_stage_all_compat", "/stage/all"
    )


def test_v022_endpoints_are_safe_for_production():
    for ep in ("/v022/stage/gemma", "/v022/stage/flux", "/v022/stage/qwen"):
        assert_production_mode_does_not_call_stage_all(PRODUCTION_MODE, ep)


# ---------------------------------------------------------------------------
# S09.02 — legacy modes are not conflated in trace fields
# ---------------------------------------------------------------------------


def test_legacy_modes_not_conflated_in_trace():
    # Trace field declared as stage_all_compat must match mode.
    assert_legacy_modes_are_distinct(
        "legacy_stage_all_compat", trace_field="legacy_stage_all_compat"
    )
    with pytest.raises(ExecutionModeError, match="conflated"):
        assert_legacy_modes_are_distinct(
            "legacy_cartesian_compat", trace_field="legacy_stage_all_compat"
        )
    with pytest.raises(ExecutionModeError, match="conflated"):
        assert_legacy_modes_are_distinct(
            "legacy_stage_all_compat", trace_field="legacy_cartesian_compat"
        )


# ---------------------------------------------------------------------------
# S09.03 — runtime trace field
# ---------------------------------------------------------------------------


def test_trace_for_production_marks_is_production_true():
    trace = trace_for_mode(PRODUCTION_MODE)
    d = trace.as_trace_dict()
    assert d == {
        "execution_mode": "v022_stage_major",
        "schema_version": "v0.2.2",
        "is_production": True,
    }


@pytest.mark.parametrize("mode", ["legacy_stage_all_compat", "legacy_cartesian_compat"])
def test_trace_for_legacy_marks_is_production_false(mode):
    trace = trace_for_mode(mode)
    d = trace.as_trace_dict()
    assert d["execution_mode"] == mode
    assert d["is_production"] is False
