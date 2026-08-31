"""V0.2.2 execution-mode isolation (S09.01–S09.05).

Three execution modes (constants live in ``canonical_paths``):

* ``v022_stage_major``        — production default; the only mode that
                                runs the V0.2.2 orchestrator.
* ``legacy_stage_all_compat`` — replay tooling that POSTs ``/stage/all``.
                                Production code paths must never enter
                                this mode.
* ``legacy_cartesian_compat`` — replay tooling that uses the legacy
                                cartesian survivor builder. Distinct
                                from ``legacy_stage_all_compat`` —
                                this module enforces they are never
                                conflated in trace fields (S09.02).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .remote._vendored import canonical_paths as cp


PRODUCTION_MODE: str = cp.PRODUCTION_EXECUTION_MODE  # "v022_stage_major"


class ExecutionModeError(RuntimeError):
    """Raised on configuration / runtime mode violations."""


def resolve_execution_mode(config: Mapping[str, Any]) -> str:
    """Return the resolved execution mode for *config*.

    Defaults to ``PRODUCTION_MODE`` when ``execution.mode`` is unset
    (S09.04). Unknown values are rejected.
    """
    if not isinstance(config, Mapping):
        return PRODUCTION_MODE
    execution = config.get("execution", {})
    if not isinstance(execution, Mapping):
        return PRODUCTION_MODE
    mode = execution.get("mode")
    if mode is None:
        return PRODUCTION_MODE
    if mode not in cp.ALLOWED_EXECUTION_MODES:
        raise ExecutionModeError(
            f"unknown execution.mode: {mode!r} "
            f"(allowed: {cp.ALLOWED_EXECUTION_MODES})"
        )
    return mode


def assert_production_mode_does_not_call_stage_all(mode: str, endpoint: str) -> None:
    """Hard-fail (S05.06 / S09.05) if production mode tries ``/stage/all``."""
    if mode == PRODUCTION_MODE and "/stage/all" in endpoint:
        raise ExecutionModeError(
            f"production mode ({PRODUCTION_MODE}) must not call {endpoint!r}"
        )


def assert_legacy_modes_are_distinct(mode: str, *, trace_field: str) -> None:
    """S09.02: trace field must not conflate the two legacy modes."""
    legacy_set = {
        cp.EXECUTION_MODE_LEGACY_STAGE_ALL,
        cp.EXECUTION_MODE_LEGACY_CARTESIAN,
    }
    if mode not in legacy_set:
        return
    if trace_field == cp.EXECUTION_MODE_LEGACY_STAGE_ALL and mode != cp.EXECUTION_MODE_LEGACY_STAGE_ALL:
        raise ExecutionModeError(
            f"trace_field={trace_field!r} conflated with mode={mode!r}"
        )
    if trace_field == cp.EXECUTION_MODE_LEGACY_CARTESIAN and mode != cp.EXECUTION_MODE_LEGACY_CARTESIAN:
        raise ExecutionModeError(
            f"trace_field={trace_field!r} conflated with mode={mode!r}"
        )


@dataclass(frozen=True)
class ExecutionModeTrace:
    """S09.03 — runtime trace fields identifying the execution mode."""

    execution_mode: str
    schema_version: str = cp.SCHEMA_VERSION
    is_production: bool = True

    def as_trace_dict(self) -> dict[str, Any]:
        return {
            "execution_mode": self.execution_mode,
            "schema_version": self.schema_version,
            "is_production": self.is_production,
        }


def trace_for_mode(mode: str) -> ExecutionModeTrace:
    if mode not in cp.ALLOWED_EXECUTION_MODES:
        raise ExecutionModeError(f"unknown execution mode: {mode!r}")
    return ExecutionModeTrace(
        execution_mode=mode,
        is_production=(mode == PRODUCTION_MODE),
    )
