"""V0.2.2 lifecycle schema enforcement (S08.01–S08.11).

The canonical lifecycle enum + mode/state matrix lives in
``canonical_paths`` (S00.10 / S00.10a). This module is the runtime
gate that:

* Rejects ``execution.lifecycle_mode="warm"`` before any stage dispatch
  (S08.05a).
* Rejects ``execution.warm_mode_enabled=true`` unconditionally in
  V0.2.2, even when a fresh benchmark-pass artifact is present.
* Refuses any executor variant that would emit ``cpu_prefetched`` or
  ``gpu_unloaded_cpu_retained`` under cold mode.
* Provides the warm-mode benchmark scaffolding — measurement
  primitives, host-RAM budget, eviction order — so the future warm
  state machine has a concrete artifact to ship against. The benchmark
  itself does not run unless explicitly invoked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .remote._vendored import canonical_paths as cp


WARM_MODE_DISABLED_ERROR = cp.WARM_MODE_DISABLED_ERROR


class LifecycleConfigError(ValueError):
    """Raised when execution.lifecycle_mode / warm_mode_enabled fail the
    V0.2.2 hard-disable gate."""


# ---------------------------------------------------------------------------
# Runtime gate (S08.05a)
# ---------------------------------------------------------------------------


def assert_v022_lifecycle_config(config: Mapping[str, Any]) -> None:
    """Run before any stage dispatch.

    Rejects ``execution.lifecycle_mode="warm"`` and any truthy
    ``execution.warm_mode_enabled`` regardless of benchmark artifact.
    """
    execution = config.get("execution", {}) if isinstance(config, Mapping) else {}
    mode = execution.get("lifecycle_mode", cp.LIFECYCLE_MODE_COLD)
    enabled = execution.get("warm_mode_enabled", False)
    if mode == cp.LIFECYCLE_MODE_WARM:
        raise LifecycleConfigError(WARM_MODE_DISABLED_ERROR)
    if enabled:
        raise LifecycleConfigError(WARM_MODE_DISABLED_ERROR)


def assert_cold_mode_state(state_after: str) -> None:
    """Refuse states cold mode must never emit (S08.10a)."""
    if state_after in (cp.LIFECYCLE_CPU_PREFETCHED, cp.LIFECYCLE_GPU_UNLOADED_CPU_RETAINED):
        raise LifecycleConfigError(
            f"cold-mode runs must not emit lifecycle_state_after={state_after!r}"
        )
    if state_after == cp.LIFECYCLE_GPU_LOADED:
        raise LifecycleConfigError(
            "gpu_loaded is never an allowed lifecycle_state_after value"
        )
    cp.validate_lifecycle_state_after(cp.LIFECYCLE_MODE_COLD, state_after)


# ---------------------------------------------------------------------------
# Warm-mode benchmark scaffolding (S08.06 / S08.07 / S08.08 / S08.09)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmModeBudget:
    host_ram_gib_total: float
    min_free_host_ram_gib: float
    eviction_order: tuple[str, ...] = (
        cp.STAGE_GEMMA, cp.STAGE_FLUX, cp.STAGE_QWEN
    )


@dataclass(frozen=True)
class StageLoadTimings:
    stage: str
    disk_to_gpu_seconds: float
    cpu_to_gpu_seconds: float

    @property
    def cpu_prefetch_speedup_seconds(self) -> float:
        return max(0.0, self.disk_to_gpu_seconds - self.cpu_to_gpu_seconds)


@dataclass
class BenchmarkResult:
    timings: list[StageLoadTimings] = field(default_factory=list)
    host_ram_free_gib: float = 0.0
    budget: WarmModeBudget | None = None

    def passes_thresholds(self, *, min_speedup_seconds: float) -> bool:
        if self.budget is None:
            return False
        if self.host_ram_free_gib < self.budget.min_free_host_ram_gib:
            return False
        return all(
            t.cpu_prefetch_speedup_seconds >= min_speedup_seconds
            for t in self.timings
        )

    def at_most_one_cpu_resident_next_stage(self) -> bool:
        """S08.08: at most one next-stage model may be CPU-resident
        unless the benchmark proves combined residency is safe."""
        return True  # benchmark scaffolding default — combined residency unproven


def fallback_to_cold_mode_if_unsafe(
    benchmark: BenchmarkResult, *, min_speedup_seconds: float
) -> str:
    """S08.09 — return the safe lifecycle mode given a benchmark result.
    Always returns ``"cold"`` in V0.2.2 because warm mode is hard-disabled.
    """
    _ = benchmark.passes_thresholds(min_speedup_seconds=min_speedup_seconds)
    return cp.LIFECYCLE_MODE_COLD
