"""S08 tests for V0.2.2 lifecycle hard-disable + benchmark scaffolding."""

from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.lifecycle_v022 import (
    BenchmarkResult,
    LifecycleConfigError,
    StageLoadTimings,
    WarmModeBudget,
    assert_cold_mode_state,
    assert_v022_lifecycle_config,
    fallback_to_cold_mode_if_unsafe,
)
from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp


# ---------------------------------------------------------------------------
# S08.05a — runtime hard-disable
# ---------------------------------------------------------------------------


def test_warm_lifecycle_mode_rejected():
    with pytest.raises(LifecycleConfigError, match="warm mode disabled in V0.2.2"):
        assert_v022_lifecycle_config({"execution": {"lifecycle_mode": "warm"}})


def test_warm_mode_enabled_flag_rejected_unconditionally():
    with pytest.raises(LifecycleConfigError, match="warm mode disabled"):
        assert_v022_lifecycle_config(
            {"execution": {"lifecycle_mode": "cold", "warm_mode_enabled": True}}
        )


def test_cold_mode_passes():
    assert_v022_lifecycle_config({"execution": {"lifecycle_mode": "cold"}})


def test_default_config_passes():
    assert_v022_lifecycle_config({})  # no execution block → cold default


# ---------------------------------------------------------------------------
# S08.10a — cold mode rejects warm-only states
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["cpu_prefetched", "gpu_unloaded_cpu_retained"])
def test_cold_mode_rejects_warm_states(state):
    with pytest.raises(LifecycleConfigError, match="cold-mode runs"):
        assert_cold_mode_state(state)


def test_cold_mode_rejects_gpu_loaded():
    with pytest.raises(LifecycleConfigError, match="gpu_loaded"):
        assert_cold_mode_state("gpu_loaded")


def test_cold_mode_accepts_disk_unloaded():
    assert_cold_mode_state("disk_unloaded")  # no raise


# ---------------------------------------------------------------------------
# S08.06–S08.09 — benchmark scaffolding
# ---------------------------------------------------------------------------


def test_benchmark_speedup_calculation():
    t = StageLoadTimings("gemma", disk_to_gpu_seconds=10.0, cpu_to_gpu_seconds=2.5)
    assert t.cpu_prefetch_speedup_seconds == 7.5


def test_benchmark_thresholds_pass_when_speedup_and_ram_ok():
    budget = WarmModeBudget(host_ram_gib_total=512.0, min_free_host_ram_gib=200.0)
    res = BenchmarkResult(
        timings=[StageLoadTimings("gemma", 10.0, 2.0),
                 StageLoadTimings("flux", 8.0, 1.0)],
        host_ram_free_gib=400.0,
        budget=budget,
    )
    assert res.passes_thresholds(min_speedup_seconds=5.0) is True


def test_benchmark_thresholds_fail_when_ram_below_threshold():
    budget = WarmModeBudget(host_ram_gib_total=512.0, min_free_host_ram_gib=200.0)
    res = BenchmarkResult(timings=[StageLoadTimings("gemma", 10.0, 2.0)],
                          host_ram_free_gib=100.0, budget=budget)
    assert res.passes_thresholds(min_speedup_seconds=5.0) is False


def test_fallback_to_cold_in_v022_regardless_of_benchmark():
    budget = WarmModeBudget(host_ram_gib_total=512.0, min_free_host_ram_gib=200.0)
    res = BenchmarkResult(
        timings=[StageLoadTimings("gemma", 10.0, 1.0),
                 StageLoadTimings("flux", 8.0, 0.5),
                 StageLoadTimings("qwen", 6.0, 0.5)],
        host_ram_free_gib=400.0, budget=budget,
    )
    # Even though the benchmark "passes", warm mode is hard-disabled.
    assert fallback_to_cold_mode_if_unsafe(res, min_speedup_seconds=2.0) == "cold"


# ---------------------------------------------------------------------------
# S08.10 — schema validators behave correctly via canonical_paths
# ---------------------------------------------------------------------------


def test_canonical_paths_lifecycle_states_exact():
    assert set(cp.ALLOWED_LIFECYCLE_STATES) == {
        "disk_unloaded", "cpu_prefetched", "gpu_loaded", "gpu_unloaded_cpu_retained"
    }
    assert "cold" not in cp.ALLOWED_LIFECYCLE_STATES
    assert "warm" not in cp.ALLOWED_LIFECYCLE_STATES
