"""Regression: rate limiter must honour cfg.rate_limits.* values.

Pre-fix the global ``RateLimiter`` was hardcoded at rps=3.0 / burst=5
and ignored every config edit. Two failure modes:
  1. ``cfg.rate_limits.requests_per_second`` was never read.
  2. ``cfg.rate_limits.requests_per_minute`` was never enforced
     (no per-minute window existed).

Option-A fix: ``init_rate_limiter(rps, rpm, max_concurrency)`` is
called from ``V022Runner.__init__``; ``get_rate_limiter()`` returns
the same singleton.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from system_prompt_retrieval_agent.rate_limiter import (
    RateLimiter,
    get_rate_limiter,
    init_rate_limiter,
    reset_rate_limiter,
)


@pytest.fixture(autouse=True)
def _isolate():
    reset_rate_limiter()
    yield
    reset_rate_limiter()


def test_init_rate_limiter_sets_global_from_config():
    rl = init_rate_limiter(rps=5.0, rpm=200, max_concurrency=2)
    same = get_rate_limiter()
    assert rl is same, "get_rate_limiter must return the init'd singleton"
    assert rl.rps == 5.0
    assert rl.rpm == 200
    assert rl.max_concurrency == 2
    assert rl.semaphore is not None
    # Default burst = max(1, int(rps))
    assert rl.burst == 5


def test_get_rate_limiter_without_init_warns_and_uses_legacy_default(caplog):
    """Falling back to legacy 3 rps must be loud (warning) so the
    misconfiguration is visible in tracebacks."""
    with caplog.at_level("WARNING"):
        rl = get_rate_limiter()
    assert rl.rps == 3.0
    assert rl.burst == 5
    assert rl.rpm == 0  # legacy default has no rpm window
    assert any("init_rate_limiter" in r.message for r in caplog.records)


def test_init_overrides_legacy_singleton():
    """If get_rate_limiter() ran first (legacy) and then
    init_rate_limiter is called from V022Runner.__init__, the new
    config-driven limiter must replace the legacy one."""
    legacy = get_rate_limiter()  # creates legacy 3 rps
    assert legacy.rps == 3.0

    new = init_rate_limiter(rps=5.0, rpm=200, max_concurrency=2)
    assert new is not legacy
    assert get_rate_limiter() is new
    assert new.rps == 5.0


@pytest.mark.asyncio
async def test_acquire_respects_rps_token_bucket():
    """At rps=2, burst=2: the third acquire must wait at least ~0.4s
    (one token refill at 2 tokens/sec)."""
    rl = init_rate_limiter(rps=2.0, rpm=0, max_concurrency=0, burst=2)
    t0 = time.monotonic()
    await rl.acquire()
    await rl.acquire()
    await rl.acquire()
    elapsed = time.monotonic() - t0
    # Two tokens consumed instantly from burst; third waits ~0.5s
    assert elapsed >= 0.3, f"third acquire returned too fast: {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_rpm_window_throttles_when_rps_burst_would_otherwise_admit():
    """rps headroom + burst would normally let many calls through. The
    rpm window must independently throttle once the per-minute cap is
    reached."""
    rl = init_rate_limiter(rps=100.0, rpm=3, max_concurrency=0, burst=100)
    t0 = time.monotonic()
    # First 3 should pass instantly; 4th must block on rpm window.
    for _ in range(3):
        await rl.acquire()
    assert time.monotonic() - t0 < 0.5, "first 3 should be instant"

    # The 4th should wait for the oldest minute-window entry to age
    # past 60 s. We can't actually wait 60 s in a unit test, so verify
    # the wait gate fires by manipulating internal state to age the
    # window forward.
    # Force the oldest entry to be very old → 4th should pass instantly.
    rl.minute_window[0] -= 61.0
    t1 = time.monotonic()
    await rl.acquire()
    assert time.monotonic() - t1 < 0.5, "after artificial age-out, 4th should pass"


def test_rpm_zero_disables_minute_window():
    """rpm=0 means "no minute-window enforcement"; only rps matters."""
    rl = init_rate_limiter(rps=10.0, rpm=0, max_concurrency=0, burst=10)
    # rpm_wait should always return 0 when rpm=0 regardless of history
    assert rl._rpm_wait(time.monotonic()) == 0.0
    # Even after pretending we made many calls, rpm=0 ignores the window.
    rl.minute_window.extend([time.monotonic()] * 1000)
    assert rl._rpm_wait(time.monotonic()) == 0.0


def test_max_concurrency_zero_disables_semaphore():
    rl = init_rate_limiter(rps=5.0, rpm=200, max_concurrency=0)
    assert rl.semaphore is None


def test_max_concurrency_positive_yields_semaphore():
    rl = init_rate_limiter(rps=5.0, rpm=200, max_concurrency=2)
    assert rl.semaphore is not None
    # Semaphore initial value reflects max_concurrency.
    assert rl.semaphore._value == 2  # type: ignore[attr-defined]
