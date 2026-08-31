"""Rate limiter — config-aware dual token bucket.

Honours both ``rate_limits.requests_per_second`` (rps) and
``rate_limits.requests_per_minute`` (rpm) from ``config.yaml``. Both
windows must allow a request before ``acquire()`` returns. The rps
side is a leaky-bucket token reservoir with burst headroom; the rpm
side is a 60-second sliding-window counter. ``max_concurrency`` is
also honoured via an ``asyncio.Semaphore`` returned from
``acquire()`` so callers can bound in-flight requests on top of the
rate cap.

Initialization
--------------

The runner MUST call :func:`init_rate_limiter` once at startup before
any code calls :func:`get_rate_limiter`. The first ``get_rate_limiter``
without prior init still works (legacy-compat 3 rps / burst 5) but
emits a warning so the misconfiguration is visible in tracebacks.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    rps: float = 3.0
    burst: int = 5
    rpm: int = 0  # 0 disables the per-minute window
    max_concurrency: int = 0  # 0 disables the in-flight cap
    tokens: float = field(default=5.0)
    last_refill: float = field(default_factory=time.monotonic)
    minute_window: collections.deque = field(default_factory=collections.deque)
    _async_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _sync_lock: threading.Lock = field(default_factory=threading.Lock)
    _semaphore: asyncio.Semaphore | None = None
    usd_spent: float = 0.0

    def __post_init__(self) -> None:
        if self.max_concurrency and self.max_concurrency > 0:
            self._semaphore = asyncio.Semaphore(int(self.max_concurrency))

    # ---- internal helpers ----------------------------------------------------

    def _refill_rps(self) -> None:
        now = time.monotonic()
        delta = now - self.last_refill
        self.tokens = min(float(self.burst), self.tokens + delta * self.rps)
        self.last_refill = now

    def _evict_old_minute_entries(self, now: float | None = None) -> None:
        """Drop minute-window entries older than 60 s."""
        now = now if now is not None else time.monotonic()
        cutoff = now - 60.0
        while self.minute_window and self.minute_window[0] < cutoff:
            self.minute_window.popleft()

    def _rpm_wait(self, now: float) -> float:
        """Return seconds to wait before the rpm window has headroom.
        ``rpm == 0`` disables this check (returns 0)."""
        if not self.rpm or self.rpm <= 0:
            return 0.0
        self._evict_old_minute_entries(now)
        if len(self.minute_window) < self.rpm:
            return 0.0
        # Wait until the oldest in-window entry ages past the 60 s line.
        return max(0.0, (self.minute_window[0] + 60.0) - now)

    # ---- async API -----------------------------------------------------------

    async def acquire(self) -> None:
        """Block until both rps and rpm windows admit one request.

        After return, the call is "spent" against both windows. If
        ``max_concurrency`` is configured, callers should use
        :meth:`semaphore` as ``async with`` to additionally bound
        in-flight requests.
        """
        while True:
            async with self._async_lock:
                now = time.monotonic()
                self._refill_rps()
                rpm_wait = self._rpm_wait(now)
                if rpm_wait <= 0 and self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self.minute_window.append(now)
                    return
                # Compute the next earliest admit time.
                if self.tokens >= 1.0:
                    rps_wait = 0.0
                else:
                    needed = 1.0 - self.tokens
                    rps_wait = needed / max(self.rps, 0.001)
                wait = max(rps_wait, rpm_wait)
            await asyncio.sleep(min(wait, 1.0))

    def acquire_sync(self) -> None:
        """Sync variant — usable from the prompt-pair generator's
        retry loop (which is sync)."""
        while True:
            with self._sync_lock:
                now = time.monotonic()
                self._refill_rps()
                rpm_wait = self._rpm_wait(now)
                if rpm_wait <= 0 and self.tokens >= 1.0:
                    self.tokens -= 1.0
                    self.minute_window.append(now)
                    return
                if self.tokens >= 1.0:
                    rps_wait = 0.0
                else:
                    needed = 1.0 - self.tokens
                    rps_wait = needed / max(self.rps, 0.001)
                wait = max(rps_wait, rpm_wait)
            time.sleep(min(wait, 1.0))

    # ---- ancillary -----------------------------------------------------------

    def add_cost(self, usd: float) -> None:
        self.usd_spent += float(usd)

    @property
    def semaphore(self) -> asyncio.Semaphore | None:
        """Optional ``asyncio.Semaphore`` reflecting ``max_concurrency``.
        ``None`` when ``max_concurrency`` is 0."""
        return self._semaphore


# ---------------------------------------------------------------------------
# Module-level singleton + init
# ---------------------------------------------------------------------------


_GLOBAL: RateLimiter | None = None


def init_rate_limiter(
    *,
    rps: float,
    rpm: int = 0,
    burst: int | None = None,
    max_concurrency: int = 0,
) -> RateLimiter:
    """Initialize (or replace) the process-global rate limiter from config.

    Must be called at agent startup BEFORE any caller invokes
    :func:`get_rate_limiter`. Replacing an existing global is
    intentional: tests and the V022Runner constructor are the only
    legitimate callers.
    """
    global _GLOBAL
    if burst is None:
        # Allow short bursts up to one full second of rps.
        burst = max(1, int(rps))
    _GLOBAL = RateLimiter(
        rps=float(rps),
        burst=int(burst),
        rpm=int(rpm),
        max_concurrency=int(max_concurrency),
        tokens=float(burst),
    )
    logger.info(
        "rate_limiter initialised: rps=%s burst=%s rpm=%s max_concurrency=%s",
        rps, burst, rpm, max_concurrency,
    )
    return _GLOBAL


def get_rate_limiter(rps: float = 3.0, burst: int = 5) -> RateLimiter:
    """Return the global rate limiter.

    If :func:`init_rate_limiter` has been called, the singleton it
    created is returned and the legacy ``rps`` / ``burst`` kwargs are
    IGNORED. If not, a legacy-compat limiter is created on first
    call with a warning so the missing ``init_rate_limiter`` call is
    visible in logs.
    """
    global _GLOBAL
    if _GLOBAL is None:
        logger.warning(
            "get_rate_limiter() called before init_rate_limiter(); "
            "falling back to legacy defaults rps=%s burst=%s. "
            "Wire init_rate_limiter(cfg) at agent startup to honour "
            "config.yaml's rate_limits.* values.",
            rps, burst,
        )
        _GLOBAL = RateLimiter(rps=rps, burst=burst, tokens=float(burst))
    return _GLOBAL


def reset_rate_limiter() -> None:
    """Test hook."""
    global _GLOBAL
    _GLOBAL = None
