"""Outbound HTTP POST for job-status callbacks with retry / backoff.

Uses ``httpx.AsyncClient`` — do NOT swap to ``requests`` (sync) or ``aiohttp``.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx


# ── Core send function ────────────────────────────────────────────────────────

async def send_callback(
    url: str,
    payload: dict,
    timeout_s: int = 10,
    retries: int = 3,
    backoff_s: Optional[list[float]] = None,
) -> dict:
    """POST *payload* as JSON to *url* with retry / exponential backoff.

    Args:
        url:        Destination URL.
        payload:    Serialisable dict (e.g. a ``CallbackPayload`` model dump).
        timeout_s:  Per-request timeout in seconds.
        retries:    Maximum number of attempts (including the first).
        backoff_s:  List of wait durations (seconds) before each retry.
                    Defaults to [0.5, 2.0, 8.0]. If the list is shorter than
                    ``retries - 1``, the last element is reused.

    Returns a dict with keys:
        ok (bool):              True if any attempt returned 2xx.
        status_code (int|None): HTTP status of the last attempt, or None if
                                every attempt raised a network error.
        attempts (int):         Number of attempts made.
        error (str|None):       Error message from the last attempt, or None
                                on success.
    """
    if backoff_s is None:
        backoff_s = [0.5, 2.0, 8.0]

    last_status: Optional[int] = None
    last_error: Optional[str] = None

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        for attempt in range(1, retries + 1):
            print(
                f"[callback] attempt {attempt}/{retries} POST {url} "
                f"payload_keys={list(payload.keys())}"
            )
            try:
                response = await client.post(url, json=payload)
                last_status = response.status_code
                if 200 <= response.status_code < 300:
                    print(
                        f"[callback] attempt {attempt} succeeded "
                        f"status={response.status_code}"
                    )
                    return {
                        "ok": True,
                        "status_code": last_status,
                        "attempts": attempt,
                        "error": None,
                    }
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                print(f"[callback] attempt {attempt} non-2xx: {last_error}")
            except httpx.TimeoutException as exc:
                last_error = f"Timeout: {exc}"
                print(f"[callback] attempt {attempt} timeout: {exc}")
            except httpx.RequestError as exc:
                last_error = f"RequestError: {exc}"
                print(f"[callback] attempt {attempt} request error: {exc}")

            if attempt < retries:
                # Determine backoff duration for this retry gap
                idx = min(attempt - 1, len(backoff_s) - 1)
                wait = backoff_s[idx]
                print(f"[callback] backing off {wait}s before attempt {attempt + 1}")
                await asyncio.sleep(wait)

    return {
        "ok": False,
        "status_code": last_status,
        "attempts": retries,
        "error": last_error,
    }


# ── Wrapper with required / optional semantics ────────────────────────────────

async def send_callback_if_required(
    url: str,
    payload: dict,
    required: bool = True,
    **kwargs,
) -> dict:
    """Send a callback and handle failure according to *required*.

    Args:
        url:      Destination URL.
        payload:  Serialisable dict.
        required: When True, raises ``RuntimeError`` if all retries are
                  exhausted. When False, logs the failure and returns normally.
        **kwargs: Forwarded verbatim to :func:`send_callback`
                  (``timeout_s``, ``retries``, ``backoff_s``).

    Returns the ``send_callback`` result dict regardless of *required*.
    """
    result = await send_callback(url, payload, **kwargs)

    if not result["ok"]:
        msg = (
            f"Callback to {url} failed after {result['attempts']} attempt(s): "
            f"{result['error']}"
        )
        if required:
            raise RuntimeError(msg)
        print(f"[callback] WARNING (non-required): {msg}")

    return result
