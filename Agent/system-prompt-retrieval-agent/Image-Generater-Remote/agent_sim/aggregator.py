"""
agent_sim/aggregator.py — Multi-GPU completion aggregator.

Coordinates inference across multiple supervisor hosts by sending /infer/list
requests in parallel and aggregating per-host BatchResponse results into a
StageAggregate-compatible dict.

Usage:
    result = await await_all_workers(
        stage_id="gemma",
        cell_id="sp01_neg00",
        expected_hosts=[17610, 17611, 17612],
        expected_shards={17610: [0, 3, 6], 17611: [1, 4], 17612: [2, 5]},
        timeout_s=600,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)


# ── Low-level HTTP helper ──────────────────────────────────────────────────────

async def send_stage_request(host_url: str, payload: dict, timeout_s: float) -> dict:
    """POST payload to ``host_url/infer/list`` and return the response dict.

    Args:
        host_url:  Base URL of the supervisor, e.g. ``http://127.0.0.1:17610``.
                   The path ``/infer/list`` is appended automatically.
        payload:   JSON-serialisable request body.
        timeout_s: Total HTTP timeout in seconds.

    Returns:
        Parsed JSON response as a plain dict.

    Raises:
        httpx.HTTPError: On network or HTTP-level errors.
        httpx.TimeoutException: When the request exceeds ``timeout_s``.
    """
    url = f"{host_url.rstrip('/')}/infer/list"
    timeout = httpx.Timeout(timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        return response.json()


# ── Per-host request builder ───────────────────────────────────────────────────

def _build_host_payload(
    stage_id: str,
    cell_id: str,
    host_port: int,
    shard_indices: list[int],
) -> dict:
    """Build the /infer/list payload for a single host.

    The payload carries the shard_indices so the supervisor knows which items
    to process.  No manifest_hash is included here — callers should extend the
    returned dict with model-specific fields as needed.
    """
    return {
        "stage_id": stage_id,
        "cell_id": cell_id,
        "shard_id": host_port,
        "host_port": host_port,
        "shard_indices": shard_indices,
        "items": [{"index": idx, "shard_id": host_port} for idx in shard_indices],
    }


# ── Per-host result validator ──────────────────────────────────────────────────

def _validate_host_response(host_port: int, resp: dict) -> dict:
    """Validate a single BatchResponse dict returned by a supervisor.

    Raises:
        RuntimeError: If the response signals an error or ok_count mismatches
                      the expected sample_count.

    Returns:
        The validated response dict (unchanged).
    """
    if not isinstance(resp, dict):
        raise RuntimeError(
            f"Host {host_port}: expected a dict response, got {type(resp).__name__}"
        )

    if not resp.get("ok", False):
        error_count = resp.get("error_count", "?")
        raise RuntimeError(
            f"Host {host_port}: response ok=False (error_count={error_count})"
        )

    ok_count = resp.get("ok_count", 0)
    total = resp.get("total", 0)
    if ok_count != total:
        raise RuntimeError(
            f"Host {host_port}: ok_count={ok_count} != total={total} — "
            "some samples failed"
        )

    return resp


# ── Main aggregator ────────────────────────────────────────────────────────────

async def await_all_workers(
    stage_id: str,
    cell_id: str,
    expected_hosts: list[int],
    expected_shards: dict[int, list[int]],
    timeout_s: float = 600,
    base_url_template: str = "http://127.0.0.1:{port}",
    extra_payload: dict[str, Any] | None = None,
) -> dict:
    """Send /infer/list to every expected host and aggregate the results.

    All HTTP requests are issued concurrently via ``asyncio.gather``.  If any
    host fails to respond, returns an error response, or reports ok_count !=
    sample_count the entire call raises immediately — no partial-success
    semantics.

    Args:
        stage_id:          Pipeline stage identifier (e.g. ``"gemma"``).
        cell_id:           Grid cell identifier (e.g. ``"sp01_neg00"``).
        expected_hosts:    Ordered list of supervisor port numbers.
        expected_shards:   Mapping port -> list[int] of shard indices assigned
                           to that host.
        timeout_s:         Per-host HTTP timeout in seconds (default 600).
        base_url_template: URL template with ``{port}`` placeholder.
        extra_payload:     Optional dict merged into every host payload.

    Returns:
        StageAggregate-compatible dict::

            {
                "stage_id": str,
                "cell_id": str,
                "overall_ok": bool,
                "expected_host_count": int,
                "actual_host_count": int,
                "total_samples": int,
                "ok_count": int,
                "error_count": int,
                "hosts": [<per-host BatchResponse dict>, ...],
                "latency_ms": float,
            }

    Raises:
        RuntimeError: On any host error, missing host response, ok_count
                      mismatch, or timeout.
    """
    if not expected_hosts:
        raise ValueError("expected_hosts must not be empty")

    t0 = time.monotonic()

    # Build one coroutine per host
    async def _fetch_host(port: int) -> dict:
        host_url = base_url_template.format(port=port)
        shard_indices = expected_shards.get(port, [])
        payload: dict[str, Any] = _build_host_payload(stage_id, cell_id, port, shard_indices)
        if extra_payload:
            payload.update(extra_payload)

        log.debug(
            "await_all_workers: requesting host_port=%d shard_count=%d",
            port,
            len(shard_indices),
        )
        try:
            resp = await send_stage_request(host_url, payload, timeout_s)
        except (httpx.TimeoutException, asyncio.TimeoutError) as exc:
            raise RuntimeError(
                f"Host {port}: request timed out after {timeout_s}s"
            ) from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Host {port}: HTTP error — {exc}") from exc

        _validate_host_response(port, resp)
        return resp

    # Gather without suppressing exceptions — any error propagates immediately.
    # NOTE: return_exceptions is intentionally left at its default (False).
    coroutines = [_fetch_host(port) for port in expected_hosts]
    host_results: list[dict] = await asyncio.gather(*coroutines)

    latency_ms = (time.monotonic() - t0) * 1000

    # Aggregate counts
    total_samples = sum(r.get("total", 0) for r in host_results)
    ok_count = sum(r.get("ok_count", 0) for r in host_results)
    error_count = sum(r.get("error_count", 0) for r in host_results)
    overall_ok = error_count == 0 and ok_count == total_samples

    return {
        "stage_id": stage_id,
        "cell_id": cell_id,
        "overall_ok": overall_ok,
        "expected_host_count": len(expected_hosts),
        "actual_host_count": len(host_results),
        "total_samples": total_samples,
        "ok_count": ok_count,
        "error_count": error_count,
        "hosts": host_results,
        "latency_ms": round(latency_ms, 1),
    }
