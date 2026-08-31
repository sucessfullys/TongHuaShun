"""
tests/test_aggregator.py — Unit tests for agent_sim.aggregator.

Covers:
  - send_stage_request: successful POST, HTTP error, url normalisation.
  - await_all_workers: all-ok case, one-host-missing case, ok_count mismatch.
  - Structural assertion: asyncio.gather is NOT called with return_exceptions=True.
"""

from __future__ import annotations

import asyncio
import inspect
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from agent_sim.aggregator import (
    await_all_workers,
    send_stage_request,
    _validate_host_response,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_batch_response(
    port: int = 17610,
    ok: bool = True,
    ok_count: int = 3,
    error_count: int = 0,
    total: int = 3,
) -> dict:
    return {
        "stage_id": "gemma",
        "cell_id": "sp01_neg00",
        "host_port": port,
        "ok": ok,
        "ok_count": ok_count,
        "error_count": error_count,
        "total": total,
        "items": [],
        "latency_ms": 42.0,
    }


# ── send_stage_request ─────────────────────────────────────────────────────────

class TestSendStageRequest(unittest.IsolatedAsyncioTestCase):
    """Tests for the low-level HTTP helper."""

    async def test_success_posts_to_infer_list_and_returns_json(self):
        """A 200 response body is returned as a dict."""
        expected = {"ok": True, "ok_count": 2, "total": 2}

        mock_response = MagicMock()
        mock_response.json.return_value = expected
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("agent_sim.aggregator.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await send_stage_request(
                "http://127.0.0.1:17610", {"stage_id": "gemma"}, timeout_s=10
            )

        assert result == expected
        mock_client.post.assert_called_once()
        call_url = mock_client.post.call_args[0][0]
        assert call_url.endswith("/infer/list"), (
            f"Expected URL to end with /infer/list, got: {call_url}"
        )

    async def test_http_error_propagates(self):
        """An HTTP 500 triggers raise_for_status and propagates the error."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(),
        )

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("agent_sim.aggregator.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            with pytest.raises(httpx.HTTPStatusError):
                await send_stage_request(
                    "http://127.0.0.1:17610", {}, timeout_s=5
                )

    async def test_trailing_slash_stripped_from_base_url(self):
        """Trailing slashes in host_url are stripped before appending /infer/list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("agent_sim.aggregator.httpx.AsyncClient") as MockClient:
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await send_stage_request("http://127.0.0.1:17610/", {}, timeout_s=5)

        call_url = mock_client.post.call_args[0][0]
        scheme_stripped = call_url.split("://", 1)[1]
        assert "//" not in scheme_stripped, (
            f"Double-slash in path portion: {call_url}"
        )
        assert call_url.endswith("/infer/list")


# ── _validate_host_response ────────────────────────────────────────────────────

class TestValidateHostResponse(unittest.TestCase):
    def test_valid_response_passes_through(self):
        resp = _make_batch_response(ok=True, ok_count=3, total=3)
        assert _validate_host_response(17610, resp) is resp

    def test_ok_false_raises(self):
        resp = _make_batch_response(ok=False, ok_count=2, error_count=1, total=3)
        with pytest.raises(RuntimeError, match="ok=False"):
            _validate_host_response(17610, resp)

    def test_ok_count_mismatch_raises(self):
        # ok=True but ok_count < total — some items silently failed
        resp = _make_batch_response(ok=True, ok_count=2, error_count=0, total=3)
        with pytest.raises(RuntimeError, match="ok_count=2 != total=3"):
            _validate_host_response(17610, resp)

    def test_non_dict_raises(self):
        with pytest.raises(RuntimeError, match="expected a dict"):
            _validate_host_response(17610, "not a dict")


# ── await_all_workers — success case ──────────────────────────────────────────

class TestAwaitAllWorkersSuccess(unittest.IsolatedAsyncioTestCase):
    """All hosts respond with ok BatchResponses."""

    def _make_responses(self) -> dict[int, dict]:
        return {
            17610: _make_batch_response(port=17610, ok=True, ok_count=3, total=3),
            17611: _make_batch_response(port=17611, ok=True, ok_count=2, total=2),
            17612: _make_batch_response(port=17612, ok=True, ok_count=2, total=2),
        }

    async def test_returns_stage_aggregate_dict(self):
        responses = self._make_responses()

        async def _fake_send(host_url, payload, timeout_s):
            port = int(host_url.rstrip("/").rsplit(":", 1)[-1])
            return responses[port]

        with patch("agent_sim.aggregator.send_stage_request", side_effect=_fake_send):
            result = await await_all_workers(
                stage_id="gemma",
                cell_id="sp01_neg00",
                expected_hosts=[17610, 17611, 17612],
                expected_shards={17610: [0, 3, 6], 17611: [1, 4], 17612: [2, 5]},
                timeout_s=10,
            )

        assert result["stage_id"] == "gemma"
        assert result["cell_id"] == "sp01_neg00"
        assert result["overall_ok"] is True
        assert result["expected_host_count"] == 3
        assert result["actual_host_count"] == 3
        assert result["total_samples"] == 7
        assert result["ok_count"] == 7
        assert result["error_count"] == 0
        assert len(result["hosts"]) == 3
        assert "latency_ms" in result

    async def test_hosts_list_preserves_per_host_data(self):
        responses = self._make_responses()

        async def _fake_send(host_url, payload, timeout_s):
            port = int(host_url.rstrip("/").rsplit(":", 1)[-1])
            return responses[port]

        with patch("agent_sim.aggregator.send_stage_request", side_effect=_fake_send):
            result = await await_all_workers(
                stage_id="gemma",
                cell_id="sp01_neg00",
                expected_hosts=[17610, 17611, 17612],
                expected_shards={17610: [0, 3, 6], 17611: [1, 4], 17612: [2, 5]},
                timeout_s=10,
            )

        host_ports = {h["host_port"] for h in result["hosts"]}
        assert host_ports == {17610, 17611, 17612}


# ── await_all_workers — failure cases ─────────────────────────────────────────

class TestAwaitAllWorkersFailure(unittest.IsolatedAsyncioTestCase):
    """Error propagation tests."""

    async def test_missing_host_raises(self):
        """One host raises an HTTP error; the whole call must raise."""

        async def _fake_send(host_url, payload, timeout_s):
            port = int(host_url.rstrip("/").rsplit(":", 1)[-1])
            if port == 17611:
                raise httpx.ConnectError("Connection refused", request=MagicMock())
            return _make_batch_response(port=port, ok=True, ok_count=3, total=3)

        with pytest.raises(RuntimeError, match="Host 17611"):
            with patch("agent_sim.aggregator.send_stage_request", side_effect=_fake_send):
                await await_all_workers(
                    stage_id="gemma",
                    cell_id="sp01_neg00",
                    expected_hosts=[17610, 17611, 17612],
                    expected_shards={
                        17610: [0, 3, 6],
                        17611: [1, 4],
                        17612: [2, 5],
                    },
                    timeout_s=10,
                )

    async def test_host_returns_ok_false_raises(self):
        """A host returning ok=False causes await_all_workers to raise."""

        async def _fake_send(host_url, payload, timeout_s):
            port = int(host_url.rstrip("/").rsplit(":", 1)[-1])
            if port == 17612:
                return _make_batch_response(
                    port=port, ok=False, ok_count=1, error_count=1, total=2
                )
            return _make_batch_response(port=port, ok=True, ok_count=3, total=3)

        with pytest.raises(RuntimeError, match="ok=False"):
            with patch("agent_sim.aggregator.send_stage_request", side_effect=_fake_send):
                await await_all_workers(
                    stage_id="gemma",
                    cell_id="sp01_neg00",
                    expected_hosts=[17610, 17611, 17612],
                    expected_shards={
                        17610: [0, 3, 6],
                        17611: [1, 4],
                        17612: [2, 5],
                    },
                    timeout_s=10,
                )

    async def test_ok_count_mismatch_raises(self):
        """ok_count != total in any host raises RuntimeError."""

        async def _fake_send(host_url, payload, timeout_s):
            port = int(host_url.rstrip("/").rsplit(":", 1)[-1])
            if port == 17610:
                # ok=True but only 2 of 3 samples reported success
                return _make_batch_response(
                    port=port, ok=True, ok_count=2, error_count=0, total=3
                )
            return _make_batch_response(port=port, ok=True, ok_count=2, total=2)

        with pytest.raises(RuntimeError, match="ok_count=2 != total=3"):
            with patch("agent_sim.aggregator.send_stage_request", side_effect=_fake_send):
                await await_all_workers(
                    stage_id="gemma",
                    cell_id="sp01_neg00",
                    expected_hosts=[17610, 17611, 17612],
                    expected_shards={
                        17610: [0, 3, 6],
                        17611: [1, 4],
                        17612: [2, 5],
                    },
                    timeout_s=10,
                )

    async def test_empty_expected_hosts_raises_value_error(self):
        with pytest.raises(ValueError, match="expected_hosts must not be empty"):
            await await_all_workers(
                stage_id="gemma",
                cell_id="c0",
                expected_hosts=[],
                expected_shards={},
            )


# ── asyncio.gather return_exceptions assertion ─────────────────────────────────

class TestGatherNotCalledWithReturnExceptions(unittest.TestCase):
    """Static check: asyncio.gather must NOT be invoked with return_exceptions=True."""

    def test_source_does_not_contain_return_exceptions_true(self):
        """Inspect aggregator source to confirm the constraint is met."""
        import agent_sim.aggregator as mod

        source = inspect.getsource(mod)
        assert "return_exceptions=True" not in source, (
            "asyncio.gather must NOT be called with return_exceptions=True in aggregator.py"
        )


class TestGatherRaisesOnException(unittest.IsolatedAsyncioTestCase):
    """Behavioural check: exceptions propagate from await_all_workers."""

    async def test_exception_from_one_host_propagates(self):
        """Confirms gather raises rather than returning the exception as a value."""

        async def _fail(host_url, payload, timeout_s):
            port = int(host_url.rstrip("/").rsplit(":", 1)[-1])
            if port == 17611:
                raise RuntimeError("simulated_host_failure")
            await asyncio.sleep(0)  # yield so both coroutines start
            return _make_batch_response(port=port, ok=True, ok_count=2, total=2)

        with pytest.raises(RuntimeError):
            with patch("agent_sim.aggregator.send_stage_request", side_effect=_fail):
                await await_all_workers(
                    stage_id="s",
                    cell_id="c",
                    expected_hosts=[17610, 17611],
                    expected_shards={17610: [0, 2], 17611: [1, 3]},
                    timeout_s=5,
                )


# Run directly
if __name__ == "__main__":
    unittest.main()
