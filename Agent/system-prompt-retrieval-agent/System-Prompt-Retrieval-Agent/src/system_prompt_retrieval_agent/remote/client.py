"""HTTP client for the remote image-generation controller."""
from __future__ import annotations

import asyncio
import random
from typing import Optional, Union

import httpx

from system_prompt_retrieval_agent.schemas import (
    FluxStageRequest,
    GemmaStageRequest,
    QwenStageRequest,
    RemoteStageRequest,
    RemoteStageResponse,
    StageManifest,
)

# V0.2.1 batched request type alias.
_V021StageRequest = Union[GemmaStageRequest, FluxStageRequest, QwenStageRequest]


class StageTimeoutError(RuntimeError):
    """Raised when the stage endpoint times out and retries are exhausted."""


class StageLockBusy(RuntimeError):
    """Raised when the stage endpoint reports a lock-contention error after retries."""


class StageServerError(RuntimeError):
    """Raised when the stage endpoint returns a 5xx error after retries."""


class RemoteControllerClient:
    """Async HTTP client wrapping the remote controller REST API.

    Parameters
    ----------
    base_url:
        Base URL of the remote controller (e.g. ``http://127.0.0.1:17700``).
    request_timeout_s:
        Timeout for a single HTTP request in seconds.
    stage_retry_max:
        How many times to retry on 5xx / timeout errors.
    lock_contention_retry_max:
        How many times to retry on 400 "locked"/"busy" responses.
    lock_contention_wait_s:
        Seconds to wait between lock-contention retries.
    http_client:
        Optional pre-built ``httpx.AsyncClient`` (injected for testing).
    """

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_s: int = 7200,
        stage_retry_max: int = 2,
        lock_contention_retry_max: int = 6,
        lock_contention_wait_s: float = 10.0,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._request_timeout_s = request_timeout_s
        self._stage_retry_max = stage_retry_max
        self._lock_contention_retry_max = lock_contention_retry_max
        self._lock_contention_wait_s = lock_contention_wait_s
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(request_timeout_s),
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        """GET /health → dict."""
        resp = await self._client.get(f"{self._base_url}/health")
        resp.raise_for_status()
        return resp.json()

    async def status(self) -> dict:
        """GET /status → dict."""
        resp = await self._client.get(f"{self._base_url}/status")
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    async def run_prompt_pair_pipeline(
        self, req: RemoteStageRequest
    ) -> RemoteStageResponse:
        """POST /stage/all with *req* and return a parsed RemoteStageResponse."""
        return await self.run_stage("all", req)

    async def run_stage(
        self, stage: str, req: RemoteStageRequest
    ) -> RemoteStageResponse:
        """POST /stage/{stage} with *req*.

        Retry logic
        -----------
        - HTTP 400 containing ``"locked"`` or ``"busy"`` in the response body:
          wait ``lock_contention_wait_s`` and retry up to
          ``lock_contention_retry_max`` times, then raise ``StageLockBusy``.
        - HTTP 5xx or ``httpx.TimeoutException``: retry up to
          ``stage_retry_max`` times, then raise ``StageServerError`` /
          ``StageTimeoutError`` respectively.
        - Any other 4xx: raise immediately (not retryable).
        - HTTP 200: parse and return ``RemoteStageResponse`` regardless of
          the ``ok`` flag (barrier / validator decides).
        """
        url = f"{self._base_url}/stage/{stage}"
        payload = req.model_dump()

        server_attempts = 0
        lock_attempts = 0

        while True:
            try:
                resp = await self._client.post(url, json=payload)
            except httpx.TimeoutException as exc:
                server_attempts += 1
                if server_attempts > self._stage_retry_max:
                    raise StageTimeoutError(
                        f"Stage '{stage}' timed out after {server_attempts} attempt(s)"
                    ) from exc
                await asyncio.sleep(0)
                continue

            if resp.status_code == 200:
                data = resp.json()
                return RemoteStageResponse.model_validate(data)

            if resp.status_code == 400:
                try:
                    body = resp.json()
                    detail = str(body.get("detail", ""))
                except Exception:
                    detail = resp.text or ""

                if "locked" in detail.lower() or "busy" in detail.lower():
                    lock_attempts += 1
                    if lock_attempts > self._lock_contention_retry_max:
                        raise StageLockBusy(
                            f"Stage '{stage}' locked after "
                            f"{lock_attempts} contention retries"
                        )
                    await asyncio.sleep(self._lock_contention_wait_s)
                    continue
                # Other 400 — not retryable.
                resp.raise_for_status()

            if resp.status_code >= 500:
                server_attempts += 1
                if server_attempts > self._stage_retry_max:
                    raise StageServerError(
                        f"Stage '{stage}' returned {resp.status_code} after "
                        f"{server_attempts} attempt(s): {resp.text[:200]}"
                    )
                await asyncio.sleep(0)
                continue

            # Any other 4xx — raise immediately.
            resp.raise_for_status()

    # ------------------------------------------------------------------
    # V0.2.1 stage dispatch
    # ------------------------------------------------------------------

    async def post_stage_v021(
        self,
        req: _V021StageRequest,
        endpoint_url: Optional[str] = None,
    ) -> StageManifest:
        """POST a V0.2.1 batched stage request and parse the ``StageManifest``.

        This is the V0.2.1 primary dispatch path.  The legacy
        :meth:`run_prompt_pair_pipeline` / :meth:`run_stage` methods are
        retained for backward compatibility but are **not** called by V0.2.1
        orchestration code.

        The method applies the same retry logic as :meth:`run_stage` (lock
        contention, 5xx, timeout) but returns a ``StageManifest`` directly
        rather than a ``RemoteStageResponse`` wrapper.

        Parameters
        ----------
        req:
            One of ``GemmaStageRequest``, ``FluxStageRequest``, or
            ``QwenStageRequest``.
        endpoint_url:
            Explicit URL override (e.g. ``http://127.0.0.1:17700/stage/gemma``).
            When ``None``, the stage is derived from the request class name.

        Returns
        -------
        StageManifest
            Parsed V0.2.1 stage manifest.

        Raises
        ------
        StageLockBusy
            Lock retries exhausted.
        StageTimeoutError
            Request timeout retries exhausted.
        StageServerError
            5xx retries exhausted.
        ValueError
            If any FLUX / Qwen pair carries ``system_prompt_text``.
        """
        from system_prompt_retrieval_agent.remote.stage_dispatcher import (
            CorpusDriftDetected,
            StageDispatcher,
        )

        # Validate system_prompt_text constraint before sending.
        if isinstance(req, (FluxStageRequest, QwenStageRequest)):
            for p in req.prompt_pairs:
                if p.system_prompt_text is not None:
                    raise ValueError(
                        f"system_prompt_text is forbidden on "
                        f"{type(req).__name__} pair '{p.prompt_pair_id}'"
                    )

        # Derive stage name and URL.
        if endpoint_url is not None:
            url = endpoint_url
            # Extract stage from the URL for retry key.
            stage = url.rstrip("/").rsplit("/", 1)[-1]
        elif isinstance(req, GemmaStageRequest):
            stage = "gemma"
            url = f"{self._base_url}/stage/gemma"
        elif isinstance(req, FluxStageRequest):
            stage = "flux"
            url = f"{self._base_url}/stage/flux"
        elif isinstance(req, QwenStageRequest):
            stage = "qwen"
            url = f"{self._base_url}/stage/qwen"
        else:
            raise TypeError(f"Unsupported request type: {type(req)}")

        payload = req.model_dump(mode="json")

        server_attempts = 0
        lock_attempts = 0

        while True:
            try:
                resp = await self._client.post(url, json=payload)
            except httpx.TimeoutException as exc:
                server_attempts += 1
                if server_attempts > self._stage_retry_max:
                    raise StageTimeoutError(
                        f"Stage '{stage}' timed out after {server_attempts} attempt(s)"
                    ) from exc
                await asyncio.sleep(0)
                continue

            if resp.status_code == 200:
                body = resp.json()
                # Response body may be:
                #   (a) {"ok": ..., "manifest": {...}} (RemoteStageResponse-style)
                #   (b) A StageManifest dict directly.
                if "manifest" in body:
                    manifest_data = body["manifest"]
                else:
                    manifest_data = body

                manifest = StageManifest.model_validate(manifest_data)

                # Corpus-hash drift detection.
                received_hash = manifest.user_prompt_corpus_hash
                if received_hash is not None and received_hash != req.user_prompt_corpus_hash:
                    raise CorpusDriftDetected(
                        expected=req.user_prompt_corpus_hash,
                        received=received_hash,
                    )

                return manifest

            if resp.status_code == 400:
                try:
                    body = resp.json()
                    detail = str(body.get("detail", ""))
                except Exception:
                    detail = resp.text or ""

                if "locked" in detail.lower() or "busy" in detail.lower():
                    lock_attempts += 1
                    if lock_attempts > self._lock_contention_retry_max:
                        raise StageLockBusy(
                            f"Stage '{stage}' locked after "
                            f"{lock_attempts} contention retries"
                        )
                    await asyncio.sleep(self._lock_contention_wait_s)
                    continue
                # Other 400 — not retryable.
                resp.raise_for_status()

            if resp.status_code >= 500:
                server_attempts += 1
                if server_attempts > self._stage_retry_max:
                    raise StageServerError(
                        f"Stage '{stage}' returned {resp.status_code} after "
                        f"{server_attempts} attempt(s): {resp.text[:200]}"
                    )
                await asyncio.sleep(0)
                continue

            # Any other 4xx — raise immediately.
            resp.raise_for_status()

    # ------------------------------------------------------------------
    # V0.2.2 stage dispatch (S05B.02)
    # ------------------------------------------------------------------

    async def post_v022_stage(self, stage: str, payload: dict) -> dict:
        """POST a V0.2.2 stage request to ``/v022/stage/{stage}``.

        The V0.2.2 production transport. Reuses the same retry policy as
        :meth:`run_stage` (lock contention, 5xx, timeout) and returns the
        parsed JSON dict (the V0.2.2 stage manifest body) without
        wrapping it in ``RemoteStageResponse`` — V0.2.2 manifests are
        the response body directly per ``/v022/stage`` controller.

        ``stage`` must be one of ``"gemma"``, ``"flux"``, ``"qwen"``.
        Any other value posts to that literal path and is rejected by
        the controller; we don't pre-validate so smoke / integration
        tests can exercise the controller's own error paths.
        """
        url = f"{self._base_url}/v022/stage/{stage}"
        server_attempts = 0
        lock_attempts = 0

        while True:
            try:
                resp = await self._client.post(url, json=payload)
            except httpx.TimeoutException as exc:
                server_attempts += 1
                if server_attempts > self._stage_retry_max:
                    raise StageTimeoutError(
                        f"Stage '{stage}' timed out after {server_attempts} attempt(s)"
                    ) from exc
                await asyncio.sleep(0)
                continue

            if resp.status_code == 200:
                body = resp.json()
                if isinstance(body, dict) and "manifest" in body and isinstance(body["manifest"], dict):
                    return body["manifest"]
                return body

            if resp.status_code == 400:
                try:
                    body = resp.json()
                    detail = str(body.get("detail", ""))
                except Exception:
                    detail = resp.text or ""
                if "locked" in detail.lower() or "busy" in detail.lower():
                    lock_attempts += 1
                    if lock_attempts > self._lock_contention_retry_max:
                        raise StageLockBusy(
                            f"Stage '{stage}' locked after "
                            f"{lock_attempts} contention retries"
                        )
                    await asyncio.sleep(self._lock_contention_wait_s)
                    continue
                resp.raise_for_status()

            if resp.status_code >= 500:
                server_attempts += 1
                if server_attempts > self._stage_retry_max:
                    raise StageServerError(
                        f"Stage '{stage}' returned {resp.status_code} after "
                        f"{server_attempts} attempt(s): {resp.text[:200]}"
                    )
                await asyncio.sleep(0)
                continue

            resp.raise_for_status()

    # ------------------------------------------------------------------
    # Async context manager support
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "RemoteControllerClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._owns_client:
            await self._client.aclose()
