"""Stage-major dispatcher for V0.2.1.

Three async dispatch methods, one per stage:
  - ``dispatch_gemma(req)``  → POST /stage/gemma → ``StageManifest``
  - ``dispatch_flux(req)``   → POST /stage/flux  → ``StageManifest``
  - ``dispatch_qwen(req)``   → POST /stage/qwen  → ``StageManifest``

Each method:
1. Validates that FLUX / Qwen requests carry no ``system_prompt_text`` on any
   pair (Gemma-only field).  Raises ``ValueError`` if violated.
2. POSTs the batched request to the per-stage endpoint.
3. Parses the JSON response body into a ``StageManifest``.
4. Checks that the response ``user_prompt_corpus_hash`` matches the request
   hash; raises ``CorpusDriftDetected`` on mismatch.

Plan reference: §6.2, §6.3, §1.2, S04.04.
"""
from __future__ import annotations

import httpx

from system_prompt_retrieval_agent.schemas import (
    FluxStageRequest,
    GemmaStageRequest,
    QwenStageRequest,
    StageManifest,
)


class CorpusDriftDetected(RuntimeError):
    """Raised when the remote controller returns a user_prompt_corpus_hash
    that differs from the hash on the request.

    Attributes
    ----------
    expected:
        Hash sent in the request.
    received:
        Hash echoed by the controller.
    """

    def __init__(self, expected: str, received: str) -> None:
        super().__init__(
            f"corpus hash drift: expected {expected!r} but remote returned {received!r}"
        )
        self.expected = expected
        self.received = received


class StageDispatcher:
    """Posts per-stage batched requests and returns parsed ``StageManifest``.

    Parameters
    ----------
    http_client:
        A configured ``httpx.AsyncClient``.  The caller owns the client
        lifecycle; ``StageDispatcher`` never closes it.
    base_url:
        Base URL of the remote controller (e.g. ``http://127.0.0.1:17700``).
        Used only when *http_client* is constructed externally with a
        different base and an absolute URL is needed.  When ``http_client``
        already has a ``base_url``, leave this ``None``.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str = "",
    ) -> None:
        self._client = http_client
        self._base = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self, stage: str) -> str:
        """Return the absolute stage URL."""
        return f"{self._base}/stage/{stage}"

    async def _post_stage(
        self,
        stage: str,
        payload: dict,
        expected_hash: str,
    ) -> StageManifest:
        """POST the payload and parse the response body as a ``StageManifest``."""
        url = self._url(stage)
        resp = await self._client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json()

        # Response body may be either:
        #   (a) {"ok": ..., "manifest": {...}} (RemoteStageResponse-style)
        #   (b) A StageManifest dict directly.
        if "manifest" in body:
            manifest_data = body["manifest"]
        else:
            manifest_data = body

        manifest = StageManifest.model_validate(manifest_data)

        # Corpus-hash drift detection.
        received_hash = manifest.user_prompt_corpus_hash
        if received_hash is not None and received_hash != expected_hash:
            raise CorpusDriftDetected(expected=expected_hash, received=received_hash)

        return manifest

    # ------------------------------------------------------------------
    # Public dispatch methods
    # ------------------------------------------------------------------

    async def dispatch_gemma(self, req: GemmaStageRequest) -> StageManifest:
        """POST ``/stage/gemma`` with *req* and return the ``StageManifest``.

        Gemma is the only stage that allows ``system_prompt_text`` on pairs.
        No additional validation beyond the pydantic model is applied here.

        Parameters
        ----------
        req:
            Fully-populated ``GemmaStageRequest``.

        Returns
        -------
        StageManifest
            Parsed stage manifest from the controller response.

        Raises
        ------
        CorpusDriftDetected
            When the response corpus hash differs from the request hash.
        httpx.HTTPStatusError
            On any non-2xx response.
        """
        payload = req.model_dump(mode="json")
        return await self._post_stage("gemma", payload, req.user_prompt_corpus_hash)

    async def dispatch_flux(self, req: FluxStageRequest) -> StageManifest:
        """POST ``/stage/flux`` with *req* and return the ``StageManifest``.

        Raises ``ValueError`` if any pair in *req* carries a non-null
        ``system_prompt_text`` (enforced by the pydantic model on construction,
        but re-validated defensively here).

        Parameters
        ----------
        req:
            Fully-populated ``FluxStageRequest``.

        Returns
        -------
        StageManifest
            Parsed stage manifest from the controller response.

        Raises
        ------
        ValueError
            If any pair carries a non-null ``system_prompt_text``.
        CorpusDriftDetected
            When the response corpus hash differs from the request hash.
        httpx.HTTPStatusError
            On any non-2xx response.
        """
        for p in req.prompt_pairs:
            if p.system_prompt_text is not None:
                raise ValueError(
                    f"system_prompt_text is forbidden on FluxStageRequest "
                    f"pair '{p.prompt_pair_id}' — Gemma-only field"
                )
        payload = req.model_dump(mode="json")
        return await self._post_stage("flux", payload, req.user_prompt_corpus_hash)

    async def dispatch_qwen(self, req: QwenStageRequest) -> StageManifest:
        """POST ``/stage/qwen`` with *req* and return the ``StageManifest``.

        Raises ``ValueError`` if any pair in *req* carries a non-null
        ``system_prompt_text`` (enforced by the pydantic model on construction,
        but re-validated defensively here).

        Parameters
        ----------
        req:
            Fully-populated ``QwenStageRequest``.

        Returns
        -------
        StageManifest
            Parsed stage manifest from the controller response.

        Raises
        ------
        ValueError
            If any pair carries a non-null ``system_prompt_text``.
        CorpusDriftDetected
            When the response corpus hash differs from the request hash.
        httpx.HTTPStatusError
            On any non-2xx response.
        """
        for p in req.prompt_pairs:
            if p.system_prompt_text is not None:
                raise ValueError(
                    f"system_prompt_text is forbidden on QwenStageRequest "
                    f"pair '{p.prompt_pair_id}' — Gemma-only field"
                )
        payload = req.model_dump(mode="json")
        return await self._post_stage("qwen", payload, req.user_prompt_corpus_hash)
