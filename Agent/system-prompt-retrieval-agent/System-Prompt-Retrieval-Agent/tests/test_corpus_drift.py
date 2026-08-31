"""S09 / S12 Probe G — corpus-hash drift detection.

The local agent pins user_prompt_corpus_hash at startup (S00.10) and
echoes it in every /stage/* request. The controller MUST echo the
same hash in every response; a mismatch raises CorpusDriftDetected
and aborts the round.
"""
from __future__ import annotations

import json
import pytest
import respx
import httpx
import asyncio

from system_prompt_retrieval_agent.remote.client import RemoteControllerClient
from system_prompt_retrieval_agent.remote.stage_dispatcher import CorpusDriftDetected
from system_prompt_retrieval_agent.schemas import (
    GemmaStageRequest,
    GemmaUserPrompt,
    PromptPairRequest,
)


def _make_req(corpus_hash: str = "a" * 64) -> GemmaStageRequest:
    return GemmaStageRequest(
        run_id="r1",
        round_id=1,
        prompt_pairs=[PromptPairRequest(prompt_pair_id="A", system_prompt_id="s1")],
        user_prompts=[GemmaUserPrompt(user_prompt_id="zh_001", language="zh", text="x")],
        user_prompt_corpus_hash=corpus_hash,
        dataset_root="/d",
        output_root="/o",
    )


@respx.mock
def test_corpus_hash_mismatch_raises_corpus_drift():
    """Controller echoes a different hash → CorpusDriftDetected."""
    pinned = "a" * 64
    drifted = "b" * 64
    req = _make_req(corpus_hash=pinned)

    respx.post("http://127.0.0.1:17700/stage/gemma").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "stage": "gemma",
                "message": "drift",
                "run_id": "r1",
                "manifest": {
                    "stage": "gemma",
                    "user_prompt_corpus_hash": drifted,  # MISMATCH
                    "pairs": {},
                    "surviving_pairs": [],
                    "failed_pairs": [],
                },
            },
        )
    )

    client = RemoteControllerClient(
        base_url="http://127.0.0.1:17700",
        request_timeout_s=5,
        http_client=httpx.AsyncClient(
            base_url="http://127.0.0.1:17700",
            timeout=httpx.Timeout(5),
            transport=httpx.AsyncHTTPTransport(),
        ),
    )

    async def _go():
        await client.post_stage_v021(req, endpoint_url="/stage/gemma")

    with pytest.raises(CorpusDriftDetected):
        asyncio.run(_go())


@respx.mock
def test_corpus_hash_match_no_raise():
    """Controller echoes the pinned hash → success."""
    pinned = "a" * 64
    req = _make_req(corpus_hash=pinned)

    respx.post("http://127.0.0.1:17700/stage/gemma").mock(
        return_value=httpx.Response(
            200,
            json={
                "ok": True,
                "stage": "gemma",
                "message": "ok",
                "run_id": "r1",
                "manifest": {
                    "stage": "gemma",
                    "user_prompt_corpus_hash": pinned,
                    "pairs": {},
                    "surviving_pairs": [],
                    "failed_pairs": [],
                },
            },
        )
    )

    client = RemoteControllerClient(
        base_url="http://127.0.0.1:17700",
        request_timeout_s=5,
        http_client=httpx.AsyncClient(
            base_url="http://127.0.0.1:17700",
            timeout=httpx.Timeout(5),
            transport=httpx.AsyncHTTPTransport(),
        ),
    )

    async def _go():
        return await client.post_stage_v021(req, endpoint_url="/stage/gemma")

    resp = asyncio.run(_go())
    assert resp is not None


def test_corpus_hash_pinned_format():
    """Pinned corpus hash must be sha256 = 64 hex chars."""
    valid = "a" * 64
    req = _make_req(corpus_hash=valid)
    assert len(req.user_prompt_corpus_hash) == 64
    assert all(c in "0123456789abcdef" for c in req.user_prompt_corpus_hash)
