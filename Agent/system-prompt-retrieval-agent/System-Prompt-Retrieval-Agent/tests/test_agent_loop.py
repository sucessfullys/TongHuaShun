"""End-to-end mocked agent loop test."""
import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from conftest import write_mock_config
from system_prompt_retrieval_agent.agent_loop import AgentLoop
from system_prompt_retrieval_agent.config import load_config
from system_prompt_retrieval_agent.schemas import PromptPair


class _StubChoiceMessage:
    def __init__(self, content: str):
        self.content = content


class _StubChoice:
    def __init__(self, content: str):
        self.message = _StubChoiceMessage(content)


class _StubCompletion:
    def __init__(self, content: str):
        self.choices = [_StubChoice(content)]


class StubChatCompletions:
    def __init__(self, content: str):
        self._content = content
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return _StubCompletion(self._content)


class StubChat:
    def __init__(self, content: str):
        self.completions = StubChatCompletions(content)


class StubOpenAI:
    def __init__(self, content: str = None):
        if content is None:
            content = json.dumps({
                "prompt_pairs": [
                    {
                        "system_prompt": "You are a careful assistant that writes concise clothing-transfer prompts with identifiable garment types.",
                        "negative_prompt": "distorted body",
                        "rationale": "Focus on brevity",
                        "expected_improvement_target": "prompt conciseness",
                        "risk": "possibly too terse",
                    }
                ] * 3
            })
        self.chat = StubChat(content)


async def _stub_stage_all(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "ok": True,
            "stage": "all",
            "run_id": "runX",
            "message": "ok",
            "manifest": {
                "stage": "qwen",
                "ok": 2,
                "errors": 0,
                "total": 2,
                "entries": [],
                "workers": [{"gpu": 0, "status": "ok"}, {"gpu": 1, "status": "ok"}, {"gpu": 2, "status": "ok"}],
                "vram_free_gib": [72.0, 73.0, 74.0],
            },
        },
    )


@pytest.mark.asyncio
async def test_agent_loop_one_round_mocked(tmp_path: Path, monkeypatch):
    cfg_path = write_mock_config(tmp_path)
    cfg = load_config(cfg_path)
    # Force non-empty env to exercise redaction
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-xyz")

    # Disable local API eval and rsync to keep test offline
    cfg.evaluation.run_local_api_eval = False

    # Patch copy-back to a no-op
    from system_prompt_retrieval_agent import agent_loop as loop_mod

    def _fake_rsync(remote_path, local_path, *, ssh_alias="3h100", required_file=None, max_retries=1, subprocess_run=None):
        Path(local_path).mkdir(parents=True, exist_ok=True)
        # create a minimal stage_summary.json
        stage3 = Path(local_path) / "stage3_qwen"
        stage3.mkdir(exist_ok=True)
        (stage3 / "stage_summary.json").write_text(json.dumps({
            "yes": 2, "no": 0, "total_evaluated": 2,
            "samples": [
                {"sample_id": "upper/a", "status": "yes", "raw_response": "yes", "category": "upper"},
                {"sample_id": "upper/b", "status": "yes", "raw_response": "yes", "category": "upper"},
            ],
        }))

    monkeypatch.setattr(loop_mod, "rsync_copyback", _fake_rsync)

    # Patch generate_prompt_pairs to return fixed pairs quickly (bypass OpenAI stub plumbing depth)
    async def _fake_gen(cfg, mgr, round_id, N, existing_pairs, openai_client=None):
        pairs = []
        for i in range(N):
            pairs.append(PromptPair(
                prompt_pair_id=f"pair_r{round_id:02d}_{i:03d}",
                system_prompt_id=f"sys_r{round_id:02d}_{i:03d}",
                negative_prompt_id=f"neg_r{round_id:02d}_{i:03d}",
                round_id=round_id,
                selection_role="seed",
                system_prompt="a careful assistant, describe clothing replacement crisply",
                negative_prompt="distorted body, duplicated clothing",
            ))
        return pairs, False

    monkeypatch.setattr(loop_mod, "generate_prompt_pairs", _fake_gen)

    transport = httpx.MockTransport(_stub_stage_all)
    http_client = httpx.AsyncClient(transport=transport, base_url=cfg.remote.controller_base_url)

    loop = AgentLoop(cfg, limit=2, max_rounds=1, http_client=http_client, openai_client=StubOpenAI())
    rc = await loop.run()
    assert rc == 0

    best_yaml = Path(cfg.paths.output_root) / "best_pair.yaml"
    assert best_yaml.is_file()
    import yaml as _y

    content = _y.safe_load(best_yaml.read_text())
    assert content["run_id"] == loop.run_id
    # One round ran; best_pair either set (pairs scored) or None (all failed); either way no API key in file
    raw = best_yaml.read_text()
    assert "test-key-xyz" not in raw
