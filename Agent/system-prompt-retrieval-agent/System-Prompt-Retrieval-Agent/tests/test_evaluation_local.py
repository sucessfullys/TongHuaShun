"""Tests for LocalApiEvaluator."""
from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from system_prompt_retrieval_agent.evaluation.local_api_eval import LocalApiEvaluator, _USD_PER_CALL
from system_prompt_retrieval_agent.evaluation.budget_guard import BudgetGuard, CostExhausted
from system_prompt_retrieval_agent.rate_limiter import reset_rate_limiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canned_response(overrides: dict | None = None) -> MagicMock:
    """Return a fake OpenAI chat completion response."""
    payload = {
        "edit_correctness": 0.80,
        "garment_transfer_correctness": 0.75,
        "preservation": 0.90,
        "artifact_penalty": 0.10,
        "notes": "Test evaluation notes.",
    }
    if overrides:
        payload.update(overrides)
    msg = MagicMock()
    msg.content = json.dumps(payload)
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_client(responses: list) -> MagicMock:
    """Create a fake async OpenAI client that returns responses in order."""
    client = MagicMock()
    # Each call to .chat.completions.create() returns next response
    responses_iter = iter(responses)

    async def _create(**kwargs):
        return next(responses_iter)

    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


def _make_fake_image(tmp_path: Path, name: str = "img.png") -> Path:
    """Create a tiny valid PNG in tmp_path."""
    from PIL import Image

    p = tmp_path / name
    img = Image.new("RGB", (4, 4), color=(255, 0, 0))
    img.save(p, format="PNG")
    return p


def _make_samples(tmp_path: Path, n: int = 3) -> list[dict]:
    model_img = _make_fake_image(tmp_path, "model.png")
    cloth_img = _make_fake_image(tmp_path, "cloth.png")
    return [
        {
            "sample_id": f"dress/sample_{i:03d}",
            "model_image_path": str(model_img),
            "cloth_image_path": str(cloth_img),
            "generated_image_path": str(_make_fake_image(tmp_path, f"gen_{i}.png")),
            "intermediate_prompt": f"Prompt for sample {i}",
        }
        for i in range(n)
    ]


def _make_cfg(tmp_path: Path, run_local_api_eval: bool = True, max_concurrent: int = 2):
    from conftest import write_mock_config
    from system_prompt_retrieval_agent.config import load_config

    cfg_path = write_mock_config(
        tmp_path,
        evaluation={
            "categories": ["dress", "lower", "upper"],
            "run_local_api_eval": run_local_api_eval,
            "max_concurrent": max_concurrent,
            "low_score_examples_per_pair": 10,
            "weights": {
                "qwen_pass_rate": 0.40,
                "edit_correctness": 0.20,
                "garment_transfer_correctness": 0.15,
                "preservation": 0.15,
                "artifact_penalty": -0.10,
                "category_balance_bonus": 0.10,
            },
        },
    )
    return load_config(cfg_path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_rl():
    """Reset the global rate limiter before each test."""
    reset_rate_limiter()
    yield
    reset_rate_limiter()


@pytest.fixture
def fast_rate_limiter(monkeypatch):
    """Replace rate limiter with one that never sleeps."""
    from system_prompt_retrieval_agent import rate_limiter as rl_mod

    fake_rl = MagicMock()
    fake_rl.acquire = AsyncMock()
    fake_rl.add_cost = MagicMock()

    monkeypatch.setattr(rl_mod, "_GLOBAL", fake_rl)
    return fake_rl


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLocalApiEvaluatorBasic:
    @pytest.mark.asyncio
    async def test_evaluate_many_returns_three_results(
        self, tmp_path: Path, fast_rate_limiter
    ):
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=2)
        samples = _make_samples(tmp_path, n=3)
        client = _make_client([_make_canned_response() for _ in range(3)])

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many(samples)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_evaluate_many_result_has_all_subscores(
        self, tmp_path: Path, fast_rate_limiter
    ):
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=2)
        samples = _make_samples(tmp_path, n=1)
        client = _make_client([_make_canned_response()])

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many(samples)

        r = results[0]
        for key in ("edit_correctness", "garment_transfer_correctness", "preservation", "artifact_penalty", "notes", "sample_id", "usd_spent"):
            assert key in r, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_evaluate_many_sample_ids_preserved(
        self, tmp_path: Path, fast_rate_limiter
    ):
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=2)
        samples = _make_samples(tmp_path, n=3)
        client = _make_client([_make_canned_response() for _ in range(3)])

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many(samples)

        result_ids = {r["sample_id"] for r in results}
        expected_ids = {s["sample_id"] for s in samples}
        assert result_ids == expected_ids

    @pytest.mark.asyncio
    async def test_evaluate_many_score_values(self, tmp_path: Path, fast_rate_limiter):
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=2)
        samples = _make_samples(tmp_path, n=1)
        client = _make_client([_make_canned_response({"edit_correctness": 0.95})])

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many(samples)

        assert abs(results[0]["edit_correctness"] - 0.95) < 1e-9

    @pytest.mark.asyncio
    async def test_evaluate_many_disabled_returns_empty(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """When run_local_api_eval=False, evaluate_many must return []."""
        cfg = _make_cfg(tmp_path, run_local_api_eval=False)
        samples = _make_samples(tmp_path, n=3)
        client = _make_client([])

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many(samples)

        assert results == []


class TestLocalApiEvaluatorConcurrency:
    @pytest.mark.asyncio
    async def test_semaphore_limit_respected(self, tmp_path: Path, fast_rate_limiter):
        """Verify that no more than max_concurrent tasks run at the same time."""
        max_concurrent = 2
        n_samples = 6
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=max_concurrent)
        samples = _make_samples(tmp_path, n=n_samples)

        peak_concurrent = 0
        currently_running = 0
        lock = asyncio.Lock()

        async def _mock_create(**kwargs):
            nonlocal peak_concurrent, currently_running
            async with lock:
                currently_running += 1
                if currently_running > peak_concurrent:
                    peak_concurrent = currently_running
            await asyncio.sleep(0)  # yield to let other tasks start
            async with lock:
                currently_running -= 1
            return _make_canned_response()

        client = MagicMock()
        client.chat = MagicMock()
        client.chat.completions = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_mock_create)

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many(samples)

        assert len(results) == n_samples
        assert peak_concurrent <= max_concurrent


class TestLocalApiEvaluatorBudget:
    @pytest.mark.asyncio
    async def test_budget_guard_raises_cost_exhausted(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """When cumulative cost exceeds daily_cap, CostExhausted is raised."""
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=1)
        n = 5
        samples = _make_samples(tmp_path, n=n)

        # Set daily cap to allow only 2 calls worth of spending
        daily_cap = _USD_PER_CALL * 2.5  # 2 calls succeed, 3rd fails
        guard = BudgetGuard(daily_usd_cap=daily_cap, per_round_usd_cap=100.0)
        client = _make_client([_make_canned_response() for _ in range(n)])

        # With max_concurrent=1 and gather, some will succeed then raise
        evaluator = LocalApiEvaluator(cfg, openai_client=client, budget_guard=guard)

        with pytest.raises(CostExhausted):
            await evaluator.evaluate_many(samples)

    @pytest.mark.asyncio
    async def test_rate_limiter_add_cost_called(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """RateLimiter.add_cost should be called for each successful evaluation."""
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=2)
        samples = _make_samples(tmp_path, n=2)
        client = _make_client([_make_canned_response() for _ in range(2)])

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        await evaluator.evaluate_many(samples)

        assert fast_rate_limiter.add_cost.call_count == 2


class TestLocalApiEvaluatorCellKeyed:
    """V0.2.1: cell-keyed evaluation with prompt_pair_id + user_prompt_id."""

    def _make_cell_samples(self, tmp_path: Path, n: int = 2) -> list[dict]:
        model_img = _make_fake_image(tmp_path, "model.png")
        cloth_img = _make_fake_image(tmp_path, "cloth.png")
        return [
            {
                "prompt_pair_id": "pair_001",
                "user_prompt_id": f"up_{i:02d}",
                "sample_id": f"dress/sample_{i:03d}",
                "model_image_path": str(model_img),
                "cloth_image_path": str(cloth_img),
                "generated_image_path": str(_make_fake_image(tmp_path, f"gen_{i}.png")),
                "intermediate_prompt": f"Prompt {i}",
            }
            for i in range(n)
        ]

    @pytest.mark.asyncio
    async def test_evaluate_many_propagates_cell_keys(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """evaluate_many must pass through prompt_pair_id and user_prompt_id."""
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=2)
        cells = self._make_cell_samples(tmp_path, n=2)
        client = _make_client([_make_canned_response() for _ in range(2)])

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many(cells)

        assert len(results) == 2
        for r in results:
            assert "prompt_pair_id" in r
            assert "user_prompt_id" in r
            assert r["prompt_pair_id"] == "pair_001"

    @pytest.mark.asyncio
    async def test_evaluate_many_cells_persists_jsonl(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """evaluate_many_cells must write per-phrasing JSONL when api_eval_root given."""
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=2)
        cells = self._make_cell_samples(tmp_path, n=2)
        client = _make_client([_make_canned_response() for _ in range(2)])
        api_eval_root = tmp_path / "api_eval"

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many_cells(cells, api_eval_root=api_eval_root)

        assert len(results) == 2
        # Each result has its own user_prompt_id, so two separate JSONL files
        for r in results:
            jsonl = api_eval_root / r["prompt_pair_id"] / f"{r['user_prompt_id']}.jsonl"
            assert jsonl.exists(), f"Expected JSONL at {jsonl}"

    @pytest.mark.asyncio
    async def test_evaluate_many_cells_no_persist_without_root(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """evaluate_many_cells with api_eval_root=None must not create any files."""
        cfg = _make_cfg(tmp_path, run_local_api_eval=True, max_concurrent=2)
        cells = self._make_cell_samples(tmp_path, n=2)
        client = _make_client([_make_canned_response() for _ in range(2)])

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many_cells(cells, api_eval_root=None)

        assert len(results) == 2
        # No api_eval dir should have been created
        assert not (tmp_path / "api_eval").exists()

    @pytest.mark.asyncio
    async def test_evaluate_many_cells_disabled_returns_empty(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """evaluate_many_cells must return [] when run_local_api_eval=False."""
        cfg = _make_cfg(tmp_path, run_local_api_eval=False)
        cells = self._make_cell_samples(tmp_path, n=2)
        client = _make_client([])

        evaluator = LocalApiEvaluator(cfg, openai_client=client)
        results = await evaluator.evaluate_many_cells(cells, api_eval_root=tmp_path / "api_eval")

        assert results == []
