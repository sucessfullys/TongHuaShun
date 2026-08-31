"""Tests for failure_summary: select_low_score_examples, summarize_failures, append_summaries_jsonl."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from system_prompt_retrieval_agent.evaluation.failure_summary import (
    append_summaries_jsonl,
    select_low_score_examples,
    summarize_failures,
)
from system_prompt_retrieval_agent.schemas import FailureSummary
from system_prompt_retrieval_agent.rate_limiter import reset_rate_limiter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(tmp_path: Path, max_concurrent: int = 4):
    from conftest import write_mock_config
    from system_prompt_retrieval_agent.config import load_config

    cfg_path = write_mock_config(
        tmp_path,
        evaluation={
            "categories": ["dress", "lower", "upper"],
            "run_local_api_eval": True,
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


def _make_samples(n: int, base_score: float = 0.5) -> list[dict]:
    """Create n samples with scores from base_score incrementing by 0.01."""
    return [
        {
            "sample_id": f"dress/sample_{i:03d}",
            "category": "dress",
            "overall_score": round(base_score + i * 0.01, 4),
            "notes": f"Sample {i} notes",
            "edit_correctness": 0.7,
            "garment_transfer_correctness": 0.6,
            "preservation": 0.8,
            "artifact_penalty": 0.1,
        }
        for i in range(n)
    ]


def _make_summary_client(n_responses: int) -> MagicMock:
    """Create a fake OpenAI client returning canned failure summaries."""
    call_count = 0

    async def _create(**kwargs):
        nonlocal call_count
        call_count += 1
        payload = {
            "summary": f"Failure summary {call_count}",
            "failure_tags": ["tag_a", "tag_b"],
        }
        msg = MagicMock()
        msg.content = json.dumps(payload)
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=_create)
    return client


@pytest.fixture(autouse=True)
def reset_rl():
    reset_rate_limiter()
    yield
    reset_rate_limiter()


@pytest.fixture
def fast_rate_limiter(monkeypatch):
    from system_prompt_retrieval_agent import rate_limiter as rl_mod

    fake_rl = MagicMock()
    fake_rl.acquire = AsyncMock()
    fake_rl.add_cost = MagicMock()
    monkeypatch.setattr(rl_mod, "_GLOBAL", fake_rl)
    return fake_rl


# ---------------------------------------------------------------------------
# select_low_score_examples
# ---------------------------------------------------------------------------


class TestSelectLowScoreExamples:
    def test_selects_k_lowest(self):
        samples = _make_samples(20)
        selected = select_low_score_examples(samples, k=10)
        assert len(selected) == 10
        # All selected should have lower scores than unselected
        selected_scores = [s["overall_score"] for s in selected]
        unselected = [s for s in samples if s not in selected]
        unselected_scores = [s["overall_score"] for s in unselected]
        assert max(selected_scores) <= min(unselected_scores)

    def test_fewer_samples_than_k(self):
        samples = _make_samples(5)
        selected = select_low_score_examples(samples, k=10)
        assert len(selected) == 5

    def test_stable_tiebreak_by_sample_id(self):
        """Tie-break on same score should be stable and lexicographic by sample_id."""
        samples = [
            {"sample_id": "dress/sample_zzz", "overall_score": 0.5},
            {"sample_id": "dress/sample_aaa", "overall_score": 0.5},
            {"sample_id": "dress/sample_mmm", "overall_score": 0.5},
        ]
        selected = select_low_score_examples(samples, k=2)
        # Should be lexicographically lowest sample_ids
        assert selected[0]["sample_id"] == "dress/sample_aaa"
        assert selected[1]["sample_id"] == "dress/sample_mmm"

    def test_none_score_sorted_last(self):
        """Samples with None overall_score should be treated as infinity (sorted last)."""
        samples = [
            {"sample_id": "dress/s01", "overall_score": 0.1},
            {"sample_id": "dress/s02", "overall_score": None},
            {"sample_id": "dress/s03", "overall_score": 0.2},
        ]
        selected = select_low_score_examples(samples, k=2)
        ids = [s["sample_id"] for s in selected]
        assert "dress/s01" in ids
        assert "dress/s03" in ids
        assert "dress/s02" not in ids


# ---------------------------------------------------------------------------
# summarize_failures
# ---------------------------------------------------------------------------


class TestSummarizeFailures:
    @pytest.mark.asyncio
    async def test_selects_10_lowest_from_15(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """Given 15 samples, exactly 10 summaries should be produced."""
        cfg = _make_cfg(tmp_path, max_concurrent=4)
        samples = _make_samples(15)
        client = _make_summary_client(10)

        summaries = await summarize_failures(
            samples,
            prompt_pair_id="pair_001",
            cfg=cfg,
            k=10,
            openai_client=client,
        )

        assert len(summaries) == 10

    @pytest.mark.asyncio
    async def test_summaries_are_failure_summary_instances(
        self, tmp_path: Path, fast_rate_limiter
    ):
        cfg = _make_cfg(tmp_path)
        samples = _make_samples(15)
        client = _make_summary_client(10)

        summaries = await summarize_failures(
            samples,
            prompt_pair_id="pair_001",
            cfg=cfg,
            k=10,
            openai_client=client,
        )

        for fs in summaries:
            assert isinstance(fs, FailureSummary)

    @pytest.mark.asyncio
    async def test_summary_has_failure_tags(
        self, tmp_path: Path, fast_rate_limiter
    ):
        cfg = _make_cfg(tmp_path)
        samples = _make_samples(5)
        client = _make_summary_client(5)

        summaries = await summarize_failures(
            samples,
            prompt_pair_id="pair_001",
            cfg=cfg,
            k=5,
            openai_client=client,
        )

        for fs in summaries:
            assert isinstance(fs.failure_tags, list)
            assert len(fs.failure_tags) > 0

    @pytest.mark.asyncio
    async def test_summaries_have_correct_sample_ids(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """Summaries should correspond to the k lowest-scoring samples."""
        cfg = _make_cfg(tmp_path)
        samples = _make_samples(15)
        # 10 lowest are sample_000 through sample_009
        expected_ids = {f"dress/sample_{i:03d}" for i in range(10)}
        client = _make_summary_client(10)

        summaries = await summarize_failures(
            samples,
            prompt_pair_id="pair_001",
            cfg=cfg,
            k=10,
            openai_client=client,
        )

        result_ids = {fs.sample_id for fs in summaries}
        assert result_ids == expected_ids

    @pytest.mark.asyncio
    async def test_prompt_pair_id_attached_to_samples(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """Each sample should have prompt_pair_id set during processing."""
        cfg = _make_cfg(tmp_path)
        samples = _make_samples(3)
        client = _make_summary_client(3)

        await summarize_failures(
            samples,
            prompt_pair_id="pair_XYZ",
            cfg=cfg,
            k=3,
            openai_client=client,
        )

        # After call, samples should have prompt_pair_id attached
        for s in samples:
            assert s.get("prompt_pair_id") == "pair_XYZ"

    @pytest.mark.asyncio
    async def test_score_attached_to_summary(
        self, tmp_path: Path, fast_rate_limiter
    ):
        cfg = _make_cfg(tmp_path)
        samples = [
            {"sample_id": "dress/s01", "category": "dress", "overall_score": 0.25, "notes": ""}
        ]
        client = _make_summary_client(1)

        summaries = await summarize_failures(
            samples,
            prompt_pair_id="pair_001",
            cfg=cfg,
            k=1,
            openai_client=client,
        )

        assert abs(summaries[0].score - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# append_summaries_jsonl
# ---------------------------------------------------------------------------


class TestAppendSummariesJsonl:
    def test_creates_jsonl_file(self, tmp_path: Path):
        summaries = [
            FailureSummary(
                sample_id="dress/s01",
                category="dress",
                summary="Test summary",
                failure_tags=["tag_a"],
                score=0.3,
            )
        ]
        out = tmp_path / "failures.jsonl"
        append_summaries_jsonl(out, summaries, "pair_001", round_id=1)
        assert out.exists()

    def test_jsonl_content_has_schema_version(self, tmp_path: Path):
        summaries = [
            FailureSummary(
                sample_id="dress/s01",
                category="dress",
                summary="Test summary",
                failure_tags=["tag_a"],
                score=0.3,
            )
        ]
        out = tmp_path / "failures.jsonl"
        append_summaries_jsonl(out, summaries, "pair_001", round_id=1)

        lines = [json.loads(line) for line in out.read_text().strip().splitlines()]
        assert lines[0]["schema_version"] == "1.0"

    def test_jsonl_round_and_prompt_pair_id(self, tmp_path: Path):
        summaries = [
            FailureSummary(
                sample_id="dress/s01",
                category="dress",
                summary="Test",
                failure_tags=[],
                score=0.4,
            )
        ]
        out = tmp_path / "failures.jsonl"
        append_summaries_jsonl(out, summaries, "pair_999", round_id=7)

        record = json.loads(out.read_text().strip())
        assert record["round"] == 7
        assert record["prompt_pair_id"] == "pair_999"

    def test_jsonl_appends_multiple_calls(self, tmp_path: Path):
        """Calling append twice should append, not overwrite."""
        summaries_a = [
            FailureSummary(sample_id="a/s01", category="a", summary="A", failure_tags=[], score=0.1)
        ]
        summaries_b = [
            FailureSummary(sample_id="b/s01", category="b", summary="B", failure_tags=[], score=0.2)
        ]
        out = tmp_path / "failures.jsonl"
        append_summaries_jsonl(out, summaries_a, "pair_001", round_id=1)
        append_summaries_jsonl(out, summaries_b, "pair_002", round_id=2)

        lines = [json.loads(line) for line in out.read_text().strip().splitlines()]
        assert len(lines) == 2

    def test_jsonl_filter_by_prompt_pair_id(self, tmp_path: Path):
        """Basic round-trip: filter lines by prompt_pair_id."""
        summaries = [
            FailureSummary(sample_id=f"dress/s{i:02d}", category="dress", summary=f"S{i}", failure_tags=[], score=float(i) / 10)
            for i in range(5)
        ]
        out = tmp_path / "failures.jsonl"
        append_summaries_jsonl(out, summaries[:3], "pair_A", round_id=1)
        append_summaries_jsonl(out, summaries[3:], "pair_B", round_id=2)

        all_lines = [json.loads(line) for line in out.read_text().strip().splitlines()]
        pair_a_lines = [l for l in all_lines if l["prompt_pair_id"] == "pair_A"]
        pair_b_lines = [l for l in all_lines if l["prompt_pair_id"] == "pair_B"]

        assert len(pair_a_lines) == 3
        assert len(pair_b_lines) == 2

    def test_creates_parent_dirs(self, tmp_path: Path):
        """Parent directories are created if they don't exist."""
        out = tmp_path / "nested" / "deep" / "failures.jsonl"
        summaries = [
            FailureSummary(sample_id="dress/s01", category="dress", summary="S", failure_tags=[], score=0.1)
        ]
        append_summaries_jsonl(out, summaries, "pair_001", round_id=1)
        assert out.exists()

    def test_jsonl_includes_user_prompt_id_when_provided(self, tmp_path: Path):
        """V0.2.1: user_prompt_id should appear in the record when supplied."""
        summaries = [
            FailureSummary(sample_id="dress/s01", category="dress", summary="S", failure_tags=[], score=0.1)
        ]
        out = tmp_path / "failures.jsonl"
        append_summaries_jsonl(out, summaries, "pair_001", round_id=1, user_prompt_id="up_zh_01")

        record = json.loads(out.read_text().strip())
        assert record["user_prompt_id"] == "up_zh_01"

    def test_jsonl_omits_user_prompt_id_when_not_provided(self, tmp_path: Path):
        """When user_prompt_id is None, the key should be absent from the record."""
        summaries = [
            FailureSummary(sample_id="dress/s01", category="dress", summary="S", failure_tags=[], score=0.1)
        ]
        out = tmp_path / "failures.jsonl"
        append_summaries_jsonl(out, summaries, "pair_001", round_id=1)

        record = json.loads(out.read_text().strip())
        assert "user_prompt_id" not in record


class TestSummarizeFailuresUserPromptId:
    """V0.2.1: user_prompt_id is attached to samples during summarize_failures."""

    @pytest.mark.asyncio
    async def test_user_prompt_id_attached_to_samples(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """When user_prompt_id is passed, each sample should have it set."""
        cfg = _make_cfg(tmp_path)
        samples = _make_samples(3)
        client = _make_summary_client(3)

        await summarize_failures(
            samples,
            prompt_pair_id="pair_001",
            cfg=cfg,
            k=3,
            openai_client=client,
            user_prompt_id="up_en_01",
        )

        for s in samples:
            assert s.get("user_prompt_id") == "up_en_01"

    @pytest.mark.asyncio
    async def test_user_prompt_id_not_overwritten_if_already_set(
        self, tmp_path: Path, fast_rate_limiter
    ):
        """setdefault ensures existing user_prompt_id is not overwritten."""
        cfg = _make_cfg(tmp_path)
        samples = [
            {
                "sample_id": "dress/s01",
                "overall_score": 0.3,
                "user_prompt_id": "existing_up",
            }
        ]
        client = _make_summary_client(1)

        await summarize_failures(
            samples,
            prompt_pair_id="pair_001",
            cfg=cfg,
            k=1,
            openai_client=client,
            user_prompt_id="new_up",
        )

        # setdefault means original value is preserved
        assert samples[0]["user_prompt_id"] == "existing_up"
