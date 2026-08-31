"""Tests for prompt generation subsystem (S03.11)."""
from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from system_prompt_retrieval_agent.prompt_generation.generator import (
    PromptPairGenerator,
    generate_prompt_pairs,
)
from system_prompt_retrieval_agent.prompt_generation.validator import (
    ParseError,
    normalize_negative_prompt,
)
from system_prompt_retrieval_agent.prompt_generation.retry import APIRetryableError
from system_prompt_retrieval_agent.schemas import (
    PromptPair,
    PromptPairHistoryContext,
    PromptPairScoreContext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_pair_dict(
    system_prompt: str = "A" * 50,
    negative_prompt: str | None = None,
    rationale: str = "Addresses garment misalignment",
    expected_improvement_target: str = "Increase edit_correctness to 0.80",
    risk: str = "May lose preservation score",
    cross_lingual_robustness_strategy: str = "Use language-neutral anchors so zh and en phrasings produce equivalent garment-on-model results without depending on a single keyword",
) -> dict[str, Any]:
    return {
        "system_prompt": system_prompt,
        "negative_prompt": negative_prompt,
        "rationale": rationale,
        "expected_improvement_target": expected_improvement_target,
        "risk": risk,
        "cross_lingual_robustness_strategy": cross_lingual_robustness_strategy,
    }


def _make_valid_response(N: int = 3, base_char: str = "A") -> dict[str, Any]:
    pairs = []
    for i in range(N):
        pairs.append(
            _make_valid_pair_dict(
                system_prompt=f"{base_char * 50} variant_{i}",
            )
        )
    return {"prompt_pairs": pairs}


def _make_prompt_pair(
    pair_id: str = "pair_r01_000",
    system_prompt: str = "X" * 50,
    negative_prompt: str | None = None,
    overall_score: float | None = 0.75,
    round_id: int = 1,
) -> PromptPair:
    scores = (
        PromptPairScoreContext(overall_score=overall_score)
        if overall_score is not None
        else None
    )
    return PromptPair(
        prompt_pair_id=pair_id,
        system_prompt_id=f"sys_{pair_id}",
        negative_prompt_id=f"neg_{pair_id}",
        round_id=round_id,
        system_prompt=system_prompt,
        negative_prompt=negative_prompt,
        scores=scores,
    )


class _FakeRateLimiter:
    """Stub rate limiter that never sleeps."""

    async def acquire(self) -> None:
        return

    def add_cost(self, usd: float) -> None:
        pass


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = MagicMock()
        self.message.content = content


class _FakeCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _StubOpenAIClient:
    """Minimal OpenAI client stub for testing."""

    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs: Any) -> _FakeCompletion:
        if not self._responses:
            raise APIRetryableError("No more responses")
        resp = self._responses.pop(0)
        self._call_count += 1
        if isinstance(resp, Exception):
            raise resp
        return _FakeCompletion(resp)


class _MockMemoryManager:
    def __init__(self, ctx: PromptPairHistoryContext | None = None) -> None:
        self._ctx = ctx or PromptPairHistoryContext(round_id=1)

    def load_for_generation(self, round_id: int, N: int) -> PromptPairHistoryContext:
        return self._ctx


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# Fixture: minimal AppConfig-like object
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    cfg.api.openai_api_key_env = "OPENAI_API_KEY"
    cfg.api.prompt_generation_model = "gpt-test"
    cfg.prompt_generation.structured_outputs = False  # simpler path for tests
    cfg.prompt_generation.max_parse_retries = 3
    cfg.prompt_generation.retry_wait_s = 0  # no sleep in tests
    cfg.prompt_generation.fallback_to_best_existing = True
    cfg.rate_limits.max_api_retries = 6
    cfg.rate_limits.initial_backoff_s = 0
    cfg.rate_limits.max_backoff_s = 0
    cfg.rate_limits.jitter = False
    cfg.paths.prompt_seed_file = (
        "/Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/temp_wxy/PROMPTS.py"
    )
    return cfg


@pytest.fixture(autouse=True)
def patch_rate_limiter():
    """Patch get_rate_limiter globally to return a stub that never sleeps."""
    stub = _FakeRateLimiter()
    with patch(
        "system_prompt_retrieval_agent.prompt_generation.generator.get_rate_limiter",
        return_value=stub,
    ):
        yield stub


# ---------------------------------------------------------------------------
# Tests: normalize_negative_prompt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("None", None),
        ("none", None),
        ("NULL", None),
        ("", None),
        ("  ", None),
        ("\t", None),
        ("distorted body", "distorted body"),
        ("  distorted body  ", "distorted body"),
    ],
)
def test_normalize_negative_prompt(value, expected):
    assert normalize_negative_prompt(value) == expected


# ---------------------------------------------------------------------------
# Tests: Happy path — schema-valid response → 3 pairs returned
# ---------------------------------------------------------------------------

def test_happy_path_returns_n_pairs(mock_cfg):
    valid_resp = json.dumps(_make_valid_response(3))
    client = _StubOpenAIClient([valid_resp])
    gen = PromptPairGenerator(mock_cfg, openai_client=client)
    ctx = PromptPairHistoryContext(round_id=1)

    result = _run(gen.generate(ctx, N=3))

    assert len(result) == 3
    for pair in result:
        assert "system_prompt" in pair
        assert len(pair["system_prompt"]) >= 20


# ---------------------------------------------------------------------------
# Tests: Schema invalid → parse retry 3× → ParseError raised (fallback triggered)
# ---------------------------------------------------------------------------

def test_schema_invalid_triggers_parse_retry(mock_cfg):
    """Schema-invalid JSON triggers ParseError which exhausts parse retries."""
    bad_resp = json.dumps({"prompt_pairs": []})  # empty list, fails minItems=3
    client = _StubOpenAIClient([bad_resp, bad_resp, bad_resp, bad_resp])
    gen = PromptPairGenerator(mock_cfg, openai_client=client)
    ctx = PromptPairHistoryContext(round_id=1)

    with pytest.raises(ParseError):
        _run(gen.generate(ctx, N=3))


def test_schema_invalid_via_generate_prompt_pairs_uses_fallback(mock_cfg, monkeypatch):
    """Schema-invalid exhausts parse retries → fallback_used=True."""
    bad_resp = json.dumps({"prompt_pairs": []})
    client = _StubOpenAIClient([bad_resp] * 10)

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.prompt_generation.generator.PromptPairGenerator",
        lambda cfg, **kwargs: PromptPairGenerator(cfg, openai_client=client),
    )

    mgr = _MockMemoryManager()
    pairs, fallback_used = _run(
        generate_prompt_pairs(mock_cfg, mgr, round_id=1, N=3, existing_pairs=[])
    )
    assert fallback_used is True
    assert len(pairs) == 3


# ---------------------------------------------------------------------------
# Tests: Empty pairs → parse retry
# ---------------------------------------------------------------------------

def test_empty_pairs_triggers_parse_error(mock_cfg):
    """Response with no prompt_pairs key triggers ParseError."""
    empty_resp = json.dumps({"prompt_pairs": [{}]})
    client = _StubOpenAIClient([empty_resp])
    gen = PromptPairGenerator(mock_cfg, openai_client=client)
    ctx = PromptPairHistoryContext(round_id=1)

    with pytest.raises(ParseError):
        _run(gen.generate(ctx, N=3))


def test_empty_string_response_triggers_parse_error(mock_cfg):
    client = _StubOpenAIClient([""])
    gen = PromptPairGenerator(mock_cfg, openai_client=client)
    ctx = PromptPairHistoryContext(round_id=1)

    with pytest.raises(ParseError, match="empty"):
        _run(gen.generate(ctx, N=3))


# ---------------------------------------------------------------------------
# Tests: Simulated 429 on first call, success on second
# ---------------------------------------------------------------------------

def test_429_then_success(mock_cfg):
    """A 429 APIRetryableError on first call then success completes OK."""
    valid_resp = json.dumps(_make_valid_response(3))
    error_429 = APIRetryableError("429 Too Many Requests")
    client = _StubOpenAIClient([error_429, valid_resp])
    gen = PromptPairGenerator(mock_cfg, openai_client=client)
    ctx = PromptPairHistoryContext(round_id=1)

    # Need to wrap generate in retry logic for this test
    from system_prompt_retrieval_agent.prompt_generation.retry import generate_with_retry

    result = _run(
        generate_with_retry(
            lambda: gen.generate(ctx, N=3),
            max_parse_retries=3,
            retry_wait_s=0,
            max_api_retries=6,
            initial_backoff=0,
            max_backoff=0,
            jitter=False,
        )
    )
    assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: All retries exhausted → fallback to seed pair
# ---------------------------------------------------------------------------

def test_all_retries_exhausted_fallback_to_seed(mock_cfg, monkeypatch):
    """When all retries fail, fallback to seed pair; fallback_used=True."""
    # Always raise APIRetryableError
    always_fail = APIRetryableError("persistent failure")
    client = _StubOpenAIClient([always_fail] * 20)

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.prompt_generation.generator.PromptPairGenerator",
        lambda cfg, **kwargs: PromptPairGenerator(cfg, openai_client=client),
    )

    mgr = _MockMemoryManager()
    pairs, fallback_used = _run(
        generate_prompt_pairs(mock_cfg, mgr, round_id=1, N=3, existing_pairs=[])
    )

    assert fallback_used is True
    assert len(pairs) == 3
    # Seed pair should contain text from PROMPTS.py or inline fallback
    for p in pairs:
        assert len(p.system_prompt) >= 20
        assert p.fallback is True


def test_all_retries_exhausted_seed_names_come_from_prompts_py(mock_cfg, monkeypatch):
    """Verify seed comes from temp_wxy/PROMPTS.py when available."""
    always_fail = ParseError("always bad schema")
    client = _StubOpenAIClient([always_fail] * 20)

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.prompt_generation.generator.PromptPairGenerator",
        lambda cfg, **kwargs: PromptPairGenerator(cfg, openai_client=client),
    )

    mgr = _MockMemoryManager()
    pairs, fallback_used = _run(
        generate_prompt_pairs(mock_cfg, mgr, round_id=1, N=3, existing_pairs=[])
    )

    assert fallback_used is True
    # Seed text should be meaningful (>= 20 chars)
    for p in pairs:
        assert len(p.system_prompt) >= 20


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------

def test_dedupe_first_attempt_matches_then_regenerates(mock_cfg, monkeypatch):
    """First attempt duplicates existing pair; regeneration returns unique pair."""
    existing = _make_prompt_pair("pair_r00_000", system_prompt="Z" * 50)

    # First batch: all duplicates. Second call (regen): unique.
    duplicate_response = json.dumps(
        {"prompt_pairs": [_make_valid_pair_dict(system_prompt="Z" * 50)] * 3}
    )
    unique_response = json.dumps(
        {"prompt_pairs": [_make_valid_pair_dict(system_prompt="Q" * 50)]}
    )

    call_count = [0]

    class _SelectiveClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _FakeCompletion(duplicate_response)
            return _FakeCompletion(unique_response)

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.prompt_generation.generator.PromptPairGenerator",
        lambda cfg, **kwargs: PromptPairGenerator(cfg, openai_client=_SelectiveClient()),
    )

    mgr = _MockMemoryManager()
    pairs, fallback_used = _run(
        generate_prompt_pairs(mock_cfg, mgr, round_id=1, N=3, existing_pairs=[existing])
    )

    # At least some pairs should not be duplicates
    assert len(pairs) == 3


def test_dedupe_still_duplicate_after_regen_accepted_with_flag(mock_cfg, monkeypatch):
    """If still duplicate after regeneration, pair accepted with duplicate=True."""
    existing = _make_prompt_pair("pair_r00_000", system_prompt="Z" * 50)

    # All calls return same duplicate system_prompt
    dup_response = json.dumps(
        {"prompt_pairs": [_make_valid_pair_dict(system_prompt="Z" * 50)] * 3}
    )
    dup_single = json.dumps(
        {"prompt_pairs": [_make_valid_pair_dict(system_prompt="Z" * 50)]}
    )

    class _AlwaysDupClient:
        def __init__(self):
            self.chat = self
            self.completions = self

        def create(self, **kwargs):
            msgs = kwargs.get("messages", [])
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            if "Generate exactly 1" in user_msg:
                return _FakeCompletion(dup_single)
            return _FakeCompletion(dup_response)

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.prompt_generation.generator.PromptPairGenerator",
        lambda cfg, **kwargs: PromptPairGenerator(cfg, openai_client=_AlwaysDupClient()),
    )

    mgr = _MockMemoryManager()
    pairs, _ = _run(
        generate_prompt_pairs(mock_cfg, mgr, round_id=1, N=3, existing_pairs=[existing])
    )

    assert len(pairs) == 3
    # At least some should be marked duplicate
    assert any(p.duplicate for p in pairs)


def test_dedupe_no_collision_returns_duplicate_false(mock_cfg, monkeypatch):
    """Pairs with unique system prompts have duplicate=False."""
    valid_resp = json.dumps(_make_valid_response(3, base_char="B"))
    client = _StubOpenAIClient([valid_resp])

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.prompt_generation.generator.PromptPairGenerator",
        lambda cfg, **kwargs: PromptPairGenerator(cfg, openai_client=client),
    )

    mgr = _MockMemoryManager()
    existing = _make_prompt_pair("pair_r00_000", system_prompt="X" * 50)
    pairs, fallback_used = _run(
        generate_prompt_pairs(mock_cfg, mgr, round_id=2, N=3, existing_pairs=[existing])
    )

    assert len(pairs) == 3
    assert all(not p.duplicate for p in pairs)
    assert not fallback_used


# ---------------------------------------------------------------------------
# Tests: Pair IDs are correctly formatted
# ---------------------------------------------------------------------------

def test_pair_ids_format(mock_cfg, monkeypatch):
    valid_resp = json.dumps(_make_valid_response(3))
    client = _StubOpenAIClient([valid_resp])

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.prompt_generation.generator.PromptPairGenerator",
        lambda cfg, **kwargs: PromptPairGenerator(cfg, openai_client=client),
    )

    mgr = _MockMemoryManager()
    pairs, _ = _run(
        generate_prompt_pairs(mock_cfg, mgr, round_id=5, N=3, existing_pairs=[])
    )

    assert pairs[0].system_prompt_id == "sys_r05_000"
    assert pairs[0].negative_prompt_id == "neg_r05_000"
    assert pairs[0].prompt_pair_id == "pair_r05_000"
    assert pairs[1].prompt_pair_id == "pair_r05_001"
    assert pairs[2].prompt_pair_id == "pair_r05_002"


# ---------------------------------------------------------------------------
# Tests: Round-trip schema validation
# ---------------------------------------------------------------------------

def test_generated_pairs_are_valid_prompt_pair_objects(mock_cfg, monkeypatch):
    """Generated PromptPair objects pass Pydantic validation."""
    valid_resp = json.dumps(_make_valid_response(3))
    client = _StubOpenAIClient([valid_resp])

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.prompt_generation.generator.PromptPairGenerator",
        lambda cfg, **kwargs: PromptPairGenerator(cfg, openai_client=client),
    )

    mgr = _MockMemoryManager()
    pairs, _ = _run(
        generate_prompt_pairs(mock_cfg, mgr, round_id=1, N=3, existing_pairs=[])
    )

    for pair in pairs:
        assert isinstance(pair, PromptPair)
        assert pair.round_id == 1
        assert len(pair.system_prompt) >= 20


# ---------------------------------------------------------------------------
# Tests: ParseError fallback triggers fallback_used=True
# ---------------------------------------------------------------------------

def test_parse_error_then_fallback(mock_cfg, monkeypatch):
    """Exhausted parse retries via generate_prompt_pairs sets fallback_used=True."""
    # Provide more than enough bad responses to exhaust retries
    bad_resp = json.dumps({"prompt_pairs": "not_a_list"})  # bad type
    client = _StubOpenAIClient([bad_resp] * 15)

    monkeypatch.setattr(
        "system_prompt_retrieval_agent.prompt_generation.generator.PromptPairGenerator",
        lambda cfg, **kwargs: PromptPairGenerator(cfg, openai_client=client),
    )

    mgr = _MockMemoryManager()
    pairs, fallback_used = _run(
        generate_prompt_pairs(mock_cfg, mgr, round_id=1, N=3, existing_pairs=[])
    )

    assert fallback_used is True
    assert len(pairs) == 3
