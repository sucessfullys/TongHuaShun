"""generator.py — PromptPairGenerator and generate_prompt_pairs entry point."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from system_prompt_retrieval_agent.config import AppConfig
from system_prompt_retrieval_agent.prompt_generation.context_builder import build_history_context
from system_prompt_retrieval_agent.prompt_generation.dedupe import is_duplicate
from system_prompt_retrieval_agent.prompt_generation.retry import (
    APIRetryableError,
    generate_with_retry,
    with_fallback,
)
from system_prompt_retrieval_agent.prompt_generation.seed_loader import load_seed_prompts
from system_prompt_retrieval_agent.prompt_generation.validator import (
    ParseError,
    normalize_negative_prompt,
    validate_response,
)
from system_prompt_retrieval_agent.rate_limiter import get_rate_limiter
from system_prompt_retrieval_agent.schemas import (
    PromptPair,
    PromptPairHistoryContext,
    PromptPairScoreContext,
)

logger = logging.getLogger(__name__)

# src/system_prompt_retrieval_agent/prompt_generation/generator.py
# Up: prompt_generation → system_prompt_retrieval_agent → src (..) → project_root (...)
# Primary: skills/ lives alongside the package under src/system_prompt_retrieval_agent/
_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "skills/generating-prompt-pairs/system_prompt.md"
)
# Fallback: project-root-level skills/ (legacy path)
_SYSTEM_PROMPT_PATH_FALLBACK = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "skills/generating-prompt-pairs/system_prompt.md"
)
_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "skills/generating-prompt-pairs/schema.json"
)
# Fallback: project-root-level skills/ (legacy path)
_SCHEMA_PATH_FALLBACK = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "skills/generating-prompt-pairs/schema.json"
)

_CACHED_SYSTEM_PROMPT: str | None = None
_CACHED_SCHEMA: dict[str, Any] | None = None


def _load_system_prompt() -> str:
    global _CACHED_SYSTEM_PROMPT
    if _CACHED_SYSTEM_PROMPT is not None:
        return _CACHED_SYSTEM_PROMPT
    # Try primary path first, then fallback
    for candidate in [_SYSTEM_PROMPT_PATH, _SYSTEM_PROMPT_PATH_FALLBACK]:
        if candidate.exists():
            _CACHED_SYSTEM_PROMPT = candidate.read_text(encoding="utf-8")
            return _CACHED_SYSTEM_PROMPT
    _CACHED_SYSTEM_PROMPT = (
        "You are an expert prompt engineer for virtual try-on AI systems. "
        "Generate {N} distinct prompt pairs as JSON with keys: system_prompt, "
        "negative_prompt, rationale, expected_improvement_target, risk, "
        "cross_lingual_robustness_strategy."
    )
    return _CACHED_SYSTEM_PROMPT


def _load_schema() -> dict[str, Any]:
    global _CACHED_SCHEMA
    if _CACHED_SCHEMA is not None:
        return _CACHED_SCHEMA
    # Try primary path first, then fallback
    for candidate in [_SCHEMA_PATH, _SCHEMA_PATH_FALLBACK]:
        if candidate.exists():
            with candidate.open() as fh:
                _CACHED_SCHEMA = json.load(fh)
            return _CACHED_SCHEMA  # noqa: return inside loop is intentional
    if _SCHEMA_PATH.exists():
        with _SCHEMA_PATH.open() as fh:
            _CACHED_SCHEMA = json.load(fh)
    else:
        _CACHED_SCHEMA = {
            "type": "object",
            "required": ["prompt_pairs"],
            "properties": {
                "prompt_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "system_prompt",
                            "negative_prompt",
                            "rationale",
                            "expected_improvement_target",
                            "risk",
                            "cross_lingual_robustness_strategy",
                        ],
                        "properties": {
                            "system_prompt": {"type": "string"},
                            "negative_prompt": {"type": ["string", "null"]},
                            "rationale": {"type": "string"},
                            "expected_improvement_target": {"type": "string"},
                            "risk": {"type": "string"},
                            "cross_lingual_robustness_strategy": {"type": "string"},
                        },
                        # additionalProperties: False rejects user_prompts and any
                        # other undeclared field the model might propose.
                        "additionalProperties": False,
                    },
                }
            },
            # Top-level additionalProperties: False ensures user_prompts at the
            # root level is also rejected.
            "additionalProperties": False,
        }
    return _CACHED_SCHEMA


class PromptPairGenerator:
    """Calls the OpenAI API to generate prompt pairs for the next round."""

    def __init__(self, cfg: AppConfig, *, openai_client=None) -> None:
        self._cfg = cfg
        if openai_client is not None:
            self._client = openai_client
        else:
            import openai  # type: ignore

            api_key = os.environ.get(cfg.api.openai_api_key_env)
            self._client = openai.OpenAI(
                api_key=api_key,
                max_retries=int(cfg.rate_limits.max_api_retries),
            )

    def _build_user_message(self, ctx: PromptPairHistoryContext, N: int) -> str:
        """Serialize the five history blocks into a user message."""
        data = ctx.model_dump(by_alias=False)
        return (
            f"Generate exactly {N} prompt pairs.\n\n"
            f"History context:\n{json.dumps(data, indent=2, default=str)}"
        )

    def _build_system_message(self, N: int) -> str:
        tmpl = _load_system_prompt()
        return tmpl.replace("{N}", str(N))

    async def generate(self, ctx: PromptPairHistoryContext, N: int) -> list[dict[str, Any]]:
        """Generate N raw prompt pair dicts from the model.

        Rate-limited via get_rate_limiter().acquire() before each API call.

        Args:
            ctx: History context for the generation call.
            N: Number of pairs to generate.

        Returns:
            List of N raw pair dicts (validated by validate_response).

        Raises:
            ParseError: If response is not schema-valid after structured/json fallback.
            APIRetryableError: If the API returns a retryable error.
        """
        system_msg = self._build_system_message(N)
        user_msg = self._build_user_message(ctx, N)
        model = self._cfg.api.prompt_generation_model

        import copy

        schema = copy.deepcopy(_load_schema())
        schema["properties"]["prompt_pairs"]["minItems"] = N
        schema["properties"]["prompt_pairs"]["maxItems"] = N

        # Acquire rate limiter token before the call
        await get_rate_limiter().acquire()

        raw_text: str | None = None

        if self._cfg.prompt_generation.structured_outputs:
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "prompt_pairs",
                            "strict": True,
                            "schema": schema,
                        },
                    },
                )
                raw_text = response.choices[0].message.content
            except Exception as exc:
                exc_str = str(exc).lower()
                if "strict" in exc_str or "unsupported" in exc_str or "json_schema" in exc_str:
                    logger.warning(
                        "Structured outputs not supported by model %s, falling back to json_object: %s",
                        model,
                        exc,
                    )
                    raw_text = None
                elif "429" in str(exc) or "rate" in exc_str or "timeout" in exc_str or "5" in str(type(exc).__name__):
                    raise APIRetryableError(str(exc)) from exc
                else:
                    # Try to detect HTTP 5xx or connection errors
                    raise APIRetryableError(str(exc)) from exc

        if raw_text is None:
            # Fallback: json_object mode
            try:
                await get_rate_limiter().acquire()
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={"type": "json_object"},
                )
                raw_text = response.choices[0].message.content
            except Exception as exc:
                exc_str = str(exc).lower()
                if "429" in str(exc) or "rate" in exc_str or "timeout" in exc_str:
                    raise APIRetryableError(str(exc)) from exc
                raise APIRetryableError(str(exc)) from exc

        if not raw_text or not raw_text.strip():
            raise ParseError("Model returned empty response.")

        try:
            raw_json = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ParseError(f"Response is not valid JSON: {exc}") from exc

        if isinstance(raw_json, dict) and "refusal" in raw_json:
            raise ParseError(f"Model refused: {raw_json['refusal']}")

        return validate_response(raw_json, N)


def _make_pair_ids(round_id: int, slot: int) -> tuple[str, str, str]:
    """Return (system_prompt_id, negative_prompt_id, prompt_pair_id)."""
    sys_id = f"sys_r{round_id:02d}_{slot:03d}"
    neg_id = f"neg_r{round_id:02d}_{slot:03d}"
    pair_id = f"pair_r{round_id:02d}_{slot:03d}"
    return sys_id, neg_id, pair_id


def _best_existing_pair(existing_pairs: list[PromptPair]) -> PromptPair | None:
    """Return the pair with the highest overall_score, or None."""
    scored = [
        p
        for p in existing_pairs
        if p.scores is not None and p.scores.overall_score is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda p: p.scores.overall_score)  # type: ignore[union-attr]


def _best_from_context(ctx: PromptPairHistoryContext) -> PromptPair | None:
    """Return the best-scored pair from long_memory or previous_top blocks."""
    candidates: list[PromptPair] = []
    for block in [ctx.long_memory_prompt_pairs, ctx.previous_top_prompt_pairs]:
        if block:
            candidates.extend(block)
    scored = [
        p
        for p in candidates
        if p.scores is not None and p.scores.overall_score is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda p: p.scores.overall_score)  # type: ignore[union-attr]


def _seed_pair(cfg: AppConfig) -> PromptPair:
    """Load a seed pair from temp_wxy/PROMPTS.py or use inline fallback."""
    seed_file = Path(cfg.paths.prompt_seed_file)
    seeds = load_seed_prompts(seed_file)
    # Pick the first available seed
    seed_name = next(iter(seeds))
    seed_text = seeds[seed_name]
    return PromptPair(
        prompt_pair_id="pair_r00_seed",
        system_prompt_id="sys_r00_seed",
        negative_prompt_id="neg_r00_seed",
        round_id=0,
        selection_role="seed",
        system_prompt=seed_text,
        negative_prompt=None,
        rationale=f"Seed pair from {seed_name}",
        expected_improvement_target="Establish baseline performance",
        risk="Seed may not reflect latest improvements",
        fallback=True,
    )


async def generate_prompt_pairs(
    cfg: AppConfig,
    memory_manager,
    round_id: int,
    N: int,
    existing_pairs: list[PromptPair],
) -> tuple[list[PromptPair], bool]:
    """Generate N PromptPairs for the given round.

    Steps:
      1. Build history context via build_history_context.
      2. Call PromptPairGenerator.generate through retry.generate_with_retry.
      3. Validate via validator.validate_response (done inside generator).
      4. Dedupe against existing_pairs — one regeneration slot; accept with
         duplicate=True if still colliding.
      5. Convert raw dicts into PromptPair objects with proper IDs.
      6. On full failure → fallback chain: best existing → best in context → seed.

    Args:
        cfg: Application configuration.
        memory_manager: Memory manager with load_for_generation method.
        round_id: Current round number.
        N: Number of pairs to generate.
        existing_pairs: Already-accepted pairs for deduplication.

    Returns:
        Tuple of (list[PromptPair], fallback_used: bool).
    """
    fallback_used = False
    ctx = build_history_context(memory_manager, round_id, N)
    generator = PromptPairGenerator(cfg)

    async def _primary() -> list[dict[str, Any]]:
        return await generate_with_retry(
            lambda: generator.generate(ctx, N),
            max_parse_retries=cfg.prompt_generation.max_parse_retries,
            retry_wait_s=cfg.prompt_generation.retry_wait_s,
            max_api_retries=cfg.rate_limits.max_api_retries,
            initial_backoff=cfg.rate_limits.initial_backoff_s,
            max_backoff=cfg.rate_limits.max_backoff_s,
            jitter=cfg.rate_limits.jitter,
        )

    def _mark_fallback() -> None:
        nonlocal fallback_used
        fallback_used = True

    async def _fallback_existing() -> list[dict[str, Any]]:
        best = _best_existing_pair(existing_pairs) or _best_from_context(ctx)
        if best is None:
            raise ValueError("No existing pairs available for fallback.")
        return [
            {
                "system_prompt": best.system_prompt,
                "negative_prompt": best.negative_prompt,
                "rationale": f"Fallback to best existing pair {best.prompt_pair_id}",
                "expected_improvement_target": "Maintain best known performance",
                "risk": "No improvement over existing best",
                "cross_lingual_robustness_strategy": "fallback — inherited from best existing pair",
            }
            for _ in range(N)
        ]

    async def _fallback_seed() -> list[dict[str, Any]]:
        seed = _seed_pair(cfg)
        return [
            {
                "system_prompt": seed.system_prompt,
                "negative_prompt": seed.negative_prompt,
                "rationale": seed.rationale or "Seed fallback",
                "expected_improvement_target": seed.expected_improvement_target or "Establish baseline",
                "risk": seed.risk or "Seed may not reflect improvements",
                "cross_lingual_robustness_strategy": "fallback — seed pair; uses generic phrasing",
            }
            for _ in range(N)
        ]

    raw_pairs = await with_fallback(
        _primary,
        fallback_chain=[_fallback_existing, _fallback_seed],
        mark_fallback_cb=_mark_fallback,
    )

    # Build PromptPair objects with deduplication
    result: list[PromptPair] = []
    for slot, raw in enumerate(raw_pairs):
        sys_id, neg_id, pair_id = _make_pair_ids(round_id, slot)
        sys_prompt = raw["system_prompt"]
        neg_prompt = normalize_negative_prompt(raw.get("negative_prompt"))

        # Dedupe check: one regeneration slot
        pair_duplicate = False
        if is_duplicate(existing_pairs + result, sys_prompt, neg_prompt):
            # Try to regenerate once synchronously by using the same generator
            logger.warning("Slot %d is a duplicate — attempting one regeneration.", slot)
            try:
                regen_raw = await generate_with_retry(
                    lambda: generator.generate(ctx, 1),
                    max_parse_retries=1,
                    retry_wait_s=0,
                    max_api_retries=1,
                    initial_backoff=0,
                    max_backoff=0,
                    jitter=False,
                )
                regen = regen_raw[0]
                new_sys = regen["system_prompt"]
                new_neg = normalize_negative_prompt(regen.get("negative_prompt"))
                if not is_duplicate(existing_pairs + result, new_sys, new_neg):
                    sys_prompt = new_sys
                    neg_prompt = new_neg
                    raw = regen
                else:
                    pair_duplicate = True
                    logger.warning("Slot %d still duplicate after regeneration; accepting with duplicate=True.", slot)
            except Exception as exc:
                pair_duplicate = True
                logger.warning("Slot %d regeneration failed (%s); accepting with duplicate=True.", slot, exc)

        pair = PromptPair(
            prompt_pair_id=pair_id,
            system_prompt_id=sys_id,
            negative_prompt_id=neg_id,
            round_id=round_id,
            selection_role=None,
            system_prompt=sys_prompt,
            negative_prompt=neg_prompt,
            rationale=raw.get("rationale"),
            expected_improvement_target=raw.get("expected_improvement_target"),
            risk=raw.get("risk"),
            fallback=fallback_used,
            duplicate=pair_duplicate,
        )
        result.append(pair)

    return result, fallback_used
