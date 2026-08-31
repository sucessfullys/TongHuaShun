"""retry.py — Retry wrappers for generator API and parse errors."""
from __future__ import annotations

import asyncio
import logging
import random
from typing import Any, Callable

from system_prompt_retrieval_agent.prompt_generation.validator import ParseError

logger = logging.getLogger(__name__)


class APIRetryableError(Exception):
    """Raised for retryable API errors: 429, 5xx, timeout."""


async def generate_with_retry(
    generator_fn: Callable[[], Any],
    *,
    max_parse_retries: int = 3,
    retry_wait_s: float = 60.0,
    max_api_retries: int = 6,
    initial_backoff: float = 1.0,
    max_backoff: float = 60.0,
    jitter: bool = True,
) -> Any:
    """Retry wrapper for generator_fn with separate parse and API error policies.

    Parse errors (ParseError / schema / empty / refusal):
      Fixed wait of retry_wait_s seconds, up to max_parse_retries attempts.

    API errors (APIRetryableError / 429 / 5xx / timeout):
      Exponential backoff with optional jitter, up to max_api_retries attempts.

    Args:
        generator_fn: Async callable returning the generation result.
        max_parse_retries: Max retries on ParseError.
        retry_wait_s: Wait seconds between parse retries.
        max_api_retries: Max retries on APIRetryableError.
        initial_backoff: Starting backoff seconds for API errors.
        max_backoff: Maximum backoff seconds for API errors.
        jitter: If True, add uniform random jitter to API backoff.

    Returns:
        Result of generator_fn on success.

    Raises:
        ParseError: If all parse retries exhausted.
        APIRetryableError: If all API retries exhausted.
    """
    parse_attempts = 0
    api_attempts = 0
    backoff = initial_backoff

    while True:
        try:
            return await generator_fn()
        except ParseError as exc:
            parse_attempts += 1
            if parse_attempts >= max_parse_retries:
                logger.error(
                    "Parse error on attempt %d/%d (exhausted): %s",
                    parse_attempts,
                    max_parse_retries,
                    exc,
                )
                raise
            logger.warning(
                "Parse error on attempt %d/%d, retrying in %.0fs: %s",
                parse_attempts,
                max_parse_retries,
                retry_wait_s,
                exc,
            )
            await asyncio.sleep(retry_wait_s)
        except APIRetryableError as exc:
            api_attempts += 1
            if api_attempts >= max_api_retries:
                logger.error(
                    "API error on attempt %d/%d (exhausted): %s",
                    api_attempts,
                    max_api_retries,
                    exc,
                )
                raise
            wait = min(backoff, max_backoff)
            if jitter:
                wait = wait * (0.5 + random.random() * 0.5)
            logger.warning(
                "API error on attempt %d/%d, retrying in %.2fs: %s",
                api_attempts,
                max_api_retries,
                wait,
                exc,
            )
            await asyncio.sleep(wait)
            backoff = min(backoff * 2.0, max_backoff)


async def with_fallback(
    primary_fn: Callable[[], Any],
    fallback_chain: list[Callable[[], Any]],
    mark_fallback_cb: Callable[[], None],
) -> Any:
    """Try primary_fn; if it raises, try each fallback in order.

    Calls mark_fallback_cb() when a fallback is used.

    Args:
        primary_fn: Primary async callable.
        fallback_chain: Ordered list of fallback async callables.
        mark_fallback_cb: Callback invoked once when any fallback is used.

    Returns:
        Result from primary_fn or the first successful fallback.

    Raises:
        Exception: If all fallbacks also fail.
    """
    try:
        return await primary_fn()
    except Exception as primary_exc:
        logger.warning("Primary generator failed: %s — trying fallbacks.", primary_exc)
        mark_fallback_cb()
        last_exc = primary_exc
        for i, fallback_fn in enumerate(fallback_chain):
            try:
                result = await fallback_fn()
                logger.info("Fallback[%d] succeeded.", i)
                return result
            except Exception as exc:
                logger.warning("Fallback[%d] failed: %s", i, exc)
                last_exc = exc
        raise last_exc
