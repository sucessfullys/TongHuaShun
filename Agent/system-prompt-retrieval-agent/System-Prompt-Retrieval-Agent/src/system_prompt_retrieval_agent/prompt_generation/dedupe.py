"""dedupe.py — Hashing and deduplication for PromptPair objects."""
from __future__ import annotations

import hashlib

from system_prompt_retrieval_agent.schemas import PromptPair


def pair_hash(
    system_prompt: str | None,
    negative_prompt: str | None,
) -> tuple[str, str]:
    """Compute short SHA-256 hashes for a (system_prompt, negative_prompt) pair.

    Args:
        system_prompt: The system prompt text (stripped before hashing).
        negative_prompt: The negative prompt text or None.

    Returns:
        Tuple of (sys_hash[:12], neg_hash[:12]).
    """
    sys_text = (system_prompt or "").strip()
    neg_text = (negative_prompt or "").strip()

    sys_hash = hashlib.sha256(sys_text.encode("utf-8")).hexdigest()[:12]
    neg_hash = hashlib.sha256(neg_text.encode("utf-8")).hexdigest()[:12]
    return (sys_hash, neg_hash)


def is_duplicate(
    existing_pairs: list[PromptPair],
    new_sys: str,
    new_neg: str | None,
) -> bool:
    """Check whether (new_sys, new_neg) collides with any pair in existing_pairs.

    Comparison is hash-based (pair_hash) for both system_prompt and
    negative_prompt.

    Args:
        existing_pairs: List of already-accepted PromptPair objects.
        new_sys: System prompt text to check.
        new_neg: Negative prompt text (or None) to check.

    Returns:
        True if a collision is found, False otherwise.
    """
    new_h = pair_hash(new_sys, new_neg)
    for p in existing_pairs:
        if pair_hash(p.system_prompt, p.negative_prompt) == new_h:
            return True
    return False
