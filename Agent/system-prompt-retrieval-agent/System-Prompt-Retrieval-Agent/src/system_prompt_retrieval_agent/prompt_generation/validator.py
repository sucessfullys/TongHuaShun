"""validator.py — JSON schema and semantic validation for generator responses."""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import jsonschema

logger = logging.getLogger(__name__)

# src/system_prompt_retrieval_agent/prompt_generation/validator.py
# Go up: prompt_generation → system_prompt_retrieval_agent → src → project_root
_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent.parent / (
    "skills/generating-prompt-pairs/schema.json"
)
# Fallback: one more level up (repo root / System-Prompt-Retrieval-Agent)
_SCHEMA_PATH_ALT = Path(__file__).resolve().parent.parent.parent.parent.parent / (
    "System-Prompt-Retrieval-Agent/skills/generating-prompt-pairs/schema.json"
)

_BASE_SCHEMA: dict[str, Any] | None = None


def _load_base_schema() -> dict[str, Any]:
    global _BASE_SCHEMA
    if _BASE_SCHEMA is not None:
        return _BASE_SCHEMA
    for candidate in [_SCHEMA_PATH, _SCHEMA_PATH_ALT]:
        if candidate.exists():
            with candidate.open() as fh:
                _BASE_SCHEMA = json.load(fh)
            return _BASE_SCHEMA
    raise FileNotFoundError(
        f"Cannot find schema.json. Tried: {_SCHEMA_PATH}, {_SCHEMA_PATH_ALT}"
    )


class ParseError(ValueError):
    """Raised when the generator response fails schema or semantic validation."""


def normalize_negative_prompt(v: Any) -> str | None:
    """Normalize negative_prompt values to Python None or a stripped string.

    Maps: None, "None", "none", "NULL", "", whitespace-only → Python None.
    Otherwise returns stripped string.
    """
    if v is None:
        return None
    if not isinstance(v, str):
        return None
    stripped = v.strip()
    if stripped in ("", "None", "none", "NULL"):
        return None
    return stripped


def validate_response(raw_json: dict[str, Any], N: int) -> list[dict[str, Any]]:
    """Validate and normalize a raw generator response dict.

    Runs JSON Schema validation (with minItems=maxItems=N), then semantic checks:
      - Exactly N items
      - Each system_prompt: 20 ≤ len ≤ 4000
      - rationale non-empty
      - No duplicate system_prompt within response
      - negative_prompt normalized via normalize_negative_prompt

    Args:
        raw_json: Parsed response dict from the model.
        N: Expected number of pairs.

    Returns:
        Normalized list of N pair dicts.

    Raises:
        ParseError: If schema or semantic validation fails.
    """
    # Patch schema with actual N
    base = _load_base_schema()
    schema = copy.deepcopy(base)
    schema["properties"]["prompt_pairs"]["minItems"] = N
    schema["properties"]["prompt_pairs"]["maxItems"] = N

    try:
        jsonschema.validate(instance=raw_json, schema=schema)
    except jsonschema.ValidationError as exc:
        raise ParseError(f"Schema validation failed: {exc.message}") from exc

    pairs = raw_json.get("prompt_pairs", [])

    # Semantic: count
    if len(pairs) != N:
        raise ParseError(f"Expected {N} pairs, got {len(pairs)}.")

    seen_prompts: set[str] = set()
    normalized: list[dict[str, Any]] = []

    for idx, pair in enumerate(pairs):
        sys_prompt = pair.get("system_prompt", "")

        # Length checks
        if len(sys_prompt) < 20:
            raise ParseError(
                f"Pair[{idx}] system_prompt too short ({len(sys_prompt)} chars, min 20)."
            )
        if len(sys_prompt) > 4000:
            raise ParseError(
                f"Pair[{idx}] system_prompt too long ({len(sys_prompt)} chars, max 4000)."
            )

        # Rationale non-empty
        rationale = pair.get("rationale", "")
        if not rationale or not rationale.strip():
            raise ParseError(f"Pair[{idx}] rationale is empty.")

        # Duplicate check
        if sys_prompt in seen_prompts:
            raise ParseError(
                f"Pair[{idx}] system_prompt is a duplicate of an earlier pair in this response."
            )
        seen_prompts.add(sys_prompt)

        # cross_lingual_robustness_strategy must be present and non-empty
        clr_strategy = pair.get("cross_lingual_robustness_strategy", "")
        if not clr_strategy or not str(clr_strategy).strip():
            raise ParseError(
                f"Pair[{idx}] cross_lingual_robustness_strategy is missing or empty."
            )

        # Normalize negative_prompt
        norm = dict(pair)
        norm["negative_prompt"] = normalize_negative_prompt(pair.get("negative_prompt"))
        normalized.append(norm)

    return normalized
