"""Tests for mcp_tools.stage_dispatch_mcp (S11a.03).

Covers:
- Schema accepts a valid V0.2.1 payload for each stage.
- Schema rejects sample_manifest_path set without sample_manifest_path_purpose.
- Schema rejects unknown user_prompt language values.
- Schema rejects user_prompt_corpus_hash with wrong length / non-hex chars.
- mcp_dispatch_* round-trips a valid payload to the corresponding pydantic request.
"""
from __future__ import annotations

import json
import copy
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from system_prompt_retrieval_agent.mcp_tools.stage_dispatch_mcp import (
    mcp_dispatch_flux,
    mcp_dispatch_gemma,
    mcp_dispatch_qwen,
    _GEMMA_SCHEMA,
    _FLUX_SCHEMA,
    _QWEN_SCHEMA,
)
from system_prompt_retrieval_agent.schemas import (
    FluxStageRequest,
    GemmaStageRequest,
    QwenStageRequest,
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

VALID_HASH = "a" * 64  # 64 hex chars — valid SHA-256 placeholder

_BASE_PAYLOAD: dict[str, Any] = {
    "run_id": "test-run-001",
    "round_id": 1,
    "prompt_pairs": [
        {
            "prompt_pair_id": "pp-01",
            "system_prompt_id": "sp-01",
        }
    ],
    "user_prompts": [
        {
            "user_prompt_id": "up-01",
            "language": "en",
            "text": "Edit the image.",
        }
    ],
    "user_prompt_corpus_hash": VALID_HASH,
    "dataset_root": "/data/dataset",
    "output_root": "/data/outputs",
    "lifecycle_mode": "cold",
}

_GEMMA_PAYLOAD: dict[str, Any] = {
    **_BASE_PAYLOAD,
    "prompt_pairs": [
        {
            "prompt_pair_id": "pp-01",
            "system_prompt_id": "sp-01",
            "system_prompt_text": "You are a helpful assistant.",
        }
    ],
}

_FLUX_PAYLOAD: dict[str, Any] = {
    **_BASE_PAYLOAD,
    "prompt_pairs": [
        {
            "prompt_pair_id": "pp-01",
            "system_prompt_id": "sp-01",
            "system_prompt_text": None,
        }
    ],
}

_QWEN_PAYLOAD: dict[str, Any] = {
    **_BASE_PAYLOAD,
    "prompt_pairs": [
        {
            "prompt_pair_id": "pp-01",
            "system_prompt_id": "sp-01",
            "system_prompt_text": None,
        }
    ],
}


def _validate(payload: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Return list of validation error messages (empty = valid)."""
    validator = jsonschema.Draft202012Validator(schema)
    return [e.message for e in validator.iter_errors(payload)]


# ---------------------------------------------------------------------------
# S11a.03-A: Schema accepts valid payloads
# ---------------------------------------------------------------------------


class TestSchemaAcceptsValidPayload:
    def test_gemma_valid(self) -> None:
        errors = _validate(_GEMMA_PAYLOAD, _GEMMA_SCHEMA)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_flux_valid(self) -> None:
        errors = _validate(_FLUX_PAYLOAD, _FLUX_SCHEMA)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_qwen_valid(self) -> None:
        errors = _validate(_QWEN_PAYLOAD, _QWEN_SCHEMA)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_gemma_with_manifest_and_purpose(self) -> None:
        payload = copy.deepcopy(_GEMMA_PAYLOAD)
        payload["sample_manifest_path"] = "/data/manifest.json"
        payload["sample_manifest_path_purpose"] = "resume_missing_cells"
        errors = _validate(payload, _GEMMA_SCHEMA)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_gemma_warm_lifecycle(self) -> None:
        payload = copy.deepcopy(_GEMMA_PAYLOAD)
        payload["lifecycle_mode"] = "warm"
        errors = _validate(payload, _GEMMA_SCHEMA)
        assert errors == [], f"Unexpected errors: {errors}"


# ---------------------------------------------------------------------------
# S11a.03-B: Schema rejects sample_manifest_path without purpose
# ---------------------------------------------------------------------------


class TestSchemaRejectsManifestWithoutPurpose:
    def test_gemma_manifest_without_purpose(self) -> None:
        payload = copy.deepcopy(_GEMMA_PAYLOAD)
        payload["sample_manifest_path"] = "/data/manifest.json"
        # sample_manifest_path_purpose is absent/null — should fail if/then
        errors = _validate(payload, _GEMMA_SCHEMA)
        assert errors, "Expected validation error when manifest set without purpose"

    def test_flux_manifest_without_purpose(self) -> None:
        payload = copy.deepcopy(_FLUX_PAYLOAD)
        payload["sample_manifest_path"] = "/data/manifest.json"
        errors = _validate(payload, _FLUX_SCHEMA)
        assert errors, "Expected validation error when manifest set without purpose"

    def test_qwen_manifest_without_purpose(self) -> None:
        payload = copy.deepcopy(_QWEN_PAYLOAD)
        payload["sample_manifest_path"] = "/data/manifest.json"
        errors = _validate(payload, _QWEN_SCHEMA)
        assert errors, "Expected validation error when manifest set without purpose"

    def test_gemma_manifest_with_null_purpose(self) -> None:
        payload = copy.deepcopy(_GEMMA_PAYLOAD)
        payload["sample_manifest_path"] = "/data/manifest.json"
        payload["sample_manifest_path_purpose"] = None
        errors = _validate(payload, _GEMMA_SCHEMA)
        assert errors, "Expected validation error when manifest set but purpose is null"


# ---------------------------------------------------------------------------
# S11a.03-C: Schema rejects unknown user_prompt language values
# ---------------------------------------------------------------------------


class TestSchemaRejectsUnknownLanguage:
    def test_gemma_unknown_language(self) -> None:
        payload = copy.deepcopy(_GEMMA_PAYLOAD)
        payload["user_prompts"][0]["language"] = "fr"
        errors = _validate(payload, _GEMMA_SCHEMA)
        assert errors, "Expected validation error for unknown language 'fr'"

    def test_flux_unknown_language(self) -> None:
        payload = copy.deepcopy(_FLUX_PAYLOAD)
        payload["user_prompts"][0]["language"] = "de"
        errors = _validate(payload, _FLUX_SCHEMA)
        assert errors, "Expected validation error for unknown language 'de'"

    def test_qwen_unknown_language(self) -> None:
        payload = copy.deepcopy(_QWEN_PAYLOAD)
        payload["user_prompts"][0]["language"] = "ja"
        errors = _validate(payload, _QWEN_SCHEMA)
        assert errors, "Expected validation error for unknown language 'ja'"


# ---------------------------------------------------------------------------
# S11a.03-D: Schema rejects invalid user_prompt_corpus_hash
# ---------------------------------------------------------------------------


class TestSchemaRejectsInvalidCorpusHash:
    def test_gemma_wrong_length(self) -> None:
        payload = copy.deepcopy(_GEMMA_PAYLOAD)
        payload["user_prompt_corpus_hash"] = "abc123"  # too short
        errors = _validate(payload, _GEMMA_SCHEMA)
        assert errors, "Expected validation error for short corpus hash"

    def test_flux_non_hex_chars(self) -> None:
        payload = copy.deepcopy(_FLUX_PAYLOAD)
        payload["user_prompt_corpus_hash"] = "z" * 64  # 'z' is not hex
        errors = _validate(payload, _FLUX_SCHEMA)
        assert errors, "Expected validation error for non-hex corpus hash"

    def test_qwen_63_chars(self) -> None:
        payload = copy.deepcopy(_QWEN_PAYLOAD)
        payload["user_prompt_corpus_hash"] = "a" * 63  # one char short
        errors = _validate(payload, _QWEN_SCHEMA)
        assert errors, "Expected validation error for 63-char hash"

    def test_gemma_65_chars(self) -> None:
        payload = copy.deepcopy(_GEMMA_PAYLOAD)
        payload["user_prompt_corpus_hash"] = "a" * 65  # one char too long
        errors = _validate(payload, _GEMMA_SCHEMA)
        assert errors, "Expected validation error for 65-char hash"

    def test_flux_uppercase_hex_accepted(self) -> None:
        payload = copy.deepcopy(_FLUX_PAYLOAD)
        payload["user_prompt_corpus_hash"] = "A" * 64  # uppercase hex is valid
        errors = _validate(payload, _FLUX_SCHEMA)
        assert errors == [], f"Uppercase hex should be accepted: {errors}"


# ---------------------------------------------------------------------------
# S11a.03-E: mcp_dispatch_* round-trip tests
# ---------------------------------------------------------------------------


class TestDispatchRoundTrip:
    def test_mcp_dispatch_gemma(self) -> None:
        result = mcp_dispatch_gemma(_GEMMA_PAYLOAD)
        assert result["stage"] == "gemma"
        assert result["run_id"] == "test-run-001"
        assert result["round_id"] == 1
        assert result["pair_count"] == 1
        assert isinstance(result["request"], GemmaStageRequest)

    def test_mcp_dispatch_flux(self) -> None:
        result = mcp_dispatch_flux(_FLUX_PAYLOAD)
        assert result["stage"] == "flux"
        assert result["run_id"] == "test-run-001"
        assert result["round_id"] == 1
        assert result["pair_count"] == 1
        assert isinstance(result["request"], FluxStageRequest)

    def test_mcp_dispatch_qwen(self) -> None:
        result = mcp_dispatch_qwen(_QWEN_PAYLOAD)
        assert result["stage"] == "qwen"
        assert result["run_id"] == "test-run-001"
        assert result["round_id"] == 1
        assert result["pair_count"] == 1
        assert isinstance(result["request"], QwenStageRequest)

    def test_mcp_dispatch_gemma_with_manifest(self) -> None:
        payload = copy.deepcopy(_GEMMA_PAYLOAD)
        payload["sample_manifest_path"] = "/data/manifest.json"
        payload["sample_manifest_path_purpose"] = "prior_stage_survivor_cells"
        result = mcp_dispatch_gemma(payload)
        assert result["stage"] == "gemma"
        req: GemmaStageRequest = result["request"]
        assert req.sample_manifest_path == "/data/manifest.json"
        assert req.sample_manifest_path_purpose == "prior_stage_survivor_cells"

    def test_mcp_dispatch_gemma_invalid_raises_value_error(self) -> None:
        payload = copy.deepcopy(_GEMMA_PAYLOAD)
        payload["user_prompt_corpus_hash"] = "not_a_hash"
        with pytest.raises(ValueError, match="schema validation failed"):
            mcp_dispatch_gemma(payload)

    def test_mcp_dispatch_flux_invalid_raises_value_error(self) -> None:
        payload = copy.deepcopy(_FLUX_PAYLOAD)
        payload["sample_manifest_path"] = "/data/manifest.json"
        # Missing purpose — should fail JSON-Schema validation
        with pytest.raises(ValueError, match="schema validation failed"):
            mcp_dispatch_flux(payload)

    def test_mcp_dispatch_qwen_invalid_raises_value_error(self) -> None:
        payload = copy.deepcopy(_QWEN_PAYLOAD)
        payload["lifecycle_mode"] = "turbo"  # not in enum
        with pytest.raises(ValueError, match="schema validation failed"):
            mcp_dispatch_qwen(payload)

    def test_dispatch_gemma_corpus_hash_in_pydantic(self) -> None:
        """Pydantic model correctly stores the corpus hash from the round-trip."""
        result = mcp_dispatch_gemma(_GEMMA_PAYLOAD)
        req: GemmaStageRequest = result["request"]
        assert req.user_prompt_corpus_hash == VALID_HASH

    def test_dispatch_flux_no_system_prompt_text_on_pairs(self) -> None:
        """Flux pairs must have system_prompt_text = None after round-trip."""
        result = mcp_dispatch_flux(_FLUX_PAYLOAD)
        req: FluxStageRequest = result["request"]
        for pair in req.prompt_pairs:
            assert pair.system_prompt_text is None
