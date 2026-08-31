"""MCP tool wrapper for V0.2.1 per-stage dispatch.

Exposes three callable tool functions:
    mcp_dispatch_gemma   -- validates and constructs GemmaStageRequest
    mcp_dispatch_flux    -- validates and constructs FluxStageRequest
    mcp_dispatch_qwen    -- validates and constructs QwenStageRequest

Each function:
  1. Validates the incoming kwargs dict against the corresponding JSON-Schema
     (from mcp_tools/schemas/).
  2. Constructs the corresponding Pydantic stage-request model to catch any
     Python-level post-init invariants (e.g. missing manifest purpose).
  3. Returns a dispatch-metadata dict.  Actual remote I/O is the caller's
     responsibility — this module never performs network calls.
"""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any

import jsonschema

from system_prompt_retrieval_agent.schemas import (
    FluxStageRequest,
    GemmaStageRequest,
    QwenStageRequest,
)

# ---------------------------------------------------------------------------
# Schema loading
# ---------------------------------------------------------------------------

_SCHEMA_DIR = Path(__file__).parent / "schemas"


def _load_schema(name: str) -> dict[str, Any]:
    """Load a JSON schema from the schemas/ directory by file basename."""
    schema_path = _SCHEMA_DIR / name
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


_GEMMA_SCHEMA: dict[str, Any] = _load_schema("gemma_stage_request.json")
_FLUX_SCHEMA: dict[str, Any] = _load_schema("flux_stage_request.json")
_QWEN_SCHEMA: dict[str, Any] = _load_schema("qwen_stage_request.json")

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

_VALIDATOR_CLS = jsonschema.Draft202012Validator


def _validate(payload: dict[str, Any], schema: dict[str, Any], stage: str) -> None:
    """Validate *payload* against *schema*.  Raises ValueError on failure."""
    validator = _VALIDATOR_CLS(schema)
    errors = list(validator.iter_errors(payload))
    if errors:
        # Collect up to 3 error messages for a concise report.
        msgs = [e.message for e in errors[:3]]
        raise ValueError(
            f"MCP schema validation failed for stage '{stage}': "
            + "; ".join(msgs)
        )


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------


def mcp_dispatch_gemma(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate *kwargs* as a GemmaStageRequest and return dispatch metadata.

    Parameters
    ----------
    kwargs:
        Raw tool-call arguments dict.

    Returns
    -------
    dict with keys:
        stage       -- "gemma"
        run_id      -- copied from the validated request
        round_id    -- copied from the validated request
        pair_count  -- number of prompt pairs
        request     -- the constructed GemmaStageRequest instance
    """
    _validate(kwargs, _GEMMA_SCHEMA, "gemma")
    request = GemmaStageRequest(**kwargs)
    return {
        "stage": "gemma",
        "run_id": request.run_id,
        "round_id": request.round_id,
        "pair_count": len(request.prompt_pairs),
        "request": request,
    }


def mcp_dispatch_flux(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate *kwargs* as a FluxStageRequest and return dispatch metadata.

    Parameters
    ----------
    kwargs:
        Raw tool-call arguments dict.

    Returns
    -------
    dict with keys:
        stage       -- "flux"
        run_id      -- copied from the validated request
        round_id    -- copied from the validated request
        pair_count  -- number of prompt pairs
        request     -- the constructed FluxStageRequest instance
    """
    _validate(kwargs, _FLUX_SCHEMA, "flux")
    request = FluxStageRequest(**kwargs)
    return {
        "stage": "flux",
        "run_id": request.run_id,
        "round_id": request.round_id,
        "pair_count": len(request.prompt_pairs),
        "request": request,
    }


def mcp_dispatch_qwen(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Validate *kwargs* as a QwenStageRequest and return dispatch metadata.

    Parameters
    ----------
    kwargs:
        Raw tool-call arguments dict.

    Returns
    -------
    dict with keys:
        stage       -- "qwen"
        run_id      -- copied from the validated request
        round_id    -- copied from the validated request
        pair_count  -- number of prompt pairs
        request     -- the constructed QwenStageRequest instance
    """
    _validate(kwargs, _QWEN_SCHEMA, "qwen")
    request = QwenStageRequest(**kwargs)
    return {
        "stage": "qwen",
        "run_id": request.run_id,
        "round_id": request.round_id,
        "pair_count": len(request.prompt_pairs),
        "request": request,
    }
