"""Detached entry point for the ERA image-annotation web server.

Launched by :func:`era.orchestration.annotate.serve_annotate` as a separate,
session-detached process (``python -m era.annotate.server``); its parameters
arrive via ``ERA_ANNOTATE_*`` environment variables so the launcher keeps the
clean one-JSON-in / one-JSON-out CLI contract.
"""

from __future__ import annotations

import json
import os

import uvicorn

from .app import build_app


def _parse_overrides_env(name: str) -> dict[str, str] | None:
    """Decode a JSON-encoded env var into a {str: str} dict, or None."""
    raw = os.environ.get(name, "")
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def main() -> None:
    dataset_root = os.environ.get("ERA_ANNOTATE_DATASET_ROOT", "")
    host = os.environ.get("ERA_ANNOTATE_HOST", "127.0.0.1")
    port = int(os.environ.get("ERA_ANNOTATE_PORT", "8801"))
    if not dataset_root:
        raise SystemExit("ERA_ANNOTATE_DATASET_ROOT is not set")
    # Operator-confirmed overrides (set by serve-annotate when the slash
    # command asked the operator to disambiguate a needs_confirmation probe).
    output_overrides = _parse_overrides_env("ERA_ANNOTATE_OUTPUT_OVERRIDES")
    input_role_overrides = _parse_overrides_env(
        "ERA_ANNOTATE_INPUT_ROLE_OVERRIDES")
    app = build_app(
        dataset_root,
        output_overrides=output_overrides,
        input_role_overrides=input_role_overrides,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
