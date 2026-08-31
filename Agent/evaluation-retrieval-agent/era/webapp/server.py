"""Detached entry point for the ERA Stage 8 feedback web server.

Launched by ``era.orchestration.human_feedback.serve_feedback`` as a separate,
session-detached process (``python -m era.webapp.server``); its parameters
arrive via ``ERA_FEEDBACK_*`` environment variables so the launcher keeps the
clean one-JSON-in / one-JSON-out CLI contract.
"""

from __future__ import annotations

import os

import uvicorn

from .app import build_app


def main() -> None:
    workspace = os.environ.get("ERA_FEEDBACK_WORKSPACE", "")
    iteration = os.environ.get("ERA_FEEDBACK_ITERATION") or None
    host = os.environ.get("ERA_FEEDBACK_HOST", "127.0.0.1")
    port = int(os.environ.get("ERA_FEEDBACK_PORT", "8731"))
    if not workspace:
        raise SystemExit("ERA_FEEDBACK_WORKSPACE is not set")
    app = build_app(workspace, iteration=iteration)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
