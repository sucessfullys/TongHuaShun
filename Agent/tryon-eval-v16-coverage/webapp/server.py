"""In-process uvicorn entry for the standalone review web app."""
from __future__ import annotations

from pathlib import Path


def serve(run_dir: str | Path, host: str = "0.0.0.0", port: int = 8731) -> None:
    import uvicorn

    from .app import build_app

    app = build_app(run_dir)
    uvicorn.run(app, host=host, port=port, log_level="info")
