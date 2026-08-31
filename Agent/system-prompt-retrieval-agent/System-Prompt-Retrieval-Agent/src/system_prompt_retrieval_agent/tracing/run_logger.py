from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .redaction import redact_secrets


class RunTraceLogger:
    """Per-run JSONL trace. All writes redacted."""

    def __init__(self, trace_path: Path, env_keys: list[str] | None = None) -> None:
        self.path = Path(trace_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.env_keys = list(env_keys or ["OPENAI_API_KEY", "Google_API_KEY"])

    def emit(self, event: str, **fields: Any) -> None:
        record = {"ts": datetime.utcnow().isoformat() + "Z", "event": event, **fields}
        safe = redact_secrets(record, env_keys=self.env_keys)
        with self.path.open("a") as fh:
            fh.write(json.dumps(safe, default=str) + "\n")
