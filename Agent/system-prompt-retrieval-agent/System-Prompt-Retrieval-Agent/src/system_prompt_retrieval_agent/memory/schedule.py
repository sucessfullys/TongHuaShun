"""schedule.py — Read/write the run-schedule JSON file atomically."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "1.0"


def default_schedule(max_rounds: int = 6) -> dict:
    """Return an initial schedule document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "current_round": 0,
        "max_rounds": max_rounds,
        "status": "in_progress",
        "fallback_prompt_generation": False,
        "consecutive_fallback_rounds": 0,
        "blockers": [],
        "stop_reason": None,
        "next_action": "generate_prompts",
        "last_updated": _now_iso(),
        "history": [],
    }


def read_schedule(path: Path) -> dict:
    """Read schedule JSON from *path*. Raises FileNotFoundError if absent."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_schedule(path: Path, data: dict) -> None:
    """Atomically write *data* to *path* via a sibling .tmp file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
