"""failures.py — Append-only JSONL failure log per round."""
from __future__ import annotations

import json
from pathlib import Path


def append_failure(failures_jsonl: Path, record: dict) -> None:
    """Append one JSON-lines record to *failures_jsonl*."""
    failures_jsonl = Path(failures_jsonl)
    failures_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(failures_jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_failures_for_pair(failures_jsonl: Path, prompt_pair_id: str) -> list[dict]:
    """Return all failure records for *prompt_pair_id* from *failures_jsonl*."""
    all_records = read_all_failures(failures_jsonl)
    return [r for r in all_records if r.get("prompt_pair_id") == prompt_pair_id]


def read_all_failures(failures_jsonl: Path) -> list[dict]:
    """Return all failure records from *failures_jsonl*; [] if file missing."""
    failures_jsonl = Path(failures_jsonl)
    if not failures_jsonl.exists():
        return []
    records: list[dict] = []
    with open(failures_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records
