"""shared.py — Shared-rules CSV (cross-round learned rules)."""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = "1.0"

FIELDNAMES = [
    "schema_version",
    "rule_id",
    "text",
    "source_round",
    "confidence",
    "last_seen",
    "tags",
    # Linkage columns — let a rule resolve to the concrete prompt artifact it
    # was learned from. `prompt_path` is read by context_builder to inline
    # the real system_prompt text into the LLM-visible shared_rules block.
    "system_prompt_id",
    "prompt_pair_id",
    "source_run_id",
    "n_cells",
    "pair_overall",
    "prompt_path",
    # Actionable feedback for next-round prompt generation.
    "improvement_summary",       # 2-4 sentence "what to change" directive (gpt-5.4-mini)
    "axis_means_json",           # {qwen, edit, transfer, preservation, artifact}
    "worst_failure_tags",        # comma-list "tag:count, tag:count, ..."
    "worst_user_prompt_id",      # which of the 8 user phrasings failed worst
    "zh_mean_score",
    "en_mean_score",
    "delta_vs_prev",             # pair_overall change vs the previous round's row
]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_prompt_path(prompt_path: str | None, memory_root: Path) -> Path | None:
    """Resolve a ``prompt_path`` cell value to an absolute :class:`Path`.

    The CSV invariant is that ``prompt_path`` is persisted **relative** to
    ``memory_root`` (e.g. ``"pairs/<run_id>/<pid>.yaml"``) so the file is
    portable across machines. Callers that need to open the file resolve it
    through this helper at use-time. Already-absolute paths (legacy rows or
    explicit overrides) pass through unchanged.

    Returns ``None`` if ``prompt_path`` is empty/None so callers can branch
    on a single check.
    """
    if not prompt_path:
        return None
    p = Path(prompt_path)
    return p if p.is_absolute() else memory_root / p


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def read_rules(csv_path: Path) -> list[dict]:
    """Return all rows from *csv_path*.  Returns [] if file missing."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


def write_rules(csv_path: Path, rows: list[dict]) -> None:
    """Atomically write *rows* to *csv_path* (full replace)."""
    csv_path = Path(csv_path)
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, csv_path)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_rules(
    rows: list[dict],
    current_round: int,
    min_confidence: float = 0.5,
    max_staleness: int = 3,
) -> tuple[list[dict], list[dict]]:
    """Drop rows where last_seen < current_round - max_staleness AND confidence < min_confidence.

    Returns (kept, evicted).
    """
    kept: list[dict] = []
    evicted: list[dict] = []
    staleness_cutoff = current_round - max_staleness
    for row in rows:
        try:
            confidence = float(row.get("confidence", 1.0))
            last_seen = int(row.get("last_seen", current_round))
        except (ValueError, TypeError):
            kept.append(row)
            continue
        if last_seen < staleness_cutoff and confidence < min_confidence:
            evicted.append(row)
        else:
            kept.append(row)
    return kept, evicted


# ---------------------------------------------------------------------------
# Markdown digest
# ---------------------------------------------------------------------------

def write_markdown_digest(md_path: Path, rows: list[dict]) -> None:
    """Write a bulleted list of rules grouped by tag to *md_path*."""
    md_path = Path(md_path)
    # Group by tags (comma-separated string in the CSV)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        tags_raw = row.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else ["untagged"]
        for tag in tags:
            groups.setdefault(tag, []).append(row)

    lines: list[str] = ["# Shared Rules Digest\n"]
    for tag in sorted(groups):
        lines.append(f"\n## {tag}\n")
        for row in groups[tag]:
            text = row.get("text", "")
            confidence = row.get("confidence", "")
            lines.append(f"- [{confidence}] {text}")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
