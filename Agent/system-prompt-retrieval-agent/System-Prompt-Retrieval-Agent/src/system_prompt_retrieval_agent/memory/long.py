"""long.py — Long-term memory CSV + per-pair YAML store.

Schema version history:
  0.2   — legacy; dress_weighted / lower_weighted / upper_weighted as category cols.
  0.2.1 — canonical; full per-category sub-score columns + 7 user-prompt-aware cols.
  1.0   — same as 0.2 (internal draft name, treated as alias for 0.2).
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "0.2.1"

# Sub-score field names (shared across top-level and per-category).
_SUB_SCORE_KEYS = [
    "qwen_pass_rate",
    "edit_correctness",
    "garment_transfer_correctness",
    "preservation",
    "artifact_penalty",
]

# Category names.
_CATEGORIES = ["dress", "lower", "upper"]

# 7 user-prompt-aware columns added in V0.2.1.
_USER_PROMPT_COLS = [
    "zh_mean_score",
    "en_mean_score",
    "prompt_sensitivity",
    "cross_lingual_gap",
    "worst_user_prompt_id",
    "worst_user_prompt_score",
    "language_brittle",
]


def _build_fieldnames() -> list[str]:
    """Return the canonical 0.2.1 column ordering per plan §9.1."""
    cols: list[str] = [
        "schema_version",
        "prompt_pair_id",
        "system_prompt_id",
        "negative_prompt_id",
        "round",
        "overall_score",
        "total_score",
    ]
    # Top-level sub-scores.
    cols.extend(_SUB_SCORE_KEYS)
    # Per-category blocks.
    for cat in _CATEGORIES:
        cols.append(f"{cat}_weighted_score")
        cols.append(f"{cat}_total_score")
        for k in _SUB_SCORE_KEYS:
            cols.append(f"{cat}_{k}")
        cols.append(f"{cat}_missing_score_reason")
    # Tail: top-level missing reason, fallback, timestamp, user-prompt-aware.
    cols.append("missing_score_reason")
    cols.append("fallback")
    cols.append("timestamp")
    cols.extend(_USER_PROMPT_COLS)
    return cols


FIELDNAMES: list[str] = _build_fieldnames()

# Legacy column aliases: old name → canonical name (read-only; never written back).
_LEGACY_ALIASES: dict[str, str] = {
    "dress_weighted": "dress_weighted_score",
    "lower_weighted": "lower_weighted_score",
    "upper_weighted": "upper_weighted_score",
}

# Columns that should exist in legacy rows (0.2 / 1.0) but are absent in legacy CSV.
_LEGACY_MISSING_COLS = set(FIELDNAMES) - {
    "schema_version", "prompt_pair_id", "system_prompt_id", "negative_prompt_id",
    "round", "overall_score", "total_score",
    *_SUB_SCORE_KEYS,
    "dress_weighted_score", "lower_weighted_score", "upper_weighted_score",
    "missing_score_reason", "fallback", "timestamp",
}


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class LongMemoryConsistencyError(Exception):
    """Raised by verify_long_memory_consistency() on score mismatch."""


# ---------------------------------------------------------------------------
# Row flattening
# ---------------------------------------------------------------------------

def flatten_pair_to_row(pair: Any) -> Dict[str, Any]:
    """Flatten a PromptPair (with PromptPairScoreContext) into a canonical CSV row.

    - Emits ``null`` (Python ``None``) for absent fields.
    - Populates ``{col}_missing_score_reason`` when a block is absent.
    - Never raises on partial scores.
    """
    row: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "prompt_pair_id": pair.prompt_pair_id,
        "system_prompt_id": pair.system_prompt_id,
        "negative_prompt_id": pair.negative_prompt_id,
        "round": pair.round_id,
        "fallback": pair.fallback,
        "timestamp": _utc_now_iso(),
    }

    scores = getattr(pair, "scores", None)

    # ------------------------------------------------------------------
    # No scores at all
    # ------------------------------------------------------------------
    if scores is None:
        for col in FIELDNAMES:
            if col not in row:
                row[col] = None
        row["missing_score_reason"] = "scores_missing"
        for cat in _CATEGORIES:
            row[f"{cat}_missing_score_reason"] = "scores_missing"
        return row

    # ------------------------------------------------------------------
    # Top-level scores
    # ------------------------------------------------------------------
    row["overall_score"] = scores.overall_score
    row["total_score"] = scores.total_score
    row["missing_score_reason"] = scores.missing_score_reason

    sub = getattr(scores, "sub_scores", None)
    for k in _SUB_SCORE_KEYS:
        row[k] = getattr(sub, k, None) if sub is not None else None

    # ------------------------------------------------------------------
    # Per-category scores
    # ------------------------------------------------------------------
    cat_scores = getattr(scores, "category_scores", None) or {}
    for cat in _CATEGORIES:
        cs = cat_scores.get(cat) if cat_scores else None
        if cs is None:
            row[f"{cat}_weighted_score"] = None
            row[f"{cat}_total_score"] = None
            for k in _SUB_SCORE_KEYS:
                row[f"{cat}_{k}"] = None
            row[f"{cat}_missing_score_reason"] = "category_scores_missing"
        else:
            row[f"{cat}_weighted_score"] = cs.weighted_score
            row[f"{cat}_total_score"] = cs.total_score
            row[f"{cat}_missing_score_reason"] = cs.missing_score_reason
            cat_sub = getattr(cs, "sub_scores", None)
            for k in _SUB_SCORE_KEYS:
                row[f"{cat}_{k}"] = getattr(cat_sub, k, None) if cat_sub is not None else None

    # ------------------------------------------------------------------
    # User-prompt-aware columns (V0.2.1)
    # ------------------------------------------------------------------
    row["zh_mean_score"] = getattr(scores, "zh_mean_score", None)
    row["en_mean_score"] = getattr(scores, "en_mean_score", None)
    row["prompt_sensitivity"] = getattr(scores, "prompt_sensitivity", None)
    row["cross_lingual_gap"] = getattr(scores, "cross_lingual_gap", None)
    row["worst_user_prompt_id"] = getattr(scores, "worst_user_prompt_id", None)
    row["worst_user_prompt_score"] = getattr(scores, "worst_user_prompt_score", None)

    # language_brittle: derive from cross_lingual_gap > 0.10 if not directly available;
    # but PromptPairScoreContext does not carry it directly — it lives on RoundState.
    # Store None unless the caller has already set it on scores (not part of schema).
    row["language_brittle"] = None

    return row


def _utc_now_iso() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def append_long_memory_row(csv_path: Path, row: dict) -> None:
    """Append *row* to *csv_path*; writes header if file is new/empty."""
    csv_path = Path(csv_path)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def read_long_memory(csv_path: Path) -> list[dict]:
    """Return all rows from *csv_path*; returns [] if file missing.

    Applies legacy alias mapping: ``dress_weighted`` → ``dress_weighted_score``,
    ``lower_weighted`` → ``lower_weighted_score``, ``upper_weighted`` → ``upper_weighted_score``.
    New columns absent in legacy rows are populated as empty strings (CSV convention).
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row = dict(raw)
            # Apply legacy aliases (read-only translation).
            for old_col, new_col in _LEGACY_ALIASES.items():
                if old_col in row and new_col not in row:
                    row[new_col] = row[old_col]
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Pruning / archiving
# ---------------------------------------------------------------------------

def prune_top_n(rows: list[dict], n: int = 50) -> tuple[list[dict], list[dict]]:
    """Return (kept top-n by overall_score desc, evicted).

    Rows with missing/unparseable overall_score are treated as 0.0.
    """
    def _score(r: dict) -> float:
        try:
            return float(r.get("overall_score") or 0.0)
        except (ValueError, TypeError):
            return 0.0

    sorted_rows = sorted(rows, key=_score, reverse=True)
    kept = sorted_rows[:n]
    evicted = sorted_rows[n:]
    return kept, evicted


def archive_rows(archive_csv: Path, evicted_rows: list[dict]) -> None:
    """Append *evicted_rows* to *archive_csv*."""
    archive_csv = Path(archive_csv)
    if not evicted_rows:
        return
    needs_header = not archive_csv.exists() or archive_csv.stat().st_size == 0
    with open(archive_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerows(evicted_rows)


# ---------------------------------------------------------------------------
# Per-pair YAML
# ---------------------------------------------------------------------------

def write_pair_yaml(pairs_dir: Path, prompt_pair_id: str, data: dict) -> None:
    """Write *data* as YAML to ``pairs_dir / f"{prompt_pair_id}.yaml"``."""
    pairs_dir = Path(pairs_dir)
    pairs_dir.mkdir(parents=True, exist_ok=True)
    out = pairs_dir / f"{prompt_pair_id}.yaml"
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def read_pair_yaml(pairs_dir: Path, prompt_pair_id: str) -> dict | None:
    """Return YAML data for *prompt_pair_id* or None if not found."""
    path = Path(pairs_dir) / f"{prompt_pair_id}.yaml"
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate_long_memory_csv(memory_root: Path) -> None:
    """Rebuild long_memory.csv from memory/pairs/*.yaml.

    Algorithm:
      1. Write ``long_memory.csv.bak.<unix_ts>`` before touching the live CSV.
      2. Load all existing CSV rows into a lookup by prompt_pair_id (preserving
         prompt IDs, round, fallback, timestamp from legacy rows).
      3. For each *.yaml in pairs/, build the canonical 0.2.1 row (YAML wins for
         score columns; legacy CSV supplies prompt IDs / round / fallback / timestamp
         when YAML lacks them).
      4. Rows present only in the legacy CSV (no YAML) are migrated with alias
         mapping and ``null`` for new columns.
      5. Write the merged result back to long_memory.csv atomically.
    """
    memory_root = Path(memory_root)
    csv_path = memory_root / "long_memory.csv"
    pairs_dir = memory_root / "pairs"

    # Step 1: backup existing CSV.
    ts = int(time.time())
    if csv_path.exists() and csv_path.stat().st_size > 0:
        bak = csv_path.with_name(f"long_memory.csv.bak.{ts}")
        import shutil
        shutil.copy2(csv_path, bak)

    # Step 2: load legacy rows.
    legacy_rows: dict[str, dict] = {}
    for row in read_long_memory(csv_path):
        pid = row.get("prompt_pair_id", "")
        if pid:
            legacy_rows[pid] = row

    # Step 3: process YAML files.
    yaml_pair_ids: set[str] = set()
    new_rows: list[dict] = []

    if pairs_dir.exists():
        for yaml_path in sorted(pairs_dir.glob("*.yaml")):
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            pid = data.get("prompt_pair_id") or yaml_path.stem
            yaml_pair_ids.add(pid)

            # Pull persisted fields from legacy CSV when YAML doesn't have them.
            legacy = legacy_rows.get(pid, {})
            row: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "prompt_pair_id": pid,
                "system_prompt_id": data.get("system_prompt_id") or legacy.get("system_prompt_id", ""),
                "negative_prompt_id": data.get("negative_prompt_id") or legacy.get("negative_prompt_id", ""),
                "round": data.get("round_id") or data.get("round") or legacy.get("round", ""),
                "fallback": data.get("fallback", legacy.get("fallback", False)),
                "timestamp": data.get("timestamp") or legacy.get("timestamp") or _utc_now_iso(),
            }

            # Score columns from YAML (if present).
            scores_data = data.get("scores") or {}
            row["overall_score"] = scores_data.get("overall_score")
            row["total_score"] = scores_data.get("total_score")
            row["missing_score_reason"] = scores_data.get("missing_score_reason")

            sub_data = scores_data.get("sub_scores") or {}
            for k in _SUB_SCORE_KEYS:
                row[k] = sub_data.get(k)

            cat_scores_data = scores_data.get("category_scores") or {}
            for cat in _CATEGORIES:
                cs = cat_scores_data.get(cat) if cat_scores_data else None
                if cs is None:
                    row[f"{cat}_weighted_score"] = None
                    row[f"{cat}_total_score"] = None
                    for k in _SUB_SCORE_KEYS:
                        row[f"{cat}_{k}"] = None
                    row[f"{cat}_missing_score_reason"] = "category_scores_missing"
                else:
                    row[f"{cat}_weighted_score"] = cs.get("weighted_score")
                    row[f"{cat}_total_score"] = cs.get("total_score")
                    row[f"{cat}_missing_score_reason"] = cs.get("missing_score_reason")
                    cat_sub = cs.get("sub_scores") or {}
                    for k in _SUB_SCORE_KEYS:
                        row[f"{cat}_{k}"] = cat_sub.get(k)

            # User-prompt-aware columns.
            row["zh_mean_score"] = scores_data.get("zh_mean_score")
            row["en_mean_score"] = scores_data.get("en_mean_score")
            row["prompt_sensitivity"] = scores_data.get("prompt_sensitivity")
            row["cross_lingual_gap"] = scores_data.get("cross_lingual_gap")
            row["worst_user_prompt_id"] = scores_data.get("worst_user_prompt_id")
            row["worst_user_prompt_score"] = scores_data.get("worst_user_prompt_score")
            row["language_brittle"] = scores_data.get("language_brittle")

            new_rows.append(row)

    # Step 4: rows in legacy CSV with no YAML → migrate with nulls for new columns.
    for pid, row in legacy_rows.items():
        if pid not in yaml_pair_ids:
            migrated: Dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "prompt_pair_id": pid,
                "system_prompt_id": row.get("system_prompt_id", ""),
                "negative_prompt_id": row.get("negative_prompt_id", ""),
                "round": row.get("round", ""),
                "overall_score": row.get("overall_score"),
                "total_score": row.get("total_score"),
                "missing_score_reason": row.get("missing_score_reason"),
                "fallback": row.get("fallback", False),
                "timestamp": row.get("timestamp") or _utc_now_iso(),
            }
            for k in _SUB_SCORE_KEYS:
                migrated[k] = row.get(k)
            for cat in _CATEGORIES:
                # Use alias mapping (already applied by read_long_memory).
                migrated[f"{cat}_weighted_score"] = row.get(f"{cat}_weighted_score")
                migrated[f"{cat}_total_score"] = row.get(f"{cat}_total_score")
                for k in _SUB_SCORE_KEYS:
                    migrated[f"{cat}_{k}"] = row.get(f"{cat}_{k}")
                migrated[f"{cat}_missing_score_reason"] = row.get(f"{cat}_missing_score_reason")
            for col in _USER_PROMPT_COLS:
                migrated[col] = row.get(col)
            new_rows.append(migrated)

    # Step 5: atomic write.
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(new_rows)
    os.replace(tmp, csv_path)


# ---------------------------------------------------------------------------
# Consistency verification
# ---------------------------------------------------------------------------

def verify_long_memory_consistency(memory_root: Path) -> None:
    """Assert that each CSV row matches its per-pair YAML.

    Raises LongMemoryConsistencyError on any mismatch.
    Tolerance: <= 1e-6 for float comparisons.
    """
    memory_root = Path(memory_root)
    csv_path = memory_root / "long_memory.csv"
    pairs_dir = memory_root / "pairs"

    rows = read_long_memory(csv_path)
    csv_by_id: dict[str, dict] = {r.get("prompt_pair_id", ""): r for r in rows if r.get("prompt_pair_id")}

    for pid, csv_row in csv_by_id.items():
        yaml_path = pairs_dir / f"{pid}.yaml"
        if not yaml_path.exists():
            continue  # CSV-only row: no YAML to compare against.

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue

        scores_data = data.get("scores") or {}

        # Check top-level scores.
        for col in ("overall_score", "total_score"):
            _assert_match(pid, col, csv_row.get(col), scores_data.get(col))

        # Check sub-scores.
        sub_data = scores_data.get("sub_scores") or {}
        for k in _SUB_SCORE_KEYS:
            _assert_match(pid, k, csv_row.get(k), sub_data.get(k))

        # Check per-category scores.
        cat_scores_data = scores_data.get("category_scores") or {}
        for cat in _CATEGORIES:
            cs = cat_scores_data.get(cat) if cat_scores_data else None
            if cs is None:
                # Expect null + missing_score_reason in CSV.
                for col in [f"{cat}_weighted_score", f"{cat}_total_score"] + [f"{cat}_{k}" for k in _SUB_SCORE_KEYS]:
                    csv_val = csv_row.get(col)
                    if csv_val not in (None, "", "None"):
                        raise LongMemoryConsistencyError(
                            f"pair {pid}: expected null for {col} (category absent in YAML) "
                            f"but CSV has {csv_val!r}"
                        )
            else:
                _assert_match(pid, f"{cat}_weighted_score", csv_row.get(f"{cat}_weighted_score"), cs.get("weighted_score"))
                _assert_match(pid, f"{cat}_total_score", csv_row.get(f"{cat}_total_score"), cs.get("total_score"))
                cat_sub = cs.get("sub_scores") or {}
                for k in _SUB_SCORE_KEYS:
                    _assert_match(pid, f"{cat}_{k}", csv_row.get(f"{cat}_{k}"), cat_sub.get(k))

        # Check user-prompt-aware columns.
        for col in ["zh_mean_score", "en_mean_score", "prompt_sensitivity",
                    "cross_lingual_gap", "worst_user_prompt_score"]:
            _assert_match(pid, col, csv_row.get(col), scores_data.get(col))


def _assert_match(pair_id: str, col: str, csv_val: Any, yaml_val: Any) -> None:
    """Raise LongMemoryConsistencyError if csv_val and yaml_val differ beyond 1e-6."""
    # Both absent.
    if csv_val in (None, "", "None") and yaml_val is None:
        return
    # Both present as numerics.
    if csv_val not in (None, "", "None") and yaml_val is not None:
        try:
            if abs(float(csv_val) - float(yaml_val)) <= 1e-6:
                return
        except (ValueError, TypeError):
            if str(csv_val) == str(yaml_val):
                return
    # Mismatch.
    raise LongMemoryConsistencyError(
        f"pair {pair_id}: column {col!r} mismatch — CSV={csv_val!r} YAML={yaml_val!r}"
    )
