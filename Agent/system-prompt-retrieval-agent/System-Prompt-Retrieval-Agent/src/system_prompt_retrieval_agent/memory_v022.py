"""V0.2.2 long-memory wiring (S07.01–S07.10).

Extends the existing 0.2.1 long-memory schema with ``run_id`` (S07.01),
implements one-time legacy migration that stamps pre-V0.2.2 rows with
``run_id="legacy:v021"`` (S07.01a), and provides idempotent
upsert / archive / atomic-write semantics keyed by
``(run_id, round_id, prompt_pair_id)`` (S07.03–S07.05).

Readers must treat ``run_id="legacy:v021"`` as a sentinel: visible to
prompt-generation context but never upserted, archived, or
deduplicated against new V0.2.2 rows (S07.01b).
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .memory.long import FIELDNAMES as _LEGACY_FIELDNAMES
from .memory.long import flatten_pair_to_row as _legacy_flatten_pair_to_row
from .remote._vendored import canonical_paths as cp


# ---------------------------------------------------------------------------
# V0.2.2 schema
# ---------------------------------------------------------------------------

V022_SCHEMA_VERSION: str = "0.2.2"
V022_RUN_ID_COL: str = "run_id"
V022_ROUND_ID_COL: str = "round_id"
V022_ARCHIVE_COL: str = "archived_by_run_id"

LEGACY_SENTINEL: str = cp.LEGACY_LONG_MEMORY_RUN_ID  # "legacy:v021"


def _v022_fieldnames() -> list[str]:
    """Insert run_id + round_id at the front (after schema_version) and
    append archived_by_run_id at the end. Preserves legacy column order
    for everything else."""
    head = ["schema_version", V022_RUN_ID_COL, V022_ROUND_ID_COL]
    middle = [c for c in _LEGACY_FIELDNAMES if c not in head]
    return head + middle + [V022_ARCHIVE_COL]


V022_FIELDNAMES: list[str] = _v022_fieldnames()


# ---------------------------------------------------------------------------
# Row construction (S07.02)
# ---------------------------------------------------------------------------


def flatten_pair_to_row_v022(pair: Any, *, run_id: str, round_id: int) -> dict[str, Any]:
    base = _legacy_flatten_pair_to_row(pair)
    base[V022_RUN_ID_COL] = run_id
    base[V022_ROUND_ID_COL] = round_id
    base["schema_version"] = V022_SCHEMA_VERSION
    base.setdefault(V022_ARCHIVE_COL, None)
    # Ensure every V0.2.2 column is present (missing ones default to None).
    for col in V022_FIELDNAMES:
        base.setdefault(col, None)
    return base


# ---------------------------------------------------------------------------
# Atomic write helpers (S07.05)
# ---------------------------------------------------------------------------


def _atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".lm_", suffix=".csv", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(fieldnames), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


# ---------------------------------------------------------------------------
# Read + legacy migration (S07.01a / S07.01b)
# ---------------------------------------------------------------------------


def read_long_memory(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """Read the long-memory CSV. Returns (fieldnames, rows). Empty when
    the file does not exist."""
    if not path.is_file():
        return list(V022_FIELDNAMES), []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(r) for r in reader]
        fieldnames = reader.fieldnames or list(V022_FIELDNAMES)
    return list(fieldnames), rows


def migrate_legacy_rows_in_place(path: Path) -> int:
    """One-time migration (S07.01a). Idempotent.

    For every row missing ``run_id``, set ``run_id=LEGACY_SENTINEL`` and
    promote the schema to V0.2.2. Already-stamped rows untouched.
    Returns the number of rows touched.
    """
    if not path.is_file():
        return 0
    fieldnames, rows = read_long_memory(path)
    touched = 0
    out_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row.get(V022_RUN_ID_COL):
            row[V022_RUN_ID_COL] = LEGACY_SENTINEL
            row[V022_ROUND_ID_COL] = row.get(V022_ROUND_ID_COL) or row.get("round")
            row["schema_version"] = V022_SCHEMA_VERSION
            row.setdefault(V022_ARCHIVE_COL, None)
            touched += 1
        for col in V022_FIELDNAMES:
            row.setdefault(col, None)
        out_rows.append(row)
    if touched:
        _atomic_write_csv(path, V022_FIELDNAMES, out_rows)
    return touched


def filter_out_legacy_for_writes(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return rows that are NOT the legacy sentinel — used by upsert /
    archive paths so legacy rows are never re-written or deduplicated
    against new V0.2.2 rows (S07.01b)."""
    return [dict(r) for r in rows if r.get(V022_RUN_ID_COL) != LEGACY_SENTINEL]


def filter_legacy_only(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(r) for r in rows if r.get(V022_RUN_ID_COL) == LEGACY_SENTINEL]


# ---------------------------------------------------------------------------
# Upsert + archive (S07.03 / S07.04)
# ---------------------------------------------------------------------------


def upsert_long_memory_rows(
    path: Path, new_rows: Sequence[Mapping[str, Any]]
) -> int:
    """Idempotent upsert keyed by ``(run_id, round_id, prompt_pair_id)``.

    Existing rows with the same key are *archived* (marked with
    ``archived_by_run_id`` set to the new row's ``run_id``) instead of
    being deleted, then the new row is appended. Legacy sentinel rows
    are preserved as-is.
    """
    _, existing = read_long_memory(path)
    legacy = filter_legacy_only(existing)
    non_legacy = filter_out_legacy_for_writes(existing)

    new_keys: set[tuple[str, str, str]] = {
        (r[V022_RUN_ID_COL], str(r[V022_ROUND_ID_COL]), r["prompt_pair_id"])
        for r in new_rows
    }
    archived: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    for row in non_legacy:
        key = (
            row.get(V022_RUN_ID_COL),
            str(row.get(V022_ROUND_ID_COL)),
            row.get("prompt_pair_id"),
        )
        if key in new_keys and not row.get(V022_ARCHIVE_COL):
            row = dict(row)
            row[V022_ARCHIVE_COL] = row.get(V022_RUN_ID_COL)  # superseded by same run
            archived.append(row)
        else:
            kept.append(row)

    final_rows = legacy + kept + archived + [dict(r) for r in new_rows]
    _atomic_write_csv(path, V022_FIELDNAMES, final_rows)
    return len(archived) + len(new_rows)


# ---------------------------------------------------------------------------
# Reader resilience (S07.06)
# ---------------------------------------------------------------------------


def read_long_memory_strict(path: Path) -> list[dict[str, Any]]:
    """Read long-memory; raise on partial/corrupt rows.

    A row is rejected when:
      * required keys (``schema_version``, ``prompt_pair_id``,
        ``run_id`` for non-legacy schema versions) are missing.
      * any cell contains an unmatched newline / unterminated quote
        (csv.Error).
    """
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8")
    try:
        rows = list(csv.DictReader(io.StringIO(raw)))
    except csv.Error as exc:
        raise ValueError(f"long_memory.csv corrupt: {exc}") from exc
    out: list[dict[str, Any]] = []
    for r in rows:
        sv = r.get("schema_version", "")
        if not r.get("prompt_pair_id"):
            raise ValueError("long_memory row missing prompt_pair_id")
        if sv == V022_SCHEMA_VERSION and not r.get(V022_RUN_ID_COL):
            raise ValueError(
                f"long_memory v0.2.2 row missing run_id: {r.get('prompt_pair_id')}"
            )
        out.append(dict(r))
    return out


# ---------------------------------------------------------------------------
# AsyncOpenAI shim for prompt generation (S07.08)
# ---------------------------------------------------------------------------


@dataclass
class AsyncOpenAIBridge:
    """Lazy, optional shim that prefers ``AsyncOpenAI`` and falls back
    to running the sync SDK in a bounded executor.

    Tests inject ``client_factory`` to avoid importing the OpenAI SDK.
    """

    client_factory: Any = None

    def get_async_client(self):
        if self.client_factory is not None:
            return self.client_factory()
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover — defensive
            raise RuntimeError(
                "openai SDK missing; install openai>=1.50 to use the async bridge"
            ) from exc
        return AsyncOpenAI()
