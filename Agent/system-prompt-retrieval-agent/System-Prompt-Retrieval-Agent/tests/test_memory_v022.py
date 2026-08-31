"""S07 tests for V0.2.2 long-memory wiring."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.memory_v022 import (
    AsyncOpenAIBridge,
    LEGACY_SENTINEL,
    V022_ARCHIVE_COL,
    V022_FIELDNAMES,
    V022_ROUND_ID_COL,
    V022_RUN_ID_COL,
    filter_legacy_only,
    filter_out_legacy_for_writes,
    flatten_pair_to_row_v022,
    migrate_legacy_rows_in_place,
    read_long_memory,
    read_long_memory_strict,
    upsert_long_memory_rows,
)


_RID = "20260426T010000Z-deadbeef"


def _legacy_csv(tmp_path: Path) -> Path:
    """Seed a long_memory.csv with a legacy 0.2.1 row missing run_id."""
    path = tmp_path / "long_memory.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        legacy_cols = [c for c in V022_FIELDNAMES if c not in (V022_RUN_ID_COL, V022_ARCHIVE_COL)]
        # Legacy file omits run_id and archived_by_run_id columns.
        writer = csv.DictWriter(fh, fieldnames=legacy_cols)
        writer.writeheader()
        writer.writerow({"schema_version": "0.2.1", "prompt_pair_id": "PP_OLD",
                         "round": 0, "overall_score": 0.7})
    return path


# ---------------------------------------------------------------------------
# S07.01 — schema includes run_id
# ---------------------------------------------------------------------------


def test_v022_fieldnames_includes_run_id_and_round_id_and_archive():
    assert V022_RUN_ID_COL in V022_FIELDNAMES
    assert V022_ROUND_ID_COL in V022_FIELDNAMES
    assert V022_ARCHIVE_COL in V022_FIELDNAMES
    # run_id appears just after schema_version
    assert V022_FIELDNAMES.index("schema_version") + 1 == V022_FIELDNAMES.index(V022_RUN_ID_COL)


# ---------------------------------------------------------------------------
# S07.02 — flatten_pair_to_row_v022 includes run_id + round_id
# ---------------------------------------------------------------------------


class _StubPair:
    prompt_pair_id = "PP1"
    system_prompt_id = "SP1"
    negative_prompt_id = None
    round_id = 3
    fallback = False
    scores = None


def test_flatten_pair_to_row_v022_includes_run_id_and_round_id():
    row = flatten_pair_to_row_v022(_StubPair(), run_id=_RID, round_id=3)
    assert row[V022_RUN_ID_COL] == _RID
    assert row[V022_ROUND_ID_COL] == 3
    assert row["schema_version"] == "0.2.2"
    assert row["prompt_pair_id"] == "PP1"


# ---------------------------------------------------------------------------
# S07.01a — one-time legacy migration
# ---------------------------------------------------------------------------


def test_migrate_legacy_rows_assigns_legacy_sentinel(tmp_path):
    path = _legacy_csv(tmp_path)
    touched = migrate_legacy_rows_in_place(path)
    assert touched == 1
    _, rows = read_long_memory(path)
    assert rows[0][V022_RUN_ID_COL] == LEGACY_SENTINEL
    assert rows[0]["schema_version"] == "0.2.2"


def test_migrate_legacy_is_idempotent(tmp_path):
    path = _legacy_csv(tmp_path)
    migrate_legacy_rows_in_place(path)
    second = migrate_legacy_rows_in_place(path)
    assert second == 0  # nothing to migrate the second time


def test_migrate_legacy_preserves_other_columns(tmp_path):
    path = _legacy_csv(tmp_path)
    migrate_legacy_rows_in_place(path)
    _, rows = read_long_memory(path)
    assert rows[0]["prompt_pair_id"] == "PP_OLD"
    assert rows[0]["overall_score"] == "0.7"  # csv reads as string


# ---------------------------------------------------------------------------
# S07.01b — legacy rows visible to readers, never re-upserted
# ---------------------------------------------------------------------------


def test_filter_out_legacy_for_writes_excludes_sentinel():
    rows = [
        {V022_RUN_ID_COL: LEGACY_SENTINEL, "prompt_pair_id": "OLD"},
        {V022_RUN_ID_COL: _RID, "prompt_pair_id": "NEW"},
    ]
    assert filter_out_legacy_for_writes(rows) == [
        {V022_RUN_ID_COL: _RID, "prompt_pair_id": "NEW"}
    ]


def test_filter_legacy_only_returns_only_sentinel():
    rows = [
        {V022_RUN_ID_COL: LEGACY_SENTINEL, "prompt_pair_id": "OLD"},
        {V022_RUN_ID_COL: _RID, "prompt_pair_id": "NEW"},
    ]
    assert filter_legacy_only(rows) == [
        {V022_RUN_ID_COL: LEGACY_SENTINEL, "prompt_pair_id": "OLD"}
    ]


# ---------------------------------------------------------------------------
# S07.03 / S07.04 / S07.05 — upsert with archive + atomic write
# ---------------------------------------------------------------------------


def test_upsert_archives_existing_row_under_same_key(tmp_path):
    path = tmp_path / "long_memory.csv"
    initial_row = flatten_pair_to_row_v022(_StubPair(), run_id=_RID, round_id=3)
    upsert_long_memory_rows(path, [initial_row])

    # Re-upsert under same key with mutated value
    second_pair = type(_StubPair)("X", (object,), {**_StubPair.__dict__, "fallback": True})()
    second_row = flatten_pair_to_row_v022(second_pair, run_id=_RID, round_id=3)
    upsert_long_memory_rows(path, [second_row])

    _, rows = read_long_memory(path)
    by_archived = [r for r in rows if r.get(V022_ARCHIVE_COL)]
    fresh = [r for r in rows if not r.get(V022_ARCHIVE_COL)]
    assert len(by_archived) == 1
    assert len(fresh) == 1
    assert fresh[0]["fallback"] == "True"


def test_upsert_idempotent_on_rerun(tmp_path):
    path = tmp_path / "long_memory.csv"
    row = flatten_pair_to_row_v022(_StubPair(), run_id=_RID, round_id=3)
    upsert_long_memory_rows(path, [row])
    upsert_long_memory_rows(path, [row])
    upsert_long_memory_rows(path, [row])
    _, rows = read_long_memory(path)
    fresh = [r for r in rows if not r.get(V022_ARCHIVE_COL)]
    assert len(fresh) == 1  # always exactly one fresh row for the key


def test_upsert_preserves_legacy_sentinel_rows(tmp_path):
    legacy_path = _legacy_csv(tmp_path)
    migrate_legacy_rows_in_place(legacy_path)
    new_row = flatten_pair_to_row_v022(_StubPair(), run_id=_RID, round_id=3)
    upsert_long_memory_rows(legacy_path, [new_row])
    _, rows = read_long_memory(legacy_path)
    legacy_rows = [r for r in rows if r.get(V022_RUN_ID_COL) == LEGACY_SENTINEL]
    new_rows = [r for r in rows if r.get(V022_RUN_ID_COL) == _RID]
    assert len(legacy_rows) == 1 and len(new_rows) == 1


# ---------------------------------------------------------------------------
# S07.06 — reader rejects partial/corrupt rows
# ---------------------------------------------------------------------------


def test_reader_strict_rejects_v022_row_missing_run_id(tmp_path):
    path = tmp_path / "long_memory.csv"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("schema_version,prompt_pair_id,run_id,round_id\n")
        fh.write("0.2.2,PP1,,5\n")
    with pytest.raises(ValueError, match="missing run_id"):
        read_long_memory_strict(path)


def test_reader_strict_rejects_row_missing_prompt_pair_id(tmp_path):
    path = tmp_path / "long_memory.csv"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("schema_version,prompt_pair_id,run_id\n")
        fh.write("0.2.2,,RUN\n")
    with pytest.raises(ValueError, match="missing prompt_pair_id"):
        read_long_memory_strict(path)


# ---------------------------------------------------------------------------
# S07.08 — async openai bridge
# ---------------------------------------------------------------------------


def test_async_openai_bridge_uses_injected_factory():
    sentinel = object()
    bridge = AsyncOpenAIBridge(client_factory=lambda: sentinel)
    assert bridge.get_async_client() is sentinel
