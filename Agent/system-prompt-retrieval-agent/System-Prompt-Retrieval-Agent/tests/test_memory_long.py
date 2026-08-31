"""S02.08 — Tests for memory.long: CSV round-trip, YAML round-trip, prune_top_n.

V0.2.1 additions:
  (a) canonical 0.2.1 columns present.
  (b) category_scores={} writes null + missing_score_reason.
  (c) full dress/lower/upper category scores round-trip.
  (d) legacy 0.2 alias read.
  (e) migrate_long_memory_csv.
  (f) verify_long_memory_consistency raises on mismatch.
  Round-trip with 7 user-prompt-aware columns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest
import yaml

from system_prompt_retrieval_agent.memory.long import (
    FIELDNAMES,
    SCHEMA_VERSION,
    LongMemoryConsistencyError,
    append_long_memory_row,
    archive_rows,
    flatten_pair_to_row,
    migrate_long_memory_csv,
    prune_top_n,
    read_long_memory,
    read_pair_yaml,
    verify_long_memory_consistency,
    write_pair_yaml,
)


def _make_row(pair_id: str, round_: int, score: float) -> dict:
    return {
        "schema_version": "1.0",
        "prompt_pair_id": pair_id,
        "system_prompt_id": f"sp-{pair_id}",
        "negative_prompt_id": "neg-001",
        "round": str(round_),
        "overall_score": str(score),
        "total_score": str(score),
        "qwen_pass_rate": "0.8",
        "edit_correctness": "0.7",
        "garment_transfer_correctness": "0.6",
        "preservation": "0.9",
        "artifact_penalty": "0.1",
        "dress_weighted": "0.5",
        "lower_weighted": "0.4",
        "upper_weighted": "0.3",
        "fallback": "False",
        "timestamp": "2026-04-25T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------

def test_append_creates_header(tmp_path: Path):
    csv_path = tmp_path / "long.csv"
    row = _make_row("pair-001", 1, 0.75)
    append_long_memory_row(csv_path, row)

    assert csv_path.exists()
    lines = csv_path.read_text().splitlines()
    assert lines[0].startswith("schema_version")


def test_append_multiple_rows(tmp_path: Path):
    csv_path = tmp_path / "long.csv"
    for i in range(3):
        append_long_memory_row(csv_path, _make_row(f"pair-{i:03d}", i, 0.5 + i * 0.1))

    rows = read_long_memory(csv_path)
    assert len(rows) == 3


def test_read_returns_correct_values(tmp_path: Path):
    csv_path = tmp_path / "long.csv"
    row = _make_row("pair-xyz", 2, 0.88)
    append_long_memory_row(csv_path, row)

    rows = read_long_memory(csv_path)
    assert rows[0]["prompt_pair_id"] == "pair-xyz"
    assert rows[0]["overall_score"] == "0.88"


def test_read_empty_file_returns_list(tmp_path: Path):
    csv_path = tmp_path / "long.csv"
    assert read_long_memory(csv_path) == []


# ---------------------------------------------------------------------------
# YAML round-trip
# ---------------------------------------------------------------------------

def test_write_and_read_pair_yaml(tmp_path: Path):
    pairs_dir = tmp_path / "pairs"
    data = {"system_prompt": "hello", "round_id": 1}
    write_pair_yaml(pairs_dir, "pair-001", data)

    loaded = read_pair_yaml(pairs_dir, "pair-001")
    assert loaded == data


def test_read_pair_yaml_missing_returns_none(tmp_path: Path):
    pairs_dir = tmp_path / "pairs"
    pairs_dir.mkdir()
    assert read_pair_yaml(pairs_dir, "nonexistent") is None


def test_write_pair_yaml_creates_dir(tmp_path: Path):
    pairs_dir = tmp_path / "nested" / "pairs"
    write_pair_yaml(pairs_dir, "pair-abc", {"x": 1})
    assert (pairs_dir / "pair-abc.yaml").exists()


# ---------------------------------------------------------------------------
# prune_top_n
# ---------------------------------------------------------------------------

def test_prune_top_n_keeps_highest_scores(tmp_path: Path):
    rows = [_make_row(f"pair-{i:03d}", 1, float(i) / 10) for i in range(10)]
    kept, evicted = prune_top_n(rows, n=3)
    assert len(kept) == 3
    assert len(evicted) == 7
    # Highest scores kept
    kept_scores = sorted([float(r["overall_score"]) for r in kept], reverse=True)
    assert kept_scores[0] == pytest.approx(0.9)
    assert kept_scores[1] == pytest.approx(0.8)
    assert kept_scores[2] == pytest.approx(0.7)


def test_prune_top_n_fewer_than_n(tmp_path: Path):
    rows = [_make_row("p1", 1, 0.5)]
    kept, evicted = prune_top_n(rows, n=10)
    assert len(kept) == 1
    assert evicted == []


def test_prune_top_n_empty():
    kept, evicted = prune_top_n([], n=5)
    assert kept == []
    assert evicted == []


def test_archive_rows_creates_file(tmp_path: Path):
    archive = tmp_path / "archive.csv"
    evicted = [_make_row("pair-old", 1, 0.3)]
    archive_rows(archive, evicted)
    assert archive.exists()
    rows = read_long_memory(archive)
    assert len(rows) == 1
    assert rows[0]["prompt_pair_id"] == "pair-old"


def test_archive_rows_appends(tmp_path: Path):
    archive = tmp_path / "archive.csv"
    archive_rows(archive, [_make_row("p1", 1, 0.3)])
    archive_rows(archive, [_make_row("p2", 2, 0.2)])
    rows = read_long_memory(archive)
    assert len(rows) == 2


# ===========================================================================
# V0.2.1 tests
# ===========================================================================

# ---------------------------------------------------------------------------
# Helpers: stub objects to simulate PromptPair / PromptPairScoreContext.
# ---------------------------------------------------------------------------

class _SubScores:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _CategoryScore:
    def __init__(self, weighted_score=None, total_score=None,
                 sub_scores=None, missing_score_reason=None):
        self.weighted_score = weighted_score
        self.total_score = total_score
        self.sub_scores = sub_scores or _SubScores(
            qwen_pass_rate=None, edit_correctness=None,
            garment_transfer_correctness=None, preservation=None, artifact_penalty=None,
        )
        self.missing_score_reason = missing_score_reason


class _Scores:
    def __init__(self, overall_score=None, total_score=None, sub_scores=None,
                 category_scores=None, missing_score_reason=None,
                 zh_mean_score=None, en_mean_score=None, prompt_sensitivity=None,
                 cross_lingual_gap=None, worst_user_prompt_id=None,
                 worst_user_prompt_score=None):
        self.overall_score = overall_score
        self.total_score = total_score
        self.sub_scores = sub_scores or _SubScores(
            qwen_pass_rate=None, edit_correctness=None,
            garment_transfer_correctness=None, preservation=None, artifact_penalty=None,
        )
        self.category_scores = category_scores if category_scores is not None else {}
        self.missing_score_reason = missing_score_reason
        self.zh_mean_score = zh_mean_score
        self.en_mean_score = en_mean_score
        self.prompt_sensitivity = prompt_sensitivity
        self.cross_lingual_gap = cross_lingual_gap
        self.worst_user_prompt_id = worst_user_prompt_id
        self.worst_user_prompt_score = worst_user_prompt_score


class _Pair:
    def __init__(self, prompt_pair_id, system_prompt_id="sp-001",
                 negative_prompt_id="neg-001", round_id=1, fallback=False, scores=None):
        self.prompt_pair_id = prompt_pair_id
        self.system_prompt_id = system_prompt_id
        self.negative_prompt_id = negative_prompt_id
        self.round_id = round_id
        self.fallback = fallback
        self.scores = scores


def _full_sub_scores(**overrides) -> _SubScores:
    defaults = dict(qwen_pass_rate=0.8, edit_correctness=0.7,
                    garment_transfer_correctness=0.6, preservation=0.9,
                    artifact_penalty=0.1)
    defaults.update(overrides)
    return _SubScores(**defaults)


def _full_category_score(weight=0.5, total=0.6) -> _CategoryScore:
    return _CategoryScore(
        weighted_score=weight,
        total_score=total,
        sub_scores=_full_sub_scores(),
        missing_score_reason=None,
    )


# ---------------------------------------------------------------------------
# Test (a): canonical 0.2.1 columns present
# ---------------------------------------------------------------------------

_EXPECTED_TOP_LEVEL = [
    "schema_version", "prompt_pair_id", "system_prompt_id", "negative_prompt_id",
    "round", "overall_score", "total_score",
    "qwen_pass_rate", "edit_correctness", "garment_transfer_correctness",
    "preservation", "artifact_penalty",
]
_EXPECTED_PER_CAT = [
    "{cat}_weighted_score", "{cat}_total_score",
    "{cat}_qwen_pass_rate", "{cat}_edit_correctness",
    "{cat}_garment_transfer_correctness", "{cat}_preservation",
    "{cat}_artifact_penalty", "{cat}_missing_score_reason",
]
_EXPECTED_TAIL = [
    "missing_score_reason", "fallback", "timestamp",
]
_EXPECTED_USER_PROMPT = [
    "zh_mean_score", "en_mean_score", "prompt_sensitivity",
    "cross_lingual_gap", "worst_user_prompt_id", "worst_user_prompt_score",
    "language_brittle",
]


def test_v021_canonical_columns_present():
    """Test (a): all canonical 0.2.1 column names are in FIELDNAMES."""
    for col in _EXPECTED_TOP_LEVEL:
        assert col in FIELDNAMES, f"Missing top-level column: {col}"
    for cat in ("dress", "lower", "upper"):
        for tpl in _EXPECTED_PER_CAT:
            col = tpl.format(cat=cat)
            assert col in FIELDNAMES, f"Missing per-category column: {col}"
    for col in _EXPECTED_TAIL:
        assert col in FIELDNAMES, f"Missing tail column: {col}"
    for col in _EXPECTED_USER_PROMPT:
        assert col in FIELDNAMES, f"Missing user-prompt-aware column: {col}"
    assert SCHEMA_VERSION == "0.2.1"


# ---------------------------------------------------------------------------
# Test (b): category_scores={} → all category columns null + missing_score_reason
# ---------------------------------------------------------------------------

def test_v021_empty_category_scores_writes_nulls(tmp_path: Path):
    """Test (b): empty category_scores writes null + 'category_scores_missing'."""
    pair = _Pair(
        "pair-b",
        scores=_Scores(overall_score=0.75, category_scores={}),
    )
    row = flatten_pair_to_row(pair)
    for cat in ("dress", "lower", "upper"):
        assert row[f"{cat}_weighted_score"] is None, f"{cat}_weighted_score should be null"
        assert row[f"{cat}_total_score"] is None
        for k in ("qwen_pass_rate", "edit_correctness", "garment_transfer_correctness",
                  "preservation", "artifact_penalty"):
            assert row[f"{cat}_{k}"] is None
        assert row[f"{cat}_missing_score_reason"] == "category_scores_missing"


# ---------------------------------------------------------------------------
# Test (c): full dress/lower/upper → all category columns populated
# ---------------------------------------------------------------------------

def test_v021_full_category_scores_populated(tmp_path: Path):
    """Test (c): full category scores write all category total + sub-score columns."""
    pair = _Pair(
        "pair-c",
        scores=_Scores(
            overall_score=0.80,
            category_scores={
                "dress": _full_category_score(0.55, 0.60),
                "lower": _full_category_score(0.45, 0.50),
                "upper": _full_category_score(0.65, 0.70),
            },
        ),
    )
    row = flatten_pair_to_row(pair)
    for cat in ("dress", "lower", "upper"):
        assert row[f"{cat}_weighted_score"] is not None
        assert row[f"{cat}_total_score"] is not None
        assert row[f"{cat}_missing_score_reason"] is None
        for k in ("qwen_pass_rate", "edit_correctness", "garment_transfer_correctness",
                  "preservation", "artifact_penalty"):
            assert row[f"{cat}_{k}"] is not None, f"{cat}_{k} should not be null"


# ---------------------------------------------------------------------------
# Test (d): legacy 0.2 alias read
# ---------------------------------------------------------------------------

def test_v021_legacy_alias_read(tmp_path: Path):
    """Test (d): legacy 0.2 rows with dress_weighted/lower_weighted/upper_weighted
    are read correctly via aliases (canonical names populated in result dict).
    """
    import csv as csv_mod
    # Write a CSV with legacy column headers.
    legacy_path = tmp_path / "legacy.csv"
    legacy_fieldnames = [
        "schema_version", "prompt_pair_id", "system_prompt_id", "negative_prompt_id",
        "round", "overall_score", "total_score",
        "qwen_pass_rate", "edit_correctness", "garment_transfer_correctness",
        "preservation", "artifact_penalty",
        "dress_weighted", "lower_weighted", "upper_weighted",
        "fallback", "timestamp",
    ]
    with open(legacy_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=legacy_fieldnames)
        writer.writeheader()
        writer.writerow({
            "schema_version": "0.2",
            "prompt_pair_id": "pair-d",
            "system_prompt_id": "sp-d",
            "negative_prompt_id": "neg-d",
            "round": "1",
            "overall_score": "0.77",
            "total_score": "0.66",
            "qwen_pass_rate": "0.8",
            "edit_correctness": "0.7",
            "garment_transfer_correctness": "0.6",
            "preservation": "0.9",
            "artifact_penalty": "0.1",
            "dress_weighted": "0.55",
            "lower_weighted": "0.44",
            "upper_weighted": "0.33",
            "fallback": "False",
            "timestamp": "2026-01-01T00:00:00Z",
        })

    rows = read_long_memory(legacy_path)
    assert len(rows) == 1
    r = rows[0]
    # Aliases should be applied.
    assert r.get("dress_weighted_score") == "0.55"
    assert r.get("lower_weighted_score") == "0.44"
    assert r.get("upper_weighted_score") == "0.33"
    # Original legacy keys are still present too (read_long_memory does not strip them).
    assert r.get("overall_score") == "0.77"


# ---------------------------------------------------------------------------
# Test (e): migrate_long_memory_csv produces valid 0.2.1 CSV with backup
# ---------------------------------------------------------------------------

def test_v021_migrate_produces_021_csv_with_backup(tmp_path: Path):
    """Test (e): migrate_long_memory_csv from pairs/*.yaml produces 0.2.1 CSV."""
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    pairs_dir = memory_root / "pairs"
    pairs_dir.mkdir()
    csv_path = memory_root / "long_memory.csv"

    # Write a legacy CSV with one row.
    import csv as csv_mod
    legacy_fields = [
        "schema_version", "prompt_pair_id", "system_prompt_id", "negative_prompt_id",
        "round", "overall_score", "total_score",
        "qwen_pass_rate", "edit_correctness", "garment_transfer_correctness",
        "preservation", "artifact_penalty",
        "dress_weighted", "lower_weighted", "upper_weighted",
        "fallback", "timestamp",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv_mod.DictWriter(fh, fieldnames=legacy_fields)
        writer.writeheader()
        writer.writerow({
            "schema_version": "0.2",
            "prompt_pair_id": "pair-e1",
            "system_prompt_id": "sp-e1",
            "negative_prompt_id": "neg-e1",
            "round": "2",
            "overall_score": "0.82",
            "total_score": "0.70",
            "qwen_pass_rate": "0.9",
            "edit_correctness": "0.8",
            "garment_transfer_correctness": "0.7",
            "preservation": "0.85",
            "artifact_penalty": "0.05",
            "dress_weighted": "0.6",
            "lower_weighted": "0.5",
            "upper_weighted": "0.4",
            "fallback": "False",
            "timestamp": "2026-01-02T00:00:00Z",
        })

    # Write a YAML for pair-e1.
    yaml_data_e1 = {
        "prompt_pair_id": "pair-e1",
        "system_prompt_id": "sp-e1",
        "negative_prompt_id": "neg-e1",
        "round_id": 2,
        "fallback": False,
        "timestamp": "2026-01-02T00:00:00Z",
        "scores": {
            "overall_score": 0.82,
            "total_score": 0.70,
            "sub_scores": {
                "qwen_pass_rate": 0.9, "edit_correctness": 0.8,
                "garment_transfer_correctness": 0.7, "preservation": 0.85,
                "artifact_penalty": 0.05,
            },
            "category_scores": {
                "dress": {
                    "weighted_score": 0.6, "total_score": 0.55,
                    "sub_scores": {
                        "qwen_pass_rate": 0.9, "edit_correctness": 0.8,
                        "garment_transfer_correctness": 0.7, "preservation": 0.85,
                        "artifact_penalty": 0.05,
                    },
                },
            },
        },
    }
    (pairs_dir / "pair-e1.yaml").write_text(
        yaml.safe_dump(yaml_data_e1, allow_unicode=True), encoding="utf-8"
    )

    # Also write a YAML for a new pair not in legacy CSV.
    yaml_data_e2 = {
        "prompt_pair_id": "pair-e2",
        "system_prompt_id": "sp-e2",
        "negative_prompt_id": "neg-e2",
        "round_id": 3,
        "fallback": False,
        "scores": {
            "overall_score": 0.91,
            "total_score": 0.88,
            "sub_scores": {
                "qwen_pass_rate": 0.95, "edit_correctness": 0.9,
                "garment_transfer_correctness": 0.85, "preservation": 0.92,
                "artifact_penalty": 0.02,
            },
            "category_scores": {},
        },
    }
    (pairs_dir / "pair-e2.yaml").write_text(
        yaml.safe_dump(yaml_data_e2, allow_unicode=True), encoding="utf-8"
    )

    migrate_long_memory_csv(memory_root)

    # Backup should exist.
    bak_files = list(memory_root.glob("long_memory.csv.bak.*"))
    assert len(bak_files) == 1, "Expected one backup file"

    # Resulting CSV must be 0.2.1.
    rows = read_long_memory(csv_path)
    pair_ids = {r["prompt_pair_id"] for r in rows}
    assert "pair-e1" in pair_ids
    assert "pair-e2" in pair_ids

    # All canonical columns present in header.
    with open(csv_path, "r", encoding="utf-8") as fh:
        header_line = fh.readline().strip()
    header_cols = header_line.split(",")
    for col in FIELDNAMES:
        assert col in header_cols, f"Missing 0.2.1 column in migrated CSV: {col}"

    # pair-e1: schema_version should be 0.2.1.
    e1 = next(r for r in rows if r["prompt_pair_id"] == "pair-e1")
    assert e1["schema_version"] == "0.2.1"
    # pair-e2: category_scores was {} so dress/lower/upper should be category_scores_missing.
    e2 = next(r for r in rows if r["prompt_pair_id"] == "pair-e2")
    assert e2.get("dress_missing_score_reason") == "category_scores_missing"


# ---------------------------------------------------------------------------
# Test (f): verify_long_memory_consistency raises on mismatch
# ---------------------------------------------------------------------------

def test_v021_consistency_check_raises_on_mismatch(tmp_path: Path):
    """Test (f): verify raises LongMemoryConsistencyError on sub-score mismatch."""
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    pairs_dir = memory_root / "pairs"
    pairs_dir.mkdir()
    csv_path = memory_root / "long_memory.csv"

    # Write a YAML.
    yaml_data = {
        "prompt_pair_id": "pair-f",
        "system_prompt_id": "sp-f",
        "negative_prompt_id": "neg-f",
        "round_id": 1,
        "fallback": False,
        "scores": {
            "overall_score": 0.75,
            "total_score": 0.70,
            "sub_scores": {
                "qwen_pass_rate": 0.8, "edit_correctness": 0.7,
                "garment_transfer_correctness": 0.6, "preservation": 0.9,
                "artifact_penalty": 0.1,
            },
            "category_scores": {
                "dress": {
                    "weighted_score": 0.55, "total_score": 0.50,
                    "sub_scores": {
                        "qwen_pass_rate": 0.8, "edit_correctness": 0.7,
                        "garment_transfer_correctness": 0.6, "preservation": 0.9,
                        "artifact_penalty": 0.1,
                    },
                },
            },
        },
    }
    (pairs_dir / "pair-f.yaml").write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True), encoding="utf-8"
    )

    # Write a CSV row with a mismatched dress_qwen_pass_rate (0.5 vs YAML's 0.8).
    row = {col: "" for col in FIELDNAMES}
    row.update({
        "schema_version": "0.2.1",
        "prompt_pair_id": "pair-f",
        "system_prompt_id": "sp-f",
        "negative_prompt_id": "neg-f",
        "round": "1",
        "overall_score": "0.75",
        "total_score": "0.70",
        "qwen_pass_rate": "0.8",
        "edit_correctness": "0.7",
        "garment_transfer_correctness": "0.6",
        "preservation": "0.9",
        "artifact_penalty": "0.1",
        "dress_weighted_score": "0.55",
        "dress_total_score": "0.50",
        "dress_qwen_pass_rate": "0.5",  # intentional mismatch
        "dress_edit_correctness": "0.7",
        "dress_garment_transfer_correctness": "0.6",
        "dress_preservation": "0.9",
        "dress_artifact_penalty": "0.1",
        "fallback": "False",
        "timestamp": "2026-01-01T00:00:00Z",
    })
    append_long_memory_row(csv_path, row)

    with pytest.raises(LongMemoryConsistencyError, match="dress_qwen_pass_rate"):
        verify_long_memory_consistency(memory_root)


def test_v021_consistency_check_passes_on_match(tmp_path: Path):
    """verify_long_memory_consistency does not raise when CSV matches YAML."""
    memory_root = tmp_path / "memory"
    memory_root.mkdir()
    pairs_dir = memory_root / "pairs"
    pairs_dir.mkdir()
    csv_path = memory_root / "long_memory.csv"

    yaml_data = {
        "prompt_pair_id": "pair-ok",
        "system_prompt_id": "sp-ok",
        "negative_prompt_id": "neg-ok",
        "round_id": 1,
        "fallback": False,
        "scores": {
            "overall_score": 0.75,
            "total_score": 0.70,
            "sub_scores": {
                "qwen_pass_rate": 0.8, "edit_correctness": 0.7,
                "garment_transfer_correctness": 0.6, "preservation": 0.9,
                "artifact_penalty": 0.1,
            },
            "category_scores": {},
        },
    }
    (pairs_dir / "pair-ok.yaml").write_text(
        yaml.safe_dump(yaml_data, allow_unicode=True), encoding="utf-8"
    )

    row = {col: "" for col in FIELDNAMES}
    row.update({
        "schema_version": "0.2.1",
        "prompt_pair_id": "pair-ok",
        "system_prompt_id": "sp-ok",
        "negative_prompt_id": "neg-ok",
        "round": "1",
        "overall_score": "0.75",
        "total_score": "0.70",
        "qwen_pass_rate": "0.8",
        "edit_correctness": "0.7",
        "garment_transfer_correctness": "0.6",
        "preservation": "0.9",
        "artifact_penalty": "0.1",
        "dress_weighted_score": "",
        "dress_missing_score_reason": "category_scores_missing",
        "lower_weighted_score": "",
        "lower_missing_score_reason": "category_scores_missing",
        "upper_weighted_score": "",
        "upper_missing_score_reason": "category_scores_missing",
        "fallback": "False",
        "timestamp": "2026-01-01T00:00:00Z",
    })
    append_long_memory_row(csv_path, row)

    # Should not raise.
    verify_long_memory_consistency(memory_root)


# ---------------------------------------------------------------------------
# Round-trip: flatten_pair_to_row → write → read including 7 user-prompt cols
# ---------------------------------------------------------------------------

def test_v021_flatten_roundtrip_user_prompt_cols(tmp_path: Path):
    """Round-trip: flatten_pair_to_row outputs all 7 user-prompt-aware columns."""
    pair = _Pair(
        "pair-rt",
        scores=_Scores(
            overall_score=0.85,
            total_score=0.80,
            sub_scores=_full_sub_scores(),
            zh_mean_score=0.83,
            en_mean_score=0.87,
            prompt_sensitivity=0.04,
            cross_lingual_gap=0.04,
            worst_user_prompt_id="zh_001",
            worst_user_prompt_score=0.70,
        ),
    )
    row = flatten_pair_to_row(pair)

    # Check all 7 user-prompt-aware columns are present and set.
    assert row["zh_mean_score"] == pytest.approx(0.83)
    assert row["en_mean_score"] == pytest.approx(0.87)
    assert row["prompt_sensitivity"] == pytest.approx(0.04)
    assert row["cross_lingual_gap"] == pytest.approx(0.04)
    assert row["worst_user_prompt_id"] == "zh_001"
    assert row["worst_user_prompt_score"] == pytest.approx(0.70)
    assert "language_brittle" in row  # present (may be None)

    # Write to CSV and read back.
    csv_path = tmp_path / "long.csv"
    append_long_memory_row(csv_path, row)
    rows = read_long_memory(csv_path)
    assert len(rows) == 1
    r = rows[0]
    assert r["zh_mean_score"] == "0.83"
    assert r["worst_user_prompt_id"] == "zh_001"
    assert r["schema_version"] == "0.2.1"


def test_v021_no_scores_row_has_all_nulls(tmp_path: Path):
    """flatten_pair_to_row with scores=None fills all score columns with None."""
    pair = _Pair("pair-ns", scores=None)
    row = flatten_pair_to_row(pair)
    assert row["overall_score"] is None
    assert row["missing_score_reason"] == "scores_missing"
    for cat in ("dress", "lower", "upper"):
        assert row[f"{cat}_weighted_score"] is None
        assert row[f"{cat}_missing_score_reason"] == "scores_missing"
    for col in ("zh_mean_score", "en_mean_score", "prompt_sensitivity",
                "cross_lingual_gap", "worst_user_prompt_id", "worst_user_prompt_score"):
        assert row[col] is None
