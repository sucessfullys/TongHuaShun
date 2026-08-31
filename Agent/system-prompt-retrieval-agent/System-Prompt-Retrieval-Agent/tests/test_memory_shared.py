"""S02.09 — Tests for memory.shared: prune_rules correctness with boundary cases."""
from __future__ import annotations

from pathlib import Path

import pytest

from system_prompt_retrieval_agent.memory.shared import (
    prune_rules,
    read_rules,
    write_rules,
    write_markdown_digest,
)


def _make_rule(rule_id: str, confidence: float, last_seen: int, tags: str = "general") -> dict:
    return {
        "schema_version": "1.0",
        "rule_id": rule_id,
        "text": f"Rule text for {rule_id}",
        "source_round": "1",
        "confidence": str(confidence),
        "last_seen": str(last_seen),
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# CSV read/write round-trip
# ---------------------------------------------------------------------------

def test_write_read_roundtrip(tmp_path: Path):
    csv_path = tmp_path / "rules.csv"
    rules = [_make_rule("r1", 0.9, 5), _make_rule("r2", 0.6, 3)]
    write_rules(csv_path, rules)

    loaded = read_rules(csv_path)
    assert len(loaded) == 2
    assert loaded[0]["rule_id"] == "r1"


def test_read_missing_returns_empty(tmp_path: Path):
    assert read_rules(tmp_path / "nonexistent.csv") == []


def test_atomic_write_no_tmp_leftover(tmp_path: Path):
    csv_path = tmp_path / "rules.csv"
    write_rules(csv_path, [_make_rule("r1", 0.9, 5)])
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    assert not tmp.exists()


# ---------------------------------------------------------------------------
# prune_rules
# ---------------------------------------------------------------------------

def test_prune_drops_stale_low_confidence():
    """Row with last_seen < current_round - 3 AND confidence < 0.5 should be evicted."""
    current_round = 10
    rows = [_make_rule("stale-low", 0.3, 5)]  # last_seen=5 < 10-3=7, conf=0.3 < 0.5
    kept, evicted = prune_rules(rows, current_round)
    assert len(evicted) == 1
    assert len(kept) == 0


def test_prune_keeps_stale_high_confidence():
    """Row with last_seen < cutoff but confidence >= 0.5 should be kept."""
    current_round = 10
    rows = [_make_rule("stale-high", 0.8, 5)]  # stale but high confidence
    kept, evicted = prune_rules(rows, current_round)
    assert len(kept) == 1
    assert len(evicted) == 0


def test_prune_keeps_fresh_low_confidence():
    """Row with last_seen >= cutoff but low confidence should be kept."""
    current_round = 10
    rows = [_make_rule("fresh-low", 0.2, 8)]  # last_seen=8 >= 10-3=7, conf=0.2
    kept, evicted = prune_rules(rows, current_round)
    assert len(kept) == 1
    assert len(evicted) == 0


def test_prune_boundary_last_seen_exactly_at_cutoff():
    """last_seen == current_round - max_staleness (boundary) should be kept."""
    current_round = 10
    # last_seen == 7 == 10-3; condition is strictly less than, so 7 is NOT stale
    rows = [_make_rule("boundary", 0.2, 7)]
    kept, evicted = prune_rules(rows, current_round)
    assert len(kept) == 1
    assert len(evicted) == 0


def test_prune_boundary_one_before_cutoff():
    """last_seen == current_round - max_staleness - 1 (just below boundary) + low conf => evict."""
    current_round = 10
    rows = [_make_rule("just-below", 0.2, 6)]  # last_seen=6 < 7, conf=0.2 < 0.5
    kept, evicted = prune_rules(rows, current_round)
    assert len(evicted) == 1


def test_prune_boundary_confidence_exactly_05():
    """confidence == 0.5 (boundary) should be kept (condition is strictly <)."""
    current_round = 10
    rows = [_make_rule("conf-boundary", 0.5, 5)]  # stale, conf=0.5 exactly
    kept, evicted = prune_rules(rows, current_round)
    assert len(kept) == 1
    assert len(evicted) == 0


def test_prune_mixed(tmp_path: Path):
    current_round = 10
    rows = [
        _make_rule("evict", 0.3, 5),   # stale + low conf => evict
        _make_rule("keep1", 0.9, 5),   # stale but high conf => keep
        _make_rule("keep2", 0.3, 8),   # fresh but low conf => keep
        _make_rule("keep3", 0.7, 9),   # fresh + high conf => keep
    ]
    kept, evicted = prune_rules(rows, current_round)
    kept_ids = {r["rule_id"] for r in kept}
    evict_ids = {r["rule_id"] for r in evicted}
    assert evict_ids == {"evict"}
    assert kept_ids == {"keep1", "keep2", "keep3"}


def test_prune_empty_rows():
    kept, evicted = prune_rules([], current_round=5)
    assert kept == []
    assert evicted == []


# ---------------------------------------------------------------------------
# write_markdown_digest
# ---------------------------------------------------------------------------

def test_write_markdown_digest(tmp_path: Path):
    md_path = tmp_path / "digest.md"
    rows = [
        _make_rule("r1", 0.9, 5, tags="garment,upper"),
        _make_rule("r2", 0.7, 4, tags="garment"),
        _make_rule("r3", 0.6, 3, tags="lower"),
    ]
    write_markdown_digest(md_path, rows)
    content = md_path.read_text()
    assert "garment" in content
    assert "upper" in content
    assert "lower" in content
    assert "r1" in content or "Rule text" in content


def test_write_markdown_digest_empty(tmp_path: Path):
    md_path = tmp_path / "digest.md"
    write_markdown_digest(md_path, [])
    assert md_path.exists()
