"""Tests for visualization.selectors (S08.07).

Covers:
- Inner join drops samples missing from either side.
- Tie-break is stable by sample_id ascending.
- select_max_improvements(k=2) returns top-2 by delta desc.
- select_max_decreases(k=3) returns bottom-3 by delta asc.
"""

from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.visualization.selectors import (
    SampleScore,
    pair_scores_by_sample,
    select_max_improvements,
    select_max_decreases,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score(sample_id: str, score: float, category: str | None = None) -> SampleScore:
    return SampleScore(sample_id=sample_id, overall_score=score, category=category)


# ---------------------------------------------------------------------------
# pair_scores_by_sample
# ---------------------------------------------------------------------------

class TestPairScoresBySample:
    def test_inner_join_drops_missing_from_prev(self):
        prev = [_score("a", 0.5), _score("b", 0.6)]
        curr = [_score("a", 0.7), _score("c", 0.8)]  # "c" only in curr
        result = pair_scores_by_sample(prev, curr)
        ids = {d["sample_id"] for d in result}
        assert ids == {"a"}
        assert "c" not in ids

    def test_inner_join_drops_missing_from_curr(self):
        prev = [_score("a", 0.5), _score("b", 0.6), _score("d", 0.3)]
        curr = [_score("a", 0.7), _score("b", 0.65)]  # "d" only in prev
        result = pair_scores_by_sample(prev, curr)
        ids = {d["sample_id"] for d in result}
        assert ids == {"a", "b"}
        assert "d" not in ids

    def test_delta_computed_correctly(self):
        prev = [_score("x", 0.4)]
        curr = [_score("x", 0.75)]
        result = pair_scores_by_sample(prev, curr)
        assert len(result) == 1
        assert abs(result[0]["delta"] - 0.35) < 1e-9

    def test_empty_prev_returns_empty(self):
        curr = [_score("a", 0.5)]
        result = pair_scores_by_sample([], curr)
        assert result == []

    def test_empty_curr_returns_empty(self):
        prev = [_score("a", 0.5)]
        result = pair_scores_by_sample(prev, [])
        assert result == []

    def test_category_from_curr_wins(self):
        prev = [_score("a", 0.5, category="cat_prev")]
        curr = [_score("a", 0.6, category="cat_curr")]
        result = pair_scores_by_sample(prev, curr)
        assert result[0]["category"] == "cat_curr"

    def test_category_falls_back_to_prev_when_curr_none(self):
        prev = [_score("a", 0.5, category="cat_prev")]
        curr = [_score("a", 0.6, category=None)]
        result = pair_scores_by_sample(prev, curr)
        assert result[0]["category"] == "cat_prev"


# ---------------------------------------------------------------------------
# select_max_improvements
# ---------------------------------------------------------------------------

class TestSelectMaxImprovements:
    def _make_deltas(self) -> list[dict]:
        return [
            {"sample_id": "b", "prev": 0.5, "curr": 0.9, "delta": 0.4, "category": None},
            {"sample_id": "a", "prev": 0.6, "curr": 0.8, "delta": 0.2, "category": None},
            {"sample_id": "c", "prev": 0.7, "curr": 0.75, "delta": 0.05, "category": None},
            {"sample_id": "d", "prev": 0.8, "curr": 0.6, "delta": -0.2, "category": None},
        ]

    def test_top_2_returns_two_items(self):
        result = select_max_improvements(self._make_deltas(), k=2)
        assert len(result) == 2

    def test_top_2_are_highest_deltas(self):
        result = select_max_improvements(self._make_deltas(), k=2)
        assert result[0]["sample_id"] == "b"
        assert result[1]["sample_id"] == "a"

    def test_k_larger_than_list_returns_all(self):
        result = select_max_improvements(self._make_deltas(), k=100)
        assert len(result) == 4

    def test_tie_break_by_sample_id_asc(self):
        deltas = [
            {"sample_id": "z", "prev": 0.5, "curr": 0.8, "delta": 0.3, "category": None},
            {"sample_id": "a", "prev": 0.5, "curr": 0.8, "delta": 0.3, "category": None},
            {"sample_id": "m", "prev": 0.5, "curr": 0.8, "delta": 0.3, "category": None},
        ]
        result = select_max_improvements(deltas, k=2)
        assert result[0]["sample_id"] == "a"
        assert result[1]["sample_id"] == "m"


# ---------------------------------------------------------------------------
# select_max_decreases
# ---------------------------------------------------------------------------

class TestSelectMaxDecreases:
    def _make_deltas(self) -> list[dict]:
        return [
            {"sample_id": "c", "prev": 0.8, "curr": 0.4, "delta": -0.4, "category": None},
            {"sample_id": "a", "prev": 0.7, "curr": 0.5, "delta": -0.2, "category": None},
            {"sample_id": "b", "prev": 0.6, "curr": 0.55, "delta": -0.05, "category": None},
            {"sample_id": "d", "prev": 0.5, "curr": 0.7, "delta": 0.2, "category": None},
        ]

    def test_bottom_3_returns_three_items(self):
        result = select_max_decreases(self._make_deltas(), k=3)
        assert len(result) == 3

    def test_bottom_3_are_lowest_deltas(self):
        result = select_max_decreases(self._make_deltas(), k=3)
        sample_ids = [d["sample_id"] for d in result]
        assert sample_ids[0] == "c"   # delta = -0.4
        assert sample_ids[1] == "a"   # delta = -0.2
        assert sample_ids[2] == "b"   # delta = -0.05

    def test_k_larger_than_list_returns_all(self):
        result = select_max_decreases(self._make_deltas(), k=100)
        assert len(result) == 4

    def test_tie_break_by_sample_id_asc(self):
        deltas = [
            {"sample_id": "z", "prev": 0.8, "curr": 0.5, "delta": -0.3, "category": None},
            {"sample_id": "a", "prev": 0.8, "curr": 0.5, "delta": -0.3, "category": None},
            {"sample_id": "m", "prev": 0.8, "curr": 0.5, "delta": -0.3, "category": None},
        ]
        result = select_max_decreases(deltas, k=2)
        assert result[0]["sample_id"] == "a"
        assert result[1]["sample_id"] == "m"
