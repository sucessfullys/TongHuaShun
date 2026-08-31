"""Tests for scoring/ranking.py (S07.07)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from system_prompt_retrieval_agent.schemas import (
    PromptPair,
    PromptPairScoreContext,
    PromptPairSubScores,
)
from system_prompt_retrieval_agent.scoring.ranking import (
    build_next_round_context,
    rank_pairs,
    write_ranking_artifacts,
)


def _make_pair(
    pair_id: str,
    overall_score: float | None = None,
    round_id: int = 1,
    fallback: bool = False,
) -> PromptPair:
    scores = None
    if overall_score is not None:
        scores = PromptPairScoreContext(
            overall_score=overall_score,
            total_score=overall_score,
            sub_scores=PromptPairSubScores(
                qwen_pass_rate=overall_score,
                edit_correctness=overall_score,
                garment_transfer_correctness=overall_score,
                preservation=overall_score,
                artifact_penalty=0.0,
            ),
        )
    return PromptPair(
        prompt_pair_id=pair_id,
        system_prompt_id=f"sys_{pair_id}",
        negative_prompt_id=f"neg_{pair_id}",
        round_id=round_id,
        system_prompt=f"System prompt for {pair_id}",
        scores=scores,
        fallback=fallback,
    )


class TestRankPairs:
    def test_sort_descending_by_overall_score(self):
        pairs = [
            _make_pair("p1", overall_score=0.5),
            _make_pair("p2", overall_score=0.9),
            _make_pair("p3", overall_score=0.1),
        ]
        ranked = rank_pairs(pairs)
        scores = [p.scores.overall_score for p in ranked]
        assert scores == [0.9, 0.5, 0.1]

    def test_none_score_sorts_last(self):
        pairs = [
            _make_pair("p_none", overall_score=None),
            _make_pair("p_mid", overall_score=0.5),
            _make_pair("p_high", overall_score=0.9),
        ]
        ranked = rank_pairs(pairs)
        assert ranked[0].prompt_pair_id == "p_high"
        assert ranked[1].prompt_pair_id == "p_mid"
        assert ranked[-1].prompt_pair_id == "p_none"

    def test_stable_tie_break_by_pair_id_ascending(self):
        """Equal scores should be broken alphabetically by prompt_pair_id ascending."""
        pairs = [
            _make_pair("p_c", overall_score=0.7),
            _make_pair("p_a", overall_score=0.7),
            _make_pair("p_b", overall_score=0.7),
        ]
        ranked = rank_pairs(pairs)
        ids = [p.prompt_pair_id for p in ranked]
        assert ids == ["p_a", "p_b", "p_c"]

    def test_tie_break_custom_field(self):
        pairs = [
            _make_pair("p3", overall_score=0.8),
            _make_pair("p1", overall_score=0.8),
            _make_pair("p2", overall_score=0.8),
        ]
        ranked = rank_pairs(pairs, tie_break_field="prompt_pair_id")
        ids = [p.prompt_pair_id for p in ranked]
        assert ids == ["p1", "p2", "p3"]

    def test_empty_list(self):
        assert rank_pairs([]) == []

    def test_multiple_nones_stable(self):
        pairs = [
            _make_pair("pn2", overall_score=None),
            _make_pair("pn1", overall_score=None),
        ]
        ranked = rank_pairs(pairs)
        # Both None; tie-break by pair_id ascending
        ids = [p.prompt_pair_id for p in ranked]
        assert ids == ["pn1", "pn2"]


class TestWriteRankingArtifacts:
    def test_creates_expected_files(self, tmp_path: Path):
        pairs = [
            _make_pair("p1", overall_score=0.9),
            _make_pair("p2", overall_score=0.5),
        ]
        write_ranking_artifacts(tmp_path, pairs, round_id=1)

        scoring_dir = tmp_path / "scoring"
        assert (scoring_dir / "weighted_scores.csv").exists()
        assert (scoring_dir / "category_scores.csv").exists()
        assert (scoring_dir / "ranking.json").exists()

    def test_weighted_scores_csv_columns(self, tmp_path: Path):
        """weighted_scores.csv must contain all expected columns."""
        import pandas as pd

        pairs = [_make_pair("p1", overall_score=0.8)]
        write_ranking_artifacts(tmp_path, pairs, round_id=2)

        df = pd.read_csv(tmp_path / "scoring" / "weighted_scores.csv")
        expected_cols = [
            "schema_version", "prompt_pair_id", "system_prompt_id", "negative_prompt_id",
            "round", "overall_score", "total_score", "qwen_pass_rate", "edit_correctness",
            "garment_transfer_correctness", "preservation", "artifact_penalty",
            "dress_weighted", "lower_weighted", "upper_weighted", "fallback", "timestamp",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_round_trip_values(self, tmp_path: Path):
        """Values written to CSV should round-trip correctly."""
        import pandas as pd

        pairs = [_make_pair("pair_alpha", overall_score=0.75, round_id=3)]
        write_ranking_artifacts(tmp_path, pairs, round_id=3)

        df = pd.read_csv(tmp_path / "scoring" / "weighted_scores.csv")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["prompt_pair_id"] == "pair_alpha"
        assert float(row["overall_score"]) == pytest.approx(0.75, abs=1e-6)
        assert int(row["round"]) == 3

    def test_ranking_json_rank_sequence(self, tmp_path: Path):
        """ranking.json should contain rank 1..N."""
        pairs = [
            _make_pair("p3", overall_score=0.3),
            _make_pair("p1", overall_score=0.9),
            _make_pair("p2", overall_score=0.6),
        ]
        write_ranking_artifacts(tmp_path, pairs, round_id=1)

        data = json.loads((tmp_path / "scoring" / "ranking.json").read_text())
        ranks = [entry["rank"] for entry in data]
        assert ranks == list(range(1, len(pairs) + 1))

    def test_ranking_json_ordered_by_score(self, tmp_path: Path):
        """ranking.json entries should be in descending score order."""
        pairs = [
            _make_pair("low", overall_score=0.2),
            _make_pair("high", overall_score=0.9),
        ]
        write_ranking_artifacts(tmp_path, pairs, round_id=1)

        data = json.loads((tmp_path / "scoring" / "ranking.json").read_text())
        assert data[0]["prompt_pair_id"] == "high"
        assert data[0]["rank"] == 1
        assert data[1]["prompt_pair_id"] == "low"
        assert data[1]["rank"] == 2

    def test_none_scores_in_ranking_json(self, tmp_path: Path):
        """Pairs with None overall_score should appear in ranking.json with null."""
        pairs = [
            _make_pair("scored", overall_score=0.7),
            _make_pair("unscored", overall_score=None),
        ]
        write_ranking_artifacts(tmp_path, pairs, round_id=1)

        data = json.loads((tmp_path / "scoring" / "ranking.json").read_text())
        id_to_entry = {e["prompt_pair_id"]: e for e in data}
        assert id_to_entry["unscored"]["overall_score"] is None
        assert id_to_entry["scored"]["rank"] < id_to_entry["unscored"]["rank"]


class TestBuildNextRoundContext:
    def test_returns_top_n_pairs(self):
        pairs = [
            _make_pair("p1", overall_score=0.9),
            _make_pair("p2", overall_score=0.7),
            _make_pair("p3", overall_score=0.5),
            _make_pair("p4", overall_score=0.3),
        ]
        result = build_next_round_context(pairs, top_n=2)
        assert len(result) == 2
        assert result[0].prompt_pair_id == "p1"
        assert result[1].prompt_pair_id == "p2"

    def test_returns_deep_copies(self):
        """Mutating returned pairs must not affect the originals."""
        pairs = [_make_pair("p1", overall_score=0.8)]
        result = build_next_round_context(pairs, top_n=1)
        result[0].scores.overall_score = 0.0
        assert pairs[0].scores.overall_score == pytest.approx(0.8, abs=1e-6)

    def test_scores_attached_to_returned_pairs(self):
        """Returned pairs should carry their PromptPairScoreContext."""
        pairs = [_make_pair("p1", overall_score=0.85)]
        result = build_next_round_context(pairs, top_n=1)
        assert result[0].scores is not None
        assert isinstance(result[0].scores, PromptPairScoreContext)
        assert result[0].scores.overall_score == pytest.approx(0.85, abs=1e-6)

    def test_fewer_than_top_n_returns_all(self):
        pairs = [_make_pair("p1", overall_score=0.5)]
        result = build_next_round_context(pairs, top_n=5)
        assert len(result) == 1

    def test_empty_input_returns_empty(self):
        result = build_next_round_context([], top_n=3)
        assert result == []

    def test_default_top_n_is_3(self):
        pairs = [_make_pair(f"p{i}", overall_score=1.0 - i * 0.1) for i in range(5)]
        result = build_next_round_context(pairs)
        assert len(result) == 3
