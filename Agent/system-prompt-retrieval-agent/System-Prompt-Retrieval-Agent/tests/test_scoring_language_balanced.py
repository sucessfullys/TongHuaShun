"""Tests for V0.2.1 language-balanced ranking, hard gate, and CategoryAggregator."""
from __future__ import annotations

from pathlib import Path

import pytest

from system_prompt_retrieval_agent.schemas import (
    GemmaUserPrompt,
    PromptPair,
    PromptPairScoreContext,
    PromptPairSubScores,
)
from system_prompt_retrieval_agent.scoring.aggregate import build_score_context_for_pair
from system_prompt_retrieval_agent.scoring.category import CategoryAggregator
from system_prompt_retrieval_agent.scoring.ranking import (
    rank_pairs_with_gate,
    write_user_prompt_scores_csv,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _library(*args: tuple[str, str]) -> list[GemmaUserPrompt]:
    return [
        GemmaUserPrompt(user_prompt_id=uid, language=lang, text=f"t_{uid}", enabled=True)
        for uid, lang in args
    ]


class _ScoringCfg:
    def __init__(
        self,
        prompt_sensitivity_penalty_weight: float = 0.5,
        prompt_sensitivity_tolerance: float = 0.10,
        min_per_user_prompt_score: float = 0.30,
        flag_language_brittle_gap: float = 0.10,
    ) -> None:
        self.prompt_sensitivity_penalty_weight = prompt_sensitivity_penalty_weight
        self.prompt_sensitivity_tolerance = prompt_sensitivity_tolerance
        self.min_per_user_prompt_score = min_per_user_prompt_score
        self.flag_language_brittle_gap = flag_language_brittle_gap


def _make_pair_with_context(
    pair_id: str,
    overall: float = 0.8,
    total: float | None = None,
    sensitivity: float | None = None,
    worst_uid: str | None = None,
    worst_score: float | None = None,
    mean_score_by_up: dict[str, float] | None = None,
) -> PromptPair:
    ctx = PromptPairScoreContext(
        overall_score=overall,
        total_score=total if total is not None else overall,
        sub_scores=PromptPairSubScores(qwen_pass_rate=overall),
        prompt_sensitivity=sensitivity,
        worst_user_prompt_id=worst_uid,
        worst_user_prompt_score=worst_score,
        mean_score_by_user_prompt=mean_score_by_up or {},
    )
    return PromptPair(
        prompt_pair_id=pair_id,
        system_prompt_id=f"sys_{pair_id}",
        negative_prompt_id=f"neg_{pair_id}",
        round_id=1,
        system_prompt=f"prompt for {pair_id}",
        scores=ctx,
    )


# ---------------------------------------------------------------------------
# S07.05 — rank_pairs_with_gate: hard gate
# ---------------------------------------------------------------------------

class TestHardGate:
    def test_pair_below_min_rejected(self):
        """A pair with worst_user_prompt_score below threshold is moved to rejected."""
        pair = _make_pair_with_context(
            "p1", overall=0.8,
            worst_uid="en_001", worst_score=0.20,
            mean_score_by_up={"zh_001": 0.8, "en_001": 0.20},
        )
        cfg = _ScoringCfg(min_per_user_prompt_score=0.30)
        ranked, rejected = rank_pairs_with_gate([pair], cfg)
        assert len(ranked) == 0
        assert len(rejected) == 1
        assert rejected[0]["pair_id"] == "p1"
        assert rejected[0]["rejection_reason"] == "per_user_prompt_below_min"
        assert rejected[0]["worst_user_prompt_id"] == "en_001"

    def test_pair_at_threshold_accepted(self):
        """Pair with worst_score == threshold is NOT rejected."""
        pair = _make_pair_with_context(
            "p1", overall=0.8,
            worst_uid="en_001", worst_score=0.30,
            mean_score_by_up={"zh_001": 0.8, "en_001": 0.30},
        )
        cfg = _ScoringCfg(min_per_user_prompt_score=0.30)
        ranked, rejected = rank_pairs_with_gate([pair], cfg)
        assert len(ranked) == 1
        assert len(rejected) == 0

    def test_pair_above_threshold_accepted(self):
        pair = _make_pair_with_context(
            "p1", overall=0.8,
            worst_uid="en_001", worst_score=0.50,
            mean_score_by_up={"zh_001": 0.8, "en_001": 0.50},
        )
        cfg = _ScoringCfg(min_per_user_prompt_score=0.30)
        ranked, rejected = rank_pairs_with_gate([pair], cfg)
        assert len(ranked) == 1
        assert len(rejected) == 0

    def test_mixed_pairs(self):
        """Among 3 pairs, only the one below min_score is rejected."""
        pairs = [
            _make_pair_with_context("good_a", overall=0.9, worst_uid="zh_001", worst_score=0.60,
                                    mean_score_by_up={"zh_001": 0.60}),
            _make_pair_with_context("bad", overall=0.7, worst_uid="en_001", worst_score=0.10,
                                    mean_score_by_up={"en_001": 0.10}),
            _make_pair_with_context("good_b", overall=0.5, worst_uid="zh_001", worst_score=0.40,
                                    mean_score_by_up={"zh_001": 0.40}),
        ]
        cfg = _ScoringCfg(min_per_user_prompt_score=0.30)
        ranked, rejected = rank_pairs_with_gate(pairs, cfg)
        assert len(ranked) == 2
        assert len(rejected) == 1
        assert rejected[0]["pair_id"] == "bad"

    def test_no_mean_score_by_up_skips_gate(self):
        """If mean_score_by_user_prompt is empty, hard gate is not applied."""
        pair = _make_pair_with_context("p1", overall=0.8, mean_score_by_up={})
        cfg = _ScoringCfg(min_per_user_prompt_score=0.30)
        ranked, rejected = rank_pairs_with_gate([pair], cfg)
        assert len(ranked) == 1
        assert len(rejected) == 0

    def test_no_scores_skips_gate(self):
        pair = PromptPair(
            prompt_pair_id="p_null",
            system_prompt_id="s1",
            negative_prompt_id="n1",
            round_id=1,
            system_prompt="s",
            scores=None,
        )
        cfg = _ScoringCfg(min_per_user_prompt_score=0.30)
        ranked, rejected = rank_pairs_with_gate([pair], cfg)
        assert len(ranked) == 1
        assert len(rejected) == 0

    def test_rejected_pair_rejection_reason_field(self):
        pair = _make_pair_with_context(
            "p_bad", worst_uid="en_001", worst_score=0.05,
            mean_score_by_up={"en_001": 0.05},
        )
        _, rejected = rank_pairs_with_gate([pair], _ScoringCfg())
        assert rejected[0]["rejection_reason"] == "per_user_prompt_below_min"


# ---------------------------------------------------------------------------
# S07.05 — rank_pairs_with_gate: sensitivity penalty
# ---------------------------------------------------------------------------

class TestSensitivityPenaltyRanking:
    def test_sensitivity_below_tolerance_no_change(self):
        """Sensitivity below tolerance -> overall_score unchanged."""
        pair = _make_pair_with_context(
            "p1", overall=0.80, total=0.80,
            sensitivity=0.05,
            mean_score_by_up={"zh_001": 0.80},
        )
        cfg = _ScoringCfg(
            prompt_sensitivity_penalty_weight=0.5,
            prompt_sensitivity_tolerance=0.10,
        )
        ranked, _ = rank_pairs_with_gate([pair], cfg)
        assert ranked[0].scores.overall_score == pytest.approx(0.80, abs=1e-6)

    def test_sensitivity_above_tolerance_linear_penalty(self):
        """sensitivity=0.30 > tolerance=0.10 -> penalty = 0.5 * 0.20 = 0.10."""
        pair = _make_pair_with_context(
            "p1", total=0.80, overall=0.80,
            sensitivity=0.30,
            mean_score_by_up={"zh_001": 0.80},
        )
        cfg = _ScoringCfg(
            prompt_sensitivity_penalty_weight=0.5,
            prompt_sensitivity_tolerance=0.10,
        )
        ranked, _ = rank_pairs_with_gate([pair], cfg)
        assert ranked[0].scores.overall_score == pytest.approx(0.70, abs=1e-6)

    def test_penalty_does_not_go_negative(self):
        """Clamping prevents overall_score below 0.
        Use min_per_user_prompt_score=0.0 so hard gate never fires."""
        pair = _make_pair_with_context(
            "p1", total=0.10, overall=0.10,
            sensitivity=1.0,
            mean_score_by_up={"zh_001": 0.10},
            worst_uid="zh_001", worst_score=0.10,
        )
        cfg = _ScoringCfg(
            prompt_sensitivity_penalty_weight=5.0,
            prompt_sensitivity_tolerance=0.0,
            min_per_user_prompt_score=0.0,  # disable hard gate
        )
        ranked, _ = rank_pairs_with_gate([pair], cfg)
        assert len(ranked) == 1
        assert ranked[0].scores.overall_score >= 0.0

    def test_original_pair_not_mutated(self):
        """rank_pairs_with_gate must not mutate the caller's pair object."""
        pair = _make_pair_with_context(
            "p1", total=0.80, overall=0.80,
            sensitivity=0.30,
            mean_score_by_up={"zh_001": 0.80},
        )
        original_score = pair.scores.overall_score
        rank_pairs_with_gate([pair], _ScoringCfg())
        assert pair.scores.overall_score == pytest.approx(original_score, abs=1e-9)

    def test_tie_break_by_pair_id(self):
        """Equal overall_score after penalty -> tie-break by pair_id ascending."""
        pairs = [
            _make_pair_with_context("p_c", total=0.80, overall=0.80, sensitivity=0.0,
                                    mean_score_by_up={"zh_001": 0.80}),
            _make_pair_with_context("p_a", total=0.80, overall=0.80, sensitivity=0.0,
                                    mean_score_by_up={"zh_001": 0.80}),
            _make_pair_with_context("p_b", total=0.80, overall=0.80, sensitivity=0.0,
                                    mean_score_by_up={"zh_001": 0.80}),
        ]
        ranked, _ = rank_pairs_with_gate(pairs, _ScoringCfg())
        ids = [p.prompt_pair_id for p in ranked]
        assert ids == ["p_a", "p_b", "p_c"]


# ---------------------------------------------------------------------------
# S07.04d — write_user_prompt_scores_csv
# ---------------------------------------------------------------------------

class TestWriteUserPromptScoresCsv:
    def test_creates_file(self, tmp_path: Path):
        lib = _library(("zh_001", "zh"), ("en_001", "en"))
        per_up = {"zh_001": 0.9, "en_001": 0.4}
        out = tmp_path / "user_prompt_scores.csv"
        write_user_prompt_scores_csv(out, "pair_001", per_up, lib)
        assert out.exists()

    def test_csv_columns(self, tmp_path: Path):
        import pandas as pd
        lib = _library(("zh_001", "zh"), ("en_001", "en"))
        per_up = {"zh_001": 0.9, "en_001": 0.4}
        out = tmp_path / "user_prompt_scores.csv"
        write_user_prompt_scores_csv(out, "pair_001", per_up, lib)
        df = pd.read_csv(out)
        for col in ["pair_id", "user_prompt_id", "language", "mean", "n_samples"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_values(self, tmp_path: Path):
        import pandas as pd
        lib = _library(("zh_001", "zh"), ("en_001", "en"))
        per_up = {"zh_001": 0.9, "en_001": 0.4}
        out = tmp_path / "user_prompt_scores.csv"
        write_user_prompt_scores_csv(out, "pair_001", per_up, lib,
                                     cell_counts={"zh_001": 5, "en_001": 3})
        df = pd.read_csv(out)
        zh_row = df[df["user_prompt_id"] == "zh_001"].iloc[0]
        en_row = df[df["user_prompt_id"] == "en_001"].iloc[0]
        assert zh_row["language"] == "zh"
        assert zh_row["mean"] == pytest.approx(0.9, abs=1e-6)
        assert int(zh_row["n_samples"]) == 5
        assert en_row["language"] == "en"
        assert en_row["mean"] == pytest.approx(0.4, abs=1e-6)
        assert int(en_row["n_samples"]) == 3

    def test_csv_pair_id_populated(self, tmp_path: Path):
        import pandas as pd
        lib = _library(("zh_001", "zh"))
        per_up = {"zh_001": 0.8}
        out = tmp_path / "user_prompt_scores.csv"
        write_user_prompt_scores_csv(out, "pair_XYZ", per_up, lib)
        df = pd.read_csv(out)
        assert (df["pair_id"] == "pair_XYZ").all()

    def test_unknown_uid_language_empty(self, tmp_path: Path):
        """user_prompt_id not in library -> language column is empty string."""
        import pandas as pd
        per_up = {"unknown_99": 0.5}
        out = tmp_path / "user_prompt_scores.csv"
        write_user_prompt_scores_csv(out, "p", per_up, [])
        df = pd.read_csv(out)
        lang_val = df.iloc[0]["language"]
        assert str(lang_val) in ("", "nan", "")  # empty or NaN in pandas

    def test_empty_per_up_means_empty_csv(self, tmp_path: Path):
        """Empty per_up_means -> zero rows in CSV (header only)."""
        import pandas as pd
        out = tmp_path / "user_prompt_scores.csv"
        write_user_prompt_scores_csv(out, "p", {}, [])
        df = pd.read_csv(out)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# S07.06-.07 — CategoryAggregator
# ---------------------------------------------------------------------------

def _sample(cat: str, uid: str, qwen_status: str = "yes",
            edit_correctness: float = 0.8,
            garment_transfer_correctness: float = 0.75,
            preservation: float = 0.9,
            artifact_penalty: float = 0.05) -> dict:
    return {
        "sample_id": f"{cat}_{uid}",
        "category": cat,
        "user_prompt_id": uid,
        "qwen_status": qwen_status,
        "edit_correctness": edit_correctness,
        "garment_transfer_correctness": garment_transfer_correctness,
        "preservation": preservation,
        "artifact_penalty": artifact_penalty,
    }


WEIGHTS = {
    "qwen_pass_rate": 0.40,
    "edit_correctness": 0.20,
    "garment_transfer_correctness": 0.15,
    "preservation": 0.15,
    "artifact_penalty": -0.10,
    "category_balance_bonus": 0.0,
}


class TestCategoryAggregator:
    def test_add_cells_and_breakdown(self):
        agg = CategoryAggregator(WEIGHTS)
        agg.add_cells([
            _sample("dress", "zh_001"),
            _sample("dress", "en_001"),
            _sample("lower", "zh_001"),
        ])
        breakdown = agg.category_user_prompt_breakdown()
        assert "dress" in breakdown
        assert "zh_001" in breakdown["dress"]
        assert "en_001" in breakdown["dress"]
        assert "lower" in breakdown

    def test_category_user_prompt_breakdown_scores_non_none(self):
        agg = CategoryAggregator(WEIGHTS)
        agg.add_cells([_sample("dress", "zh_001"), _sample("dress", "en_001")])
        breakdown = agg.category_user_prompt_breakdown()
        for uid, ctx in breakdown["dress"].items():
            assert ctx is not None
            assert ctx.weighted_score is not None

    def test_no_cells_returns_empty_uids(self):
        agg = CategoryAggregator(WEIGHTS, categories=["dress"])
        breakdown = agg.category_user_prompt_breakdown()
        assert breakdown["dress"] == {}

    def test_brittle_categories_true_when_gap_exceeds_flag(self):
        """High gap between phrasings -> brittle=True."""
        agg = CategoryAggregator(WEIGHTS)
        # dress: zh_001 gets all-yes (high score), en_001 gets all-no (low score)
        cells_high = [_sample("dress", "zh_001", qwen_status="yes",
                               edit_correctness=1.0, garment_transfer_correctness=1.0,
                               preservation=1.0, artifact_penalty=0.0) for _ in range(5)]
        cells_low = [_sample("dress", "en_001", qwen_status="no",
                              edit_correctness=0.0, garment_transfer_correctness=0.0,
                              preservation=0.0, artifact_penalty=0.9) for _ in range(5)]
        agg.add_cells(cells_high + cells_low)
        brittle = agg.brittle_categories(flag_gap=0.10)
        assert brittle["dress"] is True

    def test_brittle_categories_false_when_gap_below_flag(self):
        """Gap within flag_gap -> brittle=False."""
        agg = CategoryAggregator(WEIGHTS)
        # Both phrasings produce identical scores
        cells = (
            [_sample("upper", "zh_001") for _ in range(3)] +
            [_sample("upper", "en_001") for _ in range(3)]
        )
        agg.add_cells(cells)
        brittle = agg.brittle_categories(flag_gap=0.10)
        assert brittle["upper"] is False

    def test_brittle_single_phrasing_is_not_brittle(self):
        """Only one phrasing -> cannot measure brittleness -> False."""
        agg = CategoryAggregator(WEIGHTS)
        agg.add_cells([_sample("lower", "zh_001") for _ in range(3)])
        brittle = agg.brittle_categories()
        assert brittle["lower"] is False

    def test_multiple_add_calls_accumulate(self):
        """Multiple add_cells calls should accumulate cells."""
        agg = CategoryAggregator(WEIGHTS)
        agg.add_cells([_sample("dress", "zh_001")])
        agg.add_cells([_sample("dress", "en_001")])
        breakdown = agg.category_user_prompt_breakdown()
        assert "zh_001" in breakdown["dress"]
        assert "en_001" in breakdown["dress"]

    def test_cells_without_user_prompt_id_bucketed_separately(self):
        """Cells missing user_prompt_id go to '__no_uid__' bucket."""
        agg = CategoryAggregator(WEIGHTS, categories=["dress"])
        cells = [{"sample_id": "s1", "category": "dress", "qwen_status": "yes",
                  "edit_correctness": 0.8, "garment_transfer_correctness": 0.75,
                  "preservation": 0.9, "artifact_penalty": 0.05}]
        agg.add_cells(cells)
        breakdown = agg.category_user_prompt_breakdown()
        assert "__no_uid__" in breakdown["dress"]

    def test_brittle_categories_custom_flag_gap(self):
        agg = CategoryAggregator(WEIGHTS)
        cells_high = [_sample("dress", "zh_001", qwen_status="yes",
                               edit_correctness=1.0, garment_transfer_correctness=1.0,
                               preservation=1.0, artifact_penalty=0.0) for _ in range(5)]
        cells_mid = [_sample("dress", "en_001", qwen_status="yes",
                              edit_correctness=0.85, garment_transfer_correctness=0.85,
                              preservation=0.85, artifact_penalty=0.05) for _ in range(5)]
        agg.add_cells(cells_high + cells_mid)
        # With very tight gap threshold, likely brittle
        brittle_tight = agg.brittle_categories(flag_gap=0.01)
        brittle_wide = agg.brittle_categories(flag_gap=0.50)
        assert brittle_tight["dress"] is True
        assert brittle_wide["dress"] is False


# ---------------------------------------------------------------------------
# Integration: build_score_context_for_pair -> rank_pairs_with_gate
# ---------------------------------------------------------------------------

class TestIntegrationScoringPipeline:
    def _make_cells_and_library(
        self,
        zh_score: float = 0.9,
        en_score: float = 0.3,
    ) -> tuple[list[dict], list[GemmaUserPrompt]]:
        cells = (
            [{"user_prompt_id": "zh_001", "overall_score": zh_score}] * 5 +
            [{"user_prompt_id": "en_001", "overall_score": en_score}] * 3
        )
        lib = _library(("zh_001", "zh"), ("en_001", "en"))
        return cells, lib

    def test_full_pipeline_language_balanced(self):
        cells, lib = self._make_cells_and_library(zh_score=0.9, en_score=0.3)
        cfg = _ScoringCfg()
        ctx = build_score_context_for_pair("pair_001", cells, lib, cfg)
        # Confirm base_mean = 0.60 (not count-weighted 0.675)
        assert ctx.total_score == pytest.approx(0.60, abs=1e-6)

    def test_full_pipeline_hard_gate_rejects(self):
        """en at 0.10 < 0.30 threshold -> hard gate rejects pair."""
        cells, lib = self._make_cells_and_library(zh_score=0.9, en_score=0.10)
        cfg = _ScoringCfg(min_per_user_prompt_score=0.30)
        ctx = build_score_context_for_pair("pair_001", cells, lib, cfg)

        from system_prompt_retrieval_agent.schemas import PromptPair
        pair = PromptPair(
            prompt_pair_id="pair_001",
            system_prompt_id="s1",
            negative_prompt_id="n1",
            round_id=1,
            system_prompt="sp",
            scores=ctx,
        )
        ranked, rejected = rank_pairs_with_gate([pair], cfg)
        assert len(ranked) == 0
        assert len(rejected) == 1
        assert rejected[0]["pair_id"] == "pair_001"

    def test_full_pipeline_accepted_with_penalty(self):
        """en at 0.50 passes gate; sensitivity > tolerance reduces score."""
        cells, lib = self._make_cells_and_library(zh_score=0.9, en_score=0.50)
        cfg = _ScoringCfg(
            min_per_user_prompt_score=0.30,
            prompt_sensitivity_penalty_weight=0.5,
            prompt_sensitivity_tolerance=0.10,
        )
        ctx = build_score_context_for_pair("pair_001", cells, lib, cfg)
        # base_mean = (0.9 + 0.5) / 2 = 0.70
        # sensitivity = 0.9 - 0.5 = 0.4
        # penalty = 0.5 * (0.4 - 0.10) = 0.15
        # overall (from build_score_context) = 0.70 - 0.15 = 0.55
        assert ctx.overall_score == pytest.approx(0.55, abs=1e-6)

        from system_prompt_retrieval_agent.schemas import PromptPair
        pair = PromptPair(
            prompt_pair_id="pair_001",
            system_prompt_id="s1",
            negative_prompt_id="n1",
            round_id=1,
            system_prompt="sp",
            scores=ctx,
        )
        ranked, rejected = rank_pairs_with_gate([pair], cfg)
        assert len(rejected) == 0
        assert len(ranked) == 1
