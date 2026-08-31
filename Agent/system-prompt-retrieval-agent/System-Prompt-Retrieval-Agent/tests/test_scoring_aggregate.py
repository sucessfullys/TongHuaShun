"""Tests for scoring/aggregate.py (S07.05)."""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.schemas import PromptPairSubScores
from system_prompt_retrieval_agent.scoring.aggregate import (
    compute_category_balance_bonus,
    compute_overall_score,
)

# Default weights that sum to 1.0 (including negative artifact_penalty offset)
WEIGHTS_BALANCED = {
    "qwen_pass_rate": 0.40,
    "edit_correctness": 0.20,
    "garment_transfer_correctness": 0.15,
    "preservation": 0.15,
    "artifact_penalty": -0.10,
    "category_balance_bonus": 0.20,
}

WEIGHTS_SIMPLE = {
    "qwen_pass_rate": 0.40,
    "edit_correctness": 0.20,
    "garment_transfer_correctness": 0.15,
    "preservation": 0.15,
    "artifact_penalty": 0.0,
    "category_balance_bonus": 0.10,
}


class TestOverallScoreRange:
    def test_all_ones_clamped_to_one(self):
        sub = {
            "qwen_pass_rate": 1.0,
            "edit_correctness": 1.0,
            "garment_transfer_correctness": 1.0,
            "preservation": 1.0,
            "artifact_penalty": 0.0,
        }
        score = compute_overall_score(sub, WEIGHTS_SIMPLE, category_balance_bonus=1.0)
        assert 0.0 <= score <= 1.0

    def test_zero_inputs_gives_zero(self):
        sub = {
            "qwen_pass_rate": 0.0,
            "edit_correctness": 0.0,
            "garment_transfer_correctness": 0.0,
            "preservation": 0.0,
            "artifact_penalty": 0.0,
        }
        score = compute_overall_score(sub, WEIGHTS_SIMPLE)
        assert score == 0.0

    def test_realistic_weights_sum_in_unit_range(self):
        """Weights summing to 1.0 with realistic sub-scores should produce overall in [0,1]."""
        sub = PromptPairSubScores(
            qwen_pass_rate=0.8,
            edit_correctness=0.7,
            garment_transfer_correctness=0.75,
            preservation=0.9,
            artifact_penalty=0.1,
        )
        score = compute_overall_score(sub, WEIGHTS_BALANCED, category_balance_bonus=0.8)
        assert 0.0 <= score <= 1.0


class TestHardGate:
    def test_hard_gate_triggered_below_threshold(self):
        """qwen_pass_rate < 0.5 should trigger multiplier."""
        sub = {
            "qwen_pass_rate": 0.4,
            "edit_correctness": 1.0,
            "garment_transfer_correctness": 1.0,
            "preservation": 1.0,
            "artifact_penalty": 0.0,
        }
        w = {
            "qwen_pass_rate": 0.40,
            "edit_correctness": 0.20,
            "garment_transfer_correctness": 0.15,
            "preservation": 0.15,
            "artifact_penalty": 0.0,
            "category_balance_bonus": 0.0,
        }
        score_no_gate = compute_overall_score(
            sub, w, hard_gate_threshold=0.5, hard_gate_multiplier=1.0
        )
        score_with_gate = compute_overall_score(
            sub, w, hard_gate_threshold=0.5, hard_gate_multiplier=0.3
        )
        assert score_with_gate == pytest.approx(score_no_gate * 0.3, abs=1e-6)

    def test_hard_gate_not_triggered_at_threshold(self):
        """qwen_pass_rate exactly at threshold should NOT trigger gate."""
        sub = {"qwen_pass_rate": 0.5, "edit_correctness": 0.0, "garment_transfer_correctness": 0.0, "preservation": 0.0, "artifact_penalty": 0.0}
        w = {"qwen_pass_rate": 1.0, "edit_correctness": 0.0, "garment_transfer_correctness": 0.0, "preservation": 0.0, "artifact_penalty": 0.0, "category_balance_bonus": 0.0}
        score = compute_overall_score(sub, w, hard_gate_threshold=0.5, hard_gate_multiplier=0.3)
        assert score == pytest.approx(0.5, abs=1e-6)

    def test_hard_gate_default_threshold(self):
        """Default hard_gate_threshold=0.5."""
        sub = {"qwen_pass_rate": 0.3}
        w = {"qwen_pass_rate": 1.0, "edit_correctness": 0.0, "garment_transfer_correctness": 0.0, "preservation": 0.0, "artifact_penalty": 0.0, "category_balance_bonus": 0.0}
        score = compute_overall_score(sub, w)
        # Without gate: 0.3 * 1.0 = 0.3; with gate: 0.3 * 0.3 = 0.09
        assert score == pytest.approx(0.09, abs=1e-6)


class TestMissingSubScore:
    def test_missing_key_treated_as_zero(self):
        """Missing sub-score key should not raise; treated as 0.0."""
        sub = {"qwen_pass_rate": 0.8}  # rest missing
        w = {
            "qwen_pass_rate": 0.40,
            "edit_correctness": 0.20,
            "garment_transfer_correctness": 0.15,
            "preservation": 0.15,
            "artifact_penalty": 0.0,
            "category_balance_bonus": 0.0,
        }
        score = compute_overall_score(sub, w)
        assert score == pytest.approx(0.40 * 0.8, abs=1e-6)

    def test_none_value_treated_as_zero(self):
        """None sub-score value should be treated as 0.0."""
        sub = PromptPairSubScores(
            qwen_pass_rate=0.8,
            edit_correctness=None,
            garment_transfer_correctness=None,
            preservation=None,
            artifact_penalty=None,
        )
        w = {
            "qwen_pass_rate": 0.40,
            "edit_correctness": 0.20,
            "garment_transfer_correctness": 0.15,
            "preservation": 0.15,
            "artifact_penalty": 0.0,
            "category_balance_bonus": 0.0,
        }
        score = compute_overall_score(sub, w)
        assert score == pytest.approx(0.40 * 0.8, abs=1e-6)


class TestArtifactPenalty:
    def test_negative_artifact_weight_lowers_score(self):
        """Higher artifact_penalty value with negative weight -> lower score."""
        w = {
            "qwen_pass_rate": 0.5,
            "edit_correctness": 0.0,
            "garment_transfer_correctness": 0.0,
            "preservation": 0.0,
            "artifact_penalty": -0.5,
            "category_balance_bonus": 0.0,
        }
        sub_low = {"qwen_pass_rate": 0.8, "artifact_penalty": 0.1}
        sub_high = {"qwen_pass_rate": 0.8, "artifact_penalty": 0.9}
        score_low = compute_overall_score(sub_low, w)
        score_high = compute_overall_score(sub_high, w)
        assert score_low > score_high


class TestClamp:
    def test_clamp_above_one(self):
        """Values that would exceed 1.0 before clamping should be clamped to 1.0."""
        # Use huge weights so raw score > 1.0
        w = {
            "qwen_pass_rate": 2.0,
            "edit_correctness": 0.0,
            "garment_transfer_correctness": 0.0,
            "preservation": 0.0,
            "artifact_penalty": 0.0,
            "category_balance_bonus": 0.0,
        }
        sub = {"qwen_pass_rate": 1.0}
        score = compute_overall_score(sub, w)
        assert score == 1.0

    def test_clamp_below_zero(self):
        """Values that would go below 0.0 should be clamped to 0.0."""
        w = {
            "qwen_pass_rate": 0.0,
            "edit_correctness": 0.0,
            "garment_transfer_correctness": 0.0,
            "preservation": 0.0,
            "artifact_penalty": -5.0,
            "category_balance_bonus": 0.0,
        }
        sub = {"qwen_pass_rate": 0.8, "artifact_penalty": 1.0}
        score = compute_overall_score(sub, w)
        assert score == 0.0


class TestCategoryBalanceBonus:
    def test_balanced_categories_returns_one(self):
        """Perfectly balanced categories should return 1.0."""
        cats = {"dress": 0.8, "lower": 0.8, "upper": 0.8}
        bonus = compute_category_balance_bonus(cats)
        assert bonus == 1.0

    def test_mostly_balanced_returns_one(self):
        """min/max ratio >= 0.70 returns 1.0."""
        cats = {"dress": 0.7, "lower": 1.0, "upper": 0.9}
        bonus = compute_category_balance_bonus(cats)
        assert bonus == 1.0

    def test_imbalanced_returns_ratio(self):
        """min/max ratio < 0.70 returns the ratio."""
        cats = {"dress": 0.3, "lower": 1.0}
        bonus = compute_category_balance_bonus(cats)
        assert 0.0 < bonus <= 1.0
        assert bonus == pytest.approx(0.3 / 1.0, abs=1e-6)

    def test_single_category_returns_zero(self):
        """Fewer than 2 categories -> 0.0."""
        bonus = compute_category_balance_bonus({"dress": 0.8})
        assert bonus == 0.0

    def test_zero_score_returns_zero(self):
        """Any zero score -> 0.0."""
        bonus = compute_category_balance_bonus({"dress": 0.0, "lower": 0.8})
        assert bonus == 0.0

    def test_empty_returns_zero(self):
        """Empty dict -> 0.0."""
        bonus = compute_category_balance_bonus({})
        assert bonus == 0.0

    def test_pydantic_weights_accepted(self):
        """EvaluationWeights pydantic model accepted via model_dump."""
        from system_prompt_retrieval_agent.config import EvaluationWeights

        ew = EvaluationWeights()
        sub = {"qwen_pass_rate": 0.8, "edit_correctness": 0.7, "garment_transfer_correctness": 0.75, "preservation": 0.9, "artifact_penalty": 0.1}
        score = compute_overall_score(sub, ew)
        assert 0.0 <= score <= 1.0
