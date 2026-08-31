"""Tests for BudgetGuard and invariance metrics (S06.04, S06.10)."""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.evaluation.budget_guard import BudgetGuard, CostExhausted
from system_prompt_retrieval_agent.evaluation.invariance import compute_invariance_metrics


class TestBudgetGuard:
    def test_charge_within_daily_cap(self):
        guard = BudgetGuard(daily_usd_cap=1.0, per_round_usd_cap=10.0)
        guard.charge(0.50)
        assert abs(guard.daily_spent - 0.50) < 1e-9
        assert abs(guard.round_spent - 0.50) < 1e-9

    def test_charge_up_to_daily_cap_succeeds(self):
        guard = BudgetGuard(daily_usd_cap=1.0, per_round_usd_cap=10.0)
        guard.charge(0.50)
        guard.charge(0.50)  # Exactly at cap
        assert abs(guard.daily_spent - 1.0) < 1e-9

    def test_charge_exceeding_daily_cap_raises(self):
        guard = BudgetGuard(daily_usd_cap=1.0, per_round_usd_cap=10.0)
        guard.charge(0.50)
        guard.charge(0.50)
        # One more cent should fail
        with pytest.raises(CostExhausted):
            guard.charge(0.01)

    def test_charge_exceeding_daily_cap_no_mutation(self):
        guard = BudgetGuard(daily_usd_cap=1.0, per_round_usd_cap=10.0)
        guard.charge(0.99)
        try:
            guard.charge(0.02)
        except CostExhausted:
            pass
        # daily_spent should NOT be mutated on failure
        assert abs(guard.daily_spent - 0.99) < 1e-9

    def test_charge_exceeding_round_cap_raises(self):
        guard = BudgetGuard(daily_usd_cap=100.0, per_round_usd_cap=1.0)
        guard.charge(0.99)
        with pytest.raises(CostExhausted):
            guard.charge(0.02)

    def test_reset_round_zeros_round_spent(self):
        guard = BudgetGuard(daily_usd_cap=10.0, per_round_usd_cap=5.0)
        guard.charge(2.0)
        guard.reset_round()
        assert abs(guard.round_spent - 0.0) < 1e-9

    def test_reset_round_preserves_daily_spent(self):
        guard = BudgetGuard(daily_usd_cap=10.0, per_round_usd_cap=5.0)
        guard.charge(2.0)
        guard.reset_round()
        assert abs(guard.daily_spent - 2.0) < 1e-9

    def test_round_cap_resets_but_daily_still_tracks(self):
        """After reset_round, round_spent resets but daily keeps accumulating."""
        guard = BudgetGuard(daily_usd_cap=10.0, per_round_usd_cap=2.0)
        guard.charge(2.0)
        guard.reset_round()
        guard.charge(2.0)
        assert abs(guard.round_spent - 2.0) < 1e-9
        assert abs(guard.daily_spent - 4.0) < 1e-9

    def test_cost_exhausted_is_runtime_error(self):
        assert issubclass(CostExhausted, RuntimeError)


# ---------------------------------------------------------------------------
# compute_invariance_metrics (S06.10)
# ---------------------------------------------------------------------------


class TestComputeInvarianceMetrics:
    def test_empty_grid_returns_defaults(self):
        result = compute_invariance_metrics({})
        assert result["per_user_prompt_score"] == {}
        assert result["prompt_sensitivity"] == 0.0
        assert result["language_brittle"] is False
        assert result["n_cells"] == 0

    def test_per_user_prompt_score_mean(self):
        grid = {
            ("up_01", "s01"): 0.8,
            ("up_01", "s02"): 0.6,
            ("up_02", "s01"): 0.4,
        }
        result = compute_invariance_metrics(grid)
        assert abs(result["per_user_prompt_score"]["up_01"] - 0.7) < 1e-9
        assert abs(result["per_user_prompt_score"]["up_02"] - 0.4) < 1e-9

    def test_prompt_sensitivity_max_minus_min(self):
        grid = {
            ("up_01", "s01"): 0.9,
            ("up_02", "s01"): 0.3,
        }
        result = compute_invariance_metrics(grid)
        assert abs(result["prompt_sensitivity"] - 0.6) < 1e-9

    def test_prompt_sensitivity_single_phrasing_is_zero(self):
        grid = {
            ("up_01", "s01"): 0.7,
            ("up_01", "s02"): 0.5,
        }
        result = compute_invariance_metrics(grid)
        assert result["prompt_sensitivity"] == 0.0

    def test_worst_user_prompt_id(self):
        grid = {
            ("up_01", "s01"): 0.9,
            ("up_02", "s01"): 0.2,
            ("up_03", "s01"): 0.6,
        }
        result = compute_invariance_metrics(grid)
        assert result["worst_user_prompt_id"] == "up_02"
        assert abs(result["worst_user_prompt_score"] - 0.2) < 1e-9

    def test_zh_en_mean_scores(self):
        grid = {
            ("zh_01", "s01"): 0.8,
            ("zh_02", "s01"): 0.6,
            ("en_01", "s01"): 0.3,
        }
        lang_map = {"zh_01": "zh", "zh_02": "zh", "en_01": "en"}
        result = compute_invariance_metrics(grid, user_prompt_language=lang_map)
        assert abs(result["zh_mean_score"] - 0.7) < 1e-9
        assert abs(result["en_mean_score"] - 0.3) < 1e-9
        assert abs(result["cross_lingual_gap"] - 0.4) < 1e-9

    def test_language_brittle_flag_set(self):
        grid = {
            ("zh_01", "s01"): 0.9,
            ("en_01", "s01"): 0.3,
        }
        lang_map = {"zh_01": "zh", "en_01": "en"}
        result = compute_invariance_metrics(
            grid,
            user_prompt_language=lang_map,
            flag_language_brittle_gap=0.10,
        )
        assert result["language_brittle"] is True
        assert abs(result["cross_lingual_gap"] - 0.6) < 1e-9

    def test_language_brittle_flag_not_set_below_threshold(self):
        grid = {
            ("zh_01", "s01"): 0.7,
            ("en_01", "s01"): 0.65,
        }
        lang_map = {"zh_01": "zh", "en_01": "en"}
        result = compute_invariance_metrics(
            grid,
            user_prompt_language=lang_map,
            flag_language_brittle_gap=0.10,
        )
        assert result["language_brittle"] is False

    def test_cross_lingual_gap_none_without_language_map(self):
        grid = {("up_01", "s01"): 0.8, ("up_02", "s01"): 0.5}
        result = compute_invariance_metrics(grid)
        assert result["cross_lingual_gap"] is None
        assert result["zh_mean_score"] is None
        assert result["en_mean_score"] is None

    def test_n_cells_count(self):
        grid = {
            ("up_01", "s01"): 0.8,
            ("up_01", "s02"): 0.7,
            ("up_02", "s01"): 0.5,
        }
        result = compute_invariance_metrics(grid)
        assert result["n_cells"] == 3

    def test_mean_score_by_user_prompt_alias(self):
        """mean_score_by_user_prompt should be the same object as per_user_prompt_score."""
        grid = {("up_01", "s01"): 0.6}
        result = compute_invariance_metrics(grid)
        assert result["mean_score_by_user_prompt"] is result["per_user_prompt_score"]

    def test_language_brittle_below_threshold_is_not_set(self):
        """gap strictly below threshold means NOT brittle."""
        grid = {
            ("zh_01", "s01"): 0.75,
            ("en_01", "s01"): 0.70,
        }
        lang_map = {"zh_01": "zh", "en_01": "en"}
        result = compute_invariance_metrics(
            grid,
            user_prompt_language=lang_map,
            flag_language_brittle_gap=0.10,
        )
        # gap = 0.05 < 0.10 → not brittle
        assert result["language_brittle"] is False
