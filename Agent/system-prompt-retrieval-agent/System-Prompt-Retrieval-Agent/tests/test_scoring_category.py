"""Tests for scoring/category.py (S07.06)."""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.schemas import CategoryScoreContext
from system_prompt_retrieval_agent.scoring.category import compute_category_scores

WEIGHTS = {
    "qwen_pass_rate": 0.40,
    "edit_correctness": 0.20,
    "garment_transfer_correctness": 0.15,
    "preservation": 0.15,
    "artifact_penalty": -0.10,
    "category_balance_bonus": 0.0,
}


def _make_sample(
    sample_id: str,
    category: str,
    qwen_status: str = "yes",
    edit_correctness: float = 0.8,
    garment_transfer_correctness: float = 0.75,
    preservation: float = 0.9,
    artifact_penalty: float = 0.05,
) -> dict:
    return {
        "sample_id": sample_id,
        "category": category,
        "qwen_status": qwen_status,
        "edit_correctness": edit_correctness,
        "garment_transfer_correctness": garment_transfer_correctness,
        "preservation": preservation,
        "artifact_penalty": artifact_penalty,
    }


class TestMissingCategory:
    def test_missing_category_returns_none(self):
        """Categories with no samples return None."""
        samples = [_make_sample("s1", "dress")]
        result = compute_category_scores(samples, WEIGHTS, categories=["dress", "lower", "upper"])
        assert result["dress"] is not None
        assert result["lower"] is None
        assert result["upper"] is None

    def test_all_categories_missing(self):
        """No samples -> all None."""
        result = compute_category_scores([], WEIGHTS, categories=["dress", "lower", "upper"])
        for cat in ["dress", "lower", "upper"]:
            assert result[cat] is None

    def test_extra_sample_category_not_in_requested(self):
        """Samples with unconfigured categories are ignored for requested categories."""
        samples = [_make_sample("s1", "accessories")]
        result = compute_category_scores(samples, WEIGHTS, categories=["dress"])
        assert result["dress"] is None


class TestQwenPassRate:
    def test_parse_fail_excluded_from_denominator(self):
        """parse_fail_* statuses must not count towards denominator."""
        samples = [
            _make_sample("s1", "dress", qwen_status="yes"),
            _make_sample("s2", "dress", qwen_status="no"),
            _make_sample("s3", "dress", qwen_status="parse_fail_json"),
            _make_sample("s4", "dress", qwen_status="parse_fail_schema"),
        ]
        result = compute_category_scores(samples, WEIGHTS, categories=["dress"])
        ctx = result["dress"]
        assert ctx is not None
        # yes=1, no=1, denom=2 -> pass_rate=0.5
        assert ctx.sub_scores.qwen_pass_rate == pytest.approx(0.5, abs=1e-6)

    def test_all_parse_fail_gives_zero_pass_rate(self):
        """All parse_fail -> denom=0 -> pass_rate=0.0."""
        samples = [
            _make_sample("s1", "upper", qwen_status="parse_fail_json"),
            _make_sample("s2", "upper", qwen_status="parse_fail_schema"),
        ]
        result = compute_category_scores(samples, WEIGHTS, categories=["upper"])
        ctx = result["upper"]
        assert ctx is not None
        assert ctx.sub_scores.qwen_pass_rate == 0.0

    def test_all_yes_gives_full_pass_rate(self):
        samples = [_make_sample(f"s{i}", "lower", qwen_status="yes") for i in range(5)]
        result = compute_category_scores(samples, WEIGHTS, categories=["lower"])
        ctx = result["lower"]
        assert ctx is not None
        assert ctx.sub_scores.qwen_pass_rate == pytest.approx(1.0, abs=1e-6)


class TestBalanceBonusNotApplied:
    def test_balance_bonus_zero_for_single_category(self):
        """Category balance bonus should be 0.0 when only one category computed."""
        from system_prompt_retrieval_agent.scoring.aggregate import compute_category_balance_bonus

        samples = [_make_sample("s1", "dress")]
        result = compute_category_scores(samples, WEIGHTS, categories=["dress"])
        cat_scores = {
            cat: ctx.weighted_score
            for cat, ctx in result.items()
            if ctx is not None and ctx.weighted_score is not None
        }
        bonus = compute_category_balance_bonus(cat_scores)
        assert bonus == 0.0

    def test_all_three_categories_valid(self):
        """Three categories with samples -> weighted_score populated for each."""
        samples = (
            [_make_sample(f"dress{i}", "dress") for i in range(3)]
            + [_make_sample(f"lower{i}", "lower") for i in range(3)]
            + [_make_sample(f"upper{i}", "upper") for i in range(3)]
        )
        result = compute_category_scores(samples, WEIGHTS, categories=["dress", "lower", "upper"])
        for cat in ["dress", "lower", "upper"]:
            ctx = result[cat]
            assert ctx is not None
            assert isinstance(ctx, CategoryScoreContext)
            assert ctx.weighted_score is not None
            assert 0.0 <= ctx.weighted_score <= 1.0


class TestSubScoreAveraging:
    def test_sub_scores_averaged_correctly(self):
        """Sub-scores should be averaged across all samples in a category."""
        samples = [
            _make_sample("s1", "upper", edit_correctness=0.6, preservation=0.8),
            _make_sample("s2", "upper", edit_correctness=0.8, preservation=1.0),
        ]
        result = compute_category_scores(samples, WEIGHTS, categories=["upper"])
        ctx = result["upper"]
        assert ctx is not None
        assert ctx.sub_scores.edit_correctness == pytest.approx(0.7, abs=1e-6)
        assert ctx.sub_scores.preservation == pytest.approx(0.9, abs=1e-6)

    def test_total_score_is_mean_of_positives(self):
        """total_score should be unweighted mean of qwen, edit, gt, preservation."""
        samples = [
            _make_sample(
                "s1",
                "dress",
                qwen_status="yes",
                edit_correctness=1.0,
                garment_transfer_correctness=1.0,
                preservation=1.0,
            )
        ]
        result = compute_category_scores(samples, WEIGHTS, categories=["dress"])
        ctx = result["dress"]
        assert ctx is not None
        # qwen=1.0, edit=1.0, gt=1.0, preservation=1.0 -> mean=1.0
        assert ctx.total_score == pytest.approx(1.0, abs=1e-6)
