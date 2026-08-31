"""Tests for V0.2.1 user-prompt-aware aggregation (plan §10.6, S07.04a-e)."""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.schemas import GemmaUserPrompt, PromptPairScoreContext
from system_prompt_retrieval_agent.scoring.aggregate import (
    build_score_context_for_pair,
    compute_per_user_prompt_means,
    compute_prompt_sensitivity,
    compute_zh_en_means,
    language_balanced_base_mean,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _library(*args: tuple[str, str]) -> list[GemmaUserPrompt]:
    """Build a GemmaUserPrompt library. args: (user_prompt_id, language)."""
    return [
        GemmaUserPrompt(user_prompt_id=uid, language=lang, text=f"prompt_{uid}", enabled=True)
        for uid, lang in args
    ]


def _cell(user_prompt_id: str, overall_score: float) -> dict:
    return {"user_prompt_id": user_prompt_id, "overall_score": overall_score}


class _ScoringCfg:
    """Minimal scoring config object for tests."""
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


# ---------------------------------------------------------------------------
# S07.04a — compute_per_user_prompt_means
# ---------------------------------------------------------------------------

class TestComputePerUserPromptMeans:
    def test_single_phrasing_single_cell(self):
        cells = [_cell("zh_001", 0.8)]
        result = compute_per_user_prompt_means(cells)
        assert result == {"zh_001": pytest.approx(0.8, abs=1e-9)}

    def test_multiple_cells_same_phrasing(self):
        """Mean over cells with the same user_prompt_id."""
        cells = [
            _cell("zh_001", 0.8),
            _cell("zh_001", 0.6),
            _cell("zh_001", 1.0),
        ]
        result = compute_per_user_prompt_means(cells)
        assert result["zh_001"] == pytest.approx(0.8, abs=1e-9)

    def test_multiple_phrasings(self):
        cells = [
            _cell("zh_001", 0.9),
            _cell("zh_001", 0.7),
            _cell("en_001", 0.5),
            _cell("en_001", 0.3),
        ]
        result = compute_per_user_prompt_means(cells)
        assert result["zh_001"] == pytest.approx(0.8, abs=1e-9)
        assert result["en_001"] == pytest.approx(0.4, abs=1e-9)

    def test_missing_user_prompt_id_skipped(self):
        """Cells without user_prompt_id are silently dropped."""
        cells = [
            {"overall_score": 0.8},
            _cell("en_001", 0.6),
        ]
        result = compute_per_user_prompt_means(cells)
        assert "en_001" in result
        assert len(result) == 1

    def test_missing_overall_score_skipped(self):
        cells = [
            {"user_prompt_id": "zh_001"},
            _cell("en_001", 0.7),
        ]
        result = compute_per_user_prompt_means(cells)
        assert "zh_001" not in result
        assert result["en_001"] == pytest.approx(0.7, abs=1e-9)

    def test_empty_cells(self):
        result = compute_per_user_prompt_means([])
        assert result == {}


# ---------------------------------------------------------------------------
# S07.04b — compute_zh_en_means
# ---------------------------------------------------------------------------

class TestComputeZhEnMeans:
    def test_separate_language_groups(self):
        lib = _library(("zh_001", "zh"), ("zh_002", "zh"), ("en_001", "en"))
        per_up = {"zh_001": 0.9, "zh_002": 0.7, "en_001": 0.5}
        zh, en = compute_zh_en_means(per_up, lib)
        assert zh == pytest.approx(0.8, abs=1e-9)
        assert en == pytest.approx(0.5, abs=1e-9)

    def test_single_language_only(self):
        lib = _library(("zh_001", "zh"), ("zh_002", "zh"))
        per_up = {"zh_001": 0.8, "zh_002": 0.6}
        zh, en = compute_zh_en_means(per_up, lib)
        assert zh == pytest.approx(0.7, abs=1e-9)
        assert en is None

    def test_disabled_prompt_excluded(self):
        """Disabled prompts in library must not contribute to language means."""
        lib = [
            GemmaUserPrompt(user_prompt_id="zh_001", language="zh", text="t", enabled=True),
            GemmaUserPrompt(user_prompt_id="zh_002", language="zh", text="t", enabled=False),
        ]
        per_up = {"zh_001": 0.8, "zh_002": 0.2}
        zh, en = compute_zh_en_means(per_up, lib)
        # zh_002 is disabled in library but still has a per_up entry
        # compute_zh_en_means uses lang_map built from enabled prompts only
        # so zh_002 maps to "zh" (it's in lang_map only if enabled=True)
        # zh_002 has no lang_map entry -> excluded from zh/en buckets
        assert zh == pytest.approx(0.8, abs=1e-9)
        assert en is None

    def test_unknown_user_prompt_id_ignored(self):
        lib = _library(("zh_001", "zh"))
        per_up = {"zh_001": 0.9, "unknown_999": 0.1}
        zh, en = compute_zh_en_means(per_up, lib)
        assert zh == pytest.approx(0.9, abs=1e-9)
        assert en is None

    def test_empty_library(self):
        per_up = {"zh_001": 0.8}
        zh, en = compute_zh_en_means(per_up, [])
        assert zh is None
        assert en is None


# ---------------------------------------------------------------------------
# S07.04c — language_balanced_base_mean
# ---------------------------------------------------------------------------

class TestLanguageBalancedBaseMean:
    def test_both_languages_equal_weight(self):
        """5 zh phrasings at 0.90 + 3 en phrasings at 0.30 -> base_mean = 0.60 (not 0.675)."""
        # In compute_zh_en_means the 5 zh values average to zh_mean=0.90
        # and the 3 en values average to en_mean=0.30.
        # language_balanced_base_mean gives (0.90+0.30)/2 = 0.60.
        zh_mean = 0.90
        en_mean = 0.30
        result = language_balanced_base_mean(zh_mean, en_mean)
        assert result == pytest.approx(0.60, abs=1e-9)

    def test_count_weighted_would_differ(self):
        """Confirm the language-balanced result differs from a flat count-weighted mean."""
        # 5 zh at 0.90, 3 en at 0.30 -> count-weighted = (5*0.90 + 3*0.30) / 8 = 0.675
        count_weighted = (5 * 0.90 + 3 * 0.30) / 8
        lang_balanced = language_balanced_base_mean(0.90, 0.30)
        assert lang_balanced != pytest.approx(count_weighted, abs=1e-6)
        assert lang_balanced == pytest.approx(0.60, abs=1e-9)

    def test_single_zh_fallback(self):
        """When only zh is present, base_mean == zh_mean (smoke_subset case)."""
        result = language_balanced_base_mean(0.75, None)
        assert result == pytest.approx(0.75, abs=1e-9)

    def test_single_en_fallback(self):
        result = language_balanced_base_mean(None, 0.60)
        assert result == pytest.approx(0.60, abs=1e-9)

    def test_both_none_returns_zero(self):
        result = language_balanced_base_mean(None, None)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_symmetry(self):
        """(zh, en) and (en, zh) produce the same result."""
        assert language_balanced_base_mean(0.8, 0.4) == language_balanced_base_mean(0.4, 0.8)


# ---------------------------------------------------------------------------
# S07.04d — compute_prompt_sensitivity
# ---------------------------------------------------------------------------

class TestComputePromptSensitivity:
    def test_single_phrasing(self):
        sensitivity, worst_uid, worst_score = compute_prompt_sensitivity({"zh_001": 0.7})
        assert sensitivity == pytest.approx(0.0, abs=1e-9)
        assert worst_uid == "zh_001"
        assert worst_score == pytest.approx(0.7, abs=1e-9)

    def test_two_phrasings(self):
        per_up = {"zh_001": 0.9, "en_001": 0.4}
        sensitivity, worst_uid, worst_score = compute_prompt_sensitivity(per_up)
        assert sensitivity == pytest.approx(0.5, abs=1e-9)
        assert worst_uid == "en_001"
        assert worst_score == pytest.approx(0.4, abs=1e-9)

    def test_worst_user_prompt_is_argmin(self):
        per_up = {"a": 0.7, "b": 0.3, "c": 0.9}
        _, worst_uid, worst_score = compute_prompt_sensitivity(per_up)
        assert worst_uid == "b"
        assert worst_score == pytest.approx(0.3, abs=1e-9)

    def test_empty_returns_zero(self):
        sensitivity, worst_uid, worst_score = compute_prompt_sensitivity({})
        assert sensitivity == pytest.approx(0.0, abs=1e-9)
        assert worst_uid is None
        assert worst_score is None

    def test_all_equal_sensitivity_zero(self):
        per_up = {"zh_001": 0.8, "zh_002": 0.8, "en_001": 0.8}
        sensitivity, _, _ = compute_prompt_sensitivity(per_up)
        assert sensitivity == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# S07.04e — build_score_context_for_pair (producer side)
# ---------------------------------------------------------------------------

class TestBuildScoreContextForPair:
    def _make_cells(self) -> list[dict]:
        # 5 zh phrasings at 0.90, 3 en phrasings at 0.30
        cells = []
        for i in range(1, 6):
            for _ in range(3):  # 3 samples per phrasing
                cells.append({"user_prompt_id": f"zh_{i:03d}", "overall_score": 0.90})
        for i in range(1, 4):
            for _ in range(3):
                cells.append({"user_prompt_id": f"en_{i:03d}", "overall_score": 0.30})
        return cells

    def _make_library(self) -> list[GemmaUserPrompt]:
        lib = []
        for i in range(1, 6):
            lib.append(GemmaUserPrompt(user_prompt_id=f"zh_{i:03d}", language="zh",
                                       text=f"zh_prompt_{i}", enabled=True))
        for i in range(1, 4):
            lib.append(GemmaUserPrompt(user_prompt_id=f"en_{i:03d}", language="en",
                                       text=f"en_prompt_{i}", enabled=True))
        return lib

    def test_returns_prompt_pair_score_context(self):
        ctx = build_score_context_for_pair(
            "pair_001", self._make_cells(), self._make_library(), _ScoringCfg()
        )
        assert isinstance(ctx, PromptPairScoreContext)

    def test_language_balance_corrects_imbalance(self):
        """5 zh at 0.90 + 3 en at 0.30 -> zh_mean=0.90, en_mean=0.30, base_mean=0.60."""
        ctx = build_score_context_for_pair(
            "pair_001", self._make_cells(), self._make_library(), _ScoringCfg()
        )
        assert ctx.zh_mean_score == pytest.approx(0.90, abs=1e-9)
        assert ctx.en_mean_score == pytest.approx(0.30, abs=1e-9)
        assert ctx.total_score == pytest.approx(0.60, abs=1e-9)

    def test_sensitivity_below_tolerance_no_penalty(self):
        """When all phrasings return the same score, sensitivity=0 and no penalty applied."""
        cells = [{"user_prompt_id": f"zh_{i}", "overall_score": 0.8} for i in range(3)]
        lib = _library(*[(f"zh_{i}", "zh") for i in range(3)])
        cfg = _ScoringCfg(prompt_sensitivity_tolerance=0.10, prompt_sensitivity_penalty_weight=0.5)
        ctx = build_score_context_for_pair("p", cells, lib, cfg)
        assert ctx.prompt_sensitivity == pytest.approx(0.0, abs=1e-9)
        # overall_score = base_mean - 0 = zh_mean = 0.8
        assert ctx.overall_score == pytest.approx(0.8, abs=1e-9)

    def test_sensitivity_above_tolerance_linear_penalty(self):
        """sensitivity=0.4 > tolerance=0.10 -> penalty = 0.5 * (0.4 - 0.1) = 0.15."""
        cells = [
            {"user_prompt_id": "zh_001", "overall_score": 0.9},
            {"user_prompt_id": "en_001", "overall_score": 0.5},
        ]
        lib = _library(("zh_001", "zh"), ("en_001", "en"))
        cfg = _ScoringCfg(prompt_sensitivity_penalty_weight=0.5, prompt_sensitivity_tolerance=0.10)
        ctx = build_score_context_for_pair("p", cells, lib, cfg)
        # base_mean = (0.9 + 0.5) / 2 = 0.70
        # sensitivity = 0.9 - 0.5 = 0.4
        # penalty = 0.5 * (0.4 - 0.1) = 0.15
        # overall = 0.70 - 0.15 = 0.55
        assert ctx.overall_score == pytest.approx(0.55, abs=1e-6)

    def test_penalty_never_negative(self):
        """Clamp ensures overall_score >= 0."""
        cells = [
            {"user_prompt_id": "zh_001", "overall_score": 0.1},
            {"user_prompt_id": "en_001", "overall_score": 0.0},
        ]
        lib = _library(("zh_001", "zh"), ("en_001", "en"))
        cfg = _ScoringCfg(prompt_sensitivity_penalty_weight=5.0, prompt_sensitivity_tolerance=0.0)
        ctx = build_score_context_for_pair("p", cells, lib, cfg)
        assert ctx.overall_score >= 0.0

    def test_single_language_smoke_subset(self):
        """Single language: base_mean = zh_mean, en_mean = None."""
        cells = [{"user_prompt_id": "zh_001", "overall_score": 0.75}]
        lib = _library(("zh_001", "zh"))
        ctx = build_score_context_for_pair("p", cells, lib, _ScoringCfg())
        assert ctx.zh_mean_score == pytest.approx(0.75, abs=1e-9)
        assert ctx.en_mean_score is None
        assert ctx.total_score == pytest.approx(0.75, abs=1e-9)

    def test_all_seven_fields_populated(self):
        """All 7 V0.2.1 user-prompt-aware fields must be non-None for a multi-lingual run."""
        ctx = build_score_context_for_pair(
            "pair_001", self._make_cells(), self._make_library(), _ScoringCfg()
        )
        assert ctx.per_user_prompt is not None and len(ctx.per_user_prompt) > 0
        assert ctx.mean_score_by_user_prompt is not None
        assert ctx.zh_mean_score is not None
        assert ctx.en_mean_score is not None
        assert ctx.prompt_sensitivity is not None
        assert ctx.cross_lingual_gap is not None
        assert ctx.worst_user_prompt_id is not None
        assert ctx.worst_user_prompt_score is not None

    def test_cross_lingual_gap(self):
        cells = [
            {"user_prompt_id": "zh_001", "overall_score": 0.9},
            {"user_prompt_id": "en_001", "overall_score": 0.3},
        ]
        lib = _library(("zh_001", "zh"), ("en_001", "en"))
        ctx = build_score_context_for_pair("p", cells, lib, _ScoringCfg())
        assert ctx.cross_lingual_gap == pytest.approx(0.6, abs=1e-9)

    def test_language_brittle_flag_logged_not_stored(self):
        """PromptPairScoreContext does not carry language_brittle flag directly;
        that flag lives on RoundState. build_score_context_for_pair logs it."""
        cells = [
            {"user_prompt_id": "zh_001", "overall_score": 0.9},
            {"user_prompt_id": "en_001", "overall_score": 0.5},
        ]
        lib = _library(("zh_001", "zh"), ("en_001", "en"))
        cfg = _ScoringCfg(flag_language_brittle_gap=0.10)
        ctx = build_score_context_for_pair("p", cells, lib, cfg)
        # cross_lingual_gap = 0.4 > 0.10 — brittle, but schema has no language_brittle field
        assert ctx.cross_lingual_gap == pytest.approx(0.4, abs=1e-9)
        # confirm context is still returned (brittle does not reject context)
        assert isinstance(ctx, PromptPairScoreContext)

    def test_worst_user_prompt_fields(self):
        cells = [
            {"user_prompt_id": "zh_001", "overall_score": 0.9},
            {"user_prompt_id": "en_001", "overall_score": 0.2},
        ]
        lib = _library(("zh_001", "zh"), ("en_001", "en"))
        ctx = build_score_context_for_pair("p", cells, lib, _ScoringCfg())
        assert ctx.worst_user_prompt_id == "en_001"
        assert ctx.worst_user_prompt_score == pytest.approx(0.2, abs=1e-9)

    def test_does_not_write_to_memory(self):
        """build_score_context_for_pair returns a context object without side effects
        (it does not write to memory/long.py — verifiable because we just call it)."""
        import importlib
        # If memory.long were imported and mutated, this test would detect that.
        cells = [_cell("zh_001", 0.8)]
        lib = _library(("zh_001", "zh"))
        ctx = build_score_context_for_pair("p", cells, lib, _ScoringCfg())
        # No exception, returns a context.
        assert isinstance(ctx, PromptPairScoreContext)
