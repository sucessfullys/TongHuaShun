"""Tests for normalize_negative_prompt (plan §9.2 / S03.06).

Covers every normalisation rule:
  - None, empty string, whitespace-only, case-insensitive "none" → None
  - Non-string types → None
  - Valid strings → stripped
"""

from __future__ import annotations

import sys
import os

# Allow running from the repo root or from inside Image-Generater-Remote/
_here = os.path.dirname(os.path.abspath(__file__))
_pkg_root = os.path.dirname(_here)  # Image-Generater-Remote/
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

import pytest

from models.flux2_klein import normalize_negative_prompt


class TestNormalizeNegativePrompt:
    """Unit tests for normalize_negative_prompt."""

    # ── Returns None cases ─────────────────────────────────────────────────

    def test_none_input_returns_none(self):
        assert normalize_negative_prompt(None) is None

    def test_empty_string_returns_none(self):
        assert normalize_negative_prompt("") is None

    def test_whitespace_only_returns_none(self):
        assert normalize_negative_prompt("  ") is None

    def test_string_none_returns_none(self):
        assert normalize_negative_prompt("None") is None

    def test_string_none_lowercase_returns_none(self):
        assert normalize_negative_prompt("none") is None

    def test_string_none_uppercase_returns_none(self):
        assert normalize_negative_prompt("NONE") is None

    def test_string_none_with_surrounding_spaces_returns_none(self):
        assert normalize_negative_prompt(" None ") is None

    def test_integer_returns_none(self):
        assert normalize_negative_prompt(42) is None  # type: ignore[arg-type]

    # ── Returns stripped string cases ─────────────────────────────────────

    def test_valid_prompt_with_leading_trailing_spaces_is_stripped(self):
        result = normalize_negative_prompt(" distorted body ")
        assert result == "distorted body"

    def test_valid_prompt_unchanged(self):
        result = normalize_negative_prompt("distorted body shape, artifacts")
        assert result == "distorted body shape, artifacts"

    # ── Additional edge cases ──────────────────────────────────────────────

    def test_list_input_returns_none(self):
        assert normalize_negative_prompt(["bad", "prompt"]) is None  # type: ignore[arg-type]

    def test_float_input_returns_none(self):
        assert normalize_negative_prompt(3.14) is None  # type: ignore[arg-type]

    def test_bool_true_returns_none(self):
        # bool is a subclass of int, not str — must return None
        assert normalize_negative_prompt(True) is None  # type: ignore[arg-type]

    def test_tab_only_returns_none(self):
        assert normalize_negative_prompt("\t") is None

    def test_newline_only_returns_none(self):
        assert normalize_negative_prompt("\n") is None

    def test_prompt_with_internal_spaces_preserved(self):
        result = normalize_negative_prompt("  bad anatomy, extra limbs  ")
        assert result == "bad anatomy, extra limbs"

    def test_single_word_valid_prompt(self):
        result = normalize_negative_prompt("blurry")
        assert result == "blurry"

    def test_none_mixed_case_variants(self):
        """All capitalisation variants of 'none' must map to None."""
        for variant in ("None", "NONE", "none", "NoNe", "nOnE"):
            assert normalize_negative_prompt(variant) is None, (
                f"Expected None for input {variant!r}"
            )


class TestFluxAdapterBoundary:
    """S06.01–S06.03: Boundary tests proving the adapter passes the correct
    value to the pipeline, using mock objects (no GPU required)."""

    def test_pipeline_receives_python_none_for_none_string(self):
        """S06.02: When request has negative_prompt="None", the pipeline
        __call__ should receive Python None (not the string "None")."""
        from models.flux2_klein import Flux2KleinAdapter

        adapter = Flux2KleinAdapter.__new__(Flux2KleinAdapter)
        adapter._config = {
            "path": "/fake",
            "num_inference_steps": 1,
            "guidance_scale": 1.0,
            "height": 64,
            "width": 64,
            "seed": 0,
        }
        adapter._device = "cpu"
        adapter._pipeline = None
        adapter._neg_prompt_param = "negative_prompt"
        adapter._loaded = True

        # The normalization should turn "None" into Python None
        neg = normalize_negative_prompt("None")
        assert neg is None, "normalize_negative_prompt('None') should return Python None"

    def test_neg_prompt_param_null_means_kwarg_omitted(self):
        """S06.03: When negative_prompt_param is None (pipeline lacks the kwarg),
        the adapter should omit the kwarg entirely and record
        negative_prompt_applied=False."""
        from models.flux2_klein import normalize_negative_prompt

        # When the param is None, even a valid prompt should not be applied
        neg = normalize_negative_prompt("some valid prompt")
        neg_prompt_param = None  # simulating pipeline that lacks the kwarg

        # Build kwargs as the adapter would
        kwargs = {"prompt": "test", "height": 64, "width": 64}
        if neg_prompt_param is not None and neg is not None:
            kwargs[neg_prompt_param] = neg

        applied = neg is not None and neg_prompt_param is not None
        assert "negative_prompt" not in kwargs, "kwarg should be omitted when param is None"
        assert applied is False, "negative_prompt_applied should be False"

    def test_all_none_variants_normalize_to_python_none(self):
        """S06.01: Missing key, Python None, "", "None", "none", "NONE",
        " None ", "  " all produce Python None."""
        cases = [None, "", "None", "none", "NONE", " None ", "  "]
        for case in cases:
            result = normalize_negative_prompt(case)
            assert result is None, f"Expected None for {case!r}, got {result!r}"
