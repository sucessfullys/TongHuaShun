"""Tests for models/eval_parser.py — Qwen yes/no response parser.

Covers the full parse-status matrix from plan §9.3:
  matched_yes, matched_no, parse_fail_empty, parse_fail_unknown,
  parse_fail_multiline.
"""

from __future__ import annotations

import pytest

from models.eval_parser import parse_eval_response


# ── matched_yes ────────────────────────────────────────────────────────────


class TestMatchedYes:
    def test_plain_yes(self):
        result = parse_eval_response("yes")
        assert result == {"parsed": "yes", "parse_status": "matched_yes"}

    def test_leading_trailing_whitespace(self):
        result = parse_eval_response(" YES ")
        assert result == {"parsed": "yes", "parse_status": "matched_yes"}

    def test_digit_one(self):
        result = parse_eval_response("1")
        assert result == {"parsed": "yes", "parse_status": "matched_yes"}

    def test_true_token(self):
        result = parse_eval_response("true")
        assert result == {"parsed": "yes", "parse_status": "matched_yes"}

    def test_ok_token_now_unknown(self):
        # L7 fix: "ok" removed from yes-tokens (too permissive)
        result = parse_eval_response("ok")
        assert result == {"parsed": None, "parse_status": "parse_fail_unknown"}

    def test_pass_token_now_unknown(self):
        # L7 fix: "pass" removed from yes-tokens (too permissive)
        result = parse_eval_response("pass")
        assert result == {"parsed": None, "parse_status": "parse_fail_unknown"}

    def test_y_token(self):
        result = parse_eval_response("y")
        assert result == {"parsed": "yes", "parse_status": "matched_yes"}

    def test_yes_with_punctuation(self):
        # e.g. "Yes." — punctuation stripped, token should still match
        result = parse_eval_response("Yes.")
        assert result == {"parsed": "yes", "parse_status": "matched_yes"}

    def test_yes_in_sentence(self):
        result = parse_eval_response("The answer is yes.")
        assert result == {"parsed": "yes", "parse_status": "matched_yes"}


# ── matched_no ─────────────────────────────────────────────────────────────


class TestMatchedNo:
    def test_plain_no(self):
        result = parse_eval_response("No")
        assert result == {"parsed": "no", "parse_status": "matched_no"}

    def test_digit_zero(self):
        result = parse_eval_response("0")
        assert result == {"parsed": "no", "parse_status": "matched_no"}

    def test_false_token(self):
        result = parse_eval_response("false")
        assert result == {"parsed": "no", "parse_status": "matched_no"}

    def test_fail_token_now_unknown(self):
        # L7 fix: "fail" removed from no-tokens (asymmetric with yes-set removal)
        result = parse_eval_response("fail")
        assert result == {"parsed": None, "parse_status": "parse_fail_unknown"}

    def test_n_token(self):
        result = parse_eval_response("n")
        assert result == {"parsed": "no", "parse_status": "matched_no"}

    def test_no_uppercase(self):
        result = parse_eval_response("NO")
        assert result == {"parsed": "no", "parse_status": "matched_no"}


# ── parse_fail_empty ───────────────────────────────────────────────────────


class TestParseFailEmpty:
    def test_empty_string(self):
        result = parse_eval_response("")
        assert result == {"parsed": None, "parse_status": "parse_fail_empty"}

    def test_spaces_only(self):
        result = parse_eval_response("  ")
        assert result == {"parsed": None, "parse_status": "parse_fail_empty"}

    def test_newline_only(self):
        result = parse_eval_response("\n")
        assert result == {"parsed": None, "parse_status": "parse_fail_empty"}

    def test_tab_only(self):
        result = parse_eval_response("\t")
        assert result == {"parsed": None, "parse_status": "parse_fail_empty"}


# ── parse_fail_unknown ─────────────────────────────────────────────────────


class TestParseFailUnknown:
    def test_maybe(self):
        result = parse_eval_response("maybe")
        assert result == {"parsed": None, "parse_status": "parse_fail_unknown"}

    def test_unrelated_word(self):
        result = parse_eval_response("excellent quality")
        assert result == {"parsed": None, "parse_status": "parse_fail_unknown"}

    def test_json_with_yes_value(self):
        # A JSON blob whose string value "yes" is embedded — after punctuation
        # stripping (including braces, quotes, colons) the token "yes" should
        # appear, so this must match yes.  This tests that we *do* extract
        # tokens from structured output.
        result = parse_eval_response('{"success": "yes", "reason": "good result"}')
        # "yes" and "reason" and "success" are tokens; only "yes" is in YES set.
        assert result["parsed"] == "yes"
        assert result["parse_status"] == "matched_yes"

    def test_random_digits(self):
        # Multi-digit numbers do not match "1" or "0".
        result = parse_eval_response("42")
        assert result == {"parsed": None, "parse_status": "parse_fail_unknown"}

    def test_no_relevant_word(self):
        result = parse_eval_response("the model looks good overall")
        assert result == {"parsed": None, "parse_status": "parse_fail_unknown"}


# ── parse_fail_multiline ───────────────────────────────────────────────────


class TestParseFailMultiline:
    def test_yes_and_no_on_separate_lines(self):
        result = parse_eval_response("yes\nno")
        assert result == {"parsed": None, "parse_status": "parse_fail_multiline"}

    def test_yes_then_fail(self):
        result = parse_eval_response("yes, but also fail")
        assert result == {"parsed": None, "parse_status": "parse_fail_multiline"}

    def test_true_and_false(self):
        result = parse_eval_response("true or false?")
        assert result == {"parsed": None, "parse_status": "parse_fail_multiline"}

    def test_pass_and_no(self):
        result = parse_eval_response("pass no")
        assert result == {"parsed": None, "parse_status": "parse_fail_multiline"}

    def test_1_and_0(self):
        result = parse_eval_response("1 0")
        assert result == {"parsed": None, "parse_status": "parse_fail_multiline"}


# ── return structure contract ──────────────────────────────────────────────


class TestReturnStructure:
    """Ensure every call returns exactly the two required keys."""

    @pytest.mark.parametrize("input_text", [
        "yes", "no", "", "maybe", "yes\nno", "  ",
    ])
    def test_keys_present(self, input_text: str):
        result = parse_eval_response(input_text)
        assert set(result.keys()) == {"parsed", "parse_status"}

    def test_parsed_values_are_canonical(self):
        """parsed is always "yes", "no", or None — never some other string."""
        for text in ["yes", "true", "1", "pass", "ok", "y"]:
            assert parse_eval_response(text)["parsed"] == "yes"
        for text in ["no", "false", "0", "fail", "n"]:
            assert parse_eval_response(text)["parsed"] == "no"
        for text in ["", "  ", "maybe", "yes\nno"]:
            assert parse_eval_response(text)["parsed"] is None
