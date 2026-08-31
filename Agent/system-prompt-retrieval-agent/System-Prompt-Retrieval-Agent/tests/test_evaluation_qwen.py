"""Tests for qwen_parser: parse_qwen_sample and parse_qwen_summary."""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from system_prompt_retrieval_agent.evaluation.qwen_parser import (
    QwenParseStatus,
    parse_qwen_sample,
    parse_qwen_summary,
)


# ---------------------------------------------------------------------------
# parse_qwen_sample: YES tokens
# ---------------------------------------------------------------------------
class TestParseQwenSampleYes:
    def test_yes_lowercase(self):
        assert parse_qwen_sample("yes") == QwenParseStatus.YES

    def test_yes_uppercase(self):
        assert parse_qwen_sample("YES") == QwenParseStatus.YES

    def test_yes_mixed(self):
        assert parse_qwen_sample("Yes") == QwenParseStatus.YES

    def test_yes_with_trailing_newline(self):
        assert parse_qwen_sample("YES\n") == QwenParseStatus.YES

    def test_yes_with_spaces(self):
        assert parse_qwen_sample(" yes ") == QwenParseStatus.YES

    def test_yes_with_period(self):
        assert parse_qwen_sample("yes.") == QwenParseStatus.YES

    def test_y_lowercase(self):
        assert parse_qwen_sample("y") == QwenParseStatus.YES

    def test_y_uppercase(self):
        assert parse_qwen_sample("Y") == QwenParseStatus.YES

    def test_yes_trailing_crlf(self):
        assert parse_qwen_sample("yes\r\n") == QwenParseStatus.YES


# ---------------------------------------------------------------------------
# parse_qwen_sample: NO tokens
# ---------------------------------------------------------------------------
class TestParseQwenSampleNo:
    def test_no_lowercase(self):
        assert parse_qwen_sample("no") == QwenParseStatus.NO

    def test_no_uppercase(self):
        assert parse_qwen_sample("NO") == QwenParseStatus.NO

    def test_no_with_trailing_newline(self):
        assert parse_qwen_sample("NO.\n") == QwenParseStatus.NO

    def test_no_with_spaces(self):
        assert parse_qwen_sample(" no ") == QwenParseStatus.NO

    def test_n_lowercase(self):
        assert parse_qwen_sample("n") == QwenParseStatus.NO

    def test_n_uppercase(self):
        assert parse_qwen_sample("N") == QwenParseStatus.NO


# ---------------------------------------------------------------------------
# parse_qwen_sample: forbidden tokens -> parse_fail_unknown
# ---------------------------------------------------------------------------
class TestParseQwenSampleForbidden:
    def test_pass_token(self):
        assert parse_qwen_sample("pass") == QwenParseStatus.PARSE_FAIL_UNKNOWN

    def test_pass_uppercase(self):
        assert parse_qwen_sample("PASS") == QwenParseStatus.PARSE_FAIL_UNKNOWN

    def test_ok_token(self):
        assert parse_qwen_sample("ok") == QwenParseStatus.PARSE_FAIL_UNKNOWN

    def test_true_token(self):
        assert parse_qwen_sample("true") == QwenParseStatus.PARSE_FAIL_UNKNOWN

    def test_one_token(self):
        assert parse_qwen_sample("1") == QwenParseStatus.PARSE_FAIL_UNKNOWN

    def test_fail_token(self):
        assert parse_qwen_sample("fail") == QwenParseStatus.PARSE_FAIL_UNKNOWN

    def test_sentence_with_yes(self):
        """'I think yes because...' is NOT a match — first token is 'I'."""
        assert parse_qwen_sample("I think yes because it looks good") == QwenParseStatus.PARSE_FAIL_UNKNOWN

    def test_empty_string(self):
        assert parse_qwen_sample("") == QwenParseStatus.PARSE_FAIL_UNKNOWN

    def test_random_word(self):
        assert parse_qwen_sample("correct") == QwenParseStatus.PARSE_FAIL_UNKNOWN


# ---------------------------------------------------------------------------
# parse_qwen_sample: multiline -> parse_fail_multiline
# ---------------------------------------------------------------------------
class TestParseQwenSampleMultiline:
    def test_yes_newline_no(self):
        assert parse_qwen_sample("yes\nno") == QwenParseStatus.PARSE_FAIL_MULTILINE

    def test_two_blank_lines(self):
        assert parse_qwen_sample("yes\n\n") == QwenParseStatus.PARSE_FAIL_MULTILINE

    def test_multiline_with_text(self):
        assert parse_qwen_sample("yes\nsome extra text") == QwenParseStatus.PARSE_FAIL_MULTILINE


# ---------------------------------------------------------------------------
# parse_qwen_summary: fixture-based tests
# ---------------------------------------------------------------------------

@pytest.fixture
def summary_json(tmp_path: Path) -> Path:
    """Create a stage_summary.json with known distribution."""
    data = {
        "samples": [
            # dress category: 5 yes, 2 no, 1 parse_fail_unknown, 1 parse_fail_multiline
            {"sample_id": "dress/s01", "raw_response": "yes", "category": "dress"},
            {"sample_id": "dress/s02", "raw_response": "yes", "category": "dress"},
            {"sample_id": "dress/s03", "raw_response": "yes", "category": "dress"},
            {"sample_id": "dress/s04", "raw_response": "yes", "category": "dress"},
            {"sample_id": "dress/s05", "raw_response": "yes", "category": "dress"},
            {"sample_id": "dress/s06", "raw_response": "no", "category": "dress"},
            {"sample_id": "dress/s07", "raw_response": "no", "category": "dress"},
            {"sample_id": "dress/s08", "raw_response": "pass", "category": "dress"},
            {"sample_id": "dress/s09", "raw_response": "yes\nno", "category": "dress"},
            # upper category: 2 yes, 3 no
            {"sample_id": "upper/s10", "raw_response": "yes", "category": "upper"},
            {"sample_id": "upper/s11", "raw_response": "yes", "category": "upper"},
            {"sample_id": "upper/s12", "raw_response": "no", "category": "upper"},
            {"sample_id": "upper/s13", "raw_response": "no", "category": "upper"},
            {"sample_id": "upper/s14", "raw_response": "no", "category": "upper"},
        ]
    }
    p = tmp_path / "stage_summary.json"
    p.write_text(json.dumps(data))
    return p


@pytest.fixture
def high_fail_summary_json(tmp_path: Path) -> Path:
    """Create a stage_summary.json with parse_fail_rate > 0.15."""
    samples = []
    for i in range(10):
        samples.append({"sample_id": f"dress/s{i:02d}", "raw_response": "yes"})
    # 4 parse failures out of 14 total = ~0.286 > 0.15
    for i in range(10, 14):
        samples.append({"sample_id": f"dress/s{i:02d}", "raw_response": "pass"})

    data = {"samples": samples}
    p = tmp_path / "stage_summary_highfail.json"
    p.write_text(json.dumps(data))
    return p


class TestParseQwenSummary:
    def test_overall_pass_rate(self, summary_json: Path):
        result = parse_qwen_summary(summary_json)
        # yes=7, no=5, pass_denom=12
        assert result["yes"] == 7
        assert result["no"] == 5
        assert abs(result["qwen_pass_rate"] - 7 / 12) < 1e-9

    def test_total_evaluated(self, summary_json: Path):
        result = parse_qwen_summary(summary_json)
        assert result["total_evaluated"] == 14

    def test_parse_fail_counts(self, summary_json: Path):
        result = parse_qwen_summary(summary_json)
        assert result["parse_fail_unknown"] == 1
        assert result["parse_fail_multiline"] == 1

    def test_parse_fail_rate(self, summary_json: Path):
        result = parse_qwen_summary(summary_json)
        # 2 failures out of 14 total
        expected = 2 / 14
        assert abs(result["parse_fail_rate"] - expected) < 1e-9

    def test_parser_stable_flag(self, summary_json: Path):
        result = parse_qwen_summary(summary_json)
        assert result["parser_unstable"] is False

    def test_parser_unstable_flag(self, high_fail_summary_json: Path):
        result = parse_qwen_summary(high_fail_summary_json)
        assert result["parser_unstable"] is True

    def test_per_category_dress(self, summary_json: Path):
        result = parse_qwen_summary(summary_json)
        dress = result["per_category"]["dress"]
        # dress: 5 yes, 2 no, 1 parse_fail_unknown, 1 parse_fail_multiline
        assert dress["yes"] == 5
        assert dress["no"] == 2
        assert dress["parse_fail_unknown"] == 1
        assert dress["parse_fail_multiline"] == 1
        assert dress["total"] == 9
        assert abs(dress["pass_rate"] - 5 / 7) < 1e-9

    def test_per_category_upper(self, summary_json: Path):
        result = parse_qwen_summary(summary_json)
        upper = result["per_category"]["upper"]
        # upper: 2 yes, 3 no
        assert upper["yes"] == 2
        assert upper["no"] == 3
        assert upper["total"] == 5
        assert abs(upper["pass_rate"] - 2 / 5) < 1e-9

    def test_raw_response_retained(self, summary_json: Path):
        result = parse_qwen_summary(summary_json)
        raw_samples = result["raw_samples"]
        assert len(raw_samples) == 14
        # Every entry must have raw_response
        for s in raw_samples:
            assert "raw_response" in s

    def test_raw_samples_have_status(self, summary_json: Path):
        result = parse_qwen_summary(summary_json)
        statuses = {s["status"] for s in result["raw_samples"]}
        assert statuses <= {e.value for e in QwenParseStatus}

    def test_category_detection_from_sample_id(self, tmp_path: Path):
        """Category should be inferred from sample_id if not explicitly provided."""
        data = {
            "samples": [
                {"sample_id": "lower/s01", "raw_response": "yes"},
                {"sample_id": "lower/s02", "raw_response": "no"},
            ]
        }
        p = tmp_path / "stage_summary_lower.json"
        p.write_text(json.dumps(data))
        result = parse_qwen_summary(p)
        assert "lower" in result["per_category"]
