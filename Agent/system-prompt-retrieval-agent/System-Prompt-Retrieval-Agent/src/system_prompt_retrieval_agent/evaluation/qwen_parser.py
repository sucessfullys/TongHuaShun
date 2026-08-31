from __future__ import annotations

import json
from enum import Enum
from pathlib import Path


class QwenParseStatus(str, Enum):
    YES = "yes"
    NO = "no"
    PARSE_FAIL_UNKNOWN = "parse_fail_unknown"
    PARSE_FAIL_MULTILINE = "parse_fail_multiline"


_YES_TOKENS = {"yes", "y"}
_NO_TOKENS = {"no", "n"}


def parse_qwen_sample(raw_response: str) -> QwenParseStatus:
    """
    STRICT rules (plan §10.5):
      yes tokens: {yes, y}  (case-insensitive, stripped, first token of single-line)
      no  tokens: {no,  n}
      Anything else:
        - multi-line → parse_fail_multiline
        - single-line but not matching → parse_fail_unknown
    DO NOT accept: pass, ok, true, 1, fail. They all → parse_fail_unknown.
    Also handles: "Yes.", "YES ", "y\n" → YES; "no.\n" → NO.
    """
    # Only strip a single trailing newline or CRLF to tolerate "yes\n" format
    # But "yes\n\n" (blank line) counts as multi-line
    if raw_response.endswith("\r\n"):
        stripped = raw_response[:-2]
    elif raw_response.endswith(("\n", "\r")):
        stripped = raw_response[:-1]
    else:
        stripped = raw_response

    # Check for multi-line: if stripped still has newlines or carriage returns it's multi-line
    if "\n" in stripped or "\r" in stripped:
        return QwenParseStatus.PARSE_FAIL_MULTILINE

    # Single line - strip whitespace
    line = stripped.strip()

    # Strip trailing punctuation (periods, exclamation marks, question marks)
    line_clean = line.rstrip(".,!?")

    token = line_clean.lower()

    if token in _YES_TOKENS:
        return QwenParseStatus.YES
    if token in _NO_TOKENS:
        return QwenParseStatus.NO

    return QwenParseStatus.PARSE_FAIL_UNKNOWN


def parse_qwen_summary(
    stage_summary_path: Path,
    *,
    per_sample_eval_dir: Path | None = None,
) -> dict:
    """
    Reads stage3_qwen/stage_summary.json AND (optionally) per-sample eval.json files.
    Returns a dict with qwen_pass_rate, parse_fail_rate, per_category breakdown, etc.
    """
    stage_summary_path = Path(stage_summary_path)
    data = json.loads(stage_summary_path.read_text())

    samples_raw: list[dict] = data.get("samples", [])

    # If per_sample_eval_dir is provided and samples lack raw_response, load from files
    if per_sample_eval_dir is not None and samples_raw:
        per_sample_eval_dir = Path(per_sample_eval_dir)
        enriched = []
        for s in samples_raw:
            if "raw_response" not in s:
                sample_id = s.get("sample_id", "")
                eval_file = per_sample_eval_dir / f"{sample_id}" / "eval.json"
                if not eval_file.exists():
                    # Try flat layout
                    eval_file = per_sample_eval_dir / f"{sample_id}.json"
                if eval_file.exists():
                    try:
                        extra = json.loads(eval_file.read_text())
                        s = {**s, "raw_response": extra.get("raw_response", "")}
                    except Exception:
                        pass
            enriched.append(s)
        samples_raw = enriched
    elif per_sample_eval_dir is not None and not samples_raw:
        # No samples in summary, try to load from per_sample_eval_dir directly
        per_sample_eval_dir = Path(per_sample_eval_dir)
        for eval_file in sorted(per_sample_eval_dir.glob("**/*.json")):
            try:
                extra = json.loads(eval_file.read_text())
                if "sample_id" in extra or "raw_response" in extra:
                    samples_raw.append(extra)
            except Exception:
                pass

    # Parse each sample
    raw_samples: list[dict] = []
    for s in samples_raw:
        raw_response = s.get("raw_response", "")
        sample_id = s.get("sample_id", "")

        # Determine status: use provided status if available, else parse raw_response
        provided_status = s.get("status")
        if provided_status in (e.value for e in QwenParseStatus):
            status = QwenParseStatus(provided_status)
        else:
            status = parse_qwen_sample(raw_response)

        # Category detection
        category = s.get("category")
        if not category and "/" in sample_id:
            category = sample_id.split("/")[0]

        raw_samples.append(
            {
                "sample_id": sample_id,
                "category": category,
                "status": status.value,
                "raw_response": raw_response,
            }
        )

    # If summary already has pre-computed totals AND no samples, use those
    if not raw_samples and "yes" in data and "no" in data:
        yes_count = int(data.get("yes", 0))
        no_count = int(data.get("no", 0))
        total = int(data.get("total_evaluated", yes_count + no_count))
        parse_fail_unknown = int(data.get("parse_fail_unknown", 0))
        parse_fail_multiline = int(data.get("parse_fail_multiline", 0))
        per_category_raw = data.get("per_category", {})
    else:
        # Compute from raw_samples
        yes_count = sum(1 for s in raw_samples if s["status"] == QwenParseStatus.YES.value)
        no_count = sum(1 for s in raw_samples if s["status"] == QwenParseStatus.NO.value)
        parse_fail_unknown = sum(
            1 for s in raw_samples if s["status"] == QwenParseStatus.PARSE_FAIL_UNKNOWN.value
        )
        parse_fail_multiline = sum(
            1 for s in raw_samples if s["status"] == QwenParseStatus.PARSE_FAIL_MULTILINE.value
        )
        total = len(raw_samples)
        per_category_raw = {}

    # Compute per_category from raw_samples
    per_category: dict[str, dict] = {}
    if raw_samples:
        cat_map: dict[str, list[dict]] = {}
        for s in raw_samples:
            cat = s.get("category") or "unknown"
            cat_map.setdefault(cat, []).append(s)

        for cat, cat_samples in cat_map.items():
            cat_yes = sum(1 for s in cat_samples if s["status"] == QwenParseStatus.YES.value)
            cat_no = sum(1 for s in cat_samples if s["status"] == QwenParseStatus.NO.value)
            cat_pfu = sum(
                1
                for s in cat_samples
                if s["status"] == QwenParseStatus.PARSE_FAIL_UNKNOWN.value
            )
            cat_pfm = sum(
                1
                for s in cat_samples
                if s["status"] == QwenParseStatus.PARSE_FAIL_MULTILINE.value
            )
            cat_total = len(cat_samples)
            cat_pass_denom = cat_yes + cat_no
            cat_fail_total = cat_pfu + cat_pfm
            per_category[cat] = {
                "pass_rate": cat_yes / cat_pass_denom if cat_pass_denom > 0 else 0.0,
                "parse_fail_rate": cat_fail_total / cat_total if cat_total > 0 else 0.0,
                "total": cat_total,
                "yes": cat_yes,
                "no": cat_no,
                "parse_fail_unknown": cat_pfu,
                "parse_fail_multiline": cat_pfm,
            }
    else:
        # Use pre-computed per_category from summary
        for cat, cat_data in per_category_raw.items():
            per_category[cat] = dict(cat_data)

    # Overall rates
    pass_denom = yes_count + no_count
    parse_fail_total = parse_fail_unknown + parse_fail_multiline
    qwen_pass_rate = yes_count / pass_denom if pass_denom > 0 else 0.0
    parse_fail_rate = parse_fail_total / total if total > 0 else 0.0

    return {
        "qwen_pass_rate": qwen_pass_rate,
        "parse_fail_rate": parse_fail_rate,
        "parse_fail_unknown": parse_fail_unknown,
        "parse_fail_multiline": parse_fail_multiline,
        "total_evaluated": total,
        "yes": yes_count,
        "no": no_count,
        "per_category": per_category,
        "parser_unstable": parse_fail_rate > 0.15,
        "raw_samples": raw_samples,
    }
