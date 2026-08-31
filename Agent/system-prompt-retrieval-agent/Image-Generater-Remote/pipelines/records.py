"""
Output record writers for the image-generation pipeline.

All writers create the output directory if it does not already exist and
write JSON with indent=2 and ensure_ascii=False for human-readable output.
"""

import json
import os


def _ensure_dir(output_dir: str) -> None:
    """Create output_dir (and any parents) if it does not exist."""
    os.makedirs(output_dir, exist_ok=True)


def _write_json(path: str, data) -> None:
    """Serialize data to path as indented JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(data, indent=2, ensure_ascii=False))


def write_intermediate_prompt(
    output_dir: str,
    sample_id: str,
    prompt_text: str,
    prompt_data: dict,
) -> None:
    """Write the intermediate prompt text and full data dict to disk.

    Creates:
        <output_dir>/intermediate_prompt.txt  — plain prompt text
        <output_dir>/intermediate_prompt.json — full data dict

    Args:
        output_dir: Directory under which files are written.
        sample_id: Identifier for the sample (unused in filenames but
                   kept as a parameter for call-site consistency and
                   future logging).
        prompt_text: The raw prompt string to write to the .txt file.
        prompt_data: Arbitrary dict containing prompt metadata to write
                     to the .json file.
    """
    _ensure_dir(output_dir)
    txt_path = os.path.join(output_dir, "intermediate_prompt.txt")
    json_path = os.path.join(output_dir, "intermediate_prompt.json")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(prompt_text)
    _write_json(json_path, prompt_data)


def write_generation_record(
    output_dir: str,
    sample_id: str,
    data: dict,
) -> None:
    """Write generation.json to disk.

    The caller is responsible for ensuring that *data* contains at minimum:
        negative_prompt_applied, negative_prompt_normalized, seed_used

    Args:
        output_dir: Directory under which the file is written.
        sample_id: Identifier for the sample (kept for call-site consistency).
        data: Dict with generation metadata. Must include
              ``negative_prompt_applied``, ``negative_prompt_normalized``,
              and ``seed_used``.
    """
    _ensure_dir(output_dir)
    required = ("negative_prompt_applied", "negative_prompt_normalized", "seed_used")
    for key in required:
        if key not in data:
            raise ValueError(
                f"write_generation_record: required key '{key}' missing from data "
                f"(sample_id={sample_id!r})"
            )
    _write_json(os.path.join(output_dir, "generation.json"), data)


def write_eval_record(
    output_dir: str,
    sample_id: str,
    data: dict,
) -> None:
    """Write eval.json to disk.

    The caller is responsible for ensuring that *data* contains at minimum:
        raw_response, parsed, parse_status

    Args:
        output_dir: Directory under which the file is written.
        sample_id: Identifier for the sample (kept for call-site consistency).
        data: Dict with evaluation metadata. Must include
              ``raw_response``, ``parsed``, and ``parse_status``.
    """
    _ensure_dir(output_dir)
    required = ("raw_response", "parsed", "parse_status")
    for key in required:
        if key not in data:
            raise ValueError(
                f"write_eval_record: required key '{key}' missing from data "
                f"(sample_id={sample_id!r})"
            )
    _write_json(os.path.join(output_dir, "eval.json"), data)


def write_run_record(
    output_dir: str,
    sample_id: str,
    data: dict,
) -> None:
    """Write run_record.json to disk.

    Args:
        output_dir: Directory under which the file is written.
        sample_id: Identifier for the sample (kept for call-site consistency).
        data: Arbitrary dict with run-level metadata.
    """
    _ensure_dir(output_dir)
    _write_json(os.path.join(output_dir, "run_record.json"), data)


def write_stage_record(
    output_dir: str,
    stage_id: str,
    data: dict,
) -> None:
    """Write stage_record_{stage_id}.json to disk.

    Args:
        output_dir: Directory under which the file is written.
        stage_id: Stage identifier used as the filename suffix.
        data: Arbitrary dict with stage-level metadata.
    """
    _ensure_dir(output_dir)
    filename = f"stage_record_{stage_id}.json"
    _write_json(os.path.join(output_dir, filename), data)


def write_eval_summary(
    output_dir: str,
    cells: list[dict],
) -> None:
    """Write eval_summary.json to disk.

    Each element in *cells* should contain:
        system_prompt_id, negative_prompt_id, cell_id,
        total, yes, no, parse_fail

    This function computes and injects ``yes_ratio`` for every cell before
    writing:
        yes_ratio = yes / (total - parse_fail)  if (total - parse_fail) > 0
                  else None

    Args:
        output_dir: Directory under which the file is written.
        cells: List of cell dicts. Each dict is written as-is with
               ``yes_ratio`` added (or overwritten).
    """
    _ensure_dir(output_dir)
    enriched: list[dict] = []
    for cell in cells:
        c = dict(cell)
        valid = c.get("total", 0) - c.get("parse_fail", 0)
        c["yes_ratio"] = c.get("yes", 0) / valid if valid > 0 else None
        enriched.append(c)
    _write_json(os.path.join(output_dir, "eval_summary.json"), enriched)
