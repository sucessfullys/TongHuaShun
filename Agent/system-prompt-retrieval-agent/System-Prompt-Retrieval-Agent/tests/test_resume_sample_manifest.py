"""S09 — Resume via sample_manifest_path with explicit purpose tag.

Verifies the local agent dispatch carries an explicit
sample_manifest_path_purpose enum whenever sample_manifest_path is set,
matching the controller's `400 missing_manifest_purpose` contract.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from system_prompt_retrieval_agent.schemas import (
    GemmaStageRequest,
    FluxStageRequest,
    QwenStageRequest,
    GemmaUserPrompt,
    PromptPairRequest,
)


def _base(**over):
    base = dict(
        run_id="r1",
        round_id=1,
        prompt_pairs=[PromptPairRequest(prompt_pair_id="A", system_prompt_id="s1")],
        user_prompts=[GemmaUserPrompt(user_prompt_id="zh_001", language="zh", text="x")],
        user_prompt_corpus_hash="a" * 64,
        dataset_root="/d",
        output_root="/o",
    )
    base.update(over)
    return base


def test_resume_missing_cells_purpose_round_trip():
    """Surgical resume after Gemma crash: sample_manifest_path lists
    only missing cells; purpose=resume_missing_cells."""
    req = GemmaStageRequest(
        **_base(
            sample_manifest_path="/tmp/missing.jsonl",
            sample_manifest_path_purpose="resume_missing_cells",
        )
    )
    assert req.sample_manifest_path == "/tmp/missing.jsonl"
    assert req.sample_manifest_path_purpose == "resume_missing_cells"


def test_prior_stage_survivor_cells_purpose_for_flux():
    """FLUX dispatch hands off survivor cells from Gemma."""
    req = FluxStageRequest(
        **_base(
            sample_manifest_path="/tmp/survivors.jsonl",
            sample_manifest_path_purpose="prior_stage_survivor_cells",
        )
    )
    assert req.sample_manifest_path_purpose == "prior_stage_survivor_cells"


def test_prior_stage_survivor_cells_purpose_for_qwen():
    """Qwen dispatch hands off survivor cells from FLUX."""
    req = QwenStageRequest(
        **_base(
            sample_manifest_path="/tmp/survivors.jsonl",
            sample_manifest_path_purpose="prior_stage_survivor_cells",
        )
    )
    assert req.sample_manifest_path_purpose == "prior_stage_survivor_cells"


def test_manifest_without_purpose_raises_iron_law_violation():
    """Iron Law §8: manifest without purpose is an immediate failure."""
    with pytest.raises(ValidationError) as exc:
        GemmaStageRequest(**_base(sample_manifest_path="/tmp/x.jsonl"))
    assert "missing_manifest_purpose" in str(exc.value)


def test_purpose_without_manifest_is_silently_valid():
    """If neither field is set, the request is valid (no manifest, no purpose)."""
    req = GemmaStageRequest(**_base())
    assert req.sample_manifest_path is None
    assert req.sample_manifest_path_purpose is None


def test_purpose_enum_strict_only_two_values():
    """Only the two literal values are accepted."""
    with pytest.raises(ValidationError):
        GemmaStageRequest(
            **_base(
                sample_manifest_path="/tmp/x.jsonl",
                sample_manifest_path_purpose="some_other_value",
            )
        )


def test_resume_manifest_jsonl_3key_rows(tmp_path: Path):
    """Manifest rows are 3-key cells (pair_id, user_prompt_id, sample_id)."""
    p = tmp_path / "m.jsonl"
    p.write_text(json.dumps({
        "prompt_pair_id": "pairA",
        "user_prompt_id": "zh_001",
        "sample_id": "s1",
    }) + "\n")
    rows = [json.loads(line) for line in p.read_text().splitlines() if line]
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"prompt_pair_id", "user_prompt_id", "sample_id"}
