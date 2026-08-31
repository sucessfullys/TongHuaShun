import pytest
from pydantic import ValidationError

from system_prompt_retrieval_agent.schemas import (
    CategoryScoreContext,
    PromptPair,
    PromptPairScoreContext,
    PromptPairSubScores,
    RemoteManifest,
    RemoteStageRequest,
)


def test_prompt_pair_extra_forbid():
    with pytest.raises(ValidationError):
        PromptPair(
            prompt_pair_id="p1",
            system_prompt_id="s1",
            negative_prompt_id="n1",
            round_id=1,
            system_prompt="hello",
            negative_prompt=None,
            unknown_field="x",
        )


def test_score_context_round_trip():
    sc = PromptPairScoreContext(
        overall_score=0.75,
        sub_scores=PromptPairSubScores(qwen_pass_rate=0.7, edit_correctness=0.8),
        category_scores={"upper": CategoryScoreContext(weighted_score=0.7)},
    )
    d = sc.model_dump()
    assert d["category_scores"]["upper"]["weighted_score"] == 0.7


def test_remote_stage_request_requires_inline_text():
    with pytest.raises(ValidationError):
        RemoteStageRequest(
            run_id="r1",
            round_id=1,
            prompt_pair_id="p1",
            system_prompt_id="s1",
            dataset_root="/x",
            output_root="/y",
        )


def test_remote_manifest_counts():
    m = RemoteManifest(stage="gemma", total=10, ok=9, errors=1)
    assert m.ok + m.errors == m.total


# ── V0.2.1 schema delta tests (plan §4.1, §6.2, §6.3, §6.4) ──────────────


def _base_v021_kwargs(**over):
    pp = {
        "prompt_pair_id": "p1",
        "system_prompt_id": "s1",
        "negative_prompt_id": "n1",
    }
    base = {
        "run_id": "r1",
        "round_id": 1,
        "prompt_pairs": [{**pp, **over.pop("pair_extra", {})}],
        "user_prompts": [
            {"user_prompt_id": "zh_001", "language": "zh", "text": "x", "enabled": True}
        ],
        "user_prompt_corpus_hash": "h" * 64,
        "dataset_root": "/d",
        "output_root": "/o",
    }
    base.update(over)
    return base


def test_gemma_user_prompt_language_enum():
    from system_prompt_retrieval_agent.schemas import GemmaUserPrompt
    GemmaUserPrompt(user_prompt_id="zh_001", language="zh", text="x")
    GemmaUserPrompt(user_prompt_id="en_001", language="en", text="y")
    with pytest.raises(ValidationError):
        GemmaUserPrompt(user_prompt_id="ja_001", language="ja", text="z")


def test_gemma_stage_request_accepts_system_prompt_text():
    from system_prompt_retrieval_agent.schemas import GemmaStageRequest
    req = GemmaStageRequest(**_base_v021_kwargs(
        pair_extra={"system_prompt_text": "you are a try-on agent"}
    ))
    assert req.prompt_pairs[0].system_prompt_text == "you are a try-on agent"


def test_flux_stage_request_rejects_system_prompt_text():
    from system_prompt_retrieval_agent.schemas import FluxStageRequest
    with pytest.raises(ValidationError):
        FluxStageRequest(**_base_v021_kwargs(
            pair_extra={"system_prompt_text": "leak"}
        ))


def test_qwen_stage_request_rejects_system_prompt_text():
    from system_prompt_retrieval_agent.schemas import QwenStageRequest
    with pytest.raises(ValidationError):
        QwenStageRequest(**_base_v021_kwargs(
            pair_extra={"system_prompt_text": "leak"}
        ))


def test_manifest_path_without_purpose_raises():
    """plan §6.4 — sample_manifest_path set without purpose mirrors 400."""
    from system_prompt_retrieval_agent.schemas import GemmaStageRequest
    with pytest.raises(ValidationError) as exc:
        GemmaStageRequest(**_base_v021_kwargs(
            sample_manifest_path="/tmp/manifest.jsonl"
        ))
    assert "missing_manifest_purpose" in str(exc.value)


def test_manifest_path_with_purpose_resume_missing_cells_valid():
    from system_prompt_retrieval_agent.schemas import GemmaStageRequest
    req = GemmaStageRequest(**_base_v021_kwargs(
        sample_manifest_path="/tmp/m.jsonl",
        sample_manifest_path_purpose="resume_missing_cells",
    ))
    assert req.sample_manifest_path_purpose == "resume_missing_cells"


def test_manifest_path_with_purpose_prior_stage_survivor_valid():
    from system_prompt_retrieval_agent.schemas import FluxStageRequest
    req = FluxStageRequest(**_base_v021_kwargs(
        sample_manifest_path="/tmp/m.jsonl",
        sample_manifest_path_purpose="prior_stage_survivor_cells",
    ))
    assert req.sample_manifest_path_purpose == "prior_stage_survivor_cells"


def test_both_manifest_fields_absent_is_valid():
    from system_prompt_retrieval_agent.schemas import GemmaStageRequest
    req = GemmaStageRequest(**_base_v021_kwargs())
    assert req.sample_manifest_path is None
    assert req.sample_manifest_path_purpose is None


def test_manifest_purpose_enum_strict():
    from system_prompt_retrieval_agent.schemas import GemmaStageRequest
    with pytest.raises(ValidationError):
        GemmaStageRequest(**_base_v021_kwargs(
            sample_manifest_path="/tmp/m.jsonl",
            sample_manifest_path_purpose="some_other_value",
        ))


def test_stage_manifest_surviving_and_failed_pairs():
    from system_prompt_retrieval_agent.schemas import StageManifest
    m = StageManifest(
        stage="gemma",
        run_id="r1",
        surviving_pairs=["p1", "p2"],
        failed_pairs=[{"prompt_pair_id": "p3", "failure_reason": "incomplete_count"}],
    )
    assert m.surviving_pairs == ["p1", "p2"]
    assert m.failed_pairs[0]["failure_reason"] == "incomplete_count"


def test_score_context_user_prompt_aware_fields():
    sc = PromptPairScoreContext(
        overall_score=0.75,
        zh_mean_score=0.80,
        en_mean_score=0.70,
        cross_lingual_gap=0.10,
        worst_user_prompt_id="en_003",
        worst_user_prompt_score=0.55,
        prompt_sensitivity=0.12,
    )
    assert sc.zh_mean_score == 0.80
    assert sc.worst_user_prompt_id == "en_003"
