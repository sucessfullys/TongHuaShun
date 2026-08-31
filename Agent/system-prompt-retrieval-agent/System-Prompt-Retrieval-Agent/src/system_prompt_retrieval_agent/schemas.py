from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

SELECTION_ROLES = Literal[
    "previous_round",
    "previous_top",
    "long_memory_reference",
    "project_memory",
    "random_history",
    "seed",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromptPairSubScores(_Strict):
    qwen_pass_rate: Optional[float] = None
    edit_correctness: Optional[float] = None
    garment_transfer_correctness: Optional[float] = None
    preservation: Optional[float] = None
    artifact_penalty: Optional[float] = None


class CategoryScoreContext(_Strict):
    weighted_score: Optional[float] = None
    total_score: Optional[float] = None
    sub_scores: PromptPairSubScores = Field(default_factory=PromptPairSubScores)
    missing_score_reason: Optional[str] = None


class PromptPairScoreContext(_Strict):
    overall_score: Optional[float] = None
    total_score: Optional[float] = None
    sub_scores: PromptPairSubScores = Field(default_factory=PromptPairSubScores)
    category_scores: dict[str, CategoryScoreContext] = Field(default_factory=dict)
    missing_score_reason: Optional[str] = None
    # V0.2.1 user-prompt-aware aggregation (plan §10.6, §8.1)
    per_user_prompt: dict[str, PromptPairSubScores] = Field(default_factory=dict)
    mean_score_by_user_prompt: dict[str, float] = Field(default_factory=dict)
    zh_mean_score: Optional[float] = None
    en_mean_score: Optional[float] = None
    prompt_sensitivity: Optional[float] = None
    cross_lingual_gap: Optional[float] = None
    worst_user_prompt_id: Optional[str] = None
    worst_user_prompt_score: Optional[float] = None


class FailureSummary(_Strict):
    sample_id: str
    category: Optional[str] = None
    summary: str
    failure_tags: list[str] = Field(default_factory=list)
    score: Optional[float] = None


class PromptPair(_Strict):
    prompt_pair_id: str
    system_prompt_id: str
    negative_prompt_id: str
    round_id: int
    selection_role: Optional[str] = None
    system_prompt: str
    negative_prompt: Optional[str] = None
    rationale: Optional[str] = None
    expected_improvement_target: Optional[str] = None
    risk: Optional[str] = None
    scores: Optional[PromptPairScoreContext] = None
    failure_summaries: list[FailureSummary] = Field(default_factory=list)
    fallback: bool = False
    duplicate: bool = False


class PromptPairHistoryContext(_Strict):
    round_id: int
    long_memory_prompt_pairs: Optional[list[PromptPair]] = None
    project_memory_prompt_pairs: Optional[list[PromptPair]] = None
    previous_top_prompt_pairs: Optional[list[PromptPair]] = None
    previous_round_prompt_pairs: Optional[list[PromptPair]] = None
    random_history_prompt_pairs: Optional[list[PromptPair]] = None
    visual_comparisons: Optional[dict[str, str]] = None  # filename -> base64 or path
    shared_rules: Optional[list[dict[str, Any]]] = None  # cross-round learned rules


class RemoteStageRequest(_Strict):
    run_id: str
    round_id: int
    prompt_pair_id: str
    system_prompt_id: str
    system_prompt_text: str
    negative_prompt_id: Optional[str] = None
    negative_prompt: Optional[str] = None
    dataset_root: str
    output_root: str
    limit: int = 0
    allow_partial: bool = False


class ManifestEntry(_Strict):
    sample_id: str
    status: str
    output_path: Optional[str] = None
    error: Optional[str] = None


class RemoteManifest(BaseModel):
    """Tolerates extra remote-specific fields (e.g. yes/no/pass_rate/timestamp/output_dir)."""
    model_config = ConfigDict(extra="allow")

    stage: str
    run_id: Optional[str] = None
    ok: int = 0
    errors: int = 0
    total: int = 0
    entries: list[ManifestEntry] = Field(default_factory=list)
    workers: list[dict[str, Any]] = Field(default_factory=list)
    vram_free_gib: Optional[list[float]] = None


class RemoteStageResponse(_Strict):
    ok: bool
    stage: str
    message: str = ""
    run_id: str = ""
    manifest: Optional[RemoteManifest] = None


class RoundState(_Strict):
    run_id: str
    round_id: int
    status: str = "initialized"
    fallback_prompt_generation: bool = False
    consecutive_fallback_rounds: int = 0
    best_overall_score: Optional[float] = None
    pairs: list[str] = Field(default_factory=list)
    vram_leak_detected: bool = False
    language_brittle: bool = False
    started_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────
# V0.2.1 additions — cell-keyed (prompt_pair_id, user_prompt_id, sample_id)
# Plan §4.1, §6.2, §6.3, §6.4. Bucket A schema declarations (≤ 20 LOC counted
# additively against the remote controller; this local mirror is not LOC-capped
# but kept tight). All models use extra="forbid".
# ─────────────────────────────────────────────────────────────────────────

ManifestPurpose = Literal["resume_missing_cells", "prior_stage_survivor_cells"]


class GemmaUserPrompt(_Strict):
    user_prompt_id: str
    language: Literal["zh", "en"]
    text: str
    enabled: bool = True


class PromptPairRequest(_Strict):
    prompt_pair_id: str
    system_prompt_id: Optional[str] = None
    system_prompt_text: Optional[str] = None
    negative_prompt_id: Optional[str] = None
    negative_prompt: Optional[str] = None
    cell_id: Optional[str] = None
    intermediate_prompt_dir: Optional[str] = None
    flux_image_dir: Optional[str] = None


class PerUserPromptManifest(_Strict):
    ok: int = 0
    errors: int = 0
    total: int = 0
    failure_reason: Optional[str] = None


class PerPairManifest(_Strict):
    prompt_pair_id: str
    ok: int = 0
    errors: int = 0
    total: int = 0
    failure_reason: Optional[str] = None
    per_user_prompt: dict[str, PerUserPromptManifest] = Field(default_factory=dict)


class StageManifest(BaseModel):
    """V0.2.1 stage manifest. extra='allow' to tolerate remote-only fields."""
    model_config = ConfigDict(extra="allow")
    stage: Literal["gemma", "flux", "qwen"]
    run_id: Optional[str] = None
    round_id: Optional[int] = None
    user_prompt_corpus_hash: Optional[str] = None
    pairs: dict[str, PerPairManifest] = Field(default_factory=dict)
    surviving_pairs: list[str] = Field(default_factory=list)
    failed_pairs: list[dict[str, Any]] = Field(default_factory=list)
    lifecycle_state_after: Optional[Literal[
        "cold", "disk_unloaded", "gpu_unloaded_cpu_retained"
    ]] = None
    cpu_resident_model: Optional[str] = None
    vram_free_gib: Optional[list[float]] = None
    host_ram_free_gib: Optional[float] = None


class _StageRequestBase(_Strict):
    """Common fields for V0.2.1 per-stage requests. Each stage subclass narrows
    which prompt-pair fields are accepted (only Gemma allows non-null
    system_prompt_text) and enforces the manifest-purpose pairing rule."""
    run_id: str
    round_id: int = 1
    prompt_pairs: list[PromptPairRequest] = Field(default_factory=list)
    user_prompts: list[GemmaUserPrompt] = Field(default_factory=list)
    user_prompt_corpus_hash: str
    sample_ids: Optional[list[str]] = None
    sample_manifest_path: Optional[str] = None
    sample_manifest_path_purpose: Optional[ManifestPurpose] = None
    dataset_root: str
    output_root: str
    lifecycle_mode: Literal["cold", "warm"] = "cold"
    limit: int = 0
    allow_partial: bool = False

    def model_post_init(self, _ctx: Any) -> None:
        if self.sample_manifest_path is not None and self.sample_manifest_path_purpose is None:
            raise ValueError(
                "missing_manifest_purpose: sample_manifest_path requires "
                "sample_manifest_path_purpose ∈ {resume_missing_cells, "
                "prior_stage_survivor_cells}"
            )


class GemmaStageRequest(_StageRequestBase):
    """Gemma is the only stage that accepts non-null system_prompt_text on its
    pairs; FLUX and Qwen receive resolved-and-cached intermediate text."""


class _NoSystemPromptTextStageRequest(_StageRequestBase):
    def model_post_init(self, _ctx: Any) -> None:
        super().model_post_init(_ctx)
        for p in self.prompt_pairs:
            if p.system_prompt_text is not None:
                raise ValueError(
                    f"system_prompt_text forbidden on {type(self).__name__} "
                    f"pair {p.prompt_pair_id} — Gemma-only field"
                )


class FluxStageRequest(_NoSystemPromptTextStageRequest):
    pass


class QwenStageRequest(_NoSystemPromptTextStageRequest):
    pass
