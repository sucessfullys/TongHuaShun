from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    local_project_root: str
    local_agent_root: str
    remote_image_service_root: str
    dataset_root: str
    dataset_candidates: list[str] = Field(default_factory=list)
    output_root: str
    memory_root: str
    prompt_seed_file: str
    env_file: str


class RemoteEndpoints(BaseModel):
    gemma: str = "/stage/gemma"
    flux: str = "/stage/flux"
    qwen: str = "/stage/qwen"


class RemoteConfig(BaseModel):
    ssh_alias: str = "3h100"
    tunnel_command: str
    controller_base_url: str = "http://127.0.0.1:17700"
    request_timeout_s: int = 7200
    stage_retry_max: int = 2
    lock_contention_retry_max: int = 6
    lock_contention_wait_s: int = 10
    require_all_gpu_workers: bool = True
    reconnect_max: int = 5
    post_unload_vram_free_min_gib: int = 70
    # V0.2.1 additions (plan §6, §12)
    lifecycle_mode: Literal["cold", "warm"] = "cold"
    post_stage_host_ram_free_gib: int = 64
    use_sample_manifest_path: bool = True
    endpoints: RemoteEndpoints = Field(default_factory=RemoteEndpoints)
    v022_artifact_root: str | None = None


class ApiConfig(BaseModel):
    openai_api_key_env: str = "OPENAI_API_KEY"
    google_api_key_env: str = "Google_API_KEY"
    prompt_generation_model: str = "gpt-5.4"
    failure_summary_model: str = "gpt-5.4-mini"
    api_eval_model: str = "gpt-5.4"


class RateLimitsConfig(BaseModel):
    requests_per_second: float = 3.0
    requests_per_minute: int = 120
    max_concurrency: int = 2
    max_api_retries: int = 6
    initial_backoff_s: float = 1.0
    max_backoff_s: float = 60.0
    jitter: bool = True
    respect_rate_limit_headers: bool = True


class PromptGenerationConfig(BaseModel):
    prompt_pairs_per_round: int = 3
    structured_outputs: bool = True
    max_parse_retries: int = 3
    retry_wait_s: int = 60
    fallback_to_best_existing: bool = True
    web_search_enabled: bool = False
    # V0.2.1 (plan §11.5, §12)
    cross_round_cell_id_collision_check: bool = True


class WorkflowConfig(BaseModel):
    max_rounds: int = 6
    score_threshold: float = 0.80
    allow_partial_samples: bool = False
    resume_from_artifacts: bool = True
    verify_existing_files: bool = True
    # V0.2.1 (plan §6.4, §11)
    resume_granularity: Literal[
        "stage_pair", "stage_pair_sample", "stage_pair_user_prompt_sample"
    ] = "stage_pair_user_prompt_sample"
    cross_round_pipelining: bool = False


class GemmaUserPromptEntry(BaseModel):
    user_prompt_id: str
    language: Literal["zh", "en"]
    text: str
    enabled: bool = True


class GemmaUserPromptsConfig(BaseModel):
    """V0.2.1 plan §4.1, §12. Mirrors temp_wxy/PROMPTS.py source-of-truth."""
    library: list[GemmaUserPromptEntry] = Field(default_factory=list)
    default_mode: Literal["smoke_subset", "balanced_subset", "full"] = "balanced_subset"
    smoke_prompt_ids: list[str] = Field(default_factory=list)
    pilot_prompt_ids: list[str] = Field(default_factory=list)
    full_prompt_ids: list[str] = Field(default_factory=list)
    corpus_id: str = "tryon_v1"
    min_surviving_user_prompts_per_pair: int = 1


class ScoringConfig(BaseModel):
    """V0.2.1 plan §10.6, §12."""
    prompt_sensitivity_penalty_weight: float = 0.5
    prompt_sensitivity_tolerance: float = 0.10
    min_per_user_prompt_score: float = 0.30
    flag_language_brittle_gap: float = 0.10


class EvaluationWeights(BaseModel):
    qwen_pass_rate: float = 0.40
    edit_correctness: float = 0.20
    garment_transfer_correctness: float = 0.15
    preservation: float = 0.15
    artifact_penalty: float = -0.10
    category_balance_bonus: float = 0.10


class EvaluationConfig(BaseModel):
    categories: list[str] = Field(default_factory=lambda: ["dress", "lower", "upper"])
    run_local_api_eval: bool = True
    max_concurrent: int = 4
    low_score_examples_per_pair: int = 10
    weights: EvaluationWeights = Field(default_factory=EvaluationWeights)
    # Level-1 eval-durability knobs. Conservative defaults; bump only after
    # one clean real-remote run validates per-cell durability.
    api_concurrency: int = 4
    api_rps_limit: Optional[float] = None  # None inherits rate_limits.requests_per_second
    streaming_api_eval: bool = False


class PipelineConfig(BaseModel):
    skip_irrecoverable_samples: bool = True
    excluded_penalty_base: float = 0.90


class LoggingConfig(BaseModel):
    level: str = "INFO"
    redact_secrets: bool = True
    redact_large_base64: bool = True
    redact_env_keys: list[str] = Field(default_factory=lambda: ["OPENAI_API_KEY", "Google_API_KEY"])


class BudgetConfig(BaseModel):
    daily_usd_cap: float = 50.0
    per_round_usd_cap: float = 10.0


class ExecutionConfig(BaseModel):
    """V0.2.2 production / compatibility execution mode (S05B.01)."""

    mode: Literal[
        "v022_stage_major",
        "legacy_stage_all_compat",
        "legacy_cartesian_compat",
    ] = "v022_stage_major"


class ProjectMeta(BaseModel):
    name: str = "System-Prompt-Retrieval-Agent"
    version: str = "V0.2.1"


class AppConfig(BaseModel):
    config_version: str = "0.2.1"
    project: ProjectMeta = Field(default_factory=ProjectMeta)
    paths: PathsConfig
    remote: RemoteConfig
    api: ApiConfig
    rate_limits: RateLimitsConfig = Field(default_factory=RateLimitsConfig)
    prompt_generation: PromptGenerationConfig = Field(default_factory=PromptGenerationConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    gemma_user_prompts: GemmaUserPromptsConfig = Field(default_factory=GemmaUserPromptsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    def model_post_init(self, _ctx: Any) -> None:
        # Strict-mode coupling guard (plan §6.4): if allow_partial_samples=False,
        # min_surviving_user_prompts_per_pair must not silently weaken the run.
        if not self.workflow.allow_partial_samples:
            enabled_count = sum(
                1 for e in self.gemma_user_prompts.library if e.enabled
            )
            if enabled_count and self.gemma_user_prompts.min_surviving_user_prompts_per_pair < enabled_count:
                # strict mode wins; warning logged at runtime by agent_loop
                pass

    def redacted_dump(self) -> dict[str, Any]:
        d = self.model_dump()
        # Remove any field value that equals a live env var value.
        env_names = self.logging.redact_env_keys
        env_values = {os.environ.get(k, "") for k in env_names}
        env_values.discard("")

        def _strip(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _strip(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_strip(x) for x in obj]
            if isinstance(obj, str) and obj in env_values:
                return "***REDACTED***"
            return obj

        return _strip(d)


def _load_dotenv(env_file: str) -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(env_file, override=False)
    except Exception:
        # Fallback minimal loader to avoid hard dep at import time in tests
        if not os.path.isfile(env_file):
            return
        with open(env_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_config(path: str | Path) -> AppConfig:
    p = Path(path)
    with p.open() as fh:
        raw = yaml.safe_load(fh)
    cfg = AppConfig(**raw)
    _load_dotenv(cfg.paths.env_file)
    return cfg
