"""Pydantic request/response schemas for the V0.1 FastAPI supervisor."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# ── Correlation IDs (mixin) ────────────────────────────────────────

class CorrelationMixin(BaseModel):
    """Fields carried by every request, log line, and callback."""
    run_id: str = ""
    stage_id: str = ""
    cell_id: str = ""
    shard_id: int = 0
    sample_id: str = ""
    host_port: int = 0
    model_name: str = ""


# ── Health / Status ────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    loaded_model: Optional[str] = None
    gpu_id: Optional[int] = None
    gpu_mem_used_gib: Optional[float] = None
    gpu_mem_total_gib: Optional[float] = None
    backend_pid: Optional[int] = None


class GpuSnapshot(BaseModel):
    gpu_id: int
    mem_used_gib: float
    mem_total_gib: float
    mem_free_gib: float
    utilization_pct: Optional[float] = None


class GpuStatusResponse(BaseModel):
    gpus: list[GpuSnapshot] = []


class StatusResponse(BaseModel):
    last_request_model: Optional[str] = None
    last_request_time: Optional[str] = None
    lock_held: bool = False
    lock_holder: Optional[str] = None


# ── Load / Unload / Prefetch ───────────────────────────────────────

class LoadRequest(BaseModel):
    model_name: str
    config_overrides: dict[str, Any] = {}


class LoadResponse(BaseModel):
    status: str
    load_latency_ms: float = 0
    adapter_attached: Optional[bool] = None
    adapter_backend_used: Optional[str] = None
    deployment_mode: str = "single_gpu_replicated"
    tp_size: int = 1
    backend_pid: Optional[int] = None


class PrefetchRequest(BaseModel):
    model_name: str
    warmup_tokens: int = 64


class UnloadRequest(BaseModel):
    force: bool = False


class UnloadResponse(BaseModel):
    status: str
    mem_before_gib: float = 0
    mem_after_gib: float = 0
    restarted: bool = False


# ── Inference — common envelope ────────────────────────────────────

class InferEnvelope(CorrelationMixin):
    manifest_hash: str = ""
    shard_indices: list[int] = []
    output_root_relpath: str = ""
    callback_url: str = ""


# ── Gemma ──────────────────────────────────────────────────────────

class GemmaSingleRequest(InferEnvelope):
    system_prompt: str
    user_instruction: str = "the model in image 1 wears the clothing from image 2"
    model_image_path: str
    cloth_image_path: str
    sample_id: str = ""


class GemmaSingleResponse(BaseModel):
    intermediate_prompt: str
    latency_ms: float = 0
    tokens_out: int = 0


# ── FLUX ───────────────────────────────────────────────────────────

class FluxSingleRequest(InferEnvelope):
    intermediate_prompt: str
    negative_prompt: Optional[str] = None
    model_image_path: str
    cloth_image_path: str
    sample_id: str = ""


class FluxSingleResponse(BaseModel):
    output_path: str
    negative_prompt_applied: bool = False
    negative_prompt_normalized: Optional[str] = None
    latency_ms: float = 0
    seed_used: int = 0


# ── Qwen ───────────────────────────────────────────────────────────

class QwenSingleRequest(InferEnvelope):
    model_image_path: str
    cloth_image_path: str
    result_image_path: str
    eval_instruction: str = ""
    sample_id: str = ""


class QwenSingleResponse(BaseModel):
    raw_response: str = ""
    parsed: Optional[str] = None  # "yes" | "no" | null
    parse_status: str = ""
    latency_ms: float = 0


# ── Batch (list / template) ───────────────────────────────────────

class InferListItem(CorrelationMixin):
    index: int = 0
    # Per-model fields are included as extra keys
    model_config = {"extra": "allow"}


class InferListRequest(InferEnvelope):
    items: list[InferListItem] = []


class InferTemplateRequest(InferEnvelope):
    template_text: str = ""
    image_root: str = ""


class BatchItemResult(BaseModel):
    index: int = 0
    sample_id: str = ""
    ok: bool = True
    error: Optional[str] = None
    result: dict[str, Any] = {}


class BatchResponse(CorrelationMixin):
    ok: bool = True
    ok_count: int = 0
    error_count: int = 0
    total: int = 0
    items: list[BatchItemResult] = []
    latency_ms: float = 0


# ── Callback payload ──────────────────────────────────────────────

class CallbackPayload(CorrelationMixin):
    status: str = "completed"
    ok: bool = True
    ok_count: int = 0
    error_count: int = 0
    total: int = 0
    latency_ms: float = 0
    error_message: Optional[str] = None


# ── Error ──────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    error_code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = {}


class ErrorResponse(BaseModel):
    ok: bool = False
    error: ErrorDetail


# Valid error codes
ERROR_CODES = [
    "invalid_input",
    "path_not_found",
    "path_not_allowed",
    "manifest_mismatch",
    "load_failed",
    "unload_failed",
    "backend_incompatible",
    "adapter_load_error",
    "gpu_memory_high",
    "oom",
    "timeout",
    "callback_failed",
    "worker_crashed",
    "worker_unresponsive",
    "internal_error",
]


# ── Stage / Eval records ──────────────────────────────────────────

class StageRecord(CorrelationMixin):
    overall_ok: bool = True
    expected_host_count: int = 3
    actual_host_count: int = 0
    total_samples: int = 0
    ok_count: int = 0
    error_count: int = 0
    hosts: list[dict[str, Any]] = []
    latency_ms: float = 0


class CellEvalSummary(BaseModel):
    system_prompt_id: str
    negative_prompt_id: str
    cell_id: str = ""
    total: int = 0
    yes: int = 0
    no: int = 0
    parse_fail: int = 0
    yes_ratio: Optional[float] = None  # yes / (total - parse_fail)


class EvalSummary(BaseModel):
    run_id: str = ""
    cells: list[CellEvalSummary] = []


# ── Sample record (walker output / input manifest row) ────────────

class SampleRecord(BaseModel):
    index: int
    sample_id: str
    category: str
    source_root: str
    relpath: str
    pair_pattern: dict[str, str]
    model_image: str
    cloth_image: str
    output_relpath: str
    model_image_sha256: str = ""
    cloth_image_sha256: str = ""
    model_image_mtime: Optional[float] = None


class InputManifest(BaseModel):
    manifest_hash: str
    source_root: str
    created_at: str = ""
    items: list[SampleRecord] = []
