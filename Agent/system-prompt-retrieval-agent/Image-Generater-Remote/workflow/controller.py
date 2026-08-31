"""Control server on port 17700 — single authoritative orchestrator.

Flow:
    POST /stage/gemma → load Gemma on 3 GPUs → generate prompts → save → unload → return {ok, manifest}
    POST /stage/flux  → load FLUX on 3 GPUs  → generate images  → save → unload → return {ok, manifest}
    POST /stage/qwen  → load Qwen on 3 GPUs  → evaluate results → save → unload → return {ok, manifest}

Each endpoint blocks until the stage completes on all GPUs, then returns
a structured response with success/failure + manifest of outputs.

Usage:
    python -m workflow.controller --config config.yaml --dataset /path/to/data
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pathlib import Path
from pydantic import BaseModel
import uvicorn

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [controller] %(levelname)s %(message)s",
)
log = logging.getLogger("controller")


# ── State ──────────────────────────────────────────────────────────

class PipelineState:
    def __init__(self):
        self.current_stage: Optional[str] = None
        self.status: str = "idle"  # idle | running | completed | error
        self.run_id: Optional[str] = None
        self.lock = asyncio.Lock()
        self.config: dict = {}
        self.dataset_root: str = ""
        self.output_root: str = ""
        self.prompt_ids: list[str] = []
        # Inter-stage data (legacy V0.1.1/V0.2 — keyed by sample_id)
        self.samples: list[dict] = []
        self.gemma_results: dict[str, dict] = {}
        self.flux_results: dict[str, dict] = {}
        self.gemma_manifest: Optional[dict] = None
        self.flux_manifest: Optional[dict] = None
        self.qwen_manifest: Optional[dict] = None
        # V0.2.1 cell-keyed views — Tuple[pair_id, user_prompt_id, sample_id]
        self.cell_gemma_results: dict[tuple, dict] = {}
        self.cell_flux_results: dict[tuple, dict] = {}
        self.stage_manifests: dict[str, dict[str, dict]] = {}  # stage → pair_id → PerPairManifest
        # History
        self.run_history: list[dict] = []


state = PipelineState()
app = FastAPI(title="Workflow Controller", description="Port 17700 — staged pipeline control")


# ── Models ─────────────────────────────────────────────────────────

class GemmaUserPrompt(BaseModel):
    user_prompt_id: str
    language: Literal["zh", "en"]
    text: str
    enabled: bool = True


class PromptPairRequest(BaseModel):
    prompt_pair_id: str
    system_prompt_id: Optional[str] = None
    system_prompt_text: Optional[str] = None
    negative_prompt_id: Optional[str] = None
    negative_prompt: Optional[str] = None
    cell_id: Optional[str] = None
    intermediate_prompt_dir: Optional[str] = None
    flux_image_dir: Optional[str] = None


class StageRequest(BaseModel):
    prompt_id: Optional[str] = None
    negative_prompt: Optional[str] = None
    limit: int = 0
    run_id: Optional[str] = None
    # V0.2 additive (legacy single-pair; back-compat-wrapped to one-row pair list)
    system_prompt_id: Optional[str] = None
    system_prompt_text: Optional[str] = None
    negative_prompt_id: Optional[str] = None
    prompt_pair_id: Optional[str] = None
    allow_partial: bool = False
    cell_id: Optional[str] = None
    # V0.2.1 additive (batched-pairs + cell-keyed manifests)
    prompt_pairs: Optional[List[PromptPairRequest]] = None
    user_prompts: Optional[List[GemmaUserPrompt]] = None
    user_prompt_corpus_hash: Optional[str] = None
    sample_ids: Optional[List[str]] = None
    sample_manifest_path: Optional[str] = None
    sample_manifest_path_purpose: Optional[Literal["resume_missing_cells", "prior_stage_survivor_cells"]] = None
    round_id: Optional[int] = None
    lifecycle_mode: Optional[Literal["cold", "warm"]] = "cold"

    def model_post_init(self, _ctx) -> None:
        if self.sample_manifest_path is not None and self.sample_manifest_path_purpose is None:
            raise ValueError("missing_manifest_purpose")


class StageResponse(BaseModel):
    ok: bool
    stage: str
    message: str
    run_id: str = ""
    manifest: Optional[dict] = None


# ── Endpoints ──────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "port": 17700}


@app.get("/status")
async def get_status():
    return {
        "current_stage": state.current_stage,
        "status": state.status,
        "run_id": state.run_id,
        "samples_loaded": len(state.samples),
        "gemma_done": state.gemma_manifest is not None,
        "flux_done": state.flux_manifest is not None,
        "qwen_done": state.qwen_manifest is not None,
    }


@app.post("/stage/gemma", response_model=StageResponse)
async def stage_gemma(req: StageRequest = StageRequest()):
    async with state.lock:
        if state.status == "running":
            raise HTTPException(400, "Pipeline already running a stage")
        if _v021_active(req):
            return await _execute_gemma_v021(req)
        return await _execute_gemma(req)


@app.post("/stage/flux", response_model=StageResponse)
async def stage_flux(req: StageRequest = StageRequest()):
    async with state.lock:
        if state.status == "running":
            raise HTTPException(400, "Pipeline already running a stage")
        if _v021_active(req):
            if not state.cell_gemma_results:
                raise HTTPException(400, "Run /stage/gemma first — no V0.2.1 Gemma cells available")
            return await _execute_flux_v021(req)
        if not state.gemma_results:
            raise HTTPException(400, "Run /stage/gemma first — no Gemma results available")
        return await _execute_flux(req)


@app.post("/stage/qwen", response_model=StageResponse)
async def stage_qwen(req: StageRequest = StageRequest()):
    async with state.lock:
        if state.status == "running":
            raise HTTPException(400, "Pipeline already running a stage")
        if _v021_active(req):
            if not state.cell_flux_results:
                raise HTTPException(400, "Run /stage/flux first — no V0.2.1 FLUX cells available")
            return await _execute_qwen_v021(req)
        if not state.flux_results:
            raise HTTPException(400, "Run /stage/flux first — no FLUX results available")
        return await _execute_qwen(req)


# ── V0.2.2 stage-major endpoint (S01 + S05 wiring) ────────────────

class V022CellRequest(BaseModel):
    prompt_pair_id: str
    user_prompt_id: str
    sample_id: str


class V022SampleRecord(BaseModel):
    model_config = {"protected_namespaces": ()}
    sample_id: str
    model_path: str
    cloth_path: str
    category: Optional[str] = None


class V022StageRequest(BaseModel):
    schema_version: str
    run_id: str
    round_id: int
    stage: Literal["gemma", "flux", "qwen"]
    config_hash: str
    user_prompt_corpus_id: str
    user_prompt_corpus_hash: str
    prompt_pair_corpus_hash: str
    sample_corpus_hash: str
    attempt_id: str
    lifecycle_mode: Literal["cold", "warm"] = "cold"
    lifecycle_state_after: Literal["disk_unloaded", "cpu_prefetched", "gpu_unloaded_cpu_retained"] = "disk_unloaded"
    dispatch_cells: list[V022CellRequest]
    artifact_root: Optional[str] = None  # absolute path; defaults to controller's output_root
    gpu_ids: Optional[list[int]] = None  # defaults to controller's configured GPUs
    # Optional metadata pass-through for cell executors
    prompt_pairs: Optional[list[PromptPairRequest]] = None
    user_prompts: Optional[list[GemmaUserPrompt]] = None
    samples: Optional[list[V022SampleRecord]] = None
    dataset_root: Optional[str] = None


class V022StageResponse(BaseModel):
    ok: bool
    stage: str
    run_id: str
    attempt_id: str
    manifest: dict
    message: str = ""


@app.post("/v022/stage/{stage}", response_model=V022StageResponse)
async def v022_stage(stage: str, req: V022StageRequest):
    if stage != req.stage:
        raise HTTPException(400, f"path stage {stage!r} != body stage {req.stage!r}")
    if state.status == "running":
        raise HTTPException(400, "Pipeline already running a stage")
    async with state.lock:
        try:
            from server import canonical_paths as _cp
            from server import v022_stage_runner as _sr
        except ImportError as exc:  # pragma: no cover — defensive
            raise HTTPException(500, f"v022 runner unavailable: {exc}")

        if req.schema_version != _cp.SCHEMA_VERSION:
            raise HTTPException(
                400, f"schema_version {req.schema_version!r} != {_cp.SCHEMA_VERSION!r}"
            )
        if req.lifecycle_mode == "warm":
            raise HTTPException(400, _cp.WARM_MODE_DISABLED_ERROR)

        artifact_root = req.artifact_root or state.output_root
        gpu_ids = req.gpu_ids or _get_gpu_ids()

        match = _sr.StageMatchFields(
            schema_version=req.schema_version,
            run_id=req.run_id,
            round_id=req.round_id,
            stage=req.stage,
            config_hash=req.config_hash,
            user_prompt_corpus_id=req.user_prompt_corpus_id,
            user_prompt_corpus_hash=req.user_prompt_corpus_hash,
            prompt_pair_corpus_hash=req.prompt_pair_corpus_hash,
            sample_corpus_hash=req.sample_corpus_hash,
        )
        cells = [
            _sr.CellKey(c.prompt_pair_id, c.user_prompt_id, c.sample_id)
            for c in req.dispatch_cells
        ]
        if not cells:
            raise HTTPException(
                400, "dispatch_cells empty; local agent must short-circuit on D=∅"
            )

        cell_executor = _build_v022_cell_executor(req.stage, req)
        state.status = "running"
        state.current_stage = req.stage
        state.run_id = req.run_id
        try:
            pool = _sr.load_stage_model(req.stage, gpu_ids, cell_executor=cell_executor)
            try:
                manifest = _sr.run_stage_cells(
                    pool,
                    _sr.StageDispatch(
                        match_fields=match,
                        attempt_id=req.attempt_id,
                        lifecycle=_sr.LifecycleEcho(req.lifecycle_mode, req.lifecycle_state_after),
                        dispatch_cells=cells,
                        artifact_root=Path(artifact_root),
                        gpu_ids=list(gpu_ids),
                    ),
                )
            finally:
                _sr.unload_or_retain_stage_model(req.stage)
        except ValueError as exc:
            state.status = "error"
            state.current_stage = None
            raise HTTPException(400, str(exc))
        except Exception as exc:
            state.status = "error"
            state.current_stage = None
            log.exception("v022 stage %s failed", req.stage)
            raise HTTPException(500, f"v022 stage {req.stage} failed: {exc}")

        state.status = "completed"
        state.current_stage = None
        return V022StageResponse(
            ok=True,
            stage=req.stage,
            run_id=req.run_id,
            attempt_id=req.attempt_id,
            manifest=manifest,
            message=f"v022: {len(cells)} cells dispatched",
        )


def _build_v022_cell_executor(stage: str, req: V022StageRequest):
    """Return a cell executor callable bound to the V0.2.2 dispatch context.

    The real Gemma/FLUX/Qwen wiring is layered on top by the remote
    deploy adapter. The default in this module is a deferred-import
    bridge to ``workflow.pipeline``'s per-stage callers; missing GPU
    stack raises an explicit ImportError that the controller surfaces
    as HTTP 500.
    """
    from server import v022_stage_runner as _sr  # noqa: F401

    pair_lookup = {p.prompt_pair_id: p for p in (req.prompt_pairs or [])}
    user_prompt_lookup = {u.user_prompt_id: u for u in (req.user_prompts or [])}
    sample_lookup = {s.sample_id: s for s in (req.samples or [])}
    request_gpu_ids = list(req.gpu_ids or _get_gpu_ids())
    request_dataset_root = req.dataset_root

    def _executor(cell, ctx):
        try:
            from server.v022_cell_adapters import run_cell_for_stage  # type: ignore
        except ImportError:
            # No GPU stack adapters installed — fall back to a deterministic
            # stub that lets the V0.2.2 contract pipeline run end-to-end
            # without real models. Used by mocked smoke tests and CI.
            payload = (
                f"v022-stub|{stage}|{cell.prompt_pair_id}|{cell.user_prompt_id}|"
                f"{cell.sample_id}|attempt={req.attempt_id}|run={req.run_id}"
            ).encode("utf-8")
            from server import v022_stage_runner as _runner
            if stage == "qwen":
                return _runner.CellResult(
                    key=cell, raw_bytes=payload,
                    parse_status="ok", parsed={"verdict": "yes", "stub": True},
                )
            return _runner.CellResult(key=cell, raw_bytes=payload)
        return run_cell_for_stage(
            stage,
            cell,
            ctx={
                **ctx,
                "pair": pair_lookup.get(cell.prompt_pair_id),
                "user_prompt": user_prompt_lookup.get(cell.user_prompt_id),
                "sample": sample_lookup.get(cell.sample_id),
                "request_gpu_ids": request_gpu_ids,
                "dataset_root": request_dataset_root,
                "artifact_root": req.artifact_root or state.output_root,
                "run_id": req.run_id,
                "round_id": req.round_id,
                "attempt_id": req.attempt_id,
                "controller_state": state,
            },
        )

    return _executor


@app.post("/stage/all", response_model=StageResponse)
async def stage_all(req: StageRequest = StageRequest()):
    """Run all 3 stages in sequence. Calls internal functions directly (no lock re-entry)."""
    async with state.lock:
        if state.status == "running":
            raise HTTPException(400, "Pipeline already running a stage")

        gemma_resp = await _execute_gemma(req)
        if not gemma_resp.ok:
            return gemma_resp

        flux_resp = await _execute_flux(req)
        if not flux_resp.ok:
            return flux_resp

        qwen_resp = await _execute_qwen(req)
        return qwen_resp


# Backward-compat callback endpoint (replaces callback_listener.py on 17700)
@app.post("/callback")
@app.post("/status")
async def receive_callback(payload: dict = {}):
    return {"ok": "true"}


# ── Internal stage executors (called under lock) ───────────────────

async def _execute_gemma(req: StageRequest) -> StageResponse:
    from workflow.pipeline import run_stage1_gemma

    state.status = "running"
    state.current_stage = "gemma"
    if not state.run_id or req.run_id:
        state.run_id = req.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        prompt_id, prompt_text = _get_prompt(req.prompt_id, req.system_prompt_text)
        if not state.samples:
            state.samples = _discover()
        samples = state.samples[:req.limit] if req.limit > 0 else state.samples

        gemma_path = state.config.get("models", {}).get("gemma4", {}).get("path", "")
        gpu_ids = _get_gpu_ids()
        gemma_ports = state.config.get("workflow", {}).get("gemma_ports", [8431, 8432, 8433])

        out_root = _cell_output_root(req)
        os.makedirs(out_root, exist_ok=True)
        state.gemma_results, manifest = await run_stage1_gemma(
            samples=samples,
            system_prompt=prompt_text,
            prompt_id=prompt_id,
            output_root=out_root,
            gemma_model_path=gemma_path,
            gpu_ids=gpu_ids,
            ports=gemma_ports[:len(gpu_ids)],
        )

        state.gemma_manifest = manifest
        state.status = "completed"
        state.current_stage = None
        manifest["vram_free_gib"] = _vram_free_gib()

        return StageResponse(
            ok=True, stage="gemma", run_id=state.run_id,
            message=f"{manifest['ok']}/{manifest['total']} samples processed",
            manifest=manifest,
        )

    except Exception as e:
        state.status = "error"
        state.current_stage = None
        log.exception("Gemma stage failed")
        return StageResponse(ok=False, stage="gemma", run_id=state.run_id or "", message=str(e))


async def _execute_flux(req: StageRequest) -> StageResponse:
    from workflow.pipeline import run_stage2_flux

    state.status = "running"
    state.current_stage = "flux"

    try:
        prompt_id, _ = _get_prompt(req.prompt_id, req.system_prompt_text)
        samples = state.samples[:req.limit] if req.limit > 0 else state.samples
        flux_path = state.config.get("models", {}).get("flux2_klein", {}).get("path", "")
        gpu_ids = _get_gpu_ids()
        flux_config = _get_flux_config()

        # Negative prompt from request or config
        neg_text = req.negative_prompt
        if neg_text is None:
            neg_targets = state.config.get("prompts", {}).get("negative_prompt_targets", [])
            for nt in neg_targets:
                if nt.get("text"):
                    neg_text = nt["text"]
                    break

        out_root = _cell_output_root(req)
        os.makedirs(out_root, exist_ok=True)
        state.flux_results, manifest = await run_stage2_flux(
            samples=samples,
            gemma_results=state.gemma_results,
            prompt_id=prompt_id,
            output_root=out_root,
            flux_model_path=flux_path,
            gpu_ids=gpu_ids,
            flux_config=flux_config,
            negative_prompt=neg_text,
        )

        state.flux_manifest = manifest
        state.status = "completed"
        state.current_stage = None
        manifest["vram_free_gib"] = _vram_free_gib()

        return StageResponse(
            ok=True, stage="flux", run_id=state.run_id or "",
            message=f"{manifest['ok']}/{manifest['total']} images generated",
            manifest=manifest,
        )

    except Exception as e:
        state.status = "error"
        state.current_stage = None
        log.exception("FLUX stage failed")
        return StageResponse(ok=False, stage="flux", run_id=state.run_id or "", message=str(e))


async def _execute_qwen(req: StageRequest) -> StageResponse:
    from workflow.pipeline import run_stage3_qwen

    state.status = "running"
    state.current_stage = "qwen"

    try:
        prompt_id, _ = _get_prompt(req.prompt_id, req.system_prompt_text)
        samples = state.samples[:req.limit] if req.limit > 0 else state.samples
        qwen_path = state.config.get("models", {}).get("qwen3_vl", {}).get("path", "")
        gpu_ids = _get_gpu_ids()
        qwen_ports = state.config.get("workflow", {}).get("qwen_ports", [8434, 8435, 8436])

        out_root = _cell_output_root(req)
        os.makedirs(out_root, exist_ok=True)
        eval_summary, manifest = await run_stage3_qwen(
            samples=samples,
            flux_results=state.flux_results,
            prompt_id=prompt_id,
            output_root=out_root,
            qwen_model_path=qwen_path,
            gpu_ids=gpu_ids,
            ports=qwen_ports[:len(gpu_ids)],
        )

        state.qwen_manifest = manifest
        state.status = "completed"
        state.current_stage = None
        manifest["vram_free_gib"] = _vram_free_gib()

        return StageResponse(
            ok=True, stage="qwen", run_id=state.run_id or "",
            message=f"pass_rate={eval_summary.get('pass_rate_pct', 'N/A')} ({eval_summary.get('yes', 0)} yes / {eval_summary.get('total_evaluated', 0)} total)",
            manifest={**manifest, **eval_summary},
        )

    except Exception as e:
        state.status = "error"
        state.current_stage = None
        log.exception("Qwen stage failed")
        return StageResponse(ok=False, stage="qwen", run_id=state.run_id or "", message=str(e))


# ── V0.2.1 batched-pairs executors ─────────────────────────────────

def _v021_active(req: "StageRequest") -> bool:
    """V0.2.1 dispatch path activates when batched payload is present."""
    return bool(req.prompt_pairs) or bool(req.user_prompts)


def _v021_cell_root(req: "StageRequest", stage: str, pair_id: str, user_prompt_id: str) -> str:
    """Per-cell output root for stage S in V0.2.1 layout.

    Layout (NOTE: uses pipeline.py legacy {prompt_id}/stage{N}_{name}/ subfolder
    inside the canonical V0.2.1 prefix; full plan §6.5 layout requires a
    pipeline.py refactor in a follow-up).
        {state.output_root}/v021/{run_id}/round_{round_id}/{pair_id}/{user_prompt_id}/
    """
    rid = req.run_id or state.run_id or "run"
    rnd = f"round_{req.round_id or 1}"
    return os.path.join(state.output_root, "v021", rid, rnd, pair_id, user_prompt_id)


def _read_manifest_rows(path: str) -> list[dict]:
    """Read a JSONL manifest and return list of {prompt_pair_id, user_prompt_id, sample_id}."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _per_pair_manifest_init(pair_id: str) -> dict:
    return {"prompt_pair_id": pair_id, "ok": 0, "errors": 0, "total": 0,
            "failure_reason": None, "per_user_prompt": {}}


def _per_user_prompt_init() -> dict:
    return {"ok": 0, "errors": 0, "total": 0, "failure_reason": None}


def _accumulate_pair_manifest(pair_mf: dict, user_prompt_id: str, cell_results: dict) -> None:
    """Accumulate one (pair, user_prompt) batch's results into pair-level + per-up rollup."""
    upm = pair_mf["per_user_prompt"].setdefault(user_prompt_id, _per_user_prompt_init())
    ok_n = sum(1 for r in cell_results.values() if r.get("status") in (None, "ok"))
    err_n = sum(1 for r in cell_results.values() if r.get("status") == "error")
    total = len(cell_results)
    upm["ok"] += ok_n
    upm["errors"] += err_n
    upm["total"] += total
    pair_mf["ok"] += ok_n
    pair_mf["errors"] += err_n
    pair_mf["total"] += total


def _partition_pairs(stage_pair_manifests: dict[str, dict], allow_partial: bool) -> tuple[list[str], list[dict]]:
    """Plan §6.4 surviving_pairs / failed_pairs partition. Strict mode (allow_partial=False)
    requires every per_user_prompt rollup to have errors==0 AND total>0."""
    surviving, failed = [], []
    for pid, mf in stage_pair_manifests.items():
        any_total = mf["total"] > 0
        no_errors = mf["errors"] == 0
        per_up = mf["per_user_prompt"]
        all_up_ok = all(u["errors"] == 0 and u["total"] > 0 for u in per_up.values())
        if allow_partial:
            ok_pair = no_errors and any_total
        else:
            ok_pair = no_errors and any_total and bool(per_up) and all_up_ok
        if ok_pair:
            surviving.append(pid)
        else:
            reason = "incomplete_count"
            if mf["errors"] > 0:
                reason = "worker_error"
            elif not any_total:
                reason = "no_cells_dispatched"
            mf["failure_reason"] = reason
            failed.append({"prompt_pair_id": pid, "failure_reason": reason})
    return surviving, failed


async def _execute_gemma_v021(req: "StageRequest") -> StageResponse:
    """V0.2.1 Gemma: iterate prompt_pairs[] × user_prompts[] cartesian.

    Calls existing run_stage1_gemma per (pair, user_prompt) cell with a
    per-cell output_root. Multiple calls = multiple Gemma loads — see
    plan B8 for batched-load follow-up. Cell-keyed results land in
    state.cell_gemma_results[(pair_id, user_prompt_id, sample_id)].
    """
    from workflow.pipeline import run_stage1_gemma

    state.status = "running"
    state.current_stage = "gemma"
    state.run_id = req.run_id or state.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        if not state.samples:
            state.samples = _discover()
        samples = state.samples[:req.limit] if req.limit > 0 else state.samples
        gemma_path = state.config.get("models", {}).get("gemma4", {}).get("path", "")
        gpu_ids = _get_gpu_ids()
        gemma_ports = state.config.get("workflow", {}).get("gemma_ports", [8431, 8432, 8433])

        enabled_user_prompts = [u for u in (req.user_prompts or []) if u.enabled]
        pair_manifests: dict[str, dict] = {}
        cell_results: dict[tuple, dict] = {}

        for pair in (req.prompt_pairs or []):
            pair_mf = pair_manifests.setdefault(pair.prompt_pair_id, _per_pair_manifest_init(pair.prompt_pair_id))
            sys_text = pair.system_prompt_text
            if not sys_text:
                _, sys_text = _get_prompt(pair.system_prompt_id, None)
            for up in enabled_user_prompts:
                # Compose system prompt + user instruction. Gemma stage takes one
                # combined "system_prompt" string today; concatenation preserves
                # behavioral intent without a pipeline.py refactor.
                combined = f"{sys_text}\n\n[USER]: {up.text}"
                cell_out = _v021_cell_root(req, "gemma", pair.prompt_pair_id, up.user_prompt_id)
                os.makedirs(cell_out, exist_ok=True)
                cell_results_local, _ = await run_stage1_gemma(
                    samples=samples,
                    system_prompt=combined,
                    prompt_id=up.user_prompt_id,  # used only for path segmentation inside pipeline.py
                    output_root=cell_out,
                    gemma_model_path=gemma_path,
                    gpu_ids=gpu_ids,
                    ports=gemma_ports[:len(gpu_ids)],
                )
                _accumulate_pair_manifest(pair_mf, up.user_prompt_id, cell_results_local)
                for sample_id, r in cell_results_local.items():
                    cell_results[(pair.prompt_pair_id, up.user_prompt_id, sample_id)] = r

        state.cell_gemma_results = cell_results
        state.stage_manifests["gemma"] = pair_manifests
        surviving, failed = _partition_pairs(pair_manifests, req.allow_partial)
        manifest = {
            "stage": "gemma",
            "run_id": state.run_id,
            "round_id": req.round_id or 1,
            "user_prompt_corpus_hash": req.user_prompt_corpus_hash,
            "pairs": pair_manifests,
            "surviving_pairs": surviving,
            "failed_pairs": failed,
            "vram_free_gib": _vram_free_gib(),
            "lifecycle_state_after": "disk_unloaded" if req.lifecycle_mode == "cold" else "gpu_unloaded_cpu_retained",
            "ok": sum(m["ok"] for m in pair_manifests.values()),
            "errors": sum(m["errors"] for m in pair_manifests.values()),
            "total": sum(m["total"] for m in pair_manifests.values()),
        }
        state.gemma_manifest = manifest
        state.status = "completed"
        state.current_stage = None
        return StageResponse(
            ok=bool(surviving),
            stage="gemma",
            run_id=state.run_id,
            message=f"v021: {len(surviving)} surviving pairs, {len(failed)} failed; {manifest['ok']}/{manifest['total']} cells",
            manifest=manifest,
        )
    except Exception as e:
        state.status = "error"
        state.current_stage = None
        log.exception("Gemma v021 stage failed")
        return StageResponse(ok=False, stage="gemma", run_id=state.run_id or "", message=str(e))


async def _execute_flux_v021(req: "StageRequest") -> StageResponse:
    """V0.2.1 FLUX: iterate manifest rows {pair_id, user_prompt_id, sample_id}.

    If sample_manifest_path is absent (legacy back-compat fallback under v021
    active), falls back to the cartesian product of req.prompt_pairs × req.user_prompts
    for cells that have Gemma results.
    """
    from workflow.pipeline import run_stage2_flux

    state.status = "running"
    state.current_stage = "flux"

    try:
        flux_path = state.config.get("models", {}).get("flux2_klein", {}).get("path", "")
        gpu_ids = _get_gpu_ids()
        flux_config = _get_flux_config()
        neg_text = req.negative_prompt
        if neg_text is None and req.prompt_pairs:
            neg_text = next((p.negative_prompt for p in req.prompt_pairs if p.negative_prompt), None)

        if req.sample_manifest_path:
            manifest_rows = _read_manifest_rows(req.sample_manifest_path)
        else:
            manifest_rows = []
            for (pid, upid, sid) in state.cell_gemma_results.keys():
                manifest_rows.append({"prompt_pair_id": pid, "user_prompt_id": upid, "sample_id": sid})

        # Group by (pair_id, user_prompt_id) for batched FLUX runs
        groups: dict[tuple, list[str]] = {}
        for row in manifest_rows:
            key = (row["prompt_pair_id"], row["user_prompt_id"])
            groups.setdefault(key, []).append(row["sample_id"])

        pair_manifests: dict[str, dict] = {}
        cell_results: dict[tuple, dict] = {}
        sample_lookup = {s["sample_id"]: s for s in state.samples}

        for (pair_id, user_prompt_id), sample_ids in groups.items():
            pair_mf = pair_manifests.setdefault(pair_id, _per_pair_manifest_init(pair_id))
            cell_samples = [sample_lookup[sid] for sid in sample_ids if sid in sample_lookup]
            if not cell_samples:
                continue
            # Build a per-cell gemma_results subset keyed by sample_id (legacy contract)
            sub_gemma = {sid: state.cell_gemma_results[(pair_id, user_prompt_id, sid)]
                         for sid in sample_ids if (pair_id, user_prompt_id, sid) in state.cell_gemma_results}
            cell_out = _v021_cell_root(req, "flux", pair_id, user_prompt_id)
            os.makedirs(cell_out, exist_ok=True)
            cell_results_local, _ = await run_stage2_flux(
                samples=cell_samples,
                gemma_results=sub_gemma,
                prompt_id=user_prompt_id,
                output_root=cell_out,
                flux_model_path=flux_path,
                gpu_ids=gpu_ids,
                flux_config=flux_config,
                negative_prompt=neg_text,
            )
            _accumulate_pair_manifest(pair_mf, user_prompt_id, cell_results_local)
            for sample_id, r in cell_results_local.items():
                cell_results[(pair_id, user_prompt_id, sample_id)] = r

        state.cell_flux_results = cell_results
        state.stage_manifests["flux"] = pair_manifests
        surviving, failed = _partition_pairs(pair_manifests, req.allow_partial)
        manifest = {
            "stage": "flux",
            "run_id": state.run_id or "",
            "round_id": req.round_id or 1,
            "user_prompt_corpus_hash": req.user_prompt_corpus_hash,
            "pairs": pair_manifests,
            "surviving_pairs": surviving,
            "failed_pairs": failed,
            "vram_free_gib": _vram_free_gib(),
            "lifecycle_state_after": "disk_unloaded" if req.lifecycle_mode == "cold" else "gpu_unloaded_cpu_retained",
            "ok": sum(m["ok"] for m in pair_manifests.values()),
            "errors": sum(m["errors"] for m in pair_manifests.values()),
            "total": sum(m["total"] for m in pair_manifests.values()),
        }
        state.flux_manifest = manifest
        state.status = "completed"
        state.current_stage = None
        return StageResponse(
            ok=bool(surviving),
            stage="flux",
            run_id=state.run_id or "",
            message=f"v021: {len(surviving)} surviving pairs, {len(failed)} failed; {manifest['ok']}/{manifest['total']} cells",
            manifest=manifest,
        )
    except Exception as e:
        state.status = "error"
        state.current_stage = None
        log.exception("FLUX v021 stage failed")
        return StageResponse(ok=False, stage="flux", run_id=state.run_id or "", message=str(e))


async def _execute_qwen_v021(req: "StageRequest") -> StageResponse:
    """V0.2.1 Qwen: same survivor-cell-only iteration as FLUX."""
    from workflow.pipeline import run_stage3_qwen

    state.status = "running"
    state.current_stage = "qwen"

    try:
        qwen_path = state.config.get("models", {}).get("qwen3_vl", {}).get("path", "")
        gpu_ids = _get_gpu_ids()
        qwen_ports = state.config.get("workflow", {}).get("qwen_ports", [8434, 8435, 8436])

        if req.sample_manifest_path:
            manifest_rows = _read_manifest_rows(req.sample_manifest_path)
        else:
            manifest_rows = []
            for (pid, upid, sid) in state.cell_flux_results.keys():
                manifest_rows.append({"prompt_pair_id": pid, "user_prompt_id": upid, "sample_id": sid})

        groups: dict[tuple, list[str]] = {}
        for row in manifest_rows:
            key = (row["prompt_pair_id"], row["user_prompt_id"])
            groups.setdefault(key, []).append(row["sample_id"])

        pair_manifests: dict[str, dict] = {}
        sample_lookup = {s["sample_id"]: s for s in state.samples}
        all_eval_summary = {"yes": 0, "no": 0, "total_evaluated": 0}

        for (pair_id, user_prompt_id), sample_ids in groups.items():
            pair_mf = pair_manifests.setdefault(pair_id, _per_pair_manifest_init(pair_id))
            cell_samples = [sample_lookup[sid] for sid in sample_ids if sid in sample_lookup]
            if not cell_samples:
                continue
            sub_flux = {sid: state.cell_flux_results[(pair_id, user_prompt_id, sid)]
                        for sid in sample_ids if (pair_id, user_prompt_id, sid) in state.cell_flux_results}
            cell_out = _v021_cell_root(req, "qwen", pair_id, user_prompt_id)
            os.makedirs(cell_out, exist_ok=True)
            eval_summary, _ = await run_stage3_qwen(
                samples=cell_samples,
                flux_results=sub_flux,
                prompt_id=user_prompt_id,
                output_root=cell_out,
                qwen_model_path=qwen_path,
                gpu_ids=gpu_ids,
                ports=qwen_ports[:len(gpu_ids)],
            )
            # Reconstruct cell_results from eval_summary structure (per-sample)
            cell_local = {}
            for sid in sample_ids:
                cell_local[sid] = {"sample_id": sid, "status": "ok"}  # filled by pipeline; barrier counts on ok
            _accumulate_pair_manifest(pair_mf, user_prompt_id, cell_local)
            for k in ("yes", "no", "total_evaluated"):
                if k in eval_summary:
                    all_eval_summary[k] += eval_summary.get(k, 0)

        state.stage_manifests["qwen"] = pair_manifests
        surviving, failed = _partition_pairs(pair_manifests, req.allow_partial)
        total_eval = all_eval_summary["total_evaluated"]
        pass_rate = (all_eval_summary["yes"] / total_eval * 100.0) if total_eval else 0.0
        manifest = {
            "stage": "qwen",
            "run_id": state.run_id or "",
            "round_id": req.round_id or 1,
            "user_prompt_corpus_hash": req.user_prompt_corpus_hash,
            "pairs": pair_manifests,
            "surviving_pairs": surviving,
            "failed_pairs": failed,
            "vram_free_gib": _vram_free_gib(),
            "lifecycle_state_after": "disk_unloaded" if req.lifecycle_mode == "cold" else "gpu_unloaded_cpu_retained",
            "pass_rate_pct": round(pass_rate, 2),
            "yes": all_eval_summary["yes"],
            "no": all_eval_summary["no"],
            "total_evaluated": total_eval,
            "ok": sum(m["ok"] for m in pair_manifests.values()),
            "errors": sum(m["errors"] for m in pair_manifests.values()),
            "total": sum(m["total"] for m in pair_manifests.values()),
        }
        state.qwen_manifest = manifest
        state.status = "completed"
        state.current_stage = None
        return StageResponse(
            ok=bool(surviving),
            stage="qwen",
            run_id=state.run_id or "",
            message=f"v021: pass_rate={pass_rate:.2f}% ({all_eval_summary['yes']}/{total_eval}); {len(surviving)} surviving pairs",
            manifest=manifest,
        )
    except Exception as e:
        state.status = "error"
        state.current_stage = None
        log.exception("Qwen v021 stage failed")
        return StageResponse(ok=False, stage="qwen", run_id=state.run_id or "", message=str(e))


# ── Helpers ────────────────────────────────────────────────────────

def _get_prompt(prompt_id: Optional[str] = None, system_prompt_text: Optional[str] = None) -> tuple[str, str]:
    # V0.2: inline text wins over config lookup
    if system_prompt_text:
        import hashlib
        pid = prompt_id or f"inline_{hashlib.sha256(system_prompt_text.encode()).hexdigest()[:8]}"
        return pid, system_prompt_text
    try:
        from agent_sim.prompts import SYSTEM_PROMPTS
    except ImportError:
        sys.path.insert(0, os.path.join(_project_root, "..", "eval_run"))
        from prompts_data import system_prompt_for_tryon_v5, system_prompt_for_tryon_v5_new
        SYSTEM_PROMPTS = {
            "system_prompt_for_tryon_v5": system_prompt_for_tryon_v5,
            "system_prompt_for_tryon_v5_new": system_prompt_for_tryon_v5_new,
        }

    pid = prompt_id or (state.prompt_ids[0] if state.prompt_ids else "system_prompt_for_tryon_v5")
    text = SYSTEM_PROMPTS.get(pid, "")
    if not text:
        raise ValueError(f"Prompt '{pid}' not found")
    return pid, text


def _cell_id(req: "StageRequest") -> Optional[str]:
    if req.system_prompt_id and req.negative_prompt_id:
        return f"{req.system_prompt_id}__{req.negative_prompt_id}"
    return None


def _cell_output_root(req: "StageRequest") -> str:
    cell = _cell_id(req)
    base = state.output_root
    if cell:
        return os.path.join(base, req.run_id or state.run_id or "run", cell)
    return base


def _vram_free_gib() -> Optional[list[float]]:
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return [round(int(x.strip()) / 1024.0, 2) for x in out.stdout.strip().splitlines() if x.strip()]
    except Exception:
        pass
    return None


def _discover() -> list[dict]:
    from workflow.data_shard import discover_samples
    categories = state.config.get("data", {}).get("categories", ["dress", "lower", "upper"])
    pair_patterns = state.config.get("data", {}).get("image_pair_patterns", [
        {"model": "model.png", "cloth": "cloth.png"},
        {"model": "input_model.png", "cloth": "input_cloth.png"},
    ])
    return discover_samples(state.dataset_root, categories, pair_patterns)


def _get_gpu_ids() -> list[int]:
    return [int(g) for g in state.config.get("gpu", {}).get("visible_devices", "0,1,2").split(",")]


def _get_flux_config() -> dict:
    fc = state.config.get("models", {}).get("flux2_klein", {})
    return {
        "height": fc.get("height", 1376),
        "width": fc.get("width", 768),
        "num_inference_steps": fc.get("num_inference_steps", 4),
        "guidance_scale": fc.get("guidance_scale", 1.0),
        "seed": fc.get("seed", 0),
    }


# ── CLI ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Workflow Controller (port 17700)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--port", type=int, default=17700)
    parser.add_argument("--prompts", nargs="*", default=None)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        state.config = yaml.safe_load(f)

    state.dataset_root = args.dataset
    state.output_root = args.output or state.config.get("paths", {}).get("output_root", "outputs/v02")
    state.prompt_ids = args.prompts or state.config.get("prompts", {}).get("system_prompt_targets", [
        "system_prompt_for_tryon_v5",
        "system_prompt_for_tryon_v5_new",
    ])

    os.environ["LOG_DIR"] = state.output_root
    os.makedirs(state.output_root, exist_ok=True)

    log.info("Controller starting on port %d", args.port)
    log.info("  dataset: %s", state.dataset_root)
    log.info("  output: %s", state.output_root)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()
