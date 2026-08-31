"""V0.2.2 cell-level adapters bridging cell-keyed dispatch to real
Gemma / FLUX / Qwen model loaders.

Lifecycle
---------
``run_cell_for_stage(stage, cell, ctx)`` is invoked once per cell by the
V0.2.2 stage runner. Per-stage GPU-resident model lifecycle is managed
lazily here: the first cell of a stage triggers vLLM server / FLUX
pipeline startup; ``atexit`` and ``teardown_stage(stage)`` (called from
the stage runner's ``unload_or_retain_stage_model``) stop the
subprocesses / drop the pipelines when the controller exits or the
containing stage completes.

Required ctx keys (injected by controller + StageWorker):
    gpu_id              int       — worker GPU id
    request_gpu_ids     list      — full GPU pool for this stage
    pair                obj       — V022PromptPair record
    user_prompt         obj       — V022UserPrompt record
    sample              obj       — V022SampleRecord (model_path, cloth_path)
    artifact_root       str/Path  — absolute root for canonical cell paths
    run_id              str
    round_id            int
    attempt_id          str

Production-mode invariant
-------------------------
Stub outputs are forbidden in production. Each adapter raises rather
than emitting ``v022-stub|...`` placeholder bytes. Stub fallback is
gated by the env flag ``SPRA_V022_ALLOW_STUB`` for isolated stage
tests; in normal operation the flag is unset and any stage-execution
error surfaces as a real cell-level error rather than a silent stub.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root must be importable so workflow.* modules resolve.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from server.v022_stage_runner import CellKey, CellResult  # noqa: E402
import server.canonical_paths as cp  # noqa: E402

log = logging.getLogger("v022_cell_adapters")

# Default vLLM ports — match start_remote.sh / pipeline defaults.
GEMMA_PORTS = [8431, 8432, 8433]
QWEN_PORTS = [8434, 8435, 8436]

_STUB_ENV_FLAG = "SPRA_V022_ALLOW_STUB"


def _stub_allowed() -> bool:
    return os.environ.get(_STUB_ENV_FLAG, "").strip() not in ("", "0", "false", "False")


def _forbid_stub(stage: str) -> None:
    raise RuntimeError(
        f"v022 adapter: {stage} stub forbidden in production mode "
        f"(set {_STUB_ENV_FLAG}=1 only for isolated stage tests)"
    )


@dataclass
class _StageState:
    stage: str
    gpu_ids: list[int]
    procs: list[Any] = field(default_factory=list)
    ports: list[int] = field(default_factory=list)
    model_path: str = ""


_LOCK = threading.Lock()
_STAGES: dict[str, _StageState] = {}

# FLUX in-process pipeline cache, keyed by gpu_id (one per StageWorker thread).
# Locking strategy:
#   _FLUX_IMPORT_LOCK   — serializes `from diffusers import Flux2KleinPipeline`
#                          and `Flux2KleinPipeline.from_pretrained(...)` across
#                          threads. Diffusers' lazy module loader and
#                          from_pretrained materialization are NOT thread-safe;
#                          concurrent calls race into ImportError or meta-tensor
#                          NotImplementedError. Construction is mostly disk I/O
#                          (~10-15s once), so serializing it is cheap.
#   _FLUX_DEVICE_LOCKS  — one lock per gpu_id so the post-construction
#                          `.to(device)` (per-GPU bf16 weight upload, ~20s)
#                          runs concurrently across GPUs.
#   _FLUX_REGISTRY_LOCK — tiny lock around lazy-creation of _FLUX_DEVICE_LOCKS
#                          entries (atomic get-or-create).
_FLUX_IMPORT_LOCK = threading.Lock()
_FLUX_REGISTRY_LOCK = threading.Lock()
_FLUX_DEVICE_LOCKS: dict[int, threading.Lock] = {}
_FLUX_LOCK = threading.Lock()  # retained: still used by _teardown_flux for the bulk drop
_FLUX_PIPELINES: dict[int, Any] = {}
_FLUX_PIPELINE_CLASS: Any = None  # cached after first import
_FLUX_MODEL_PATH: str = ""


def _flux_device_lock(gpu_id: int) -> threading.Lock:
    """Return a per-GPU lock so concurrent loads on different GPUs do not
    block each other. The registry lock is held only briefly to read or
    create the per-GPU lock entry."""
    with _FLUX_REGISTRY_LOCK:
        lock = _FLUX_DEVICE_LOCKS.get(gpu_id)
        if lock is None:
            lock = threading.Lock()
            _FLUX_DEVICE_LOCKS[gpu_id] = lock
        return lock


def _resolve_model_path(stage: str) -> str:
    """Read model path from the controller's config.yaml at the project root."""
    cfg_path = os.path.join(_PROJECT_ROOT, "config.yaml")
    try:
        import yaml
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:  # pragma: no cover — defensive
        log.warning("failed to read config.yaml at %s: %s", cfg_path, exc)
        cfg = {}
    keymap = {"gemma": "gemma4", "flux": "flux2_klein", "qwen": "qwen3_vl"}
    return cfg.get("models", {}).get(keymap[stage], {}).get("path", "")


def _resolve_flux_params() -> dict[str, Any]:
    """Read FLUX generation params from controller config.yaml; fall back to V0.1 defaults."""
    cfg_path = os.path.join(_PROJECT_ROOT, "config.yaml")
    try:
        import yaml
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        cfg = {}
    fcfg = (cfg.get("models", {}) or {}).get("flux2_klein", {}) or {}
    params = fcfg.get("gen_params", {}) or {}
    return {
        "height": int(params.get("height", 1376)),
        "width": int(params.get("width", 768)),
        "num_inference_steps": int(params.get("num_inference_steps", 4)),
        "guidance_scale": float(params.get("guidance_scale", 1.0)),
        "seed": int(params.get("seed", 0)),
    }


def _ensure_vllm_loaded(stage: str, gpu_ids: list[int]) -> _StageState:
    """Start one vLLM server per GPU for ``stage`` (idempotent)."""
    with _LOCK:
        st = _STAGES.get(stage)
        if st is not None:
            return st
        from workflow.gpu_pool import start_vllm_servers
        model_path = _resolve_model_path(stage)
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"v022 adapter: {stage} model path not found: {model_path!r}. "
                f"Configure models.{ {'gemma':'gemma4','qwen':'qwen3_vl'}[stage] }.path "
                "in Image-Generater-Remote/config.yaml."
            )
        if stage == "gemma":
            ports = GEMMA_PORTS[: len(gpu_ids)]
            extra: list[str] | None = None
            gpu_util = 0.95
            max_len = 8192
        elif stage == "qwen":
            ports = QWEN_PORTS[: len(gpu_ids)]
            extra = ["--limit-mm-per-prompt", '{"image": 3}']
            gpu_util = 0.90
            max_len = 32768
        else:
            raise ValueError(f"_ensure_vllm_loaded: not a vllm stage: {stage!r}")
        log.info(
            "v022 adapter: starting %s vLLM servers on GPUs %s ports %s model=%s",
            stage, gpu_ids, ports, model_path,
        )
        procs = start_vllm_servers(
            model_path, ports, gpu_ids,
            gpu_memory_utilization=gpu_util,
            max_model_len=max_len,
            extra_args=extra,
        )
        st = _StageState(
            stage=stage, gpu_ids=list(gpu_ids), procs=list(procs),
            ports=list(ports), model_path=model_path,
        )
        _STAGES[stage] = st
        return st


def _ensure_flux_loaded(gpu_id: int) -> Any:
    """Load the FLUX diffusers pipeline once per StageWorker (keyed by gpu_id).

    StageWorker is in-process (not a subprocess), so CUDA_VISIBLE_DEVICES
    is NOT scoped per worker. We must place each pipeline on the
    absolute device cuda:{gpu_id}; using cuda:0 collapses all workers
    onto GPU 0 and OOMs.
    """
    global _FLUX_MODEL_PATH, _FLUX_PIPELINE_CLASS
    dev_lock = _flux_device_lock(gpu_id)
    with dev_lock:
        pipe = _FLUX_PIPELINES.get(gpu_id)
        if pipe is not None:
            return pipe
        import torch  # imported lazily to avoid cost when FLUX never runs
        model_path = _resolve_model_path("flux")
        if not model_path or not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"v022 adapter: flux model path not found: {model_path!r}. "
                "Configure models.flux2_klein.path in Image-Generater-Remote/config.yaml."
            )
        device = f"cuda:{gpu_id}"
        log.info(
            "v022 adapter: loading FLUX pipeline gpu=%d device=%s model=%s",
            gpu_id, device, model_path,
        )
        # Serialize the import + from_pretrained construction (diffusers
        # lazy-import + meta-tensor materialization are not thread-safe).
        with _FLUX_IMPORT_LOCK:
            if _FLUX_PIPELINE_CLASS is None:
                from diffusers import Flux2KleinPipeline as _Cls
                _FLUX_PIPELINE_CLASS = _Cls
            pipe = _FLUX_PIPELINE_CLASS.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
            )
        # `.to(device)` (per-GPU weight upload) is safe to run concurrently.
        pipe.to(device)
        # Cap peak VRAM: attention slicing + VAE slicing/tiling.
        # Without these, a single 1376×768 4-step call can spike >40 GiB,
        # and a second call on the same GPU OOMs at 79 GiB.
        for fn_name in (
            "enable_attention_slicing", "enable_vae_slicing",
            "enable_vae_tiling", "enable_xformers_memory_efficient_attention",
        ):
            fn = getattr(pipe, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    log.info("v022 adapter: FLUX %s() enabled", fn_name)
                except Exception as exc:
                    log.info("v022 adapter: FLUX %s() skipped: %s", fn_name, exc)
        _FLUX_PIPELINES[gpu_id] = pipe
        _FLUX_MODEL_PATH = model_path
        return pipe


def teardown_stage(stage: str) -> None:
    """Tear down per-stage resources. Wired from v022_stage_runner.unload_or_retain_stage_model."""
    if stage == "flux":
        _teardown_flux()
        return
    with _LOCK:
        st = _STAGES.pop(stage, None)
    if st is None or not st.procs:
        return
    try:
        from workflow.gpu_pool import stop_servers
        stop_servers(st.procs)
        log.info("v022 adapter: stopped %s vLLM servers", stage)
    except Exception:  # pragma: no cover — defensive
        log.exception("v022 adapter: teardown_stage(%s) failed", stage)


def _teardown_flux() -> None:
    """Force-release FLUX VRAM after a stage. ``del pipe`` alone leaves
    ~18 GiB resident because diffusers' submodules hold circular refs
    and the CUDA caching allocator retains blocks until cycles break.

    Sequence: move pipeline → CPU, null every known submodule attr, drop
    the dict ref, run gc, synchronize all CUDA streams, then empty_cache.
    """
    with _FLUX_LOCK:
        if not _FLUX_PIPELINES:
            return
        try:
            import gc
            import torch
        except Exception:
            torch = None  # type: ignore
            gc = None  # type: ignore
        # Diffusers Flux2Klein pipeline known submodule attrs to null.
        FLUX_SUBMODULES = (
            "transformer", "vae", "text_encoder", "text_encoder_2",
            "text_encoder_3", "tokenizer", "tokenizer_2", "tokenizer_3",
            "image_encoder", "feature_extractor", "scheduler", "unet",
        )
        for gpu_id in list(_FLUX_PIPELINES.keys()):
            try:
                pipe = _FLUX_PIPELINES.pop(gpu_id)
                # Move pipe off GPU first so weights drop CUDA refs.
                try:
                    if hasattr(pipe, "to"):
                        pipe.to("cpu")
                except Exception:
                    pass
                # Null out every submodule attribute.
                for attr in FLUX_SUBMODULES:
                    try:
                        if hasattr(pipe, attr):
                            setattr(pipe, attr, None)
                    except Exception:
                        pass
                del pipe
            except Exception:
                pass
        if gc is not None:
            try:
                gc.collect()
            except Exception:
                pass
        if torch is not None:
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        log.info("v022 adapter: dropped FLUX pipelines + reclaimed CUDA memory")


@atexit.register
def _atexit_teardown() -> None:
    for stage in list(_STAGES.keys()):
        try:
            teardown_stage(stage)
        except Exception:
            pass
    try:
        _teardown_flux()
    except Exception:
        pass


# ---- Helpers --------------------------------------------------------------


def _stub_bytes(stage: str, cell: CellKey, ctx: dict) -> bytes:
    """Match the controller's import-failure stub format exactly. Only callable
    behind the SPRA_V022_ALLOW_STUB env flag."""
    return (
        f"v022-stub|{stage}|{cell.prompt_pair_id}|{cell.user_prompt_id}|"
        f"{cell.sample_id}|attempt={ctx.get('attempt_id', '')}|"
        f"run={ctx.get('run_id', '')}"
    ).encode("utf-8")


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read attr/key from pydantic model | dict | dataclass | None."""
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default


def _prior_stage_artifact_path(stage: str, cell: CellKey, ctx: dict) -> Path:
    """Resolve the absolute path of a prior stage's canonical artifact for this cell."""
    artifact_root = ctx.get("artifact_root")
    run_id = ctx.get("run_id")
    round_id = ctx.get("round_id")
    if not artifact_root or not run_id or round_id is None:
        raise RuntimeError(
            f"v022 adapter: ctx missing artifact_root/run_id/round_id for prior-stage "
            f"resolution (stage={stage!r}, cell={cell})"
        )
    relpath = cp.cell_artifact_path(
        run_id, stage, int(round_id),
        cell.prompt_pair_id, cell.user_prompt_id, cell.sample_id,
    )
    return Path(artifact_root) / relpath


def _validate_image_inputs(stage: str, sample: Any, cell: CellKey) -> tuple[str, str]:
    model_image = _attr(sample, "model_path", "")
    cloth_image = _attr(sample, "cloth_path", "")
    if not model_image or not cloth_image:
        raise FileNotFoundError(
            f"{stage} adapter: missing image paths for sample_id={cell.sample_id}"
        )
    if not os.path.isfile(model_image) or not os.path.isfile(cloth_image):
        raise FileNotFoundError(
            f"{stage} adapter: image files not found: model={model_image!r} "
            f"cloth={cloth_image!r}"
        )
    return model_image, cloth_image


# ---- Gemma ---------------------------------------------------------------


def _gemma_run_cell(cell: CellKey, ctx: dict) -> CellResult:
    from openai import OpenAI
    from PIL import Image
    from workflow.image_preprocess import (
        resize_model_image, resize_cloth_image, to_base64_url,
    )

    gpu_id = int(ctx["gpu_id"])
    gpu_ids = list(ctx.get("request_gpu_ids") or [gpu_id])
    st = _STAGES.get("gemma") or _ensure_vllm_loaded("gemma", gpu_ids)
    if gpu_id not in st.gpu_ids:
        raise RuntimeError(
            f"gemma adapter: gpu_id={gpu_id} not in pool {st.gpu_ids}"
        )
    port = st.ports[st.gpu_ids.index(gpu_id)]

    pair = ctx.get("pair")
    user_prompt = ctx.get("user_prompt")
    sample = ctx.get("sample")
    if pair is None or user_prompt is None or sample is None:
        raise RuntimeError(
            "gemma adapter: ctx missing pair/user_prompt/sample "
            f"(pair={pair is not None}, up={user_prompt is not None}, "
            f"sample={sample is not None})"
        )
    system_prompt = (
        _attr(pair, "system_prompt_text")
        or _attr(pair, "system_prompt")
        or ""
    )
    user_instruction = _attr(user_prompt, "text", "") or ""
    model_image, cloth_image = _validate_image_inputs("gemma", sample, cell)

    img_model = Image.open(model_image)
    img_model = resize_model_image(img_model, 1376, 768)
    img_cloth = Image.open(cloth_image)
    img_cloth = resize_cloth_image(img_cloth, 1376, 768)
    url_model = to_base64_url(img_model)
    url_cloth = to_base64_url(img_cloth)

    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{port}/v1")
    models = client.models.list()
    model_name = models.data[0].id if models.data else "unknown"

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text",
                 "text": "Image 1: Model Image - Shows the model wearing current clothing"},
                {"type": "image_url", "image_url": {"url": url_model}},
                {"type": "text",
                 "text": "Image 2: Clothing Image - Shows the garment to be tried on"},
                {"type": "image_url", "image_url": {"url": url_cloth}},
                {"type": "text", "text": user_instruction},
            ]},
        ],
        temperature=0.1, top_p=0.5, presence_penalty=1.5,
        extra_body={
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    text = response.choices[0].message.content or ""
    return CellResult(key=cell, raw_bytes=text.encode("utf-8"))


# ---- FLUX ---------------------------------------------------------------


def _read_gemma_caption(cell: CellKey, ctx: dict) -> str:
    """Read the upstream Gemma caption for this cell from the canonical artifact."""
    p = _prior_stage_artifact_path("gemma", cell, ctx)
    if not p.is_file():
        raise FileNotFoundError(
            f"flux adapter: missing upstream gemma caption at {p} "
            f"(cell={cell.prompt_pair_id}/{cell.user_prompt_id}/{cell.sample_id})"
        )
    return p.read_text(encoding="utf-8")


def _normalize_negative_prompt(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    s = value.strip()
    if s == "" or s.lower() == "none":
        return None
    return s


def _flux_run_cell(cell: CellKey, ctx: dict) -> CellResult:
    import io
    import inspect
    import torch
    from PIL import Image
    from workflow.image_preprocess import resize_model_image, resize_cloth_image

    gpu_id = int(ctx["gpu_id"])
    pipe = _ensure_flux_loaded(gpu_id)

    pair = ctx.get("pair")
    sample = ctx.get("sample")
    if sample is None:
        raise RuntimeError(
            f"flux adapter: ctx missing sample (cell={cell.sample_id})"
        )

    intermediate_prompt = _read_gemma_caption(cell, ctx)
    neg = _normalize_negative_prompt(_attr(pair, "negative_prompt", None))

    params = _resolve_flux_params()
    height = params["height"]
    width = params["width"]
    num_steps = params["num_inference_steps"]
    guidance = params["guidance_scale"]
    seed = params["seed"]

    model_image, cloth_image = _validate_image_inputs("flux", sample, cell)
    img_model = resize_model_image(Image.open(model_image), height, width)
    img_cloth = resize_cloth_image(Image.open(cloth_image), height, width)
    w, h = img_model.size

    pipe_sig = inspect.signature(pipe.__call__)
    supports_neg = "negative_prompt" in pipe_sig.parameters

    pipe_kwargs: dict[str, Any] = dict(
        image=[img_model, img_cloth],
        prompt=intermediate_prompt,
        height=h,
        width=w,
        guidance_scale=guidance,
        num_inference_steps=num_steps,
        generator=torch.Generator(device=f"cuda:{gpu_id}").manual_seed(seed),
    )
    if neg is not None and supports_neg:
        pipe_kwargs["negative_prompt"] = neg

    with torch.inference_mode():
        result_img = pipe(**pipe_kwargs).images[0]
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    raw = buf.getvalue()
    # Inter-cell VRAM cleanup — diffusers pipeline retains intermediate
    # latents/encoder caches per call; without an explicit empty_cache
    # the second cell on the same GPU OOMs at ~79 GiB.
    try:
        del result_img, buf
        import gc
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass
    return CellResult(key=cell, raw_bytes=raw)


# ---- Qwen ----------------------------------------------------------------


_QWEN_EVAL_INSTRUCTION = """Please compare three images:

1. Image 1: Original person image showing appearance and pose.
2. Image 2: Target clothing image showing garment details.
3. Image 3: Final result showing person wearing target clothing.

Tasks:
- Judge if Image 2's clothing is accurately worn by Image 1's person.
- Ensure non-clothing areas (hair, face, background) remain unchanged between Image 1 and Image 3.
- Judge if a valid try-on result was generated.

Output:
- If Image 2's clothing is perfectly worn and Image 3 has no non-clothing changes, output "yes".
- Otherwise, output "no".
"""


def _parse_yes_no(raw: str) -> str:
    """Parse yes/no from raw VLM response (mirrors workflow/stage_qwen.py)."""
    if not raw or not raw.strip():
        return "parse_fail_empty"
    text = raw.strip().lower()
    if text in ("yes", "y"):
        return "yes"
    if text in ("no", "n"):
        return "no"
    first_word = text.split()[0].strip(".,!?;:")
    if first_word in ("yes", "y", "true", "1"):
        return "yes"
    if first_word in ("no", "n", "false", "0"):
        return "no"
    has_yes = "yes" in text
    has_no = "no" in text
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    if has_yes and has_no:
        return "parse_fail_multiline"
    return "parse_fail_unknown"


def _downscale(img: Any, max_dim: int = 800) -> Any:
    from PIL import Image
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def _qwen_run_cell(cell: CellKey, ctx: dict) -> CellResult:
    import json
    import time
    from openai import OpenAI
    from PIL import Image
    from workflow.image_preprocess import to_base64_url

    gpu_id = int(ctx["gpu_id"])
    gpu_ids = list(ctx.get("request_gpu_ids") or [gpu_id])
    st = _STAGES.get("qwen") or _ensure_vllm_loaded("qwen", gpu_ids)
    if gpu_id not in st.gpu_ids:
        raise RuntimeError(
            f"qwen adapter: gpu_id={gpu_id} not in pool {st.gpu_ids}"
        )
    port = st.ports[st.gpu_ids.index(gpu_id)]

    sample = ctx.get("sample")
    if sample is None:
        raise RuntimeError(
            f"qwen adapter: ctx missing sample (cell={cell.sample_id})"
        )
    model_image, cloth_image = _validate_image_inputs("qwen", sample, cell)

    flux_path = _prior_stage_artifact_path("flux", cell, ctx)
    if not flux_path.is_file():
        raise FileNotFoundError(
            f"qwen adapter: missing upstream FLUX result at {flux_path} "
            f"(cell={cell.prompt_pair_id}/{cell.user_prompt_id}/{cell.sample_id})"
        )

    img_model = _downscale(Image.open(model_image))
    img_cloth = _downscale(Image.open(cloth_image))
    img_result = _downscale(Image.open(str(flux_path)))

    url_model = to_base64_url(img_model)
    url_cloth = to_base64_url(img_cloth)
    url_result = to_base64_url(img_result)

    client = OpenAI(api_key="EMPTY", base_url=f"http://127.0.0.1:{port}/v1")
    models = client.models.list()
    model_name = models.data[0].id if models.data else "unknown"

    t0 = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url_model}},
                {"type": "image_url", "image_url": {"url": url_cloth}},
                {"type": "image_url", "image_url": {"url": url_result}},
                {"type": "text", "text": _QWEN_EVAL_INSTRUCTION},
            ]},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    latency = round(time.time() - t0, 2)
    raw_response = response.choices[0].message.content or ""

    parsed = _parse_yes_no(raw_response)
    parse_status = f"matched_{parsed}" if parsed in ("yes", "no") else parsed

    eval_data = {
        "sample_id": cell.sample_id,
        "prompt_pair_id": cell.prompt_pair_id,
        "user_prompt_id": cell.user_prompt_id,
        "raw_response": raw_response,
        "parsed": parsed if parsed in ("yes", "no") else None,
        "parse_status": parse_status,
        "latency_s": latency,
    }
    eval_bytes = (json.dumps(eval_data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")

    if parsed in ("yes", "no"):
        runner_parse_status = "ok"
    else:
        runner_parse_status = "parse_failed"

    return CellResult(
        key=cell,
        raw_bytes=eval_bytes,
        parse_status=runner_parse_status,
        parsed={"verdict": parsed if parsed in ("yes", "no") else None,
                "raw": raw_response,
                "parse_status": parse_status},
    )


# ---- Public dispatch -----------------------------------------------------


def run_cell_for_stage(stage: str, cell: CellKey, ctx: dict) -> CellResult:
    if stage == "gemma":
        return _gemma_run_cell(cell, ctx)
    if stage == "flux":
        return _flux_run_cell(cell, ctx)
    if stage == "qwen":
        return _qwen_run_cell(cell, ctx)
    raise ValueError(f"unknown stage {stage!r}")


def run_cell_for_stage_or_stub(stage: str, cell: CellKey, ctx: dict) -> CellResult:
    """Production-mode-aware entry. Real adapter unless SPRA_V022_ALLOW_STUB is set."""
    if not _stub_allowed():
        return run_cell_for_stage(stage, cell, ctx)
    log.warning(
        "v022 adapter: %s using stub bytes (SPRA_V022_ALLOW_STUB set; isolated test)",
        stage,
    )
    if stage == "qwen":
        return CellResult(
            key=cell,
            raw_bytes=_stub_bytes(stage, cell, ctx),
            parse_status="ok",
            parsed={"verdict": "yes", "stub": True},
        )
    return CellResult(key=cell, raw_bytes=_stub_bytes(stage, cell, ctx))
