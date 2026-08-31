#!/usr/bin/env python3
"""Gradio demo: FLUX.2 Klein latent edit + PiD decode (single image)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import torch
from PIL import Image

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from csv_pid_batch_infer import (  # noqa: E402
    DEFAULT_LORA,
    DEFAULT_MODELS_ROOT,
    DEFAULT_PID_ROOT,
    apply_resolution_scale_crop,
    build_ldm_pipeline,
    build_runtime,
    crop_resolution_to_scale,
    decode_with_pid,
    ensure_pid_import,
    parse_resolution,
    patch_pid_text_encoder_local,
    resize_reference_like_diffsynth,
    set_pid_device,
    sync_cuda,
)
from input_face_preprocess import FaceCropper, ensure_rgb  # noqa: E402
from gradio_pid_infer_log import DEFAULT_LOG_DIR, GradioInferLogger  # noqa: E402

_SHENSHENG_ROOT = os.environ.get("SHENSHENG_ROOT", "/mnt/data/image-edit/datasets/shensheng")
DEFAULT_BACKBONE_MODEL = f"{DEFAULT_MODELS_ROOT}/black-forest-labs/FLUX.2-klein-9B"
DEFAULT_FLUX2_MODEL = f"{DEFAULT_MODELS_ROOT}/black-forest-labs/FLUX.2-dev"
DEFAULT_SEED = -1  # -1 表示每次生成随机 seed
DEFAULT_LDM_STEPS = 4
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_PID_STEPS = 4
DEFAULT_PID_CFG = 1.0

BACKBONE_CHOICES = ["flux2-klein-9b", "flux2-klein-4b", "flux2"]
PID_CKPT_CHOICES = ["2k", "2kto4k"]
PRELOAD_BACKBONE = "flux2-klein-9b"
PRELOAD_PID_CKPT = "2kto4k"
PID_RESOLUTIONS = {
    "2k": (2304, 2600),
    "2kto4k": (3584, 4608),
}
FIXED_SCALE = 4  # LDM→PiD 固定 4 倍，其他倍率效果差
LORA_PRESETS = {
    "none": "",
    "DreamFace v2.1 (diffusers)": (
        f"{DEFAULT_MODELS_ROOT}/hithink-image-labs/DreamFace_lora/v2.1/diffusers_lora.safetensors"
    ),
    "sft_facemask epoch-10": (
        f"{_SHENSHENG_ROOT}/code/stable/DreamFaceOmini/models/train/"
        "sft_facemask_w3_captions_dual/epoch-10.safetensors"
    ),
}
BACKBONE_MODEL_PRESETS = {
    "FLUX.2-klein-9B (local)": DEFAULT_BACKBONE_MODEL,
    "FLUX.2-klein-4B (HF id)": "black-forest-labs/FLUX.2-klein-4B",
    "FLUX.2-dev (local)": DEFAULT_FLUX2_MODEL,
    "FLUX.2-dev (HF id)": "black-forest-labs/FLUX.2-dev",
}


def resolution_for_pid_ckpt(pid_ckpt_type: str) -> tuple[int, int]:
    return PID_RESOLUTIONS.get(pid_ckpt_type, PID_RESOLUTIONS["2k"])


def resolve_seed(seed: int) -> int:
    if int(seed) < 0:
        return int(torch.randint(0, 2**63 - 1, (1,), dtype=torch.long).item())
    return int(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio: FLUX.2 + PiD single-image inference")
    parser.add_argument("--pid-root", default=os.environ.get("PID_ROOT", DEFAULT_PID_ROOT))
    parser.add_argument("--backbone", default="flux2-klein-9b", choices=BACKBONE_CHOICES)
    parser.add_argument("--backbone-model-id", default=DEFAULT_BACKBONE_MODEL)
    parser.add_argument("--pid-ckpt-type", default="2kto4k", choices=PID_CKPT_CHOICES)
    parser.add_argument("--lora", default=DEFAULT_LORA)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--scale", type=int, default=FIXED_SCALE, help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--ldm-steps", type=int, default=DEFAULT_LDM_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--pid-steps", type=int, default=DEFAULT_PID_STEPS)
    parser.add_argument("--pid-cfg", type=float, default=DEFAULT_PID_CFG)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--log-dir", default=os.environ.get("GRADIO_PID_LOG_DIR", str(DEFAULT_LOG_DIR)))
    parser.add_argument("--no-log", action="store_true", help="Disable backend run logging")
    args = parser.parse_args()
    default_w, default_h = resolution_for_pid_ckpt(args.pid_ckpt_type)
    if args.width is None:
        args.width = default_w
    if args.height is None:
        args.height = default_h
    if args.scale != FIXED_SCALE:
        print(f"[warn] scale 固定为 {FIXED_SCALE}，忽略 --scale={args.scale}")
        args.scale = FIXED_SCALE
    return args


def make_args_namespace(
    *,
    pid_root: str,
    backbone: str,
    backbone_model_id: str,
    pid_ckpt_type: str,
    lora_path: str,
    lora_scale: float,
    scale: int,
    width: int,
    height: int,
    seed: int,
    ldm_steps: int,
    guidance_scale: float,
    pid_steps: int,
    pid_cfg: float,
    low_vram: bool,
    local_files_only: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        pid_root=pid_root,
        backbone=backbone,
        backbone_model_id=backbone_model_id or None,
        pid_ckpt_type=pid_ckpt_type,
        checkpoint_path=None,
        experiment=None,
        resolution=f"{int(width)},{int(height)}",
        scale=int(scale),
        ldm_steps=int(ldm_steps),
        guidance_scale=float(guidance_scale),
        pid_steps=int(pid_steps),
        pid_cfg=float(pid_cfg),
        seed=int(seed),
        lora=lora_path or None,
        lora_scale=float(lora_scale),
        cpu_offload=low_vram,
        staged_vram=low_vram,
        low_vram=low_vram,
        max_vram_gb=None,
        compile_pid=False,
        compile_ldm=False,
        local_files_only=local_files_only,
        no_lora=not bool(lora_path),
        save_vae_baseline=False,
        patch_size=16,
    )


def format_resolution_info(width: int, height: int, scale: int) -> str:
    cropped_w, cropped_h = crop_resolution_to_scale(int(width), int(height), int(scale))
    ldm_w, ldm_h = cropped_w // scale, cropped_h // scale
    lines = [f"输出: {cropped_w} x {cropped_h}", f"LDM: {ldm_w} x {ldm_h} (scale={scale})"]
    if (cropped_w, cropped_h) != (int(width), int(height)):
        lines.insert(0, f"已对齐: {int(width)}x{int(height)} -> {cropped_w}x{cropped_h}")
    return "\n".join(lines)


def format_infer_timings(steps: list[tuple[str, float]], total_sec: float) -> str:
    lines = ["--- 耗时（自点击生成） ---"]
    for name, sec in steps:
        pct = sec / total_sec * 100 if total_sec > 0 else 0
        lines.append(f"  {name}: {sec:.2f}s ({pct:.0f}%)")
    lines.append(f"  总计: {total_sec:.2f}s")
    return "\n".join(lines)


def _elapsed_gpu(t0: float) -> float:
    sync_cuda()
    return time.perf_counter() - t0


def gradio_display_dir() -> Path:
    """Temporary JPEG cache for Gradio UI only; permanent copies go to log dir."""
    return Path(os.environ.get("GRADIO_PID_DISPLAY_DIR", Path(tempfile.gettempdir()) / "gradio_pid_display"))


def _cleanup_old_display_files(out_dir: Path, *, max_age_hours: int = 24) -> None:
    cutoff = time.time() - max_age_hours * 3600
    try:
        for path in out_dir.glob("*.jpg"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
    except OSError:
        pass


def save_gradio_output_image(img: Image.Image) -> str:
    out_dir = gradio_display_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_old_display_files(out_dir)
    out_path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}.jpg"
    img.convert("RGB").save(
        out_path,
        format="JPEG",
        quality=92,
        subsampling=2,
        optimize=False,
    )
    return str(out_path)


def gradio_allowed_paths(*extra: str | Path) -> list[str]:
    paths = {
        str(tempfile.gettempdir()),
        str(gradio_display_dir().resolve()),
    }
    for item in extra:
        if item:
            paths.add(str(Path(item).resolve()))
    return sorted(paths)


def pipeline_cache_key(args_ns: SimpleNamespace) -> tuple:
    return (
        args_ns.pid_root,
        args_ns.backbone,
        args_ns.backbone_model_id,
        args_ns.lora,
        args_ns.lora_scale,
        args_ns.low_vram,
        args_ns.local_files_only,
    )


def runtime_cache_key(args_ns: SimpleNamespace) -> tuple:
    return pipeline_cache_key(args_ns) + (args_ns.pid_ckpt_type,)


def format_runtime_status(args_ns: SimpleNamespace, *, prefix: str = "已加载") -> str:
    h_out, w_out = parse_resolution(args_ns.resolution)
    return (
        f"{prefix}\n"
        f"Backbone: {args_ns.backbone}\n"
        f"Model: {args_ns.backbone_model_id or '(default)'}\n"
        f"PiD: {args_ns.pid_ckpt_type}\n"
        f"LoRA: {args_ns.lora or 'none'} (scale={args_ns.lora_scale})\n"
        f"默认输出: {w_out}x{h_out}"
    )


class GradioModelPool:
    """Preload PiD decoders + LDM pipelines; switch configs from cache at runtime."""

    def __init__(self, cli_args):
        self.cli_args = cli_args
        self.device = f"cuda:{cli_args.gpu}"
        self.lock = threading.Lock()
        self.pipelines: dict[tuple, tuple] = {}
        self.pid_models: dict[str, object] = {}
        self.runtimes: dict[tuple, SimpleNamespace] = {}
        self.active_key: tuple | None = None
        self.status = "未加载"

    def _load_pid_decoder(self, args_ns: SimpleNamespace):
        from pid._src.inference.checkpoint_registry import get_pid_checkpoint
        from pid._src.inference.decoder import load_our_decoder

        pid_root = Path(args_ns.pid_root).resolve()
        ckpt = get_pid_checkpoint(args_ns.backbone, args_ns.pid_ckpt_type)
        checkpoint_path = args_ns.checkpoint_path or ckpt.checkpoint_path
        if not Path(checkpoint_path).is_absolute():
            checkpoint_path = str(pid_root / checkpoint_path)
        if not Path(checkpoint_path).is_file():
            raise FileNotFoundError(f"PiD checkpoint missing: {checkpoint_path}")

        config_file = "pid/_src/configs/pid/config.py"
        os.chdir(pid_root)
        decoder_args = SimpleNamespace(
            experiment=args_ns.experiment or ckpt.experiment,
            checkpoint_path=checkpoint_path,
            config_file=config_file,
            load_ema_to_reg=False,
            compile=args_ns.compile_pid,
            extra_experiment_opts=[],
        )
        print(f"[preload] PiD decoder: {args_ns.pid_ckpt_type} ({checkpoint_path})")
        return load_our_decoder(decoder_args, [], is_rank0=True)

    def _assemble_runtime(
        self,
        args_ns: SimpleNamespace,
        pipeline,
        pipe_cfg,
        pid_model,
    ) -> SimpleNamespace:
        from pid._src.inference.pipeline_registry import decode_with_pipeline_vae, extract_latent

        apply_resolution_scale_crop(args_ns)
        h_out, w_out = parse_resolution(args_ns.resolution)
        h_ldm, w_ldm = h_out // args_ns.scale, w_out // args_ns.scale
        ldm_steps = args_ns.ldm_steps or pipe_cfg.default_num_inference_steps
        guidance = (
            args_ns.guidance_scale
            if args_ns.guidance_scale is not None
            else pipe_cfg.default_guidance_scale
        )
        return SimpleNamespace(
            pipeline=pipeline,
            pipe_cfg=pipe_cfg,
            pid_model=pid_model,
            device=self.device,
            h_out=h_out,
            w_out=w_out,
            h_ldm=h_ldm,
            w_ldm=w_ldm,
            ldm_steps=ldm_steps,
            guidance=guidance,
            scale=args_ns.scale,
            pid_steps=args_ns.pid_steps,
            pid_cfg=args_ns.pid_cfg,
            seed=args_ns.seed,
            save_vae_baseline=args_ns.save_vae_baseline,
            staged_vram=args_ns.staged_vram or args_ns.low_vram,
            cpu_offload=args_ns.cpu_offload or args_ns.low_vram,
            decode_with_pipeline_vae=decode_with_pipeline_vae,
            extract_latent=extract_latent,
        )

    def _get_or_build_pipeline(self, args_ns: SimpleNamespace):
        key = pipeline_cache_key(args_ns)
        if key not in self.pipelines:
            print(
                f"[load] LDM pipeline: backbone={args_ns.backbone} "
                f"model={args_ns.backbone_model_id} lora={args_ns.lora or 'none'}"
            )
            self.pipelines[key] = build_ldm_pipeline(args_ns, self.device)
        return self.pipelines[key]

    def _get_or_build_pid_model(self, args_ns: SimpleNamespace):
        ckpt_type = args_ns.pid_ckpt_type
        if ckpt_type not in self.pid_models:
            self.pid_models[ckpt_type] = self._load_pid_decoder(args_ns)
        return self.pid_models[ckpt_type]

    def _store_runtime(self, args_ns: SimpleNamespace) -> SimpleNamespace:
        pipeline, pipe_cfg = self._get_or_build_pipeline(args_ns)
        pid_model = self._get_or_build_pid_model(args_ns)
        runtime = self._assemble_runtime(args_ns, pipeline, pipe_cfg, pid_model)
        self.runtimes[runtime_cache_key(args_ns)] = runtime
        return runtime

    def preload(self) -> str:
        cli = self.cli_args
        lora_path = cli.lora if cli.lora and Path(cli.lora).is_file() else ""
        if cli.lora and not lora_path:
            print(f"[warn] preload LoRA not found, skip: {cli.lora}")

        ensure_pid_import(Path(cli.pid_root))
        patch_pid_text_encoder_local(DEFAULT_MODELS_ROOT)

        model_id = cli.backbone_model_id or DEFAULT_BACKBONE_MODEL
        args_ns = make_args_namespace(
            pid_root=cli.pid_root,
            backbone=PRELOAD_BACKBONE,
            backbone_model_id=model_id,
            pid_ckpt_type=PRELOAD_PID_CKPT,
            lora_path=lora_path,
            lora_scale=cli.lora_scale,
            scale=cli.scale,
            width=cli.width,
            height=cli.height,
            seed=cli.seed,
            ldm_steps=cli.ldm_steps,
            guidance_scale=cli.guidance_scale,
            pid_steps=cli.pid_steps,
            pid_cfg=cli.pid_cfg,
            low_vram=cli.low_vram,
            local_files_only=cli.local_files_only,
        )

        with self.lock:
            self._store_runtime(args_ns)
            self.active_key = runtime_cache_key(args_ns)

        self.status = format_runtime_status(args_ns, prefix="启动预加载完成")
        print(self.status)
        return self.status

    @property
    def runtime(self) -> SimpleNamespace | None:
        if self.active_key is None:
            return None
        return self.runtimes.get(self.active_key)

    def ensure_runtime(self, args_ns: SimpleNamespace, *, force: bool = False) -> str:
        key = runtime_cache_key(args_ns)
        if not force and key in self.runtimes:
            self.active_key = key
            self.status = format_runtime_status(args_ns)
            return self.status

        with self.lock:
            if not force and key in self.runtimes:
                self.active_key = key
                self.status = format_runtime_status(args_ns)
                return self.status

            if key in self.runtimes and not force:
                runtime = self.runtimes[key]
            else:
                try:
                    runtime = self._store_runtime(args_ns)
                except Exception:
                    print("[load] cache miss, full build_runtime fallback")
                    ensure_pid_import(Path(args_ns.pid_root))
                    apply_resolution_scale_crop(args_ns)
                    runtime = build_runtime(args_ns, self.device)
                    self.runtimes[key] = runtime

            self.active_key = key
            self.status = format_runtime_status(args_ns)
            return self.status

    def _update_runtime_resolution(self, runtime: SimpleNamespace, args_ns: SimpleNamespace):
        apply_resolution_scale_crop(args_ns)
        h_out, w_out = parse_resolution(args_ns.resolution)
        scale = int(args_ns.scale)
        runtime.h_out = h_out
        runtime.w_out = w_out
        runtime.h_ldm = h_out // scale
        runtime.w_ldm = w_out // scale
        runtime.scale = scale
        runtime.ldm_steps = int(args_ns.ldm_steps)
        runtime.guidance = float(args_ns.guidance_scale)
        runtime.pid_steps = int(args_ns.pid_steps)
        runtime.pid_cfg = float(args_ns.pid_cfg)
        runtime.seed = int(args_ns.seed)

    def infer(
        self,
        image: Image.Image,
        prompt: str,
        args_ns: SimpleNamespace,
        face_cropper: FaceCropper,
        *,
        logger: GradioInferLogger | None = None,
    ) -> tuple[str, str, str]:
        if image is None:
            raise gr.Error("请上传输入图。")
        if not prompt or not prompt.strip():
            raise gr.Error("请输入提示词。")

        timings: list[tuple[str, float]] = []
        t_total = time.perf_counter()
        input_rgb = ensure_rgb(image)
        ref_img: Image.Image | None = None
        prep_info = ""
        pid_img: Image.Image | None = None
        output_path = ""
        info = ""

        try:
            t0 = time.perf_counter()
            self.ensure_runtime(args_ns)
            runtime = self.runtime
            if runtime is None:
                raise gr.Error("模型未加载。")
            timings.append(("确保模型", time.perf_counter() - t0))

            args_ns.seed = resolve_seed(args_ns.seed)

            try:
                t0 = time.perf_counter()
                ref_img, prep_info = face_cropper.preprocess(input_rgb)
                timings.append(("人脸裁剪", time.perf_counter() - t0))
            except RuntimeError as exc:
                raise gr.Error(str(exc)) from exc

            crop_w, crop_h = ref_img.size
            t0 = time.perf_counter()
            ref_img = resize_reference_like_diffsynth(ref_img)
            timings.append(("DiffSynth resize", time.perf_counter() - t0))
            ref_w, ref_h = ref_img.size
            prep_info = f"{prep_info} | DiffSynth resize: {crop_w}x{crop_h} -> {ref_w}x{ref_h}"

            with self.lock:
                t0 = time.perf_counter()
                self._update_runtime_resolution(runtime, args_ns)
                if runtime.staged_vram:
                    set_pid_device(runtime.pid_model, "cpu")
                    timings.append(("准备推理", _elapsed_gpu(t0)))
                else:
                    timings.append(("准备推理", time.perf_counter() - t0))

                gen = torch.Generator(device=runtime.device).manual_seed(int(args_ns.seed))
                gen_kwargs = dict(
                    prompt=prompt.strip(),
                    image=ref_img,
                    height=runtime.h_ldm,
                    width=runtime.w_ldm,
                    num_inference_steps=runtime.ldm_steps,
                    guidance_scale=runtime.guidance,
                    num_images_per_prompt=1,
                    output_type="latent",
                    generator=gen,
                )
                gen_kwargs.update(runtime.pipe_cfg.extra_generate_kwargs)

                with torch.inference_mode():
                    t0 = time.perf_counter()
                    raw = runtime.pipeline(**gen_kwargs)
                    timings.append(("LDM 推理", _elapsed_gpu(t0)))

                    t0 = time.perf_counter()
                    latent = runtime.extract_latent(
                        runtime.pipeline, raw, runtime.pipe_cfg, runtime.h_ldm, runtime.w_ldm
                    )
                    sigma = float(runtime.pipeline.scheduler.sigmas[-1].item())
                    del raw
                    timings.append(("提取 latent", _elapsed_gpu(t0)))

                    if runtime.staged_vram:
                        t0 = time.perf_counter()
                        set_pid_device(runtime.pid_model, runtime.device)
                        timings.append(("VRAM 切换→PiD", _elapsed_gpu(t0)))

                    t0 = time.perf_counter()
                    pid_img, _ = decode_with_pid(
                        runtime,
                        latent,
                        sigma,
                        prompt.strip(),
                        int(args_ns.seed),
                    )
                    del latent
                    timings.append(("PiD 解码", _elapsed_gpu(t0)))

            t0 = time.perf_counter()
            output_path = save_gradio_output_image(pid_img)
            timings.append(("Gradio JPEG 输出", time.perf_counter() - t0))

            total_sec = time.perf_counter() - t_total
            timing_text = format_infer_timings(timings, total_sec)
            print(f"[timing] {timing_text.replace(chr(10), ' | ')}", flush=True)

            h_out, w_out = parse_resolution(args_ns.resolution)
            info = f"Seed: {args_ns.seed}\n{prep_info}\n{format_resolution_info(w_out, h_out, args_ns.scale)}"

            if logger is not None:
                logger.submit_run(
                    prompt=prompt,
                    args_ns=args_ns,
                    input_image=input_rgb.copy(),
                    preprocessed_image=ref_img.copy() if ref_img is not None else None,
                    output_image=pid_img.copy() if pid_img is not None else None,
                    prep_info=prep_info,
                    resolution_info=info,
                    timings=timings,
                    total_sec=total_sec,
                    status="ok",
                )

            return output_path, info, timing_text
        except Exception as exc:
            total_sec = time.perf_counter() - t_total
            if logger is not None:
                h_out, w_out = parse_resolution(args_ns.resolution)
                fail_info = info or prep_info or format_resolution_info(w_out, h_out, args_ns.scale)
                logger.submit_run(
                    prompt=prompt,
                    args_ns=args_ns,
                    input_image=input_rgb.copy(),
                    preprocessed_image=ref_img.copy() if ref_img is not None else None,
                    output_image=pid_img.copy() if pid_img is not None else None,
                    prep_info=prep_info,
                    resolution_info=fail_info,
                    timings=timings,
                    total_sec=total_sec,
                    status="error",
                    error=str(exc),
                )
            raise


def resolve_lora_path(preset: str, custom_path: str) -> str:
    if custom_path and str(custom_path).strip():
        return str(custom_path).strip()
    return LORA_PRESETS.get(preset, "")


def resolve_backbone_model_id(preset: str, custom_path: str) -> str:
    if custom_path and str(custom_path).strip():
        return str(custom_path).strip()
    return BACKBONE_MODEL_PRESETS.get(preset, DEFAULT_BACKBONE_MODEL)


def build_args_from_ui(
    pid_root,
    backbone,
    backbone_preset,
    backbone_model_id,
    pid_ckpt_type,
    lora_preset,
    lora_path,
    lora_scale,
    no_lora,
    width,
    height,
    seed,
    low_vram,
    local_files_only,
    *,
    ldm_steps: int = DEFAULT_LDM_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    pid_steps: int = DEFAULT_PID_STEPS,
    pid_cfg: float = DEFAULT_PID_CFG,
):
    lora = "" if no_lora else resolve_lora_path(lora_preset, lora_path)
    model_id = resolve_backbone_model_id(backbone_preset, backbone_model_id)
    return make_args_namespace(
        pid_root=pid_root,
        backbone=backbone,
        backbone_model_id=model_id,
        pid_ckpt_type=pid_ckpt_type,
        lora_path=lora,
        lora_scale=lora_scale,
        scale=FIXED_SCALE,
        width=width,
        height=height,
        seed=seed,
        ldm_steps=ldm_steps,
        guidance_scale=guidance_scale,
        pid_steps=pid_steps,
        pid_cfg=pid_cfg,
        low_vram=low_vram,
        local_files_only=local_files_only,
    )


def main():
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.cuda.set_device(cli.gpu)
    pool = GradioModelPool(cli)
    face_cropper = FaceCropper(backend="yolo")
    run_logger = GradioInferLogger(cli.log_dir, enabled=not cli.no_log)
    if run_logger.enabled:
        print(f"Run logging enabled: {run_logger.log_dir}", flush=True)
        print("  永久记录: input/preprocessed/output + meta.json", flush=True)
    display_dir = gradio_display_dir()
    display_dir.mkdir(parents=True, exist_ok=True)
    print(f"Gradio display dir (temp): {display_dir.resolve()}", flush=True)
    print("Preloading PiD + LDM models, please wait ...")
    initial_status = pool.preload()
    print("Preloading YOLOv8n-face detector ...")
    face_cropper.warmup()
    initial_status = f"{initial_status}\n人脸检测: YOLOv8n-face"

    def on_resolution_change(width, height):
        return format_resolution_info(width, height, FIXED_SCALE)

    def on_apply_model(*inputs):
        args_ns = build_args_from_ui(
            *inputs,
            ldm_steps=cli.ldm_steps,
            guidance_scale=cli.guidance_scale,
            pid_steps=cli.pid_steps,
            pid_cfg=cli.pid_cfg,
        )
        try:
            status = pool.ensure_runtime(args_ns, force=False)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc
        width_val, height_val = inputs[9], inputs[10]
        return status, format_resolution_info(width_val, height_val, FIXED_SCALE)

    def on_pid_ckpt_change(pid_ckpt_type):
        width_val, height_val = resolution_for_pid_ckpt(pid_ckpt_type)
        return width_val, height_val, format_resolution_info(width_val, height_val, FIXED_SCALE)

    def on_generate(image, prompt, *model_input_values):
        args_ns = build_args_from_ui(
            *model_input_values,
            ldm_steps=cli.ldm_steps,
            guidance_scale=cli.guidance_scale,
            pid_steps=cli.pid_steps,
            pid_cfg=cli.pid_cfg,
        )
        try:
            return pool.infer(image, prompt, args_ns, face_cropper, logger=run_logger)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

    with gr.Blocks(title="DreamFace PiD Gradio") as demo:
        gr.Markdown("# DreamFace FLUX.2 + PiD 单图推理")
        gr.Markdown(
            "启动时预加载：PiD 2kto4k + FLUX.2-klein-9B + 默认 LoRA + YOLOv8n-face。"
            "输入流程：YOLO 人脸裁剪 → DiffSynth resize。"
            "LDM→PiD **固定 scale=4**。"
            "默认输出：2k=2304×2600，2kto4k=3584×4608。"
            "界面显示图为 `/tmp` 临时缓存；完整输入/输出与参数保存在日志目录。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="输入图", type="pil", sources=["upload", "clipboard"])
                prompt = gr.Textbox(label="提示词", lines=4, placeholder="描述期望的编辑效果…")
                seed = gr.Number(label="Seed（-1=每次随机）", value=cli.seed, precision=0)

                with gr.Accordion("输出尺寸", open=True):
                    with gr.Row():
                        width = gr.Number(label="Width", value=cli.width, precision=0)
                        height = gr.Number(label="Height", value=cli.height, precision=0)
                    gr.Markdown(f"**Scale (LDM→PiD)：固定 {FIXED_SCALE}×**")
                    resolution_info = gr.Textbox(
                        label="分辨率",
                        value=format_resolution_info(cli.width, cli.height, FIXED_SCALE),
                        interactive=False,
                        lines=3,
                    )

                with gr.Accordion("模型配置", open=False):
                    pid_root = gr.Textbox(label="PiD Root", value=cli.pid_root)
                    backbone = gr.Dropdown(label="Backbone", choices=BACKBONE_CHOICES, value=cli.backbone)
                    backbone_preset = gr.Dropdown(
                        label="基模型预设",
                        choices=list(BACKBONE_MODEL_PRESETS.keys()),
                        value="FLUX.2-klein-9B (local)",
                    )
                    backbone_model_id = gr.Textbox(
                        label="基模型路径 / HF id（可覆盖预设）",
                        value=cli.backbone_model_id,
                    )
                    pid_ckpt_type = gr.Dropdown(
                        label="PiD 权重",
                        choices=PID_CKPT_CHOICES,
                        value=PRELOAD_PID_CKPT,
                    )
                    lora_preset = gr.Dropdown(
                        label="LoRA 预设",
                        choices=list(LORA_PRESETS.keys()),
                        value="DreamFace v2.1 (diffusers)",
                    )
                    lora_path = gr.Textbox(label="LoRA 路径（可覆盖预设）", value=cli.lora)
                    lora_scale = gr.Slider(
                        label="LoRA Scale", minimum=0.0, maximum=2.0, value=cli.lora_scale, step=0.05
                    )
                    no_lora = gr.Checkbox(label="不使用 LoRA", value=False)
                    low_vram = gr.Checkbox(label="Low VRAM（CPU offload + staged）", value=cli.low_vram)
                    local_files_only = gr.Checkbox(label="仅本地权重", value=cli.local_files_only)
                    model_status = gr.Textbox(
                        label="模型状态", value=initial_status, interactive=False, lines=8
                    )
                    apply_model_btn = gr.Button("加载 / 切换模型", variant="secondary")

                generate_btn = gr.Button("生成", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="输出图", type="filepath", format="jpeg")
                timing_info = gr.Textbox(
                    label="耗时统计",
                    value="",
                    interactive=False,
                    lines=10,
                )

        model_inputs = [
            pid_root,
            backbone,
            backbone_preset,
            backbone_model_id,
            pid_ckpt_type,
            lora_preset,
            lora_path,
            lora_scale,
            no_lora,
            width,
            height,
            seed,
            low_vram,
            local_files_only,
        ]

        for component in (width, height):
            component.change(on_resolution_change, inputs=[width, height], outputs=resolution_info)

        pid_ckpt_type.change(
            fn=on_pid_ckpt_change,
            inputs=[pid_ckpt_type],
            outputs=[width, height, resolution_info],
        )

        apply_model_btn.click(fn=on_apply_model, inputs=model_inputs, outputs=[model_status, resolution_info])

        generate_btn.click(
            fn=on_generate,
            inputs=[input_image, prompt, *model_inputs],
            outputs=[output_image, resolution_info, timing_info],
        )

    demo.queue(max_size=4).launch(
        server_name=cli.server_name,
        server_port=cli.server_port,
        share=cli.share,
        allowed_paths=gradio_allowed_paths(cli.log_dir, display_dir),
    )


if __name__ == "__main__":
    main()
