#!/usr/bin/env python3
"""CSV batch inference tuned for 48GB L20 GPUs at 4K output (FLUX.2 Klein + PiD).

Strategy vs ../csv_pid_batch_infer.py — phase-swapped execution keeps only what
each phase needs on the GPU, so prompts can change freely at runtime (no
pre-encoding of the whole CSV):

  phase 1  prompt encode : text encoder swapped onto GPU per new prompt
  phase 2  LDM denoise   : FLUX transformer + VAE on GPU, PiD on CPU
  phase 3  PiD decode    : PiD on GPU, FLUX transformer swapped to CPU

At 4K (3584x4608, LDM 896x1152) each phase alone nearly fills 48GB, so all
three swaps default to ON. The PCIe traffic costs a few seconds per sample,
which is small against minute-scale 4K compute on L20.

Optional acceleration (Ada/L20-friendly, all OFF by default):
  --te-quant 4bit              bitsandbytes NF4 text encoder (resident, no swap)
  --quant-transformer fp8dq    torchao FP8 matmuls (needs SM89+; L20 is SM89)
  --sage-attention             SageAttention INT8 SDPA patch (LDM + PiD)
  --compile-pid / --compile-ldm  torch.compile

Per-sample timing and per-phase peak VRAM are printed so the swap/quant combo
can be tuned on real hardware.
"""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

import csv_pid_batch_infer as base  # noqa: E402

DEFAULT_OUTPUT_ROOT = str(Path(base.DEFAULT_OUTPUT_ROOT).parent / "csv-pidout-l20-4k")


def parse_args():
    p = argparse.ArgumentParser(description="CSV batch FLUX.2 Klein + PiD decode (L20 48GB, 4K)")
    p.add_argument("--csv", default=base.DEFAULT_CSV)
    p.add_argument("--output", default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--pid-root", default=base.DEFAULT_PID_ROOT, help="PiD repo root with checkpoints/")
    p.add_argument("--backbone", default="flux2-klein-9b", choices=[
        "flux2-klein-4b", "flux2-klein-9b", "flux2",
    ])
    p.add_argument("--backbone-model-id", default=None, help="Local diffusers dir or HF model id")
    p.add_argument("--pid-ckpt-type", default="2kto4k", choices=["2k", "2kto4k"],
                   help="4K output needs the 2kto4k checkpoint")
    p.add_argument("--checkpoint-path", default=None, help="Override PiD checkpoint .pth")
    p.add_argument("--experiment", default=None, help="Override PiD experiment name")
    p.add_argument("--resolution", default="3584,4608", help="Final output W,H in pixels (4K default)")
    p.add_argument("--scale", type=int, default=4, help="PiD upscale factor (LDM size = output/scale)")
    p.add_argument("--ldm-steps", type=int, default=None)
    p.add_argument("--guidance-scale", type=float, default=None)
    p.add_argument("--pid-steps", type=int, default=4)
    p.add_argument("--pid-cfg", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora", default=os.environ.get("LORA", base.DEFAULT_LORA),
                   help="Diffusers LoRA folder or .safetensors")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--gpus", default="0")
    p.add_argument("--num", type=int, default=None)
    p.add_argument("--skip", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--local-files-only", action="store_true", help="Never hit HuggingFace Hub")
    p.add_argument("--no-lora", action="store_true")
    p.add_argument("--save-vae-baseline", action="store_true", help="Also save native VAE decode")

    # --- phase-swap knobs (all ON by default; required to fit 4K in 48GB) ---
    p.add_argument("--swap-text-encoder", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep TE on CPU; move to GPU only while encoding a new prompt")
    p.add_argument("--swap-transformer", action=argparse.BooleanOptionalAction, default=True,
                   help="Move FLUX transformer to CPU during PiD decode")
    p.add_argument("--swap-pid", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep PiD decoder + Gemma on CPU during LDM denoise")
    p.add_argument("--prompt-cache-size", type=int, default=64,
                   help="LRU cache of prompt embeddings; saves TE swaps on repeated prompts (0=off)")

    # --- quantization / acceleration ---
    p.add_argument("--te-quant", default="none", choices=["none", "4bit", "8bit"],
                   help="bitsandbytes-quantize the text encoder; it then stays resident "
                        "(implies --no-swap-text-encoder). Experimental.")
    p.add_argument("--quant-transformer", default="none", choices=["none", "fp8dq", "fp8wo", "int8wo"],
                   help="torchao-quantize the FLUX transformer after LoRA fuse "
                        "(fp8* needs SM89+, ideal on L20)")
    p.add_argument("--sage-attention", action="store_true",
                   help="Patch SDPA with SageAttention (INT8) for mask-free attention calls")
    p.add_argument("--compile-pid", action="store_true",
                   help="torch.compile PiD decoder (first sample warms up, then faster)")
    p.add_argument("--compile-ldm", action="store_true",
                   help="torch.compile FLUX transformer")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quantization / attention helpers
# ---------------------------------------------------------------------------


def build_te_quant_config(mode: str):
    if mode == "none":
        return None
    from diffusers.quantizers import PipelineQuantizationConfig

    if mode == "4bit":
        backend = "bitsandbytes_4bit"
        quant_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
    else:
        backend = "bitsandbytes_8bit"
        quant_kwargs = {"load_in_8bit": True}
    return PipelineQuantizationConfig(
        quant_backend=backend,
        quant_kwargs=quant_kwargs,
        components_to_quantize=["text_encoder"],
    )


def quantize_transformer_torchao(transformer, mode: str) -> None:
    if mode == "none":
        return
    from torchao.quantization import quantize_

    def resolve_config():
        # torchao renamed the factory helpers to *Config classes; support both.
        if mode == "fp8dq":
            try:
                from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
                return Float8DynamicActivationFloat8WeightConfig()
            except ImportError:
                from torchao.quantization import float8_dynamic_activation_float8_weight
                return float8_dynamic_activation_float8_weight()
        if mode == "fp8wo":
            try:
                from torchao.quantization import Float8WeightOnlyConfig
                return Float8WeightOnlyConfig()
            except ImportError:
                from torchao.quantization import float8_weight_only
                return float8_weight_only()
        if mode == "int8wo":
            try:
                from torchao.quantization import Int8WeightOnlyConfig
                return Int8WeightOnlyConfig()
            except ImportError:
                from torchao.quantization import int8_weight_only
                return int8_weight_only()
        raise ValueError(f"Unknown transformer quant mode: {mode}")

    print(f"Quantizing FLUX transformer with torchao ({mode}) ...")
    quantize_(transformer, resolve_config())


def enable_sage_attention() -> bool:
    """Patch torch SDPA with SageAttention for mask-free bf16/fp16 calls.

    Must run BEFORE the PiD network module is imported: pixeldit_official.py
    binds `from torch.nn.functional import scaled_dot_product_attention` at
    import time, so we also rebind that module global if already imported.
    """
    try:
        from sageattention import sageattn
    except ImportError:
        print("[warn] sageattention not installed; ignoring --sage-attention")
        return False

    orig = torch.nn.functional.scaled_dot_product_attention

    def patched(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
        if (
            attn_mask is None
            and dropout_p == 0.0
            and not kwargs.get("enable_gqa", False)
            and query.dtype in (torch.float16, torch.bfloat16)
            and query.shape[-1] in (64, 96, 128)
        ):
            try:
                return sageattn(query, key, value, tensor_layout="HND", is_causal=is_causal)
            except Exception:
                pass  # head-dim/layout not supported: fall through to original SDPA
        return orig(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
                    is_causal=is_causal, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = patched
    pixeldit_mod = sys.modules.get("pid._src.networks.pixeldit_official")
    if pixeldit_mod is not None and hasattr(pixeldit_mod, "scaled_dot_product_attention"):
        pixeldit_mod.scaled_dot_product_attention = patched
    print("SageAttention SDPA patch enabled.")
    return True


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------


def force_execution_device(pipeline, device: str) -> None:
    """Pin diffusers' `_execution_device` to the compute GPU.

    The stock property returns the device of the first component module it
    finds; with the text encoder parked on CPU it can resolve to `cpu` and
    break generator/latent device checks. We swap in an instance-scoped
    subclass so only this pipeline is affected.
    """
    forced = torch.device(device)
    cls = type(pipeline)
    patched = type(
        cls.__name__ + "_L20Pinned",
        (cls,),
        {"_execution_device": property(lambda self: forced)},
    )
    pipeline.__class__ = patched


def load_pipeline_l20(args, device: str):
    import importlib

    from pid._src.inference.pipeline_registry import get_config

    cfg = get_config(args.backbone)
    model_id = base.resolve_backbone_model_id(args.backbone, args.backbone_model_id)
    if not model_id:
        if args.local_files_only:
            raise FileNotFoundError(
                f"No local diffusers backbone found under {base.DEFAULT_BACKBONE_LOCAL}. "
                "Pass --backbone-model-id or unset --local-files-only."
            )
        model_id = cfg.default_model_id

    module_path, cls_name = cfg.pipeline_class.rsplit(".", 1)
    pipeline_cls = getattr(importlib.import_module(module_path), cls_name)

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    load_kwargs = {"torch_dtype": torch.bfloat16, "token": token}
    if Path(model_id).is_dir() or args.local_files_only:
        load_kwargs["local_files_only"] = True
    quant_config = build_te_quant_config(args.te_quant)
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    print(f"Loading {cfg.pipeline_class} from {model_id} (te_quant={args.te_quant}) ...")
    pipeline = pipeline_cls.from_pretrained(model_id, **load_kwargs)
    base.configure_pipeline_low_vram(pipeline)
    pipeline = pipeline.to(device)
    force_execution_device(pipeline, device)
    print(f"Pipeline loaded on {device}.")
    return pipeline, cfg


def build_runtime(args, device: str):
    pid_root = base.ensure_pid_import(Path(args.pid_root))

    if args.sage_attention:
        enable_sage_attention()

    from pid._src.inference.checkpoint_registry import get_pid_checkpoint
    from pid._src.inference.decoder import load_our_decoder
    from pid._src.inference.pipeline_registry import (
        decode_with_pipeline_vae,
        extract_latent,
        get_config,
    )

    ckpt = get_pid_checkpoint(args.backbone, args.pid_ckpt_type)
    experiment = args.experiment or ckpt.experiment
    checkpoint_path = args.checkpoint_path or ckpt.checkpoint_path
    if not Path(checkpoint_path).is_absolute():
        checkpoint_path = str(Path(args.pid_root) / checkpoint_path)
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(
            f"PiD checkpoint missing: {checkpoint_path}\n"
            f"Run: hf download nvidia/PiD --local-dir {pid_root} --include 'checkpoints/*'"
        )

    # PiD config loader expects repo-relative path and cwd at PiD root.
    config_file = "pid/_src/configs/pid/config.py"
    os.chdir(pid_root)

    text_encoder_path = base.patch_pid_text_encoder_local(base.DEFAULT_MODELS_ROOT)
    if text_encoder_path:
        print(f"PiD text encoder (local): {text_encoder_path}")
    elif args.local_files_only:
        raise FileNotFoundError(
            f"PiD decoder needs Gemma text encoder at {base.DEFAULT_PID_TEXT_ENCODER}. "
            "Run setup_pid_env.sh or set PID_TEXT_ENCODER_PATH."
        )

    pipe_cfg = get_config(args.backbone)
    ldm_steps = args.ldm_steps or pipe_cfg.default_num_inference_steps
    guidance = args.guidance_scale if args.guidance_scale is not None else pipe_cfg.default_guidance_scale

    h_out, w_out = base.parse_resolution(args.resolution)
    h_ldm, w_ldm = h_out // args.scale, w_out // args.scale

    decoder_args = SimpleNamespace(
        experiment=experiment,
        checkpoint_path=checkpoint_path,
        config_file=config_file,
        load_ema_to_reg=False,
        compile=args.compile_pid,
        extra_experiment_opts=[],
    )
    pid_model = load_our_decoder(decoder_args, [], is_rank0=True)
    if args.swap_pid:
        # PiD lands on GPU at instantiation; park it on CPU until decode.
        base.set_pid_device(pid_model, "cpu")
        base.sync_cuda()

    pipeline, _ = load_pipeline_l20(args, device)

    lora_path = base.resolve_lora_path(args.lora, args.no_lora)
    if lora_path:
        base.load_lora_weights_local(pipeline, lora_path, args.lora_scale)

    # Quantize after LoRA fuse so the fused weights are what gets quantized.
    quantize_transformer_torchao(pipeline.transformer, args.quant_transformer)

    if args.compile_ldm:
        print("Compiling FLUX transformer with torch.compile ...")
        pipeline.transformer = torch.compile(pipeline.transformer, mode="reduce-overhead")

    swap_te = args.swap_text_encoder and args.te_quant == "none"
    if args.te_quant != "none" and args.swap_text_encoder:
        print("[info] quantized text encoder stays resident; ignoring --swap-text-encoder")
    if swap_te:
        pipeline.text_encoder.to("cpu")
        base.sync_cuda()

    runtime = SimpleNamespace(
        pipeline=pipeline,
        pipe_cfg=pipe_cfg,
        pid_model=pid_model,
        device=device,
        h_out=h_out,
        w_out=w_out,
        h_ldm=h_ldm,
        w_ldm=w_ldm,
        ldm_steps=ldm_steps,
        guidance=guidance,
        scale=args.scale,
        pid_steps=args.pid_steps,
        pid_cfg=args.pid_cfg,
        seed=args.seed,
        save_vae_baseline=args.save_vae_baseline,
        swap_te=swap_te,
        swap_transformer=args.swap_transformer,
        swap_pid=args.swap_pid,
        decode_with_pipeline_vae=decode_with_pipeline_vae,
        extract_latent=extract_latent,
        prompt_cache={},
        prompt_cache_order=[],
        prompt_cache_size=max(0, args.prompt_cache_size),
        max_seq_len=pipe_cfg.extra_generate_kwargs.get("max_sequence_length", 512),
    )
    return runtime


# ---------------------------------------------------------------------------
# Per-sample phases
# ---------------------------------------------------------------------------


def needs_cfg(runtime) -> bool:
    is_distilled = bool(dict(runtime.pipeline.config).get("is_distilled", False))
    return runtime.guidance > 1.0 and not is_distilled


def encode_prompt_cached(runtime, prompt: str):
    """Encode one prompt, swapping the TE on/off the GPU around the forward.

    Prompts vary at runtime, so encoding is per-sample; the small LRU cache
    only short-circuits exact repeats within a worker.
    """
    cached = runtime.prompt_cache.get(prompt)
    if cached is None:
        te = runtime.pipeline.text_encoder
        if runtime.swap_te:
            te.to(runtime.device)
        with torch.inference_mode():
            embeds, _ = runtime.pipeline.encode_prompt(
                prompt=prompt, device=runtime.device, max_sequence_length=runtime.max_seq_len
            )
            neg = None
            if needs_cfg(runtime):
                neg, _ = runtime.pipeline.encode_prompt(
                    prompt="", device=runtime.device, max_sequence_length=runtime.max_seq_len
                )
        if runtime.swap_te:
            te.to("cpu")
            base.sync_cuda()
        cached = (embeds.to("cpu"), neg.to("cpu") if neg is not None else None)
        if runtime.prompt_cache_size > 0:
            runtime.prompt_cache[prompt] = cached
            runtime.prompt_cache_order.append(prompt)
            if len(runtime.prompt_cache_order) > runtime.prompt_cache_size:
                evicted = runtime.prompt_cache_order.pop(0)
                runtime.prompt_cache.pop(evicted, None)

    embeds_cpu, neg_cpu = cached
    embeds = embeds_cpu.to(runtime.device)
    neg = neg_cpu.to(runtime.device) if neg_cpu is not None else None
    return embeds, neg


def run_one(row: dict[str, str], runtime, output_root: Path, force: bool) -> dict[str, str]:
    uid = row["id"] or hashlib.md5(row["home_img"].encode()).hexdigest()[:12]
    out_dir = output_root / "outputs" / uid
    result_path = out_dir / "result.webp"

    if result_path.exists() and not force:
        return {**row, "flux_result": str(result_path), "status": "skipped", "error": ""}

    try:
        ref_img, _ = base.load_image(row["home_img"], output_root / "inputs", uid)
        ref_img = base.resize_reference_like_diffsynth(ref_img)

        dev = runtime.device
        seed = runtime.seed + hash(uid) % 100000

        # Phase 0: make sure PiD is parked on CPU before the heavy LDM phase.
        if runtime.swap_pid:
            base.set_pid_device(runtime.pid_model, "cpu")
            base.sync_cuda()
        torch.cuda.reset_peak_memory_stats()

        # Phase 1: prompt encode (TE swapped in/out inside).
        t0 = time.time()
        prompt_embeds, negative_embeds = encode_prompt_cached(runtime, row["replaced_prompt"])
        t_encode = time.time() - t0

        # Phase 2: LDM denoise with transformer + VAE on GPU.
        if runtime.swap_transformer:
            runtime.pipeline.transformer.to(dev)

        gen = torch.Generator(device=dev).manual_seed(seed)
        gen_kwargs = dict(
            prompt=None,
            prompt_embeds=prompt_embeds,
            image=ref_img,
            height=runtime.h_ldm,
            width=runtime.w_ldm,
            num_inference_steps=runtime.ldm_steps,
            guidance_scale=runtime.guidance,
            num_images_per_prompt=1,
            output_type="latent",
            generator=gen,
        )
        if negative_embeds is not None:
            gen_kwargs["negative_prompt_embeds"] = negative_embeds
        gen_kwargs.update(runtime.pipe_cfg.extra_generate_kwargs)

        t0 = time.time()
        with torch.inference_mode():
            raw = runtime.pipeline(**gen_kwargs)
            latent = runtime.extract_latent(
                runtime.pipeline, raw, runtime.pipe_cfg, runtime.h_ldm, runtime.w_ldm
            )
            sigma = float(runtime.pipeline.scheduler.sigmas[-1].item())
            del raw
        t_ldm = time.time() - t0
        peak_ldm = torch.cuda.max_memory_allocated() / 1024**3

        # Phase 3: swap transformer out, PiD in, decode at full resolution.
        t0 = time.time()
        if runtime.swap_transformer:
            runtime.pipeline.transformer.to("cpu")
        base.sync_cuda()
        if runtime.swap_pid:
            base.set_pid_device(runtime.pid_model, dev)
        t_swap = time.time() - t0

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        with torch.inference_mode():
            pid_img, vae_img = base.decode_with_pid(
                runtime, latent, sigma, row["replaced_prompt"], seed
            )
            del latent
        base.sync_cuda()
        t_pid = time.time() - t0
        peak_pid = torch.cuda.max_memory_allocated() / 1024**3

        out_dir.mkdir(parents=True, exist_ok=True)
        ref_img.save(out_dir / "input.webp", format="WEBP", quality=90)
        pid_img.save(result_path, format="WEBP", quality=90)
        if vae_img is not None:
            vae_img.save(out_dir / "result_vae.webp", format="WEBP", quality=90)

        print(
            f"[mem/time] id={uid}: encode {t_encode:.1f}s | "
            f"ldm {t_ldm:.1f}s (peak {peak_ldm:.1f}GB) | swap {t_swap:.1f}s | "
            f"pid {t_pid:.1f}s (peak {peak_pid:.1f}GB)",
            flush=True,
        )
        return {**row, "flux_result": str(result_path), "status": "ok", "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {
            **row,
            "flux_result": "",
            "status": "error",
            "error": str(exc),
            "_traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Main / workers (same structure as the base script)
# ---------------------------------------------------------------------------


def worker_main(gpu_id, worker_idx, worker_total, rows, output_root, args_dict, result_queue):
    args = SimpleNamespace(**args_dict)
    base.apply_resolution_scale_crop(args)
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"
    print(f"[worker {worker_idx + 1}/{worker_total}] GPU {gpu_id}, rows={len(rows)}")
    runtime = build_runtime(args, device)
    for i, row in enumerate(rows, 1):
        print(f"[worker {worker_idx + 1}/{worker_total}] ({i}/{len(rows)}) id={row.get('id')}")
        result = run_one(row, runtime, Path(output_root), args.force)
        base.log_result(result)
        result_queue.put(result)


def main():
    args = parse_args()
    base.apply_resolution_scale_crop(args)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = base.load_csv_rows(args.csv)
    if args.skip:
        rows = rows[args.skip:]
    if args.num is not None:
        rows = rows[: args.num]

    h_out, w_out = base.parse_resolution(args.resolution)
    h_ldm, w_ldm = h_out // args.scale, w_out // args.scale
    print(f"CSV: {args.csv}")
    print(f"Rows: {len(rows)}")
    print(f"Backbone: {args.backbone} | PiD ckpt: {args.pid_ckpt_type}")
    print(f"Output: {output_root}")
    print(f"LDM resolution: {w_ldm}x{h_ldm} -> PiD output: {w_out}x{h_out} (scale={args.scale})")
    print(
        f"Phase swap: te={args.swap_text_encoder} transformer={args.swap_transformer} "
        f"pid={args.swap_pid}"
    )
    print(
        f"Accel: te_quant={args.te_quant} transformer_quant={args.quant_transformer} "
        f"sage={args.sage_attention} compile_pid={args.compile_pid} compile_ldm={args.compile_ldm}"
    )
    alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in alloc_conf:
        print(
            "[warn] PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is strongly "
            "recommended on 48GB GPUs (the PiD decode phase runs close to the limit)."
        )

    gpu_ids = base.parse_gpu_ids(args.gpus)
    if not gpu_ids:
        raise RuntimeError("No GPU configured")

    args_dict = vars(args)
    csv_writer = base.IncrementalCsvWriter(output_root, rows)
    print(f"Incremental CSV: {csv_writer.out_csv} (resume rows: {len(csv_writer.results)})")

    if len(gpu_ids) == 1:
        torch.cuda.set_device(gpu_ids[0])
        runtime = build_runtime(args, f"cuda:{gpu_ids[0]}")
        results = []
        for i, row in enumerate(rows, 1):
            print(f"[{i}/{len(rows)}] id={row.get('id')}", flush=True)
            result = run_one(row, runtime, output_root, args.force)
            base.log_result(result)
            csv_writer.upsert(result)
            results.append(result)
    else:
        row_chunks = [c for c in base.chunk_rows(rows, len(gpu_ids)) if c]
        active = gpu_ids[: len(row_chunks)]
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        procs = []
        for idx, (gpu, chunk) in enumerate(zip(active, row_chunks)):
            p = ctx.Process(
                target=worker_main,
                args=(gpu, idx, len(active), chunk, str(output_root), args_dict, q),
            )
            p.start()
            procs.append(p)
        results = []
        for _ in range(len(rows)):
            result = q.get()
            base.log_result(result)
            csv_writer.upsert(result)
            results.append(result)
        for p in procs:
            p.join()
            if p.exitcode:
                raise RuntimeError(f"worker failed: exit {p.exitcode}")
        order = {r["id"]: i for i, r in enumerate(rows)}
        results.sort(key=lambda r: order.get(r["id"], 10**9))

    ok = sum(1 for r in results if r["status"] == "ok")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    failed = sum(1 for r in results if r["status"] == "error")
    print(f"Done: ok={ok}, skipped={skipped}, error={failed}")
    print(f"Output CSV: {csv_writer.out_csv}")


if __name__ == "__main__":
    main()
