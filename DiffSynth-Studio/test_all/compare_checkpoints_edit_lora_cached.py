#!/usr/bin/env python3
"""Compare many native edit_image LoRA checkpoints with cached edit latents.

Training consistency:
  train_flux2_klein_base_4b_deblur_multi_dataset_all_edit_lora.sh trains FLUX.2
  with metadata fields:
    image: HR target
    edit_image: aligned LR condition
    edit_image_auto_resize: false

This script mirrors that inference path:
  LR image -> VAE edit_latents/edit_image_ids -> concat with noisy generation
  latents inside FLUX.2 DiT.

Optimization:
  - Load FLUX.2 base/text/VAE once per distributed rank.
  - Read LR images once.
  - VAE encode edit_image once per rank and cache edit_latents/edit_image_ids.
  - Loop over LoRA checkpoints by hotloading only LoRA weights.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
INFERENCE_DIR = REPO_ROOT / "inference"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(INFERENCE_DIR))

from accelerate import Accelerator
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from infer_flux2_base_edit_deblur_dataset import (
    DEFAULT_BASE_MODEL,
    DEFAULT_METADATA,
    load_pipeline,
    load_records,
    validate_base_model,
)


DEFAULT_OUTPUT_BASE = REPO_ROOT / "test_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cached native edit_image LoRA inference for all checkpoints in a directory."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint-pattern", default="*.safetensors")
    parser.add_argument(
        "--checkpoint-names",
        default=None,
        help="Optional comma-separated checkpoint stems/files, e.g. step-2000,step-4000.safetensors.",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--embedded-guidance", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cache-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Store cached edit latents on CPU to save VRAM, or CUDA for speed.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def natural_step_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    for prefix in ("step-", "epoch-"):
        if stem.startswith(prefix):
            value = stem[len(prefix):]
            if value.isdigit():
                return int(value), stem
    return 10**18, stem


def find_checkpoints(checkpoint_dir: Path, pattern: str, checkpoint_names: str | None) -> list[Path]:
    if checkpoint_names:
        checkpoints = []
        for raw_name in checkpoint_names.split(","):
            name = raw_name.strip()
            if not name:
                continue
            path = Path(name)
            if not path.is_absolute():
                if path.suffix != ".safetensors":
                    path = path.with_suffix(".safetensors")
                path = checkpoint_dir / path
            require_file(path, "LoRA checkpoint")
            checkpoints.append(path)
        return checkpoints
    checkpoints = sorted(checkpoint_dir.glob(pattern), key=natural_step_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints matching {pattern} in {checkpoint_dir}")
    return checkpoints


def deterministic_seed(base_seed: int, relative_path: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{relative_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def get_edit_embedder_unit(pipe):
    for unit in pipe.units:
        output_params = set(getattr(unit, "output_params", ()) or ())
        if {"edit_latents", "edit_image_ids"}.issubset(output_params):
            return unit
    raise RuntimeError("Could not find Flux2Unit_EditImageEmbedder in pipe.units")


def reflect_pad_to_multiple(image: Image.Image, multiple: int = 16) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    pad_width = (-width) % multiple
    pad_height = (-height) % multiple
    if pad_width == 0 and pad_height == 0:
        return image, (width, height)
    array = np.asarray(image)
    mode = "reflect" if width > 1 and height > 1 else "edge"
    padded = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode=mode)
    return Image.fromarray(padded), (width, height)


@torch.inference_mode()
def cache_record(pipe, edit_embedder, record: dict[str, Any], cache_device: str) -> dict[str, Any]:
    with Image.open(record["_lr_path"]) as opened:
        source = opened.convert("RGB")
    lr_image, original_size = reflect_pad_to_multiple(source)
    width, height = lr_image.size

    edit = edit_embedder.process(pipe, edit_image=lr_image, edit_image_auto_resize=False)
    edit_latents = edit["edit_latents"].detach()
    edit_image_ids = edit["edit_image_ids"].detach()
    if cache_device == "cpu":
        edit_latents = edit_latents.cpu()
        edit_image_ids = edit_image_ids.cpu()

    return {
        "relative_path": record["_relative_path"],
        "prompt": record["prompt"],
        "lr_image": lr_image,
        "original_size": original_size,
        "height": height,
        "width": width,
        "seed": deterministic_seed(42, record["_relative_path"]),
        "edit_latents": edit_latents,
        "edit_image_ids": edit_image_ids,
    }


@torch.inference_mode()
def infer_with_cached_edit(pipe, cached: dict[str, Any], output_path: Path, args: argparse.Namespace) -> None:
    edit_latents = cached["edit_latents"]
    edit_image_ids = cached["edit_image_ids"]
    if edit_latents.device != pipe.device:
        edit_latents = edit_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        edit_image_ids = edit_image_ids.to(device=pipe.device)

    height = cached["height"]
    width = cached["width"]
    cfg_scale = args.cfg_scale
    pipe.scheduler.set_timesteps(
        args.num_inference_steps,
        denoising_strength=1.0,
        dynamic_shift_len=height // 16 * width // 16,
    )

    inputs_posi = {
        "prompt": cached["prompt"],
        "kv_cache": None,
        "extra_text_embedding": None,
    }
    inputs_nega = {
        "negative_prompt": args.negative_prompt,
        "kv_cache": None,
        "extra_text_embedding": None,
    }
    inputs_shared = {
        "cfg_scale": cfg_scale,
        "embedded_guidance": args.embedded_guidance,
        "input_image": None,
        "denoising_strength": 1.0,
        "edit_image": None,
        "edit_image_auto_resize": False,
        "edit_latents": edit_latents,
        "edit_image_ids": edit_image_ids,
        "height": height,
        "width": width,
        "seed": cached["seed"],
        "rand_device": "cpu",
        "initial_noise": None,
        "num_inference_steps": args.num_inference_steps,
        "positive_only_lora": None,
        "negative_only_lora": None,
        "inpaint_mask": None,
        "inpaint_blur_size": None,
        "inpaint_blur_sigma": None,
    }
    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate((lambda values: values)(pipe.scheduler.timesteps)):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.cfg_guided_model_fn(
            pipe.model_fn,
            cfg_scale,
            inputs_shared,
            inputs_posi,
            inputs_nega,
            **models,
            timestep=timestep,
            progress_id=progress_id,
        )
        inputs_shared["latents"] = pipe.step(
            pipe.scheduler,
            progress_id=progress_id,
            noise_pred=noise_pred,
            **inputs_shared,
        )

    pipe.load_models_to_device(["vae"])
    latents = rearrange(
        inputs_shared["latents"],
        "B (H W) C -> B C H W",
        H=height // 16,
        W=width // 16,
    )
    image = pipe.vae.decode(latents)
    image = pipe.vae_output_to_image(image)
    image = image.crop((0, 0, *cached["original_size"]))
    pipe.load_models_to_device([])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    image.save(temporary_path, format="PNG")
    temporary_path.replace(output_path)


def load_lora_hot(pipe, checkpoint: Path, lora_scale: float) -> None:
    pipe.clear_lora(verbose=0)
    try:
        pipe.load_lora(
            pipe.dit,
            lora_config=str(checkpoint),
            alpha=lora_scale,
            hotload=True,
            verbose=0,
        )
    except ValueError as error:
        raise RuntimeError(
            "LoRA hotload is required for cached multi-checkpoint comparison, "
            "because normal LoRA loading fuses weights into the base DiT and "
            "cannot be safely swapped without reloading the base model. "
            "Use the original one-checkpoint-per-process script if hotload is "
            "not supported in this environment."
        ) from error
    pipe.dit.eval()


def main() -> None:
    args = parse_args()
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    output_base = args.output_base.expanduser().resolve()
    base = args.base_model.expanduser().resolve()
    require_dir(checkpoint_dir, "checkpoint directory")
    text_encoder_files = validate_base_model(base)
    checkpoints = find_checkpoints(checkpoint_dir, args.checkpoint_pattern, args.checkpoint_names)
    records = load_records(metadata, args.limit)

    output_root = output_base / checkpoint_dir.name
    output_root.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()
    if accelerator.device.type != "cuda":
        raise RuntimeError("CUDA is required for FLUX.2 inference")

    if accelerator.is_main_process:
        print("==========================================")
        print("  Cached native edit_image LoRA comparison")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Checkpoints: {len(checkpoints)}")
        print(f"  Metadata: {metadata}")
        print(f"  Records: {len(records)}")
        print(f"  Output root: {output_root}")
        print(f"  Processes: {accelerator.num_processes}")
        print(f"  Steps: {args.num_inference_steps}")
        print(f"  cfg_scale: {args.cfg_scale}")
        print(f"  embedded_guidance: {args.embedded_guidance}")
        print(f"  lora_scale: {args.lora_scale}")
        print(f"  cache_device: {args.cache_device}")
        print("==========================================")
    accelerator.wait_for_everyone()

    pipe = load_pipeline(accelerator, base, text_encoder_files)
    pipe.dit = pipe.enable_lora_hot_loading(pipe.dit)
    pipe.vram_management_enabled = pipe.check_vram_management_state()
    edit_embedder = get_edit_embedder_unit(pipe)
    accelerator.wait_for_everyone()

    shard = records[accelerator.process_index :: accelerator.num_processes]
    cache_iter = tqdm(
        shard,
        desc=f"cache rank{accelerator.process_index}",
        disable=not accelerator.is_main_process,
    )
    cached_records = []
    for record in cache_iter:
        cached = cache_record(pipe, edit_embedder, record, args.cache_device)
        cached["seed"] = deterministic_seed(args.seed, cached["relative_path"])
        cached_records.append(cached)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("edit_latents cache is ready.", flush=True)

    for checkpoint in checkpoints:
        checkpoint_stem = checkpoint.stem
        out_dir = output_root / checkpoint_stem
        if accelerator.is_main_process:
            print(f"\n>>> Testing {checkpoint.name}", flush=True)
        accelerator.wait_for_everyone()

        load_lora_hot(pipe, checkpoint, args.lora_scale)
        iterator = tqdm(
            cached_records,
            desc=f"{checkpoint_stem} rank{accelerator.process_index}",
            disable=not accelerator.is_main_process,
        )
        generated = 0
        skipped = 0
        for cached in iterator:
            output_path = out_dir / cached["relative_path"]
            if output_path.exists() and not args.overwrite:
                skipped += 1
                continue
            infer_with_cached_edit(pipe, cached, output_path, args)
            generated += 1
        print(
            f"rank={accelerator.process_index} checkpoint={checkpoint_stem} "
            f"generated={generated} skipped={skipped}",
            flush=True,
        )
        accelerator.wait_for_everyone()
        pipe.clear_lora(verbose=0)
        if accelerator.is_main_process:
            output_count = sum(1 for path in out_dir.rglob("*.png") if path.is_file())
            print(f">>> {checkpoint_stem} done: {output_count} images", flush=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("\n==========================================")
        print(f"  All done! Results: {output_root}")
        print("==========================================")


if __name__ == "__main__":
    main()
