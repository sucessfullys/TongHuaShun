#!/usr/bin/env python3
"""Compare many LRE checkpoints with shared model loading and cached LR latents.

This is an optimized replacement for launching one inference process per
checkpoint. Each distributed rank loads FLUX.2/base/template once, encodes its
own shard of LR images once into LRE initial noise, then loops over checkpoints
by only swapping the Template checkpoint weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
sys.path.insert(0, str(REPO_ROOT))

from accelerate import Accelerator
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline


DEFAULT_BASE_MODEL = Path("/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B")
DEFAULT_TEMPLATE_MODEL = REPO_ROOT / "Template-KleinBase4B-Upscaler"
DEFAULT_METADATA = REPO_ROOT / "test_all" / "测试集合" / "metadata.jsonl"
DEFAULT_OUTPUT_BASE = REPO_ROOT / "test_all"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cached LRE inference for all .safetensors checkpoints in a directory."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--template-model", type=Path, default=DEFAULT_TEMPLATE_MODEL)
    parser.add_argument("--checkpoint-pattern", default="*.safetensors")
    parser.add_argument(
        "--checkpoint-names",
        default=None,
        help="Optional comma-separated checkpoint stems/files, e.g. step-10000,step-20000.safetensors.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--embedded-guidance", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lre-strength", type=float, default=0.8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cache-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Store cached LRE initial noise on CPU to save VRAM, or CUDA for speed.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {description}: {path}")


def validate_models(base: Path, template: Path) -> list[str]:
    text_encoder = sorted((base / "text_encoder").glob("*.safetensors"))
    if not text_encoder:
        raise FileNotFoundError(f"No text encoder weights in {base / 'text_encoder'}")
    require_file(base / "transformer" / "diffusion_pytorch_model.safetensors", "FLUX.2 transformer")
    require_file(base / "vae" / "diffusion_pytorch_model.safetensors", "FLUX.2 VAE")
    require_dir(base / "tokenizer", "FLUX.2 tokenizer")
    require_file(template / "model.py", "Template model.py")
    require_file(template / "model.safetensors", "Template base weights")
    return [str(path) for path in text_encoder]


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
        paths = []
        for raw_name in checkpoint_names.split(","):
            name = raw_name.strip()
            if not name:
                continue
            path = Path(name)
            if not path.is_absolute():
                if path.suffix != ".safetensors":
                    path = path.with_suffix(".safetensors")
                path = checkpoint_dir / path
            require_file(path, "checkpoint")
            paths.append(path)
        return paths
    paths = sorted(checkpoint_dir.glob(pattern), key=natural_step_key)
    if not paths:
        raise FileNotFoundError(f"No checkpoints matching {pattern} in {checkpoint_dir}")
    return paths


def load_records(metadata_path: Path, limit: int | None) -> list[dict[str, Any]]:
    require_file(metadata_path, "metadata")
    records: list[dict[str, Any]] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                relative_path = Path(record["image"])
                template_inputs = record["template_inputs"]
                lr_path = Path(template_inputs["image"]).expanduser().resolve()
                prompt = record["prompt"]
                template_prompt = template_inputs["prompt"]
            except (KeyError, TypeError) as error:
                raise ValueError(f"{metadata_path}:{line_number}: invalid metadata: {error}") from error
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"{metadata_path}:{line_number}: image must be relative: {relative_path}")
            require_file(lr_path, f"LR image at metadata line {line_number}")
            records.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "lr_path": str(lr_path),
                    "prompt": prompt,
                    "template_prompt": template_prompt,
                }
            )
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {metadata_path}")
    return records


def deterministic_seed(base_seed: int, relative_path: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{relative_path}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


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


def load_models(
    device: torch.device,
    base: Path,
    template_dir: Path,
    text_encoder_files: list[str],
) -> tuple[Flux2ImagePipeline, TemplatePipeline]:
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=text_encoder_files),
            ModelConfig(path=str(base / "transformer" / "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=str(base / "vae" / "diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(base / "tokenizer")),
    )
    template = TemplatePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=str(template_dir))],
    )
    pipe.dit.eval()
    template.eval()
    return pipe, template


def load_checkpoint_onto(template: TemplatePipeline, checkpoint: Path) -> None:
    state_dict = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16)
    load_result = template.models[0].load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: {checkpoint}, "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    template.eval()


@torch.inference_mode()
def make_lre_initial_noise(
    pipe: Flux2ImagePipeline,
    lr_image: Image.Image,
    lre_strength: float,
    seed: int,
    cache_device: str,
) -> torch.Tensor:
    width, height = lr_image.size
    latent_h, latent_w = height // 16, width // 16
    lr_tensor = pipe.preprocess_image(
        lr_image,
        torch_dtype=pipe.torch_dtype,
        device=pipe.device,
    )
    lr_latent = pipe.vae.encode(lr_tensor)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn(
        (1, 128, latent_h, latent_w),
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device=pipe.device, dtype=pipe.torch_dtype)

    lre_noise = (1.0 - lre_strength) * lr_latent + lre_strength * noise
    if cache_device == "cpu":
        return lre_noise.detach().cpu()
    return lre_noise.detach()


@torch.inference_mode()
def cache_record(
    pipe: Flux2ImagePipeline,
    record: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    with Image.open(record["lr_path"]) as opened:
        source = opened.convert("RGB")
    lr_image, original_size = reflect_pad_to_multiple(source)
    file_seed = deterministic_seed(args.seed, record["relative_path"])
    initial_noise = make_lre_initial_noise(
        pipe,
        lr_image,
        args.lre_strength,
        file_seed,
        args.cache_device,
    )
    return {
        **record,
        "lr_image": lr_image,
        "original_size": original_size,
        "height": lr_image.height,
        "width": lr_image.width,
        "seed": file_seed,
        "initial_noise": initial_noise,
    }


@torch.inference_mode()
def infer_cached_record(
    pipe: Flux2ImagePipeline,
    template: TemplatePipeline,
    cached: dict[str, Any],
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    initial_noise = cached["initial_noise"]
    if initial_noise.device != pipe.device:
        initial_noise = initial_noise.to(device=pipe.device, dtype=pipe.torch_dtype)
    image = template(
        pipe,
        prompt=cached["prompt"],
        negative_prompt=args.negative_prompt,
        height=cached["height"],
        width=cached["width"],
        seed=cached["seed"],
        rand_device="cpu",
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        num_inference_steps=args.num_inference_steps,
        initial_noise=initial_noise,
        template_inputs=[{"image": cached["lr_image"], "prompt": cached["template_prompt"]}],
        negative_template_inputs=[{"image": cached["lr_image"], "prompt": ""}],
        progress_bar_cmd=lambda values: values,
    )
    original_width, original_height = cached["original_size"]
    image = image.crop((0, 0, original_width, original_height))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    image.save(temporary_path, format="PNG")
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    if not 0 < args.lre_strength <= 1:
        raise ValueError("--lre-strength must be in (0, 1].")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive.")

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    output_base = args.output_base.expanduser().resolve()
    base = args.base_model.expanduser().resolve()
    template_dir = args.template_model.expanduser().resolve()
    require_dir(checkpoint_dir, "checkpoint directory")
    text_encoder_files = validate_models(base, template_dir)
    checkpoints = find_checkpoints(checkpoint_dir, args.checkpoint_pattern, args.checkpoint_names)
    records = load_records(metadata, args.limit)

    checkpoint_name = checkpoint_dir.name
    output_root = output_base / checkpoint_name
    output_root.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator()
    if accelerator.device.type != "cuda":
        raise RuntimeError("Cached LRE checkpoint comparison requires CUDA.")

    if accelerator.is_main_process:
        print("==========================================")
        print("  Cached LRE Checkpoint Comparison")
        print(f"  Metadata: {metadata}")
        print(f"  Records: {len(records)}")
        print(f"  Checkpoint dir: {checkpoint_dir}")
        print(f"  Checkpoints: {len(checkpoints)}")
        print(f"  Output root: {output_root}")
        print(f"  Processes: {accelerator.num_processes}")
        print(f"  Steps: {args.num_inference_steps}")
        print(f"  cfg_scale: {args.cfg_scale}")
        print(f"  embedded_guidance: {args.embedded_guidance}")
        print(f"  lre_strength: {args.lre_strength}")
        print(f"  cache_device: {args.cache_device}")
        print("==========================================")
    accelerator.wait_for_everyone()

    pipe, template = load_models(accelerator.device, base, template_dir, text_encoder_files)
    accelerator.wait_for_everyone()

    shard = records[accelerator.process_index :: accelerator.num_processes]
    cache_iter = tqdm(
        shard,
        desc=f"cache rank{accelerator.process_index}",
        disable=not accelerator.is_main_process,
    )
    cached_records = [cache_record(pipe, record, args) for record in cache_iter]
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("LRE initial noise cache is ready.", flush=True)

    for checkpoint in checkpoints:
        checkpoint_stem = checkpoint.stem
        out_dir = output_root / checkpoint_stem
        if accelerator.is_main_process:
            print(f"\n>>> Testing {checkpoint.name}", flush=True)
        accelerator.wait_for_everyone()

        load_checkpoint_onto(template, checkpoint)
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
            infer_cached_record(pipe, template, cached, output_path, args)
            generated += 1
        print(
            f"rank={accelerator.process_index} checkpoint={checkpoint_stem} "
            f"generated={generated} skipped={skipped}",
            flush=True,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            output_count = sum(1 for path in out_dir.rglob("*.png") if path.is_file())
            print(f">>> {checkpoint_stem} done: {output_count} images", flush=True)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("\n==========================================")
        print(f"  All done! Results: {output_root}")
        print("==========================================")
        for path in sorted((p for p in output_root.iterdir() if p.is_dir()), key=natural_step_key):
            print(path)


if __name__ == "__main__":
    main()
