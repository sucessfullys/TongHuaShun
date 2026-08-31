#!/usr/bin/env python3
"""Batch image-to-image generation with FLUX models.

Given an input image directory and one fixed edit prompt, this script edits each
image and saves generated images to an output directory.
It does not load LoRA weights.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch edit images from a folder with FLUX."
    )
    parser.add_argument(
        "--backend",
        default="diffusers",
        choices=["diffusers", "diffsynth"],
        help="Inference backend. Use diffusers for klein base 4B/9B; diffsynth for FLUX.2-dev.",
    )
    parser.add_argument(
        "--model-path",
        default="/mnt/data/image-edit/datasets/shensheng/models/black-forest-labs/FLUX.2-klein-base-9B",
        help="Local model path or Hugging Face model id.",
    )
    parser.add_argument("--input-dir", required=True, help="Input image folder.")
    parser.add_argument("--output-dir", required=True, help="Output image folder.")
    parser.add_argument("--prompt", default="", help="Fixed edit prompt.")
    parser.add_argument(
        "--prompt-file",
        default="",
        help="Read fixed edit prompt from a text file. Overrides --prompt.",
    )
    parser.add_argument("--results-file", default="", help="JSONL metadata path.")
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cuda:0.")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype.",
    )
    parser.add_argument("--steps", type=int, default=28, help="Inference steps.")
    parser.add_argument("--cfg", type=float, default=1.0, help="Guidance scale.")
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output height when --preserve-size is not set.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output width when --preserve-size is not set.",
    )
    parser.add_argument(
        "--preserve-size",
        action="store_true",
        help="Use each input image's original width and height for output.",
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=0,
        help=(
            "When --preserve-size is set, keep aspect ratio but clamp both width "
            "and height to this maximum. 0 disables clamping."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all images.")
    parser.add_argument("--shard-index", type=int, default=0, help="Current shard index.")
    parser.add_argument("--shard-count", type=int, default=1, help="Total shard count.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input-dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument(
        "--keep-ext",
        action="store_true",
        help="Keep original file extension. Default saves PNG.",
    )
    return parser.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    else:
        prompt = args.prompt.strip()
    if not prompt:
        raise ValueError("Prompt is empty. Use --prompt or --prompt-file.")
    return prompt


def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = [p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda p: p.as_posix())


def output_path_for(input_path: Path, input_dir: Path, output_dir: Path, keep_ext: bool) -> Path:
    rel = input_path.relative_to(input_dir)
    suffix = input_path.suffix if keep_ext else ".png"
    return (output_dir / rel).with_suffix(suffix)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def round_to_multiple(value: int, multiple: int = 16, max_value: int | None = None) -> int:
    rounded = max(multiple, int(round(value / multiple)) * multiple)
    if max_value is not None:
        rounded = min(max_value // multiple * multiple, rounded)
    return max(multiple, rounded)


def closest_aspect_size(
    input_size: tuple[int, int],
    max_resolution: int,
    multiple: int = 16,
) -> tuple[int, int]:
    src_w, src_h = input_size
    scale = min(1.0, max_resolution / max(src_w, src_h))
    target_w = src_w * scale
    target_h = src_h * scale
    target_area = target_w * target_h
    src_ratio = src_w / src_h
    max_multiple = max_resolution // multiple * multiple

    best_size = (round_to_multiple(target_w, multiple, max_resolution), round_to_multiple(target_h, multiple, max_resolution))
    best_score = (float("inf"), float("inf"))
    for width in range(multiple, max_multiple + 1, multiple):
        ideal_height = width / src_ratio
        for height in {
            round_to_multiple(ideal_height, multiple, max_resolution),
            max(multiple, int(math.floor(ideal_height / multiple)) * multiple),
            min(max_multiple, int(math.ceil(ideal_height / multiple)) * multiple),
        }:
            if height < multiple or height > max_multiple:
                continue
            ratio_error = abs(math.log((width / height) / src_ratio))
            area_error = abs((width * height) - target_area) / max(target_area, 1.0)
            score = (ratio_error, area_error)
            if score < best_score:
                best_score = score
                best_size = (width, height)
    return best_size


def output_size_for(input_size: tuple[int, int], args: argparse.Namespace) -> tuple[int, int]:
    if not args.preserve_size:
        return round_to_multiple(args.width), round_to_multiple(args.height)

    width, height = input_size
    if args.max_resolution > 0:
        return closest_aspect_size((width, height), args.max_resolution)
    return round_to_multiple(width), round_to_multiple(height)


def atomic_save(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp.png")
    image.save(tmp_path, format="PNG")
    tmp_path.replace(output_path)


def load_pipeline(args: argparse.Namespace):
    dtype = torch_dtype(args.dtype)
    if args.backend == "diffusers":
        from diffusers import Flux2KleinPipeline

        print(f"Loading Flux2KleinPipeline: {args.model_path}")
        pipe = Flux2KleinPipeline.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
        )
        pipe.to(args.device)
        if hasattr(pipe, "transformer"):
            pipe.transformer.eval()
        return pipe

    from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

    print(f"Loading DiffSynth Flux2ImagePipeline: {args.model_path}")
    model_root = Path(args.model_path)
    if model_root.exists():
        text_encoder = sorted((model_root / "text_encoder").glob("*.safetensors"))
        transformer = sorted((model_root / "transformer").glob("*.safetensors"))
        vae = model_root / "vae" / "diffusion_pytorch_model.safetensors"
        tokenizer = model_root / "tokenizer"
        if not text_encoder:
            raise FileNotFoundError(f"No text_encoder safetensors found under: {model_root}")
        if not transformer:
            raise FileNotFoundError(f"No transformer safetensors found under: {model_root}")
        if not vae.is_file():
            fallback_vae = model_root / "ae.safetensors"
            if fallback_vae.is_file():
                vae = fallback_vae
            else:
                raise FileNotFoundError(f"No VAE safetensors found under: {model_root}")
        if not tokenizer.is_dir():
            raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer}")
        model_configs = [
            ModelConfig(path=[str(p) for p in text_encoder]),
            ModelConfig(path=[str(p) for p in transformer]),
            ModelConfig(path=str(vae)),
        ]
        tokenizer_config = ModelConfig(path=str(tokenizer))
    else:
        model_id = args.model_path
        model_configs = [
            ModelConfig(model_id=model_id, origin_file_pattern="text_encoder/*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="transformer/*.safetensors"),
            ModelConfig(model_id=model_id, origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ]
        tokenizer_config = ModelConfig(model_id=model_id, origin_file_pattern="tokenizer/")
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=dtype,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )
    return pipe


def run_pipe(
    pipe,
    backend: str,
    prompt: str,
    ref_image: Image.Image,
    width: int,
    height: int,
    cfg: float,
    steps: int,
    seed: int,
):
    if backend == "diffusers":
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return pipe(
            prompt=prompt,
            image=ref_image,
            height=height,
            width=width,
            guidance_scale=cfg,
            num_inference_steps=steps,
            generator=generator,
        ).images[0]

    return pipe(
        prompt,
        edit_image=[ref_image],
        height=height,
        width=width,
        cfg_scale=cfg,
        num_inference_steps=steps,
        seed=seed,
        rand_device="cuda" if torch.cuda.is_available() else "cpu",
    )


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    prompt = load_prompt(args)
    images = list_images(input_dir, args.recursive)
    if args.max_samples > 0:
        images = images[: args.max_samples]
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")
    indexed_images = list(enumerate(images))
    shard_items = indexed_images[args.shard_index :: args.shard_count]
    if not shard_items:
        print(
            f"No images for shard {args.shard_index}/{args.shard_count}; "
            f"total images={len(images)}"
        )
        return

    results_file = Path(args.results_file) if args.results_file else output_dir / "results.jsonl"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args)

    print(f"Input images: {len(images)}")
    print(f"Shard: {args.shard_index}/{args.shard_count}, shard images: {len(shard_items)}")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model_path}")
    print(f"Output dir: {output_dir}")
    print(f"Prompt: {prompt}")

    ok = 0
    skipped = 0
    failed = 0
    with results_file.open("a", encoding="utf-8") as fp:
        for global_index, image_path in tqdm(
            shard_items,
            desc=f"Generating shard {args.shard_index}",
            unit="image",
        ):
            out_path = output_path_for(image_path, input_dir, output_dir, args.keep_ext)
            record = {
                "index": global_index,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "input_path": str(image_path),
                "output_path": str(out_path),
                "prompt": prompt,
                "seed": args.seed + global_index,
                "steps": args.steps,
                "cfg": args.cfg,
                "height": None,
                "width": None,
                "preserve_size": args.preserve_size,
                "success": False,
                "skipped": False,
                "error": None,
                "time_s": None,
            }

            if out_path.exists() and not args.overwrite:
                record["success"] = True
                record["skipped"] = True
                skipped += 1
                fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                fp.flush()
                continue

            try:
                ref_image = load_image(image_path)
                width, height = output_size_for(ref_image.size, args)
                record["height"] = height
                record["width"] = width
                start = time.time()
                with torch.inference_mode():
                    image = run_pipe(
                        pipe=pipe,
                        backend=args.backend,
                        prompt=prompt,
                        ref_image=ref_image,
                        width=width,
                        height=height,
                        cfg=args.cfg,
                        steps=args.steps,
                        seed=args.seed + global_index,
                    )
                atomic_save(image, out_path)
                record["success"] = True
                record["time_s"] = round(time.time() - start, 3)
                ok += 1
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                failed += 1

            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            fp.flush()

    print(f"Done. generated={ok}, skipped={skipped}, failed={failed}")
    print(f"Metadata: {results_file}")


if __name__ == "__main__":
    main()
