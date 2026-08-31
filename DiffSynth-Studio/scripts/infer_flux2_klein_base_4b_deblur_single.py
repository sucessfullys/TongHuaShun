#!/usr/bin/env python3
"""Single-image inference for the fine-tuned FLUX.2 Deblur Template."""

from __future__ import annotations

import os

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
from pathlib import Path
import sys

from accelerate import Accelerator
import numpy as np
from PIL import Image
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from infer_flux2_klein_base_4b_deblur_dataset import (
    DEFAULT_BASE_MODEL,
    DEFAULT_TEMPLATE_MODEL,
    load_models,
    require_file,
    validate_models,
)


DEFAULT_INPUT = REPO_ROOT / "image.png"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "outputs"
    / "Template-KleinBase4B-Deblur_full"
    / "epoch-3.safetensors"
)
DEFAULT_PROMPT = (
    "Restore the input image to a clean, sharp, high-quality image. "
    "Remove blur and noise while preserving the original content, structure, "
    "identity, colors, and layout. Do not add, remove, or change objects."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real image through a fine-tuned FLUX.2 Deblur Template. "
            "Non-multiple-of-16 dimensions are reflect-padded and cropped back."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--template-model", type=Path, default=DEFAULT_TEMPLATE_MODEL
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--embedded-guidance", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_output(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    return (
        checkpoint.parent
        / "single_results"
        / checkpoint.stem
        / f"{input_path.stem}.png"
    )


def reflect_pad_to_multiple(
    image: Image.Image, multiple: int = 16
) -> tuple[Image.Image, tuple[int, int]]:
    width, height = image.size
    pad_width = (-width) % multiple
    pad_height = (-height) % multiple
    if pad_width == 0 and pad_height == 0:
        return image, (width, height)
    array = np.asarray(image)
    mode = "reflect" if width > 1 and height > 1 else "edge"
    padded = np.pad(
        array,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode=mode,
    )
    return Image.fromarray(padded), (width, height)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")

    input_path = args.input.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    base = args.base_model.expanduser().resolve()
    template_dir = args.template_model.expanduser().resolve()
    output_path = resolve_output(args)
    require_file(input_path, "input image")
    require_file(checkpoint, "fine-tuned Template checkpoint")
    text_encoder_files = validate_models(base, template_dir)

    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("This single-image script must run with one process")
    if accelerator.device.type != "cuda":
        raise RuntimeError("CUDA is required for FLUX.2 inference")

    with Image.open(input_path) as opened:
        source = opened.convert("RGB")
    padded, original_size = reflect_pad_to_multiple(source)
    print(f"Input: {input_path}")
    print(f"Original size: {original_size[0]}x{original_size[1]}")
    print(f"Model size: {padded.width}x{padded.height}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output: {output_path}")

    pipe, template = load_models(
        accelerator,
        base,
        template_dir,
        checkpoint,
        text_encoder_files,
    )
    result = template(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=padded.height,
        width=padded.width,
        seed=args.seed,
        rand_device="cpu",
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        num_inference_steps=args.num_inference_steps,
        template_inputs=[{"image": padded, "prompt": args.prompt}],
        negative_template_inputs=[{"image": padded, "prompt": ""}],
    )
    result = result.crop((0, 0, original_size[0], original_size[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    result.save(temporary, format="PNG")
    temporary.replace(output_path)
    print(f"Saved: {output_path} ({result.width}x{result.height})")


if __name__ == "__main__":
    main()
