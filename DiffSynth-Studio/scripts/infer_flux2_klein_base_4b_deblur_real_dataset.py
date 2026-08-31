#!/usr/bin/env python3
"""Batch inference for real-world images with original-size padding/cropping.

This script is intentionally separate from the fixed test-set inference script:
real images may be JPG/PNG and may have arbitrary dimensions. Each image is
reflect-padded to a multiple of 16 for FLUX.2 inference, then cropped back to
the original size and saved as PNG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from accelerate import Accelerator
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from infer_flux2_klein_base_4b_deblur_dataset import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_TEMPLATE_MODEL,
    load_models,
    require_file,
    validate_models,
)


DEFAULT_METADATA = Path("/mnt/image-edit/datasets/duanyufa/Real_RL_Data/metadata.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FLUX.2 Deblur Template inference on real images with original-size outputs."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--template-model", type=Path, default=DEFAULT_TEMPLATE_MODEL)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--embedded-guidance", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_records(metadata_path: Path, limit: int | None) -> list[dict]:
    require_file(metadata_path, "real dataset metadata")
    records = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                relative_path = Path(record["image"])
                input_path = Path(record["template_inputs"]["image"])
                record["prompt"]
                record["template_inputs"]["prompt"]
            except (KeyError, TypeError) as error:
                raise ValueError(f"Invalid metadata record at line {line_number}: {error}") from error
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe output path at line {line_number}: {relative_path}")
            require_file(input_path.expanduser(), f"input image at line {line_number}")
            record["_relative_path"] = relative_path.with_suffix(".png").as_posix()
            records.append(record)
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


@torch.inference_mode()
def infer_record(pipe, template, record: dict, output_path: Path, args: argparse.Namespace) -> None:
    input_path = Path(record["template_inputs"]["image"]).expanduser().resolve()
    require_file(input_path, "real input image")
    with Image.open(input_path) as opened:
        source = opened.convert("RGB")
    padded, original_size = reflect_pad_to_multiple(source)

    prompt = record["prompt"]
    template_prompt = record["template_inputs"]["prompt"]
    image = template(
        pipe,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=padded.height,
        width=padded.width,
        seed=deterministic_seed(args.seed, record["_relative_path"]),
        rand_device="cpu",
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        num_inference_steps=args.num_inference_steps,
        template_inputs=[{"image": padded, "prompt": template_prompt}],
        negative_template_inputs=[{"image": padded, "prompt": ""}],
        progress_bar_cmd=lambda values: values,
    )
    image = image.crop((0, 0, original_size[0], original_size[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    image.save(temporary_path, format="PNG")
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")

    accelerator = Accelerator()
    if accelerator.device.type != "cuda":
        raise RuntimeError("CUDA is required for FLUX.2 Deblur inference.")

    checkpoint = args.checkpoint.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    base = args.base_model.expanduser().resolve()
    template_dir = args.template_model.expanduser().resolve()

    require_file(checkpoint, "fine-tuned Template checkpoint")
    text_encoder_files = validate_models(base, template_dir)
    records = load_records(metadata, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"Checkpoint: {checkpoint}")
        print(f"Metadata: {metadata}")
        print(f"Records: {len(records)}")
        print(f"Processes: {accelerator.num_processes}")
        print(f"Output: {output_dir}")
    accelerator.wait_for_everyone()

    pipe, template = load_models(accelerator, base, template_dir, checkpoint, text_encoder_files)
    accelerator.wait_for_everyone()

    shard = records[accelerator.process_index :: accelerator.num_processes]
    iterator = tqdm(shard, desc=f"rank {accelerator.process_index}", disable=not accelerator.is_main_process)
    generated = 0
    skipped = 0
    for record in iterator:
        output_path = output_dir / record["_relative_path"]
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        infer_record(pipe, template, record, output_path, args)
        generated += 1

    print(f"rank={accelerator.process_index} generated={generated} skipped={skipped}", flush=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_count = sum(1 for path in output_dir.rglob("*.png") if path.is_file())
        print(f"Inference complete. PNG files in output directory: {output_count}")


if __name__ == "__main__":
    main()
