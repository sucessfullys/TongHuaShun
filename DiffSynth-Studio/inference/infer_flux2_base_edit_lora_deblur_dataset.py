#!/usr/bin/env python3
"""Paired-deblur FLUX.2 Base 4B LoRA inference on a metadata dataset."""

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
import torch
from tqdm import tqdm


INFERENCE_DIR = Path(__file__).resolve().parent
REPO_ROOT = INFERENCE_DIR.parent
sys.path.insert(0, str(INFERENCE_DIR))

from infer_flux2_base_edit_deblur_dataset import (
    DEFAULT_BASE_MODEL,
    DEFAULT_METADATA,
    infer_record,
    load_pipeline,
    load_records,
    require_file,
    validate_base_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fully offline, optionally multi-GPU FLUX.2 Base 4B native "
            "edit_image inference with a paired-deblur DiT LoRA."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--embedded-guidance", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    return checkpoint.parent / "test_results" / checkpoint.stem / "edit_image"


def main() -> None:
    args = parse_args()
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    accelerator = Accelerator()
    if accelerator.device.type != "cuda":
        raise RuntimeError("CUDA is required for FLUX.2 inference")

    checkpoint = args.checkpoint.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    base = args.base_model.expanduser().resolve()
    output_dir = resolve_output_dir(args)
    require_file(checkpoint, "trained LoRA checkpoint")
    text_encoder_files = validate_base_model(base)
    records = load_records(metadata, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print("Experiment: paired FLUX.2 Base 4B edit_image LoRA")
        print("Template: disabled")
        print(f"LoRA checkpoint: {checkpoint}")
        print(f"LoRA scale: {args.lora_scale}")
        print(f"Base model: {base}")
        print(f"Metadata: {metadata}")
        print(f"Records: {len(records)}")
        print(f"Processes: {accelerator.num_processes}")
        print(f"Output: {output_dir}")
    accelerator.wait_for_everyone()

    pipe = load_pipeline(accelerator, base, text_encoder_files)
    pipe.load_lora(
        pipe.dit,
        lora_config=str(checkpoint),
        alpha=args.lora_scale,
        hotload=False,
    )
    pipe.dit.eval()
    accelerator.wait_for_everyone()

    shard = records[accelerator.process_index :: accelerator.num_processes]
    iterator = tqdm(
        shard,
        desc=f"rank {accelerator.process_index}",
        disable=not accelerator.is_main_process,
    )
    generated = 0
    skipped = 0
    for record in iterator:
        output_path = output_dir / record["_relative_path"]
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        infer_record(pipe, record, output_path, args)
        generated += 1

    print(
        f"rank={accelerator.process_index} generated={generated} skipped={skipped}",
        flush=True,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_count = sum(
            1 for path in output_dir.rglob("*.png") if path.is_file()
        )
        print(f"Inference complete. PNG files: {output_count}")


if __name__ == "__main__":
    main()
