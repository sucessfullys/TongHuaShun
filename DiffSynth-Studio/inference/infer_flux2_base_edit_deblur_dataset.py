#!/usr/bin/env python3
"""Zero-shot FLUX.2 Base 4B deblurring with the native edit_image input.

This baseline intentionally loads neither Template-KleinBase4B-Upscaler nor a
LoRA checkpoint.  The degraded LR image is supplied only through FLUX.2's
native edit-image conditioning path.
"""

from __future__ import annotations

import os

# Set offline flags before importing Transformers/ModelScope/DiffSynth.
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import hashlib
import json
from pathlib import Path
import sys

from accelerate import Accelerator
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig


DEFAULT_BASE_MODEL = Path(
    "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
)
DEFAULT_METADATA = Path(
    "/mnt/image-edit/datasets/duanyufa/Face/test/metadata.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs" / "FLUX.2_Base_deblur"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fully offline, optionally multi-GPU, zero-shot FLUX.2 Base "
            "4B deblurring with LR images passed as edit_image conditions."
        )
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--embedded-guidance", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def validate_base_model(base: Path) -> list[str]:
    text_encoder_files = sorted((base / "text_encoder").glob("*.safetensors"))
    if not text_encoder_files:
        raise FileNotFoundError(
            f"No text-encoder safetensors found in {base / 'text_encoder'}"
        )
    require_file(
        base / "transformer" / "diffusion_pytorch_model.safetensors",
        "FLUX.2 transformer",
    )
    require_file(
        base / "vae" / "diffusion_pytorch_model.safetensors",
        "FLUX.2 VAE",
    )
    if not (base / "tokenizer").is_dir():
        raise FileNotFoundError(f"Missing tokenizer directory: {base / 'tokenizer'}")
    return [str(path) for path in text_encoder_files]


def resolve_lr_path(record: dict, line_number: int) -> Path:
    """Accept the current Template metadata and future edit_image metadata."""
    if "edit_image" in record:
        value = record["edit_image"]
    else:
        try:
            value = record["template_inputs"]["image"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Line {line_number} needs edit_image or template_inputs.image"
            ) from error
    return Path(value).expanduser().resolve()


def load_records(metadata_path: Path, limit: int | None) -> list[dict]:
    require_file(metadata_path, "test metadata")
    records: list[dict] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                relative_path = Path(record["image"])
                prompt = record["prompt"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid metadata record at line {line_number}: {error}"
                ) from error
            if not isinstance(prompt, str):
                raise ValueError(f"prompt must be a string at line {line_number}")
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Metadata image must be a safe relative output path at "
                    f"line {line_number}: {relative_path}"
                )
            record["_relative_path"] = relative_path.as_posix()
            record["_lr_path"] = resolve_lr_path(record, line_number)
            require_file(record["_lr_path"], "LR test image")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records found in {metadata_path}")
    return records


def deterministic_seed(base_seed: int, relative_path: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}:{relative_path}".encode("utf-8")
    ).digest()
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


def load_pipeline(
    accelerator: Accelerator,
    base: Path,
    text_encoder_files: list[str],
) -> Flux2ImagePipeline:
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=accelerator.device,
        model_configs=[
            ModelConfig(path=text_encoder_files),
            ModelConfig(
                path=str(
                    base / "transformer" / "diffusion_pytorch_model.safetensors"
                )
            ),
            ModelConfig(
                path=str(base / "vae" / "diffusion_pytorch_model.safetensors")
            ),
        ],
        tokenizer_config=ModelConfig(path=str(base / "tokenizer")),
    )
    pipe.dit.eval()
    return pipe


@torch.inference_mode()
def infer_record(
    pipe: Flux2ImagePipeline,
    record: dict,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    with Image.open(record["_lr_path"]) as opened:
        source = opened.convert("RGB")
    lr_image, original_size = reflect_pad_to_multiple(source)
    width, height = lr_image.size

    image = pipe(
        prompt=record["prompt"],
        negative_prompt=args.negative_prompt,
        edit_image=lr_image,
        edit_image_auto_resize=False,
        height=height,
        width=width,
        seed=deterministic_seed(args.seed, record["_relative_path"]),
        rand_device="cpu",
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        num_inference_steps=args.num_inference_steps,
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
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")

    accelerator = Accelerator()
    if accelerator.device.type != "cuda":
        raise RuntimeError("CUDA is required for FLUX.2 inference")

    base = args.base_model.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    text_encoder_files = validate_base_model(base)
    records = load_records(metadata, args.limit)
    output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print("Experiment: zero-shot FLUX.2 Base 4B + native edit_image")
        print("Template: disabled")
        print("LoRA: disabled")
        print(f"Base model: {base}")
        print(f"Metadata: {metadata}")
        print(f"Records: {len(records)}")
        print(f"Processes: {accelerator.num_processes}")
        print(f"Output: {output_dir}")
    accelerator.wait_for_everyone()

    pipe = load_pipeline(accelerator, base, text_encoder_files)
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
