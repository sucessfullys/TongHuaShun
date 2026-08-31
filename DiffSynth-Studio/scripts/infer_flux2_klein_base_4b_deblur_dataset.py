#!/usr/bin/env python3
"""Batch inference for a fine-tuned FLUX.2 Deblur Template checkpoint."""

from __future__ import annotations

import os

# Enforce offline loading before importing ModelScope/Transformers/DiffSynth.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from accelerate import Accelerator
from PIL import Image
import torch
from tqdm import tqdm

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline


DEFAULT_BASE_MODEL = Path(
    "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
)
DEFAULT_TEMPLATE_MODEL = REPO_ROOT / "Template-KleinBase4B-Upscaler"
DEFAULT_METADATA = Path(
    "/mnt/image-edit/datasets/duanyufa/Face/test/metadata.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fully offline, optionally multi-GPU inference over a DiffSynth "
            "Deblur Template metadata.jsonl dataset."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Fine-tuned Template checkpoint, e.g. epoch-0.safetensors.",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to CHECKPOINT_DIR/test_results/CHECKPOINT_STEM/template.",
    )
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--template-model", type=Path, default=DEFAULT_TEMPLATE_MODEL
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--embedded-guidance", type=float, default=4.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=("template", "img2img"),
        default="template",
        help=(
            "template matches training and starts from noise; img2img also uses "
            "the LR latent as initialization for stronger content preservation."
        ),
    )
    parser.add_argument(
        "--denoising-strength",
        type=float,
        default=0.15,
        help="Only used when --mode=img2img.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def validate_models(base: Path, template: Path) -> list[str]:
    text_encoder = sorted((base / "text_encoder").glob("*.safetensors"))
    if not text_encoder:
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
    require_file(template / "model.py", "Template model.py")
    require_file(template / "model.safetensors", "base Template weights")
    return [str(path) for path in text_encoder]


def load_records(metadata_path: Path, limit: int | None) -> list[dict]:
    require_file(metadata_path, "test metadata")
    records = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                relative_path = Path(record["image"])
                template_inputs = record["template_inputs"]
                Path(template_inputs["image"])
                record["prompt"]
                template_inputs["prompt"]
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid metadata record at line {line_number}: {error}"
                ) from error
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Metadata image must be a safe relative output path at "
                    f"line {line_number}: {relative_path}"
                )
            record["_relative_path"] = relative_path.as_posix()
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


def load_models(
    accelerator: Accelerator,
    base: Path,
    template_dir: Path,
    checkpoint: Path,
    text_encoder_files: list[str],
):
    device = accelerator.device
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
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
    template = TemplatePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=str(template_dir))],
    )
    state_dict = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16)
    load_result = template.models[0].load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "Template checkpoint mismatch: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    template.eval()
    pipe.dit.eval()
    return pipe, template


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    mode_suffix = (
        "template"
        if args.mode == "template"
        else f"img2img-strength-{args.denoising_strength:g}"
    )
    return (
        args.checkpoint.expanduser().resolve().parent
        / "test_results"
        / args.checkpoint.stem
        / mode_suffix
    )


@torch.inference_mode()
def infer_record(
    pipe: Flux2ImagePipeline,
    template: TemplatePipeline,
    record: dict,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    lr_path = Path(record["template_inputs"]["image"]).expanduser().resolve()
    require_file(lr_path, "LR test image")
    with Image.open(lr_path) as opened:
        lr_image = opened.convert("RGB")
    width, height = lr_image.size
    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(
            f"LR dimensions must be divisible by 16: {lr_path} is {width}x{height}"
        )

    prompt = record["prompt"]
    template_prompt = record["template_inputs"]["prompt"]
    inference_kwargs = {
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "height": height,
        "width": width,
        "seed": deterministic_seed(args.seed, record["_relative_path"]),
        "rand_device": "cpu",
        "cfg_scale": args.cfg_scale,
        "embedded_guidance": args.embedded_guidance,
        "num_inference_steps": args.num_inference_steps,
        "template_inputs": [{"image": lr_image, "prompt": template_prompt}],
        "negative_template_inputs": [{"image": lr_image, "prompt": ""}],
        "progress_bar_cmd": lambda values: values,
    }
    if args.mode == "img2img":
        inference_kwargs.update(
            input_image=lr_image,
            denoising_strength=args.denoising_strength,
        )
    image = template(pipe, **inference_kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    image.save(temporary_path, format="PNG")
    temporary_path.replace(output_path)


def main() -> None:
    args = parse_args()
    if not 0 < args.denoising_strength <= 1:
        raise ValueError("--denoising-strength must be in (0, 1].")
    if args.num_inference_steps < 1:
        raise ValueError("--num-inference-steps must be positive.")

    accelerator = Accelerator()
    if accelerator.device.type != "cuda":
        raise RuntimeError("CUDA is required for FLUX.2 Deblur inference.")

    checkpoint = args.checkpoint.expanduser().resolve()
    metadata = args.metadata.expanduser().resolve()
    base = args.base_model.expanduser().resolve()
    template_dir = args.template_model.expanduser().resolve()
    require_file(checkpoint, "fine-tuned Template checkpoint")
    text_encoder_files = validate_models(base, template_dir)
    records = load_records(metadata, args.limit)
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    if accelerator.is_main_process:
        print(f"Checkpoint: {checkpoint}")
        print(f"Metadata: {metadata}")
        print(f"Records: {len(records)}")
        print(f"Processes: {accelerator.num_processes}")
        print(f"Mode: {args.mode}")
        print(f"Output: {output_dir}")
    accelerator.wait_for_everyone()

    pipe, template = load_models(
        accelerator,
        base,
        template_dir,
        checkpoint,
        text_encoder_files,
    )
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
        infer_record(pipe, template, record, output_path, args)
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
        print(f"Inference complete. PNG files in output directory: {output_count}")


if __name__ == "__main__":
    main()
