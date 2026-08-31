#!/usr/bin/env python3
"""Minimal single-image LRE inference for FLUX.2 KleinBase4B Deblur Template.

Input:
  1. A degraded/LR image.
  2. An annotation file. Supported formats:
     - plain text: the whole file is used as prompt
     - json/jsonl: reads prompt from `prompt` or `template_inputs.prompt`

Example:
  python infer_single_lre.py \
    --input /path/to/lr.png \
    --annotation /path/to/ann.jsonl \
    --output /path/to/out.png
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

DEFAULT_HANDOFF_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = DEFAULT_HANDOFF_DIR / "step-10000.safetensors"
DEFAULT_DIFFSYNTH_ROOT = Path(os.environ.get("DIFFSYNTH_ROOT", "/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio"))
DEFAULT_BASE_MODEL = Path(os.environ.get("FLUX2_BASE_MODEL", "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"))
DEFAULT_TEMPLATE_MODEL = Path(os.environ.get("TEMPLATE_MODEL", "/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/Template-KleinBase4B-Upscaler"))

sys.path.insert(0, str(DEFAULT_DIFFSYNTH_ROOT))

import numpy as np
from PIL import Image
import torch

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline

DEFAULT_PROMPT = "Restore this degraded image to a clean, sharp, natural image while preserving all original content, colors, geometry, and composition."


def require_file(path: Path, name: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {name}: {path}")


def require_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing {name}: {path}")


def load_annotation_prompt(annotation: Path) -> tuple[str, str]:
    require_file(annotation, "annotation file")
    text = annotation.read_text(encoding="utf-8").strip()
    if not text:
        return DEFAULT_PROMPT, DEFAULT_PROMPT

    suffix = annotation.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        raw = text.splitlines()[0] if suffix == ".jsonl" else text
        item: dict[str, Any] = json.loads(raw)
        prompt = item.get("prompt") or item.get("template_inputs", {}).get("prompt") or DEFAULT_PROMPT
        template_prompt = item.get("template_inputs", {}).get("prompt") or prompt
        return prompt, template_prompt

    return text, text


def deterministic_seed(base_seed: int, key: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{key}".encode("utf-8")).digest()
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


def validate_model_paths(base_model: Path, template_model: Path) -> list[str]:
    require_dir(base_model, "FLUX.2 base model directory")
    require_dir(template_model, "template model directory")
    text_encoder = sorted((base_model / "text_encoder").glob("*.safetensors"))
    if not text_encoder:
        raise FileNotFoundError(f"No text encoder safetensors found in {base_model / 'text_encoder'}")
    require_file(base_model / "transformer" / "diffusion_pytorch_model.safetensors", "FLUX.2 transformer")
    require_file(base_model / "vae" / "diffusion_pytorch_model.safetensors", "FLUX.2 VAE")
    require_dir(base_model / "tokenizer", "FLUX.2 tokenizer")
    require_file(template_model / "model.py", "template model.py")
    require_file(template_model / "model.safetensors", "template base weights")
    return [str(p) for p in text_encoder]


def load_models(device: torch.device, base_model: Path, template_model: Path):
    text_encoder_files = validate_model_paths(base_model, template_model)
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=text_encoder_files),
            ModelConfig(path=str(base_model / "transformer" / "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=str(base_model / "vae" / "diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(base_model / "tokenizer")),
    )
    template = TemplatePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=str(template_model))],
    )
    pipe.dit.eval()
    template.eval()
    return pipe, template


def load_checkpoint(template: TemplatePipeline, checkpoint: Path) -> None:
    require_file(checkpoint, "template checkpoint")
    state_dict = load_state_dict(str(checkpoint), torch_dtype=torch.bfloat16)
    result = template.models[0].load_state_dict(state_dict, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    template.eval()


@torch.inference_mode()
def make_lre_initial_noise(pipe: Flux2ImagePipeline, image: Image.Image, lre_strength: float, seed: int) -> torch.Tensor:
    width, height = image.size
    latent_h, latent_w = height // 16, width // 16
    lr_tensor = pipe.preprocess_image(image, torch_dtype=pipe.torch_dtype, device=pipe.device)
    lr_latent = pipe.vae.encode(lr_tensor)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    noise = torch.randn((1, 128, latent_h, latent_w), generator=generator, dtype=torch.float32, device="cpu")
    noise = noise.to(device=pipe.device, dtype=pipe.torch_dtype)
    return (1.0 - lre_strength) * lr_latent + lre_strength * noise


@torch.inference_mode()
def run_inference(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    prompt, template_prompt = load_annotation_prompt(args.annotation)
    with Image.open(args.input) as opened:
        lr_source = opened.convert("RGB")
    lr_image, original_size = reflect_pad_to_multiple(lr_source)

    pipe, template = load_models(device, args.base_model, args.template_model)
    load_checkpoint(template, args.checkpoint)

    seed = deterministic_seed(args.seed, str(args.input.resolve()))
    initial_noise = make_lre_initial_noise(pipe, lr_image, args.lre_strength, seed)
    image = template(
        pipe,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=lr_image.height,
        width=lr_image.width,
        seed=seed,
        rand_device="cpu",
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        num_inference_steps=args.num_inference_steps,
        initial_noise=initial_noise,
        template_inputs=[{"image": lr_image, "prompt": template_prompt}],
        negative_template_inputs=[{"image": lr_image, "prompt": ""}],
    )
    original_width, original_height = original_size
    image = image.crop((0, 0, original_width, original_height))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".tmp")
    image.save(tmp, format="PNG")
    tmp.replace(args.output)
    print(f"Saved: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single image inference with trained LRE deblur template checkpoint.")
    parser.add_argument("--input", type=Path, required=True, help="Input degraded/LR image.")
    parser.add_argument("--annotation", type=Path, required=True, help="Text/json/jsonl annotation containing prompt.")
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--template-model", type=Path, default=DEFAULT_TEMPLATE_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--embedded-guidance", type=float, default=4.0)
    parser.add_argument("--lre-strength", type=float, default=0.8)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.lre_strength <= 1:
        raise ValueError("--lre-strength must be in (0, 1].")
    return args


if __name__ == "__main__":
    run_inference(parse_args())
