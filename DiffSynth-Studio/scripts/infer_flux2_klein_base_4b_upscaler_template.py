#!/usr/bin/env python3
"""Run the official FLUX.2 Klein Base 4B Upscaler fully offline."""

import os

# Set offline flags before importing ModelScope/Transformers/DiffSynth.
os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")

import argparse
from pathlib import Path
import sys

# Allow direct execution from a source checkout without requiring `pip install -e .`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig


DEFAULT_BASE_MODEL = Path(
    "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
)
DEFAULT_COMPONENT_MODEL = Path(
    "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
)
DEFAULT_TEMPLATE_MODEL = REPO_ROOT / "Template-KleinBase4B-Upscaler"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Enhance an image with the unmodified official "
            "Template-KleinBase4B-Upscaler weights, without network access."
        )
    )
    parser.add_argument("--input", required=True, help="Low-quality input image.")
    parser.add_argument("--output", required=True, help="Output image path.")
    parser.add_argument(
        "--template-model",
        default=str(DEFAULT_TEMPLATE_MODEL),
        help="Local directory containing official model.py and model.safetensors.",
    )
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--component-model", default=str(DEFAULT_COMPONENT_MODEL))
    parser.add_argument(
        "--prompt",
        default=(
            "Remove blur and restore sharp natural details while preserving "
            "the original content."
        ),
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Output width. Defaults to the input width rounded down to a multiple of 16.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Output height. Defaults to the input height rounded down to a multiple of 16.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--rand-device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def require_file(path: Path, description: str):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def validate_local_models(base: Path, components: Path, template: Path):
    text_encoder_files = sorted((components / "text_encoder").glob("*.safetensors"))
    if not text_encoder_files:
        raise FileNotFoundError(
            f"No local text-encoder weights found in {components / 'text_encoder'}"
        )
    require_file(
        base / "transformer" / "diffusion_pytorch_model.safetensors",
        "FLUX.2 Base transformer",
    )
    require_file(
        components / "vae" / "diffusion_pytorch_model.safetensors",
        "FLUX.2 VAE",
    )
    if not (components / "tokenizer").is_dir():
        raise FileNotFoundError(f"Missing local tokenizer: {components / 'tokenizer'}")
    require_file(template / "model.py", "official Template model.py")
    require_file(template / "model.safetensors", "official Template weights")
    return [str(path) for path in text_encoder_files]


def output_size(image: Image.Image, width: int | None, height: int | None):
    width = image.width if width is None else width
    height = image.height if height is None else height
    width = width // 16 * 16
    height = height // 16 * 16
    if width < 16 or height < 16:
        raise ValueError("Output width and height must both be at least 16 pixels.")
    return width, height


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLUX.2 Upscaler inference.")

    base = Path(args.base_model).expanduser().resolve()
    components = Path(args.component_model).expanduser().resolve()
    template_dir = Path(args.template_model).expanduser().resolve()
    text_encoder_files = validate_local_models(base, components, template_dir)

    input_path = Path(args.input).expanduser().resolve()
    require_file(input_path, "input image")
    input_image = Image.open(input_path).convert("RGB")
    width, height = output_size(input_image, args.width, args.height)

    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=text_encoder_files),
            ModelConfig(
                path=str(
                    base / "transformer" / "diffusion_pytorch_model.safetensors"
                )
            ),
            ModelConfig(
                path=str(components / "vae" / "diffusion_pytorch_model.safetensors")
            ),
        ],
        tokenizer_config=ModelConfig(path=str(components / "tokenizer")),
    )
    template = TemplatePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(path=str(template_dir))],
    )

    # This is the official Template-only inference path: the image is supplied
    # as a KV condition, not as img2img input, so denoising_strength is omitted.
    image = template(
        pipe,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=height,
        width=width,
        seed=args.seed,
        rand_device=args.rand_device,
        cfg_scale=args.cfg_scale,
        num_inference_steps=args.num_inference_steps,
        template_inputs=[{"image": input_image, "prompt": args.prompt}],
        negative_template_inputs=[{"image": input_image, "prompt": ""}],
    )

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved enhanced image to {output_path} ({width}x{height}).")


if __name__ == "__main__":
    main()
