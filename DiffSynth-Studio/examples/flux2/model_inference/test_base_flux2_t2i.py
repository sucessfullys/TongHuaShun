import argparse
from pathlib import Path

import torch

from diffsynth.core import ModelConfig
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline


DEFAULT_BASE_MODEL = (
    "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B"
)
DEFAULT_OUTPUT = (
    "/mnt/image-edit/datasets/duanyufa/outputs/"
    "flux2_klein_base_4b_self_flow_sample250k/base_test.png"
)
DEFAULT_PROMPT = (
    '42'
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-to-image inference with FLUX.2-klein-base-4B."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--embedded_guidance", type=float, default=4.0)
    parser.add_argument(
        "--rand_device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device used to sample the initial noise.",
    )
    return parser.parse_args()


def validate_args(args):
    base_model = Path(args.base_model)
    required_paths = [
        base_model / "text_encoder",
        base_model / "transformer" / "diffusion_pytorch_model.safetensors",
        base_model / "vae" / "diffusion_pytorch_model.safetensors",
        base_model / "tokenizer",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("height and width must be divisible by 16.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLUX.2-klein-base-4B inference.")


def load_pipeline(args):
    base_model = Path(args.base_model)
    text_encoder_files = sorted(
        str(path) for path in (base_model / "text_encoder").glob("*.safetensors")
    )
    if not text_encoder_files:
        raise FileNotFoundError(
            f"No text encoder weights found under {base_model / 'text_encoder'}"
        )

    print(f"Loading FLUX.2-klein-base-4B model from {base_model}")
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=text_encoder_files),
            ModelConfig(
                path=str(
                    base_model
                    / "transformer"
                    / "diffusion_pytorch_model.safetensors"
                )
            ),
            ModelConfig(
                path=str(
                    base_model / "vae" / "diffusion_pytorch_model.safetensors"
                )
            ),
        ],
        tokenizer_config=ModelConfig(path=str(base_model / "tokenizer")),
    )
    pipe.dit.eval()
    return pipe


def main():
    args = parse_args()
    validate_args(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args)
    print(
        f"Generating FLUX.2-klein-base-4B image: "
        f"{args.width}x{args.height}, steps={args.num_inference_steps}, "
        f"seed={args.seed}, cfg_scale={args.cfg_scale}, "
        f"embedded_guidance={args.embedded_guidance}"
    )
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        height=args.height,
        width=args.width,
        seed=args.seed,
        rand_device=args.rand_device,
        num_inference_steps=args.num_inference_steps,
    )
    image.save(output_path)
    print(f"Saved image to {output_path}")


if __name__ == "__main__":
    main()
