import argparse
from pathlib import Path

import torch

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline


DEFAULT_BASE_MODEL = (
    "/mnt/image-edit/datasets/dingbaojin/models/"
    "black-forest-labs/FLUX.2-klein-4B"
)
DEFAULT_CHECKPOINT = (
    "/mnt/image-edit/datasets/duanyufa/outputs/"
    "flux2_klein_4b_self_flow_sample250k/checkpoint-7813/student.safetensors"
)
DEFAULT_OUTPUT = (
    "/mnt/image-edit/datasets/duanyufa/outputs/"
    "flux2_klein_4b_self_flow_sample250k/test_t2i_1.png"
)
# DEFAULT_PROMPT = (
#     "A high-quality full-body fashion portrait of a young woman standing in "
#     "a sunlit city street, wearing a detailed blue dress, natural skin texture, "
#     "realistic photography, balanced composition, sharp focus."
# )
DEFAULT_PROMPT = (
    "A studio product shot focuses on the lower half of a person wearing black "
    "athletic joggers. The pants feature a drawstring waist, zippered side "
    "pockets, and a prominent vertical white text graphic reading \"EASI HORSE\" "
    "running down the left leg. The person is standing in a neutral pose "
    "against a plain, light-grey studio background. The lighting is bright and "
    "artificial, creating soft shadows that highlight the texture and fit of "
    "the synthetic fabric."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Text-to-image inference with a trained FLUX.2 Self-Flow student."
    )
    parser.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--embedded_guidance", type=float, default=1.0)
    parser.add_argument(
        "--rand_device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device used to sample the initial noise.",
    )
    return parser.parse_args()


def validate_args(args):
    base_model = Path(args.base_model)
    checkpoint = Path(args.checkpoint)
    required_paths = [
        base_model / "text_encoder",
        base_model / "transformer" / "diffusion_pytorch_model.safetensors",
        base_model / "vae" / "diffusion_pytorch_model.safetensors",
        base_model / "tokenizer",
        checkpoint,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("height and width must be divisible by 16.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLUX.2-klein-4B inference.")


def load_pipeline(args):
    base_model = Path(args.base_model)
    text_encoder_files = sorted(
        str(path) for path in (base_model / "text_encoder").glob("*.safetensors")
    )
    if not text_encoder_files:
        raise FileNotFoundError(
            f"No text encoder weights found under {base_model / 'text_encoder'}"
        )

    print(f"Loading base model from {base_model}")
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

    print(f"Loading trained student from {args.checkpoint}")
    state_dict = load_state_dict(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    incompatible = pipe.dit.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Student checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    del state_dict
    pipe.dit.eval()
    return pipe


def main():
    args = parse_args()
    validate_args(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = load_pipeline(args)
    print(
        f"Generating {args.width}x{args.height}, "
        f"steps={args.num_inference_steps}, seed={args.seed}"
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
