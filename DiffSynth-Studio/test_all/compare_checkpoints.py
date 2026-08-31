#!/usr/bin/env python3
"""Compare checkpoint epochs efficiently — load base models once, then swap checkpoints.

Usage:
  # Single image
  python test_all/compare_checkpoints.py \
    --images /mnt/image-edit/datasets/duanyufa/Real_RL_Data/real_rl/9.png \
    --checkpoint-dir outputs/Template-KleinBase4B-Deblur_full_1e5 \
    --epochs 0,1,2,3,4,5,6,7,8,9

  # Multiple images (comma-separated)
  python test_all/compare_checkpoints.py \
    --images img1.png,img2.png,img3.png \
    --checkpoint-dir outputs/Template-KleinBase4B-Deblur_full_1e5 \
    --epochs 0,5,9

  # From metadata (limit=N)
  python test_all/compare_checkpoints.py \
    --metadata /mnt/image-edit/datasets/duanyufa/Real_RL_Data/metadata.jsonl \
    --limit 5 \
    --checkpoint-dir outputs/Template-KleinBase4B-Deblur_full_1e5 \
    --epochs 0,5,9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from accelerate import Accelerator
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.diffusion.template import TemplatePipeline
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline

DEFAULT_BASE_MODEL = Path("/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B")
DEFAULT_TEMPLATE_MODEL = REPO_ROOT / "Template-KleinBase4B-Upscaler"

DEFAULT_PROMPT = (
    "Enhance image clarity and sharpness, recover fine details, and improve "
    "texture definition while preserving the original content exactly. "
    "Do not change the people, pose, clothing design, colors, proportions, "
    "composition, background, or layout. Do not add, remove, or invent any "
    "objects or details. No hallucinated content. Keep the image visually "
    "identical to the original, only clearer and sharper."
)


def parse_args():
    p = argparse.ArgumentParser(description="Compare checkpoint epochs on test images")
    # Image source (one of --images or --metadata required)
    p.add_argument("--images", type=str, default=None,
                   help="Direct image paths, comma-separated. Uses a default deblur prompt.")
    p.add_argument("--metadata", type=Path, default=None,
                   help="Metadata JSONL file. Uses per-record prompts.")
    p.add_argument("--limit", type=int, default=None,
                   help="Max records from metadata.")
    # Checkpoint
    p.add_argument("--checkpoint-dir", type=Path, required=True,
                   help="Directory containing epoch-*.safetensors files.")
    p.add_argument("--epochs", type=str, default="0,1,2,3,4,5,6,7,8,9",
                   help="Comma-separated epoch numbers, e.g. '0,5,9'.")
    # Models
    p.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    p.add_argument("--template-model", type=Path, default=DEFAULT_TEMPLATE_MODEL)
    # Inference
    p.add_argument("--output-dir", type=Path,
                   default=REPO_ROOT / "test_all")
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--embedded-guidance", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--prompt", type=str, default=None,
                   help="Custom prompt for --images mode. Uses default deblur prompt if not set.")
    return p.parse_args()


def require_file(path, desc):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {desc}: {path}")


def load_base_models(device, base_path, template_path, text_encoder_files):
    """Load FLUX.2 pipeline + Template *once*. Returns (pipe, template)."""
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[
            ModelConfig(path=text_encoder_files),
            ModelConfig(path=str(base_path / "transformer" / "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=str(base_path / "vae" / "diffusion_pytorch_model.safetensors")),
        ],
        tokenizer_config=ModelConfig(path=str(base_path / "tokenizer")),
    )
    template = TemplatePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=str(template_path))],
    )
    pipe.dit.eval()
    return pipe, template


def load_checkpoint_onto(template, checkpoint_path):
    """Load a fine-tuned checkpoint into the existing template model."""
    state_dict = load_state_dict(str(checkpoint_path), torch_dtype=torch.bfloat16)
    load_result = template.models[0].load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint_path}: "
            f"missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )
    template.eval()


def load_images_from_paths(paths, prompt):
    """Return list of {'image': PIL.Image, 'name': str, 'prompt': str, 'template_prompt': str}."""
    records = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        records.append({
            "image": img,
            "name": Path(p).stem + ".png",
            "prompt": prompt,
            "template_prompt": prompt,
            "original_size": img.size,
        })
    return records


def load_images_from_metadata(metadata_path, limit):
    """Return list compatible with load_images_from_paths."""
    require_file(metadata_path, "metadata")
    records = []
    with open(metadata_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            img_path = Path(r["template_inputs"]["image"]).expanduser().resolve()
            require_file(img_path, f"input image {img_path}")
            img = Image.open(img_path).convert("RGB")
            records.append({
                "image": img,
                "name": Path(r["image"]).stem + ".png",
                "prompt": r["prompt"],
                "template_prompt": r["template_inputs"]["prompt"],
                "original_size": img.size,
            })
            if limit and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"No records in {metadata_path}")
    return records


def pad_to_multiple(img, multiple=16):
    w, h = img.size
    pw, ph = (-w) % multiple, (-h) % multiple
    if pw == 0 and ph == 0:
        return img, (w, h)
    arr = np.asarray(img)
    mode = "reflect" if w > 1 and h > 1 else "edge"
    padded = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode=mode)
    return Image.fromarray(padded), (w, h)


def deterministic_seed(base_seed, name):
    digest = hashlib.sha256(f"{base_seed}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


@torch.inference_mode()
def infer_one(pipe, template, record, output_path, args):
    padded, orig_size = pad_to_multiple(record["image"])
    image = template(
        pipe,
        prompt=record["prompt"],
        negative_prompt="",
        height=padded.height,
        width=padded.width,
        seed=deterministic_seed(args.seed, record["name"]),
        rand_device="cpu",
        cfg_scale=args.cfg_scale,
        embedded_guidance=args.embedded_guidance,
        num_inference_steps=args.num_inference_steps,
        template_inputs=[{"image": padded, "prompt": record["template_prompt"]}],
        negative_template_inputs=[{"image": padded, "prompt": ""}],
        progress_bar_cmd=lambda v: v,
    )
    image = image.crop((0, 0, orig_size[0], orig_size[1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")


def main():
    args = parse_args()

    # Validate image source
    if args.images:
        paths = [Path(p.strip()).expanduser().resolve() for p in args.images.split(",")]
        for p in paths:
            require_file(p, f"input image {p}")
        records = load_images_from_paths(paths, args.prompt or DEFAULT_PROMPT)
    elif args.metadata:
        records = load_images_from_metadata(args.metadata.expanduser().resolve(), args.limit)
    else:
        raise ValueError("Either --images or --metadata is required.")

    epochs = [int(e.strip()) for e in args.epochs.split(",")]
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    base_path = args.base_model.expanduser().resolve()
    template_path = args.template_model.expanduser().resolve()
    output_base = args.output_dir.expanduser().resolve()

    # Validate models
    text_encoder = sorted((base_path / "text_encoder").glob("*.safetensors"))
    if not text_encoder:
        raise FileNotFoundError(f"No text-encoder in {base_path / 'text_encoder'}")
    require_file(template_path / "model.py", "Template model.py")

    # Checkpoints
    ckpt_paths = {}
    for e in epochs:
        ckpt = checkpoint_dir / f"epoch-{e}.safetensors"
        require_file(ckpt, f"epoch-{e} checkpoint")
        ckpt_paths[e] = ckpt

    print(f"Images: {len(records)}")
    print(f"Epochs: {epochs}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Output: {output_base}")
    print()

    accelerator = Accelerator()
    device = accelerator.device

    # ---- Load base models ONCE ----
    if accelerator.is_main_process:
        print("Loading base models (once)...", flush=True)
    pipe, template = load_base_models(
        device, base_path, template_path,
        [str(p) for p in text_encoder],
    )
    if accelerator.is_main_process:
        print("Base models loaded.", flush=True)

    # ---- Iterate epochs ----
    for epoch in epochs:
        if accelerator.is_main_process:
            print(f"\n--- Epoch {epoch} ---", flush=True)

        # Load checkpoint
        load_checkpoint_onto(template, ckpt_paths[epoch])

        out_dir = output_base / f"epoch-{epoch}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Shard images across GPUs
        shard = records[accelerator.process_index :: accelerator.num_processes]
        iterator = tqdm(shard, desc=f"epoch-{epoch} rank{accelerator.process_index}",
                        disable=not accelerator.is_main_process)

        for rec in iterator:
            out_path = out_dir / rec["name"]
            if out_path.exists():
                out_path.unlink()
            infer_one(pipe, template, rec, out_path, args)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print(f"\nDone. Results: {output_base}")
        for e in epochs:
            d = output_base / f"epoch-{e}"
            n = len(list(d.glob("*.png"))) if d.is_dir() else 0
            print(f"  epoch-{e}: {n} images")


if __name__ == "__main__":
    main()
