#!/usr/bin/env python3
"""Compute ArcFace cosine similarity between two face images."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffsynth.diffusion.loss import FaceIdentityHelper  # noqa: E402


DEFAULT_ARCFACE_CKPT = "/mnt/data/image-edit/models/arcface/weights/arcface-r100-glint360k.pth"
DEFAULT_INSIGHTFACE_ROOT = "/mnt/data/image-edit/datasets/shensheng/models/insightface"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ArcFace cosine similarity for two images.",
    )
    parser.add_argument("image_a", help="Path to the first image.")
    parser.add_argument("image_b", help="Path to the second image.")
    parser.add_argument(
        "--arcface_ckpt",
        default=DEFAULT_ARCFACE_CKPT,
        help=f"ArcFace PyTorch checkpoint path. Default: {DEFAULT_ARCFACE_CKPT}",
    )
    parser.add_argument(
        "--insightface_root",
        default=DEFAULT_INSIGHTFACE_ROOT,
        help=f"InsightFace model root. Default: {DEFAULT_INSIGHTFACE_ROOT}",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for ArcFace inference. Default: cuda if available, else cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default=None,
        help="Model dtype. Default: bfloat16 on cuda, float32 on cpu.",
    )
    parser.add_argument(
        "--det_size",
        type=int,
        default=640,
        help="InsightFace detector input size. Default: 640.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name, device):
    if dtype_name is None:
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype_name]


def load_image(path):
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    return Image.open(path).convert("RGB")


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype, device)

    helper = FaceIdentityHelper(
        arcface_ckpt_path=args.arcface_ckpt,
        insightface_root=args.insightface_root,
        det_size=(args.det_size, args.det_size),
    ).to(device=device, dtype=dtype)

    image_a = load_image(args.image_a)
    image_b = load_image(args.image_b)

    emb_a = helper.get_embedding(image_a)
    emb_b = helper.get_embedding(image_b)
    if emb_a is None:
        raise RuntimeError(f"No usable frontal face detected in image_a: {args.image_a}")
    if emb_b is None:
        raise RuntimeError(f"No usable frontal face detected in image_b: {args.image_b}")

    sim = float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8))
    print(f"arcface_cosine_similarity: {sim:.6f}")


if __name__ == "__main__":
    main()
