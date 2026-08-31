#!/usr/bin/env python3
"""Run good/bad hand classification on one image or an image directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from hand_clip import CLASS_TO_IDX, IDX_TO_CLASS, IMAGE_EXTENSIONS, resolve_device
from modeling import MLPHead


DEFAULT_OUTPUT = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/outputs/inference_predictions.jsonl"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--checkpoint", required=True, help="best_model.pt from train.py")
    parser.add_argument("--clip-checkpoint", default=None, help="Override CLIP path stored in classifier checkpoint")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bad-threshold", type=float, default=0.5)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def discover_images(input_path: str, recursive: bool) -> list[Path]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    iterator = path.rglob("*") if recursive else path.glob("*")
    images = sorted(item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No supported images found under {path}")
    return images


class InferenceDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, str(path)


@torch.inference_mode()
def predict(args):
    if not 0 <= args.bad_threshold <= 1:
        raise ValueError("--bad-threshold must be between 0 and 1")
    device = resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
    state = safe_torch_load(checkpoint_path)
    if state.get("class_to_idx") != CLASS_TO_IDX:
        raise ValueError(f"Unexpected class mapping in checkpoint: {state.get('class_to_idx')}")

    clip_path = Path(args.clip_checkpoint or state["clip_checkpoint"]).expanduser().resolve()
    if not clip_path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint not found: {clip_path}")
    print(f"Loading CLIP: {clip_path}")
    clip_model, preprocess = clip.load(str(clip_path), device=device, jit=False)
    clip_model.eval().requires_grad_(False)
    head = MLPHead(
        state["feature_dim"],
        state["hidden_dim"],
        state["bottleneck_dim"],
        state["dropout"],
    ).to(device)
    head.load_state_dict(state["head_state_dict"])
    head.eval()

    paths = discover_images(args.input, args.recursive)
    loader = DataLoader(
        InferenceDataset(paths, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    rows = []
    for images, batch_paths in loader:
        images = images.to(device, non_blocking=True)
        features = F.normalize(clip_model.encode_image(images).float(), dim=-1)
        probabilities = head(features).softmax(dim=-1).cpu()
        for path, probability in zip(batch_paths, probabilities):
            prob_good = float(probability[CLASS_TO_IDX["good"]])
            prob_bad = float(probability[CLASS_TO_IDX["bad"]])
            prediction = CLASS_TO_IDX["bad"] if prob_bad >= args.bad_threshold else CLASS_TO_IDX["good"]
            rows.append({
                "path": path,
                "prediction": IDX_TO_CLASS[prediction],
                "prediction_id": prediction,
                "prob_good": prob_good,
                "prob_bad": prob_bad,
                "bad_threshold": args.bad_threshold,
            })

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for row in rows:
        print(f"{row['prediction']:>4}  bad={row['prob_bad']:.4f}  {row['path']}")
    print(f"Saved {len(rows)} predictions to {output}")


def main():
    args = parse_args()
    predict(args)


if __name__ == "__main__":
    main()
