#!/usr/bin/env python3
"""Explain one prediction with a model-agnostic occlusion sensitivity heatmap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from hand_clip import CLASS_TO_IDX, IDX_TO_CLASS, resolve_device
from modeling import MLPHead


DEFAULT_CHECKPOINT = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/outputs/clip_vitl14_mlp/best_model.pt"
DEFAULT_OUTPUT_ROOT = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/outputs/explanations"
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Image to explain")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--clip-checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--patch-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--overlay-alpha", type=float, default=0.55)
    return parser.parse_args()


def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_models(args, device):
    state = safe_torch_load(Path(args.checkpoint).expanduser().resolve())
    clip_path = Path(args.clip_checkpoint or state["clip_checkpoint"]).expanduser().resolve()
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
    return clip_model, preprocess, head, clip_path


@torch.inference_mode()
def probabilities(clip_model, head, images):
    features = F.normalize(clip_model.encode_image(images).float(), dim=-1)
    return head(features).softmax(dim=-1)


def denormalize(tensor):
    pixels = tensor.cpu() * CLIP_STD + CLIP_MEAN
    pixels = pixels.clamp(0, 1).permute(1, 2, 0).numpy()
    return (pixels * 255).round().astype(np.uint8)


def colorize_and_overlay(base, heatmap, alpha):
    scale = float(np.max(np.abs(heatmap)))
    normalized = heatmap / scale if scale > 1e-12 else np.zeros_like(heatmap)
    color = np.zeros((*normalized.shape, 3), dtype=np.float32)
    color[..., 0] = np.clip(normalized, 0, 1) * 255
    color[..., 2] = np.clip(-normalized, 0, 1) * 255
    magnitude = np.abs(normalized)[..., None]
    overlay = base.astype(np.float32) * (1 - alpha * magnitude) + color * (alpha * magnitude)
    return color.astype(np.uint8), np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    if args.patch_size < 1 or args.stride < 1 or args.batch_size < 1:
        raise ValueError("patch-size, stride, and batch-size must be positive")
    if not 0 <= args.overlay_alpha <= 1:
        raise ValueError("overlay-alpha must be between 0 and 1")
    image_path = Path(args.input).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    device = resolve_device(args.device)
    clip_model, preprocess, head, clip_path = load_models(args, device)
    with Image.open(image_path) as image:
        base_tensor = preprocess(image.convert("RGB"))
    height, width = base_tensor.shape[-2:]
    if args.patch_size > min(height, width):
        raise ValueError("patch-size is larger than the preprocessed image")

    base_prob = probabilities(clip_model, head, base_tensor.unsqueeze(0).to(device))[0].cpu()
    bad_index = CLASS_TO_IDX["bad"]
    positions = []
    variants = []
    for y in range(0, height - args.patch_size + 1, args.stride):
        for x in range(0, width - args.patch_size + 1, args.stride):
            masked = base_tensor.clone()
            masked[:, y:y + args.patch_size, x:x + args.patch_size] = 0
            variants.append(masked)
            positions.append((x, y))

    masked_bad_probs = []
    for start in range(0, len(variants), args.batch_size):
        batch = torch.stack(variants[start:start + args.batch_size]).to(device)
        masked_bad_probs.extend(probabilities(clip_model, head, batch)[:, bad_index].cpu().tolist())

    heat_sum = np.zeros((height, width), dtype=np.float32)
    heat_count = np.zeros((height, width), dtype=np.float32)
    patches = []
    baseline_bad = float(base_prob[bad_index])
    for (x, y), masked_bad in zip(positions, masked_bad_probs):
        delta = baseline_bad - float(masked_bad)
        heat_sum[y:y + args.patch_size, x:x + args.patch_size] += delta
        heat_count[y:y + args.patch_size, x:x + args.patch_size] += 1
        patches.append({"x": x, "y": y, "delta_bad": delta, "masked_prob_bad": float(masked_bad)})
    heatmap = heat_sum / np.maximum(heat_count, 1)

    base_pixels = denormalize(base_tensor)
    heat_pixels, overlay_pixels = colorize_and_overlay(base_pixels, heatmap, args.overlay_alpha)
    output_dir = Path(args.output_dir or (Path(DEFAULT_OUTPUT_ROOT) / image_path.stem)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(base_pixels).save(output_dir / "original_clip_input.png")
    Image.fromarray(heat_pixels).save(output_dir / "heatmap.png")
    Image.fromarray(overlay_pixels).save(output_dir / "overlay.png")

    prediction_id = int(base_prob.argmax())
    payload = {
        "input": str(image_path),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "clip_checkpoint": str(clip_path),
        "prediction": IDX_TO_CLASS[prediction_id],
        "prob_good": float(base_prob[CLASS_TO_IDX["good"]]),
        "prob_bad": baseline_bad,
        "interpretation": "Red supports bad; blue supports good; black has little measured influence.",
        "patch_size": args.patch_size,
        "stride": args.stride,
        "heatmap_min": float(heatmap.min()),
        "heatmap_max": float(heatmap.max()),
        "top_bad_support": sorted(patches, key=lambda row: row["delta_bad"], reverse=True)[:10],
        "top_good_support": sorted(patches, key=lambda row: row["delta_bad"])[:10],
    }
    (output_dir / "explanation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: payload[key] for key in ["prediction", "prob_good", "prob_bad"]}, ensure_ascii=False))
    print("Saved explanation to", output_dir)


if __name__ == "__main__":
    main()
