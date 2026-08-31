#!/usr/bin/env python3
"""Visualize cached CLIP embeddings with dependency-free PCA and a PNG scatter plot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from hand_clip import IDX_TO_CLASS


DEFAULT_CACHE = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/outputs/clip_vitl14_mlp/feature_cache.pt"
DEFAULT_OUTPUT = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/outputs/clip_vitl14_mlp/feature_pca.png"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", default=DEFAULT_CACHE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def safe_torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unique_image_features(cache):
    records = defaultdict(list)
    metadata = {}
    for split_name, split in cache["splits"].items():
        for feature, label, path, group in zip(
            split["features"], split["labels"], split["paths"], split["groups"]
        ):
            records[path].append(feature.float())
            metadata[path] = (int(label), group, split_name)
    paths = sorted(records)
    features = torch.stack([torch.stack(records[path]).mean(0) for path in paths])
    features = torch.nn.functional.normalize(features, dim=-1)
    return paths, features, [metadata[path] for path in paths]


def project_pca(features):
    centered = features - features.mean(0, keepdim=True)
    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    coordinates = centered @ vh[:2].T
    variance = singular_values.square()
    ratios = variance[:2] / variance.sum().clamp_min(1e-12)
    return coordinates.numpy(), ratios.numpy()


def draw_point(draw, x, y, label, split):
    color = (45, 110, 220) if label == 0 else (220, 65, 55)
    radius = 7
    if split == "train":
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black", width=1)
    elif split == "val":
        draw.rectangle((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black", width=1)
    else:
        draw.polygon([(x, y - radius - 2), (x - radius - 1, y + radius), (x + radius + 1, y + radius)], fill=color, outline="black")


def main():
    args = parse_args()
    cache_path = Path(args.feature_cache).expanduser().resolve()
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)
    cache = safe_torch_load(cache_path)
    paths, features, metadata = unique_image_features(cache)
    coordinates, ratios = project_pca(features)

    width, height, margin = 1200, 900, 100
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    mins, maxs = coordinates.min(0), coordinates.max(0)
    spans = np.maximum(maxs - mins, 1e-8)
    scaled = (coordinates - mins) / spans
    xs = margin + scaled[:, 0] * (width - 2 * margin)
    ys = height - margin - scaled[:, 1] * (height - 2 * margin)
    draw.text((30, 25), f"CLIP feature PCA: {len(paths)} unique images", fill="black")
    draw.text((30, 48), f"PC1 variance={ratios[0]:.1%}, PC2 variance={ratios[1]:.1%}", fill="black")
    draw.text((30, 70), "Blue=good, Red=bad; circle=train, square=val, triangle=test", fill="black")
    rows = []
    for path, (label, group, split), raw_xy, x, y in zip(paths, metadata, coordinates, xs, ys):
        draw_point(draw, float(x), float(y), label, split)
        rows.append({
            "path": path,
            "group": group,
            "class_name": IDX_TO_CLASS[label],
            "label": label,
            "split": split,
            "pc1": float(raw_xy[0]),
            "pc2": float(raw_xy[1]),
        })
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    output.with_suffix(".json").write_text(
        json.dumps({"explained_variance_ratio": ratios.tolist(), "samples": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Saved", output)
    print("Saved", output.with_suffix(".json"))


if __name__ == "__main__":
    main()
