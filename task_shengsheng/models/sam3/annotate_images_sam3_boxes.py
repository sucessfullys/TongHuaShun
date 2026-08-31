#!/usr/bin/env python3
"""Annotate images with SAM3 text-prompt boxes.

The output label format is one object per line:

    x1 y1 x2 y2 score

Coordinates are pixel-space boxes on the original image. If no object is found,
an empty same-stem .txt file is written by default so images and labels remain
one-to-one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


HERE = Path(__file__).resolve().parent
BPE_PATH = Path(sam3.__file__).resolve().parent / "assets/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_CHECKPOINT_PATH = HERE / "hf_sam3.1" / "sam3.1_multiplex.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG"}


def list_images(input_dir: Path) -> list[Path]:
    return sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix in IMAGE_SUFFIXES)


def shard_items(items: list[Path], shard_index: int, shard_count: int) -> list[Path]:
    return [item for i, item in enumerate(items) if i % shard_count == shard_index]


def build_model(device: str, checkpoint_path: Path):
    print(f"Loading SAM3 model on {device}...")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = build_sam3_image_model(
        bpe_path=str(BPE_PATH),
        checkpoint_path=str(checkpoint_path),
        load_from_HF=False,
        device="cpu",
    )
    model = model.to(torch.device(device))
    model.eval()
    print(f"Model ready on {device}.")
    return model


def detect_boxes(
    model,
    image_path: Path,
    prompt: str,
    device: str,
    threshold: float,
    max_objects: int,
) -> list[tuple[float, float, float, float, float]]:
    img = Image.open(image_path).convert("RGB")
    processor = Sam3Processor(model, device=device, confidence_threshold=threshold)
    use_cuda_amp = str(device).startswith("cuda")
    with torch.inference_mode(), torch.autocast(
        device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda_amp
    ):
        state = processor.set_image(img)
        processor.reset_all_prompts(state)
        state = processor.set_text_prompt(state=state, prompt=prompt)

    scores = state.get("scores")
    boxes = state.get("boxes")
    if scores is None or boxes is None or boxes.shape[0] == 0:
        return []

    score_values = scores.detach().float().cpu().tolist()
    box_values = boxes.detach().float().cpu().tolist()
    ranked = sorted(range(len(score_values)), key=lambda i: score_values[i], reverse=True)
    if max_objects > 0:
        ranked = ranked[:max_objects]

    width, height = img.size
    results: list[tuple[float, float, float, float, float]] = []
    for i in ranked:
        x1, y1, x2, y2 = box_values[i]
        x1 = min(max(float(x1), 0.0), float(width))
        y1 = min(max(float(y1), 0.0), float(height))
        x2 = min(max(float(x2), 0.0), float(width))
        y2 = min(max(float(y2), 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            continue
        results.append((x1, y1, x2, y2, float(score_values[i])))
    return results


def write_label(path: Path, boxes: list[tuple[float, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for x1, y1, x2, y2, score in boxes:
            f.write(f"{x1:.3f} {y1:.3f} {x2:.3f} {y2:.3f} {score:.6f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--prompt", default="person")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-objects", type=int, default=0, help="0 means keep all objects.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip images whose label already exists.")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per shard; 0 means all.")
    parser.add_argument("--log-jsonl", default="")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    input_dir = Path(args.input_dir)
    label_dir = Path(args.label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    all_images = list_images(input_dir)
    images = shard_items(all_images, args.shard_index, args.shard_count)
    if args.limit > 0:
        images = images[: args.limit]
    print(
        f"Shard {args.shard_index}/{args.shard_count}: {len(images)} images "
        f"from {len(all_images)} total images."
    )

    model = build_model(args.device, Path(args.checkpoint_path))
    log_path = Path(args.log_jsonl) if args.log_jsonl else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("a") if log_path else None

    annotated = 0
    skipped = 0
    errors = 0
    try:
        for index, image_path in enumerate(images, 1):
            label_path = label_dir / f"{image_path.stem}.txt"
            if args.resume and label_path.exists():
                skipped += 1
                continue
            try:
                boxes = detect_boxes(
                    model=model,
                    image_path=image_path,
                    prompt=args.prompt,
                    device=args.device,
                    threshold=args.threshold,
                    max_objects=args.max_objects,
                )
                write_label(label_path, boxes)
                annotated += 1
                record = {
                    "image": str(image_path),
                    "label": str(label_path),
                    "prompt": args.prompt,
                    "status": "found" if boxes else "not_found",
                    "num_boxes": len(boxes),
                    "boxes": boxes,
                }
                if log_f:
                    log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_f.flush()
                print(
                    f"[{index}/{len(images)}] {image_path.name} boxes={len(boxes)}",
                    flush=True,
                )
            except Exception as exc:
                errors += 1
                record = {"image": str(image_path), "status": "error", "error": repr(exc)}
                if log_f:
                    log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_f.flush()
                print(f"[{index}/{len(images)}] {image_path.name} ERROR {exc!r}", flush=True)
    finally:
        if log_f:
            log_f.close()

    print(f"Done. annotated={annotated} skipped={skipped} errors={errors}")


if __name__ == "__main__":
    main()
