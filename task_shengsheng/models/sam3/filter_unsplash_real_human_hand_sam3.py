#!/usr/bin/env python3
"""Filter real_human images that contain hands with SAM3.

Input:
  real_human/images/*.jpg
  real_human/labels/*.txt

Output when at least one "hand" object is found:
  real_human_hand/images/<same image name>
  real_human_hand/labels/<same stem>.txt

Output when no "hand" object is found:
  real_human_no_hand/images/<same image name>
  real_human_no_hand/labels/<same stem>.txt

The original label file is copied unchanged in both branches. SAM3 boxes are
only used as a presence signal and are written to the JSONL log for traceability.
"""

from __future__ import annotations

import argparse
import json
import shutil
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
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES)


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

    masks = state.get("masks")
    scores = state.get("scores")
    boxes = state.get("boxes")
    if masks is None or scores is None or boxes is None or masks.shape[0] == 0:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-image-dir",
        default="/mnt/image-edit-hdd/datasets/duanyufa/unsplash/清洗/real_human/images",
    )
    parser.add_argument(
        "--input-label-dir",
        default="/mnt/image-edit-hdd/datasets/duanyufa/unsplash/清洗/real_human/labels",
    )
    parser.add_argument(
        "--output-image-dir",
        default="/mnt/image-edit-hdd/datasets/duanyufa/unsplash/清洗/real_human_hand/images",
    )
    parser.add_argument(
        "--output-label-dir",
        default="/mnt/image-edit-hdd/datasets/duanyufa/unsplash/清洗/real_human_hand/labels",
    )
    parser.add_argument(
        "--no-hand-image-dir",
        default="/mnt/image-edit-hdd/datasets/duanyufa/unsplash/清洗/real_human_no_hand/images",
    )
    parser.add_argument(
        "--no-hand-label-dir",
        default="/mnt/image-edit-hdd/datasets/duanyufa/unsplash/清洗/real_human_no_hand/labels",
    )
    parser.add_argument("--prompt", default="hand")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-objects", type=int, default=0, help="0 means keep all detections.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per shard; 0 means all.")
    parser.add_argument("--log-jsonl", default="")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    input_image_dir = Path(args.input_image_dir)
    input_label_dir = Path(args.input_label_dir)
    output_image_dir = Path(args.output_image_dir)
    output_label_dir = Path(args.output_label_dir)
    no_hand_image_dir = Path(args.no_hand_image_dir)
    no_hand_label_dir = Path(args.no_hand_label_dir)
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    no_hand_image_dir.mkdir(parents=True, exist_ok=True)
    no_hand_label_dir.mkdir(parents=True, exist_ok=True)

    all_images = list_images(input_image_dir)
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

    hand_count = 0
    no_hand_count = 0
    skipped_count = 0
    missing_label_count = 0
    error_count = 0
    try:
        for index, image_path in enumerate(images, 1):
            src_label = input_label_dir / f"{image_path.stem}.txt"
            dst_image = output_image_dir / image_path.name
            dst_label = output_label_dir / src_label.name
            no_hand_dst_image = no_hand_image_dir / image_path.name
            no_hand_dst_label = no_hand_label_dir / src_label.name
            if args.resume and (
                (dst_image.exists() and dst_label.exists())
                or (no_hand_dst_image.exists() and no_hand_dst_label.exists())
            ):
                skipped_count += 1
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
                if boxes:
                    if not src_label.exists():
                        missing_label_count += 1
                        status = "hand_missing_label"
                    else:
                        shutil.copy2(image_path, dst_image)
                        shutil.copy2(src_label, dst_label)
                        hand_count += 1
                        status = "hand"
                else:
                    if not src_label.exists():
                        missing_label_count += 1
                        status = "no_hand_missing_label"
                    else:
                        shutil.copy2(image_path, no_hand_dst_image)
                        shutil.copy2(src_label, no_hand_dst_label)
                        no_hand_count += 1
                        status = "no_hand"

                record = {
                    "image": str(image_path),
                    "label": str(src_label),
                    "status": status,
                    "num_boxes": len(boxes),
                    "boxes": boxes,
                }
                if log_f:
                    log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_f.flush()
                print(
                    f"[{index}/{len(images)}] {image_path.name} status={status} "
                    f"boxes={len(boxes)}",
                    flush=True,
                )
            except Exception as exc:
                error_count += 1
                record = {"image": str(image_path), "status": "error", "error": repr(exc)}
                if log_f:
                    log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_f.flush()
                print(f"[{index}/{len(images)}] {image_path.name} ERROR {exc!r}", flush=True)
    finally:
        if log_f:
            log_f.close()

    print(
        "Done. "
        f"hand={hand_count} no_hand={no_hand_count} skipped={skipped_count} "
        f"missing_label={missing_label_count} errors={error_count}"
    )


if __name__ == "__main__":
    main()
