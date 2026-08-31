#!/usr/bin/env python3
"""Batch SAM3 text-prompt segmentation, rendering only the best mask per image."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


HERE = Path(__file__).resolve().parent
BPE_PATH = Path(sam3.__file__).resolve().parent / "assets/bpe_simple_vocab_16e6.txt.gz"
DEFAULT_CHECKPOINT_PATH = HERE / "hf_sam3.1" / "sam3.1_multiplex.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="hand")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Debug limit per shard; 0 means all.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--log-jsonl", default="")
    return parser.parse_args()


def list_images(input_dir: Path, recursive: bool) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(p for p in iterator if p.is_file() and p.suffix in IMAGE_SUFFIXES)


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


def render_best_mask(
    model,
    image_path: Path,
    prompt: str,
    device: str,
    threshold: float,
    alpha: float,
) -> tuple[Image.Image | None, dict]:
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
    if masks is None or scores is None or masks.shape[0] == 0:
        return None, {"num_objects": 0, "score": None, "box": None}

    best_idx = int(torch.argmax(scores).item())
    mask = masks[best_idx].squeeze().detach().cpu().numpy().astype(bool)
    score = float(scores[best_idx].detach().float().cpu().item())
    box = None
    if boxes is not None:
        box = [round(float(x), 3) for x in boxes[best_idx].detach().float().cpu().tolist()]

    img_np = np.array(img).astype(np.float32)
    if mask.shape[:2] != img_np.shape[:2]:
        mask = np.array(
            Image.fromarray(mask.astype(np.uint8) * 255).resize(
                (img.width, img.height), Image.NEAREST
            )
        ) > 128

    alpha = min(max(alpha, 0.0), 1.0)
    overlay = img_np.copy()
    overlay[mask] = overlay[mask] * (1.0 - alpha) + np.array([255, 0, 0]) * alpha
    result = Image.fromarray(overlay.astype(np.uint8))
    meta = {"num_objects": int(masks.shape[0]), "score": score, "box": box}
    return result, meta


def output_path_for(image_path: Path, input_dir: Path, output_dir: Path) -> Path:
    rel = image_path.relative_to(input_dir)
    return (output_dir / rel).with_suffix(".png")


def main() -> None:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    all_images = list_images(input_dir, args.recursive)
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
    log_f = log_path.open("a", encoding="utf-8") if log_path else None

    rendered = 0
    no_object = 0
    skipped = 0
    errors = 0
    try:
        for idx, image_path in enumerate(images, 1):
            out_path = output_path_for(image_path, input_dir, output_dir)
            if args.resume and out_path.exists():
                skipped += 1
                continue

            record = {
                "image": str(image_path),
                "output": str(out_path),
                "prompt": args.prompt,
                "status": None,
                "meta": None,
            }
            try:
                result, meta = render_best_mask(
                    model=model,
                    image_path=image_path,
                    prompt=args.prompt,
                    device=args.device,
                    threshold=args.threshold,
                    alpha=args.alpha,
                )
                record["meta"] = meta
                if result is None:
                    no_object += 1
                    record["status"] = "no_object"
                else:
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = out_path.with_suffix(".tmp.png")
                    result.save(tmp_path)
                    tmp_path.replace(out_path)
                    rendered += 1
                    record["status"] = "rendered"
                print(
                    f"[{idx}/{len(images)}] {image_path.name} "
                    f"status={record['status']} meta={meta}",
                    flush=True,
                )
            except Exception as exc:
                errors += 1
                record["status"] = "error"
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc()
                print(f"[{idx}/{len(images)}] {image_path.name} ERROR {exc!r}", flush=True)

            if log_f:
                log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_f.flush()
    finally:
        if log_f:
            log_f.close()

    print(
        f"Done. rendered={rendered} no_object={no_object} "
        f"skipped={skipped} errors={errors}"
    )


if __name__ == "__main__":
    main()
