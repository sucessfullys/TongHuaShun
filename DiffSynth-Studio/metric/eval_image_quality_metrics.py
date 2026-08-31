#!/usr/bin/env python3
"""Evaluate PSNR, SSIM, LPIPS, MUSIQ, and MANIQA for image folders.

Examples:
    python metric/eval_image_quality_metrics.py \
        --pred_dir outputs/enhanced \
        --gt_dir data/gt \
        --output metric/results.csv

    python metric/eval_image_quality_metrics.py --download_only --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
METRIC_NAMES = ("psnr", "ssim", "lpips", "musiq", "maniqa")
REFERENCE_METRICS = ("psnr", "ssim", "lpips")
NO_REFERENCE_METRICS = ("musiq", "maniqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PSNR, SSIM, LPIPS, MUSIQ, and MANIQA on image folders.")
    parser.add_argument("--pred_dir", type=Path, help="Folder containing enhanced/restored images.")
    parser.add_argument("--gt_dir", type=Path, default=None, help="Folder containing GT images for PSNR/SSIM.")
    parser.add_argument("--output", type=Path, default=Path("metric/results.csv"), help="CSV detail output path.")
    parser.add_argument("--summary", type=Path, default=None, help="Optional JSON summary output path.")
    parser.add_argument("--txt_output", type=Path, default=None, help="TXT output path for mean metrics.")
    parser.add_argument("--device", default="cuda:0", help="Device for pyiqa metrics, e.g. cuda:0 or cpu.")
    parser.add_argument("--recursive", action="store_true", help="Scan folders recursively and match relative paths.")
    parser.add_argument(
        "--resize",
        choices=("none", "pred_to_gt", "gt_to_pred"),
        default="none",
        help="How to handle mismatched pred/GT sizes for PSNR/SSIM.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N matched predictions.")
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Instantiate metrics once to trigger pyiqa weight downloads, then exit.",
    )
    return parser.parse_args()


def require_pyiqa():
    try:
        import pyiqa  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyiqa. Install it in the active environment with:\n"
            "  pip install pyiqa\n"
            "Then rerun this script."
        ) from exc
    return pyiqa


def iter_images(root: Path, recursive: bool) -> Iterable[Path]:
    pattern = "**/*" if recursive else "*"
    for path in sorted(root.glob(pattern)):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def relative_key(path: Path, root: Path, recursive: bool) -> str:
    if recursive:
        return path.relative_to(root).as_posix()
    return path.name


def load_rgb_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = pil_to_tensor(image).float().div(255.0)
    return tensor.unsqueeze(0)


def maybe_resize(pred: torch.Tensor, gt: torch.Tensor, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if pred.shape[-2:] == gt.shape[-2:]:
        return pred, gt
    if mode == "none":
        raise ValueError(f"size mismatch: pred={tuple(pred.shape[-2:])}, gt={tuple(gt.shape[-2:])}")

    import torch.nn.functional as F

    if mode == "pred_to_gt":
        pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    elif mode == "gt_to_pred":
        gt = F.interpolate(gt, size=pred.shape[-2:], mode="bilinear", align_corners=False)
    return pred, gt


def create_metrics(device: str) -> Dict[str, object]:
    pyiqa = require_pyiqa()
    metrics = {}
    for name in METRIC_NAMES:
        metrics[name] = pyiqa.create_metric(name, device=device)
    return metrics


def score_metric(metric: object, *inputs: torch.Tensor) -> float:
    with torch.inference_mode():
        value = metric(*inputs)  # type: ignore[misc]
    if isinstance(value, torch.Tensor):
        value = value.detach().float().mean().cpu().item()
    return float(value)


def format_metric(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isfinite(value):
            return f"{value:.6f}"
        return "NA"
    return str(value)


def finite_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def main() -> None:
    args = parse_args()
    metrics = create_metrics(args.device)

    if args.download_only:
        print("Metrics initialized. If pyiqa needed model weights, they have been downloaded/cached.")
        return

    if args.pred_dir is None:
        raise SystemExit("--pred_dir is required unless --download_only is used.")
    if not args.pred_dir.is_dir():
        raise SystemExit(f"pred_dir does not exist or is not a directory: {args.pred_dir}")
    if args.gt_dir is not None and not args.gt_dir.is_dir():
        raise SystemExit(f"gt_dir does not exist or is not a directory: {args.gt_dir}")

    pred_paths = list(iter_images(args.pred_dir, args.recursive))
    if args.limit is not None:
        pred_paths = pred_paths[: args.limit]
    if not pred_paths:
        raise SystemExit(f"No images found in pred_dir: {args.pred_dir}")

    gt_map: Dict[str, Path] = {}
    if args.gt_dir is not None:
        gt_map = {relative_key(p, args.gt_dir, args.recursive): p for p in iter_images(args.gt_dir, args.recursive)}

    rows: List[Dict[str, object]] = []
    skipped_pairs = 0
    for pred_path in pred_paths:
        key = relative_key(pred_path, args.pred_dir, args.recursive)
        row: Dict[str, object] = {"file": key, "pred_path": str(pred_path), "gt_path": ""}
        pred = load_rgb_tensor(pred_path).to(args.device)

        for name in NO_REFERENCE_METRICS:
            row[name] = score_metric(metrics[name], pred)

        gt_path = gt_map.get(key)
        if gt_path is None:
            for name in REFERENCE_METRICS:
                row[name] = None
            if args.gt_dir is not None:
                skipped_pairs += 1
        else:
            row["gt_path"] = str(gt_path)
            gt = load_rgb_tensor(gt_path).to(args.device)
            try:
                pred_pair, gt_pair = maybe_resize(pred, gt, args.resize)
                for name in REFERENCE_METRICS:
                    row[name] = score_metric(metrics[name], pred_pair, gt_pair)
            except ValueError as exc:
                for name in REFERENCE_METRICS:
                    row[name] = None
                row["pair_error"] = str(exc)
                skipped_pairs += 1

        rows.append(row)
        print(
            f"{key}: "
            f"PSNR={format_metric(row['psnr'])} "
            f"SSIM={format_metric(row['ssim'])} "
            f"LPIPS={format_metric(row['lpips'])} "
            f"MUSIQ={format_metric(row['musiq'])} "
            f"MANIQA={format_metric(row['maniqa'])}",
            flush=True,
        )

    means = {name: finite_mean(row.get(name) for row in rows) for name in METRIC_NAMES}
    summary = {
        "pred_dir": str(args.pred_dir),
        "gt_dir": str(args.gt_dir) if args.gt_dir else None,
        "num_pred_images": len(pred_paths),
        "num_rows": len(rows),
        "skipped_pairs": skipped_pairs,
        "means": means,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "pred_path", "gt_path", "psnr", "ssim", "lpips", "musiq", "maniqa", "pair_error"]
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    txt_path = args.txt_output or args.output.with_suffix(".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_lines = [
        "Image Quality Metrics",
        f"pred_dir: {args.pred_dir}",
        f"gt_dir: {args.gt_dir if args.gt_dir else 'None'}",
        f"num_pred_images: {len(pred_paths)}",
        f"num_rows: {len(rows)}",
        f"skipped_pairs: {skipped_pairs}",
        "",
        "Mean metrics:",
    ]
    for name in METRIC_NAMES:
        txt_lines.append(f"{name.upper()}: {format_metric(means[name])}")
    txt_lines.extend(
        [
            "",
            "Notes:",
            "PSNR, SSIM, and LPIPS require GT images.",
            "MUSIQ and MANIQA are no-reference metrics and only use predicted images.",
            "For LPIPS, lower is better. For PSNR/SSIM/MUSIQ/MANIQA, higher is generally better.",
            "",
        ]
    )
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    print("\nMean metrics:")
    for name in METRIC_NAMES:
        print(f"  {name}: {format_metric(means[name])}")
    print(f"\nSaved CSV: {args.output}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved TXT: {txt_path}", flush=True)


if __name__ == "__main__":
    main()
