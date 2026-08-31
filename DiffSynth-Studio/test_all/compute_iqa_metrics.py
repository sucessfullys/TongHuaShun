#!/usr/bin/env python3
"""Compute no-reference IQA metrics for LR and enhanced images.

Metrics:
  - NIQE: lower is better
  - BRISQUE: lower is better
  - MUSIQ: higher is better

The script can pair images in two ways:

1. Metadata mode, recommended for this repo's test_all outputs:
   metadata.template_inputs.image points to LR input, and metadata.image is the
   relative output path under --enhanced-dir.

2. Directory mode:
   pair files by the same relative path from --lr-dir to --enhanced-dir. If that
   fails, pair by basename when unique.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_METRICS = ("niqe", "brisque", "musiq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute NIQE/BRISQUE/MUSIQ for paired LR and enhanced images.")
    parser.add_argument("--metadata", type=Path, default=None, help="JSONL metadata with template_inputs.image and image.")
    parser.add_argument("--lr-dir", type=Path, default=None, help="LR image directory for directory pairing mode.")
    parser.add_argument("--enhanced-dir", type=Path, required=True, help="Enhanced output image directory.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write metrics.csv and summary.json.")
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Comma-separated metrics: niqe,brisque,musiq")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, or cpu.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan dirs in directory pairing mode.")
    parser.add_argument("--allow-missing", action="store_true", help="Skip missing enhanced images instead of failing.")
    return parser.parse_args()


def image_files(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(p for p in root.glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_pairs_from_metadata(metadata: Path, enhanced_dir: Path, allow_missing: bool) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    missing: list[str] = []
    with metadata.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            lr_value = item.get("template_inputs", {}).get("image")
            out_value = item.get("image")
            if not lr_value or not out_value:
                raise ValueError(f"{metadata}:{line_no}: missing template_inputs.image or image")
            lr_path = Path(lr_value)
            enhanced_path = Path(out_value)
            if not enhanced_path.is_absolute():
                enhanced_path = enhanced_dir / enhanced_path
            if not enhanced_path.is_file():
                missing.append(str(enhanced_path))
                if allow_missing:
                    continue
            pairs.append(
                {
                    "id": str(out_value),
                    "group": str(Path(out_value).parent) if Path(out_value).parent != Path(".") else "",
                    "lr": str(lr_path),
                    "enhanced": str(enhanced_path),
                }
            )
    if missing and not allow_missing:
        raise FileNotFoundError(f"Missing {len(missing)} enhanced images, examples: {missing[:5]}")
    return pairs


def load_pairs_from_dirs(lr_dir: Path, enhanced_dir: Path, recursive: bool) -> list[dict[str, Any]]:
    lr_files = image_files(lr_dir, recursive)
    enhanced_files = image_files(enhanced_dir, recursive)
    enhanced_by_rel = {str(p.relative_to(enhanced_dir)): p for p in enhanced_files}

    basename_map: dict[str, list[Path]] = defaultdict(list)
    for p in enhanced_files:
        basename_map[p.name].append(p)

    pairs: list[dict[str, Any]] = []
    missing: list[str] = []
    ambiguous: list[str] = []
    for lr_path in lr_files:
        rel = str(lr_path.relative_to(lr_dir))
        enhanced_path = enhanced_by_rel.get(rel)
        if enhanced_path is None:
            candidates = basename_map.get(lr_path.name, [])
            if len(candidates) == 1:
                enhanced_path = candidates[0]
            elif len(candidates) > 1:
                ambiguous.append(str(lr_path))
                continue
        if enhanced_path is None:
            missing.append(str(lr_path))
            continue
        pairs.append(
            {
                "id": rel,
                "group": str(Path(rel).parent) if Path(rel).parent != Path(".") else "",
                "lr": str(lr_path),
                "enhanced": str(enhanced_path),
            }
        )
    if missing:
        print(f"[warn] missing enhanced pairs: {len(missing)}, examples={missing[:5]}", file=sys.stderr)
    if ambiguous:
        print(f"[warn] ambiguous basename pairs skipped: {len(ambiguous)}, examples={ambiguous[:5]}", file=sys.stderr)
    return pairs


def load_rgb_tensor(path: str, device: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype("float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def make_metric(name: str, device: str):
    try:
        import pyiqa
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pyiqa. Install it in the active environment, for example:\n"
            "  pip install pyiqa\n"
            "Some metrics may download pretrained weights on first use unless already cached."
        ) from exc
    return pyiqa.create_metric(name, device=device)


def score_image(metric, image_path: str, device: str) -> float:
    with torch.no_grad():
        value = metric(load_rgb_tensor(image_path, device))
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().item()
    return float(value)


def finite_values(values: list[float]) -> list[float]:
    return [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]


def summarize(values: list[float]) -> dict[str, Any]:
    vals = sorted(finite_values(values))
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    mid = len(vals) // 2
    median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
    return {
        "count": len(vals),
        "mean": sum(vals) / len(vals),
        "median": median,
        "min": vals[0],
        "max": vals[-1],
    }


def mean_or_none(values: list[float]) -> float | None:
    vals = finite_values(values)
    if not vals:
        return None
    return sum(vals) / len(vals)


def metric_direction(name: str) -> tuple[str, str]:
    if name in {"niqe", "brisque"}:
        return "lower_is_better", "越小越好"
    return "higher_is_better", "越大越好"


def better_by_mean(name: str, lr_mean: float | None, enhanced_mean: float | None) -> str | None:
    if lr_mean is None or enhanced_mean is None:
        return None
    if math.isclose(lr_mean, enhanced_mean, rel_tol=1e-12, abs_tol=1e-12):
        return "tie"
    direction, _ = metric_direction(name)
    if direction == "lower_is_better":
        return "enhanced" if enhanced_mean < lr_mean else "lr"
    return "enhanced" if enhanced_mean > lr_mean else "lr"


def main() -> None:
    args = parse_args()
    metric_names = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    if not metric_names:
        raise ValueError("--metrics cannot be empty")

    if args.metadata is not None:
        pairs = load_pairs_from_metadata(args.metadata, args.enhanced_dir, args.allow_missing)
    else:
        if args.lr_dir is None:
            raise ValueError("--lr-dir is required when --metadata is not set")
        pairs = load_pairs_from_dirs(args.lr_dir, args.enhanced_dir, args.recursive)
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    if not pairs:
        raise SystemExit("No image pairs found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {name: make_metric(name, args.device) for name in metric_names}

    rows: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs, 1):
        row: dict[str, Any] = dict(pair)
        print(f"[{idx}/{len(pairs)}] {pair['id']}", flush=True)
        for name, metric in metrics.items():
            lr_score = score_image(metric, pair["lr"], args.device)
            enhanced_score = score_image(metric, pair["enhanced"], args.device)
            row[f"lr_{name}"] = lr_score
            row[f"enhanced_{name}"] = enhanced_score
            row[f"delta_{name}"] = enhanced_score - lr_score
            if name in {"niqe", "brisque"}:
                row[f"improved_{name}"] = enhanced_score < lr_score
            else:
                row[f"improved_{name}"] = enhanced_score > lr_score
        rows.append(row)

    csv_path = args.output_dir / "metrics.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, Any] = {
        "num_pairs": len(rows),
        "metadata": str(args.metadata) if args.metadata else None,
        "lr_dir": str(args.lr_dir) if args.lr_dir else None,
        "enhanced_dir": str(args.enhanced_dir),
        "note": "delta_mean_enhanced_minus_lr = enhanced_mean - lr_mean",
        "metrics": [],
    }
    for name in metric_names:
        lr_mean = mean_or_none([r[f"lr_{name}"] for r in rows])
        enhanced_mean = mean_or_none([r[f"enhanced_{name}"] for r in rows])
        direction, direction_zh = metric_direction(name)
        summary["metrics"].append({
            "metric": name,
            "direction": direction,
            "direction_zh": direction_zh,
            "lr_mean": lr_mean,
            "enhanced_mean": enhanced_mean,
            "delta_mean_enhanced_minus_lr": None
            if lr_mean is None or enhanced_mean is None
            else enhanced_mean - lr_mean,
            "better_by_mean": better_by_mean(name, lr_mean, enhanced_mean),
            "improved_count": sum(bool(r[f"improved_{name}"]) for r in rows),
            "improved_rate": sum(bool(r[f"improved_{name}"]) for r in rows) / len(rows),
        })

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"metrics_csv: {csv_path}")
    print(f"summary_json: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
