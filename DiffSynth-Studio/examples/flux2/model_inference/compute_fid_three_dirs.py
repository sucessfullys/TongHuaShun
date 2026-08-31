import argparse
import json
import os
from pathlib import Path

import torch

from diffsynth.core import ModelConfig
from diffsynth.metrics import FIDMetric


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpg",
    ".jpeg",
    ".pgm",
    ".png",
    ".ppm",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Base and Self-Flow FID from three image directories."
    )
    parser.add_argument("--reference_dir", required=True)
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--self_flow_dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fid_batch_size", type=int, default=32)
    parser.add_argument("--fid_num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fid_model",
        default=None,
        help="Optional local FID model.safetensors path.",
    )
    return parser.parse_args()


def image_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")
    files = []
    for root, directories, names in os.walk(directory):
        directories.sort()
        files.extend(
            Path(root) / name
            for name in sorted(names)
            if Path(name).suffix.lower() in IMAGE_EXTENSIONS
        )
    if not files:
        raise ValueError(f"No supported images found under {directory}")
    return files


def sample_ids(files):
    ids = [path.stem for path in files]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate image stems found in an input directory.")
    return set(ids)


def validate_inputs(reference_files, base_files, self_flow_files):
    counts = {
        "reference": len(reference_files),
        "base": len(base_files),
        "self_flow": len(self_flow_files),
    }
    if len(set(counts.values())) != 1:
        raise ValueError(f"Image counts must match, got {counts}")
    reference_ids = sample_ids(reference_files)
    base_ids = sample_ids(base_files)
    self_flow_ids = sample_ids(self_flow_files)
    if reference_ids != base_ids or reference_ids != self_flow_ids:
        raise ValueError(
            "Image filename stems must describe the same sample set in all "
            "three directories."
        )
    return counts


def build_metric(args):
    model_config = (
        ModelConfig(path=args.fid_model)
        if args.fid_model
        else ModelConfig(
            model_id="DiffSynth-Studio/ImageMetrics",
            origin_file_pattern="FID/model.safetensors",
        )
    )
    return FIDMetric.from_pretrained(
        model_config=model_config,
        device=args.device,
        batch_size=args.fid_batch_size,
        num_workers=args.fid_num_workers,
    )


def to_float(value):
    return value.detach().cpu().item() if torch.is_tensor(value) else float(value)


def main():
    args = parse_args()
    reference_files = image_files(args.reference_dir)
    base_files = image_files(args.base_dir)
    self_flow_files = image_files(args.self_flow_dir)
    counts = validate_inputs(reference_files, base_files, self_flow_files)

    print(f"Validated image counts: {counts}")
    metric = build_metric(args)
    reference_stats = metric.statistics(reference_files)
    base_stats = metric.statistics(base_files)
    self_flow_stats = metric.statistics(self_flow_files)

    fid_base = to_float(
        metric.model.frechet_distance(
            reference_stats[0],
            reference_stats[1],
            base_stats[0],
            base_stats[1],
        )
    )
    fid_self_flow = to_float(
        metric.model.frechet_distance(
            reference_stats[0],
            reference_stats[1],
            self_flow_stats[0],
            self_flow_stats[1],
        )
    )
    results = {
        "num_samples": counts["reference"],
        "reference_dir": str(Path(args.reference_dir).resolve()),
        "base_dir": str(Path(args.base_dir).resolve()),
        "self_flow_dir": str(Path(args.self_flow_dir).resolve()),
        "fid_base": fid_base,
        "fid_self_flow": fid_self_flow,
        "fid_delta_self_flow_minus_base": fid_self_flow - fid_base,
    }

    output_path = (
        Path(args.output)
        if args.output
        else Path(args.self_flow_dir).parent / "fid_results_decoupled.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
