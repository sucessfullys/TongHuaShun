#!/usr/bin/env python3
"""Create metadata.jsonl for real-world deblur inference images."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_INPUT_DIR = Path("/mnt/image-edit/datasets/duanyufa/Real_RL_Data/real_rl")
DEFAULT_OUTPUT = Path("/mnt/image-edit/datasets/duanyufa/Real_RL_Data/metadata.jsonl")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_PROMPT = (
    "Restore the input image to a clean, sharp, high-quality image. "
    "Remove blur, noise, compression artifacts, and low-quality degradation "
    "while preserving the original content, structure, colors, layout, and identity. "
    "Do not add, remove, or change objects."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate metadata.jsonl for real-world deblur inference.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def iter_images(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def output_name(path: Path, root: Path, stem_counts: Counter[str], recursive: bool) -> str:
    relative = path.relative_to(root) if recursive else Path(path.name)
    output = relative.with_suffix(".png")
    if stem_counts[relative.with_suffix("").as_posix()] > 1:
        output = relative.with_name(f"{relative.stem}__{path.suffix.lower().lstrip('.')}.png")
    return output.as_posix()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    images = iter_images(input_dir, args.recursive)
    if not images:
        raise ValueError(f"No images found in {input_dir}")

    stems = Counter(
        (path.relative_to(input_dir) if args.recursive else Path(path.name)).with_suffix("").as_posix()
        for path in images
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for image_path in images:
            record = {
                "prompt": args.prompt,
                "image": output_name(image_path, input_dir, stems, args.recursive),
                "template_inputs": {
                    "image": str(image_path.resolve()),
                    "prompt": args.prompt,
                },
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Input images: {len(images)}")
    print(f"Metadata: {args.output}")


if __name__ == "__main__":
    main()
