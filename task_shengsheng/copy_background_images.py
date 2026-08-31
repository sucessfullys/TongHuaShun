#!/usr/bin/env python3
"""
Copy a fixed number of images from a source directory into a target directory.

For each source image:
    * open and validate with PIL (skip corrupted files)
    * if the longest side exceeds ``--max-side`` (default 1024), downscale so the
      longest side equals ``--max-side`` while preserving the aspect ratio
    * if the longest side is already within ``--max-side``, leave dimensions
      alone (do NOT pad or snap to any multiple of 16)

Output files keep the source stem and are written with a ``.png`` extension
via an atomic "tmp + rename" write so the target directory never contains a
half-written file.

The source directory is treated strictly read-only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError


DEFAULT_SOURCE = Path(
    "/mnt/image-edit-hdd/datasets/duanyufa/unsplash/清洗/real_no_human"
)
DEFAULT_TARGET = Path(
    "/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/背景"
)
DEFAULT_COUNT = 500
DEFAULT_MAX_SIDE = 1024

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy up to N images from SOURCE to TARGET, downscaling any whose "
            "longest side exceeds MAX_SIDE while preserving aspect ratio."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--max-side", type=int, default=DEFAULT_MAX_SIDE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be done without writing target files",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def iter_source_files(source_root: Path):
    for path in source_root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if is_image(path):
            yield path


def open_image(path: Path) -> Image.Image:
    """Open an image, validating it via ``Image.verify`` first.

    ``verify`` consumes the file handle, so we reopen for the actual decode.
    """
    with path.open("rb") as fh:
        probe = Image.open(fh)
        probe.verify()
    image = Image.open(path)
    return image.convert("RGB")


def resize_if_needed(image: Image.Image, max_side: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    return image.resize((new_w, new_h), Image.LANCZOS)


def atomic_save_png(image: Image.Image, output_path: Path) -> None:
    """Write the image to ``output_path`` as PNG without leaving partial files."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp.png")
    image.save(tmp_path, format="PNG")
    tmp_path.replace(output_path)


def main() -> int:
    args = parse_args()
    source_root = args.source.resolve()
    target_root = args.target.resolve()

    if not source_root.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_root}")
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    if args.max_side <= 0:
        raise SystemExit("--max-side must be positive")

    source_files = sorted(iter_source_files(source_root), key=lambda p: p.name)
    if not source_files:
        raise SystemExit(f"No images found under source: {source_root}")

    if not args.dry_run:
        target_root.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_existing = 0
    failed = 0
    examples_logged = 0
    MAX_EXAMPLE_LOGS = 5

    for source_file in source_files:
        if copied >= args.count:
            break

        target_file = target_root / (source_file.stem + ".png")

        if target_file.exists():
            skipped_existing += 1
            if copied < args.count:
                copied += 1  # existing outputs already satisfy the count
            continue

        try:
            image = open_image(source_file)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            failed += 1
            print(f"SKIP corrupt: {source_file} ({exc})")
            continue

        original_size = image.size
        resized = resize_if_needed(image, args.max_side)
        new_size = resized.size

        preview = (
            f"COPY {source_file.name} {original_size} -> {new_size} "
            f"-> {target_file.name}"
        )
        if examples_logged < MAX_EXAMPLE_LOGS:
            print(preview)
            examples_logged += 1
        elif examples_logged == MAX_EXAMPLE_LOGS:
            print("... (further items omitted from preview log)")
            examples_logged += 1

        if args.dry_run:
            copied += 1
            continue

        try:
            atomic_save_png(resized, target_file)
        except OSError as exc:
            failed += 1
            print(f"FAIL write: {target_file} ({exc})")
            continue

        copied += 1

    print(
        f"Done. copied={copied}, skipped_existing={skipped_existing}, "
        f"failed={failed}, scanned={len(source_files)}, target={target_root}"
    )
    return 0 if copied >= args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())
