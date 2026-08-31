#!/usr/bin/env python3
"""
Copy all images from the nested Unsplash source tree into the raw directory.

The source directory is treated as read-only: this script only scans and copies
from it. It never deletes, moves, renames, or writes anything under source.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


SOURCE_DIR = Path("/mnt/image-edit-hdd/datasets/shensheng/datasets/unsplash/images")
TARGET_DIR = Path("/mnt/image-edit-hdd/datasets/duanyufa/unsplash/raw")

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
        description="Copy nested Unsplash images into the raw directory."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=SOURCE_DIR,
        help=f"source image root, default: {SOURCE_DIR}",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=TARGET_DIR,
        help=f"target raw directory, default: {TARGET_DIR}",
    )
    parser.add_argument(
        "--preserve-dirs",
        action="store_true",
        help="preserve the source relative directory structure under target",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="copy every regular file instead of filtering by image suffix",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be copied without writing target files",
    )
    return parser.parse_args()


def is_image(path: Path, copy_all_files: bool) -> bool:
    return copy_all_files or path.suffix.lower() in IMAGE_SUFFIXES


def unique_flat_name(relative_path: Path) -> str:
    digest = hashlib.sha1(relative_path.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"{relative_path.stem}_{digest}{relative_path.suffix}"


def choose_target_path(
    source_file: Path,
    source_root: Path,
    target_root: Path,
    preserve_dirs: bool,
) -> Path:
    relative_path = source_file.relative_to(source_root)
    if preserve_dirs:
        return target_root / relative_path

    flat_path = target_root / source_file.name
    if not flat_path.exists():
        return flat_path

    try:
        if flat_path.stat().st_size == source_file.stat().st_size:
            return flat_path
    except OSError:
        pass

    return target_root / unique_flat_name(relative_path)


def iter_source_files(source_root: Path, copy_all_files: bool):
    for path in source_root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        if is_image(path, copy_all_files):
            yield path


def main() -> int:
    args = parse_args()
    source_root = args.source.resolve()
    target_root = args.target.resolve()

    if not source_root.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_root}")

    copied = 0
    skipped = 0

    if not args.dry_run:
        target_root.mkdir(parents=True, exist_ok=True)

    for source_file in iter_source_files(source_root, args.all_files):
        target_file = choose_target_path(
            source_file=source_file,
            source_root=source_root,
            target_root=target_root,
            preserve_dirs=args.preserve_dirs,
        )

        if target_file.exists() and target_file.stat().st_size == source_file.stat().st_size:
            skipped += 1
            continue

        print(f"COPY {source_file} -> {target_file}")
        if not args.dry_run:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
        copied += 1

    print(f"Done. copied={copied}, skipped_existing={skipped}, target={target_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
