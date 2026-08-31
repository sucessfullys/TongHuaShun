#!/usr/bin/env python3
"""Convert paired Template metadata to native FLUX.2 edit-image metadata."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from PIL import Image


DEFAULT_SOURCE = Path("/mnt/image-edit/datasets/duanyufa/Face/metadata.jsonl")
DEFAULT_HR_ROOT = Path("/mnt/image-edit/datasets/duanyufa/Face/HR")
DEFAULT_OUTPUT = Path(
    "/mnt/image-edit/datasets/duanyufa/Face/metadata_flux2_edit_lora.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build aligned HR-target/LR-edit_image metadata for FLUX.2 LoRA."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--hr-root", type=Path, default=DEFAULT_HR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-size-check",
        action="store_true",
        help="Skip the exact HR/LR dimension validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    hr_root = args.hr_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Missing source metadata: {source}")
    if not hr_root.is_dir():
        raise FileNotFoundError(f"Missing HR root: {hr_root}")
    if output == source:
        raise ValueError("Refusing to overwrite the source metadata")

    converted: list[dict] = []
    seen_outputs: set[str] = set()
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                prompt = record["prompt"]
                relative_hr = Path(record["image"])
                lr_path = Path(record["template_inputs"]["image"]).expanduser().resolve()
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid source record at line {line_number}: {error}"
                ) from error
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Empty or invalid prompt at line {line_number}")
            if relative_hr.is_absolute() or ".." in relative_hr.parts:
                raise ValueError(
                    f"HR image must be a safe path relative to --hr-root at "
                    f"line {line_number}: {relative_hr}"
                )
            relative_key = relative_hr.as_posix()
            if relative_key in seen_outputs:
                raise ValueError(f"Duplicate HR image at line {line_number}: {relative_key}")
            seen_outputs.add(relative_key)

            hr_path = (hr_root / relative_hr).resolve()
            if not hr_path.is_file():
                raise FileNotFoundError(f"Missing HR image at line {line_number}: {hr_path}")
            if not lr_path.is_file():
                raise FileNotFoundError(f"Missing LR image at line {line_number}: {lr_path}")

            if not args.skip_size_check:
                with Image.open(hr_path) as hr_opened, Image.open(lr_path) as lr_opened:
                    if hr_opened.size != lr_opened.size:
                        raise ValueError(
                            f"Unaligned dimensions at line {line_number}: "
                            f"HR={hr_opened.size}, LR={lr_opened.size}"
                        )
                    width, height = hr_opened.size
                    if width % 16 != 0 or height % 16 != 0:
                        raise ValueError(
                            f"Dimensions must be divisible by 16 at line "
                            f"{line_number}: {width}x{height}"
                        )

            converted.append(
                {
                    "prompt": prompt,
                    "image": relative_key,
                    "edit_image": str(lr_path),
                    "edit_image_auto_resize": False,
                }
            )

    if not converted:
        raise ValueError(f"No records found in {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for record in converted:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(output)
    print(f"Wrote {len(converted)} aligned records to {output}")


if __name__ == "__main__":
    main()
