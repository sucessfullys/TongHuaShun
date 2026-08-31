#!/usr/bin/env python3
"""Build DiffSynth FLUX.2 Template metadata for aligned Face HR/LR pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


FACE_ROOT = Path("/mnt/image-edit/datasets/duanyufa/Face")
DEFAULT_PROMPT = (
    "Enhance image clarity and sharpness, recover fine details, and improve "
    "texture definition while preserving the original content exactly. "
    "Do not change the people, pose, clothing design, colors, proportions, "
    "composition, background, or layout. Do not add, remove, or invent any "
    "objects or details. No hallucinated content. Keep the image visually "
    "identical to the original, only clearer and sharper."
)
PROMPT_VARIANTS = (
    DEFAULT_PROMPT,
    (
        "Improve the sharpness and visual quality of this image. Remove blur and "
        "noise while restoring fine details and natural textures. "
        "Do not alter any person, object, facial features, clothing, colors, "
        "background, composition, or proportions. Do not add or remove anything. "
        "The image should look identical to the original, only sharper and cleaner."
    ),
    (
        "Restore clarity and sharpness to this degraded image. Recover the natural "
        "textures and fine details that blur and noise have obscured. "
        "Keep all content identical — same people, objects, colors, pose, clothing, "
        "background, and composition. No invented details, no hallucinated objects. "
        "Only clearer and sharper than before."
    ),
    (
        "Deblur and denoise this image while enhancing sharpness. Reconstruct fine "
        "details and textures faithfully from the remaining image information. "
        "Do not change people, facial features, clothing, colors, proportions, "
        "composition, or background. Do not introduce new objects or remove "
        "existing ones. Keep the image visually unchanged except for improved "
        "clarity."
    ),
    (
        "Enhance the sharpness and clarity of this image by removing degradation. "
        "Recover fine details and textures that were originally present in the "
        "scene. "
        "Preserve exact colors, composition, layout, and all visual elements. Do "
        "not modify people, objects, clothing, or background. No hallucinated "
        "content — only clearer and sharper."
    ),
    (
        "Recover a sharp, clean image from this degraded input. Remove blur, noise, "
        "and compression artifacts while enhancing fine detail and texture "
        "definition. "
        "Do not alter the original content — same people, same clothing, same "
        "colors, same background, same composition, same layout. Do not add, "
        "remove, or change anything. Only sharper and clearer."
    ),
    (
        "Improve image clarity by deblurring and denoising, and recover natural "
        "fine details and textures. "
        "Do not change people, facial features, pose, clothing design, colors, "
        "proportions, background, composition, or layout. Do not invent or remove "
        "any objects or details. Keep the image visually identical to the original, "
        "only clearer and sharper."
    ),
    (
        "Restore this image to a clean, sharp, visually clear state. Enhance fine "
        "textures and details lost during degradation. "
        "Preserve the original content faithfully — do not alter any person, "
        "object, clothing, color, background, or composition. Do not hallucinate "
        "or add details. Only clearer and sharper, nothing else changed."
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr-dir", type=Path, default=FACE_ROOT / "HR")
    parser.add_argument("--lr-dir", type=Path, default=FACE_ROOT / "LR")
    parser.add_argument(
        "--output", type=Path, default=FACE_ROOT / "metadata.jsonl"
    )
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument(
        "--prompt-mode",
        choices=("diverse", "fixed"),
        default="diverse",
        help="Use deterministic prompt variants or one fixed prompt.",
    )
    parser.add_argument(
        "--fixed-prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used when --prompt-mode=fixed.",
    )
    parser.add_argument(
        "--skip-size-check",
        action="store_true",
        help="Skip decoding every HR/LR pair to verify identical dimensions.",
    )
    return parser.parse_args()


def image_map(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    images = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*.png")
        if path.is_file()
    }
    if not images:
        raise FileNotFoundError(f"No PNG images found under {root}")
    return images


def prompt_for(relative_path: str, args: argparse.Namespace) -> str:
    if args.prompt_mode == "fixed":
        return args.fixed_prompt
    digest = hashlib.sha256(
        f"{args.seed}:{relative_path}".encode("utf-8")
    ).digest()
    return PROMPT_VARIANTS[int.from_bytes(digest[:8], "big") % len(PROMPT_VARIANTS)]


def verify_pair(hr_path: Path, lr_path: Path, relative_path: str) -> None:
    with Image.open(hr_path) as hr, Image.open(lr_path) as lr:
        if hr.size != lr.size:
            raise ValueError(
                f"HR/LR size mismatch for {relative_path}: {hr.size} vs {lr.size}"
            )


def main() -> None:
    args = parse_args()
    hr_root = args.hr_dir.expanduser().resolve()
    lr_root = args.lr_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    hr_images = image_map(hr_root)
    lr_images = image_map(lr_root)

    missing_lr = sorted(set(hr_images) - set(lr_images))
    extra_lr = sorted(set(lr_images) - set(hr_images))
    if missing_lr or extra_lr:
        raise ValueError(
            "HR/LR filenames are not one-to-one: "
            f"missing LR={len(missing_lr)}, extra LR={len(extra_lr)}. "
            f"Examples: missing={missing_lr[:3]}, extra={extra_lr[:3]}"
        )

    records = []
    prompt_counts = {prompt: 0 for prompt in PROMPT_VARIANTS}
    for index, relative_path in enumerate(sorted(hr_images), start=1):
        if not args.skip_size_check:
            verify_pair(hr_images[relative_path], lr_images[relative_path], relative_path)
        prompt = prompt_for(relative_path, args)
        prompt_counts[prompt] = prompt_counts.get(prompt, 0) + 1
        records.append(
            {
                # Relative to --dataset_base_path, which should be Face/HR.
                "prompt": prompt,
                "image": relative_path,
                # The official Template data processor resolves this independently
                # of dataset_base_path, so use a fully offline absolute path.
                "template_inputs": {
                    "image": str(lr_images[relative_path].resolve()),
                    "prompt": prompt,
                },
            }
        )
        if index % 500 == 0 or index == len(hr_images):
            print(f"Verified {index}/{len(hr_images)} pairs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)

    print(f"Wrote {len(records)} records to {output_path}")
    print(f"Use --dataset_base_path {hr_root}")
    if args.prompt_mode == "diverse":
        print("Prompt distribution:")
        for prompt_id, prompt in enumerate(PROMPT_VARIANTS, start=1):
            print(f"  variant {prompt_id}: {prompt_counts[prompt]}")


if __name__ == "__main__":
    main()
