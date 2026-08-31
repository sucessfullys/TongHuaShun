#!/usr/bin/env python3
"""Generate same-size bicubic and realistic LR previews from DF2K HR images."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/DF2K/Download_df2k/datasets/DF2K")
DEFAULT_HR = ROOT / "DF2K_train_HR"
DEFAULT_OUTPUT = ROOT / "DF2K_realistic_lr_samples_10"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr-dir", type=Path, default=DEFAULT_HR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def anisotropic_kernel(size, sigma_x, sigma_y, angle):
    radius = size // 2
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    cosine, sine = math.cos(angle), math.sin(angle)
    rotated_x = cosine * xx + sine * yy
    rotated_y = -sine * xx + cosine * yy
    kernel = np.exp(-0.5 * ((rotated_x / sigma_x) ** 2 + (rotated_y / sigma_y) ** 2))
    return (kernel / kernel.sum()).astype(np.float32)


def jpeg_roundtrip(image, quality):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def add_noise(image, rng, params):
    array = image.astype(np.float32)
    if rng.random() < 0.75:
        sigma = float(rng.uniform(0.8, 7.0))
        gray_noise = rng.random() < 0.35
        shape = (*array.shape[:2], 1 if gray_noise else 3)
        noise = rng.normal(0.0, sigma, shape).astype(np.float32)
        array = array + noise
        params.update({"noise": "gaussian", "noise_sigma": sigma, "gray_noise": gray_noise})
    else:
        peak = float(rng.uniform(35.0, 120.0))
        array = rng.poisson(np.clip(array, 0, 255) / 255.0 * peak) / peak * 255.0
        params.update({"noise": "poisson", "poisson_peak": peak})
    return np.clip(array, 0, 255).astype(np.uint8)


def resize_interpolation(rng, down):
    if down:
        choices = [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR]
        names = ["area", "bicubic", "bilinear"]
        probabilities = [0.55, 0.25, 0.20]
    else:
        choices = [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_LANCZOS4]
        names = ["bicubic", "bilinear", "lanczos"]
        probabilities = [0.50, 0.30, 0.20]
    index = int(rng.choice(len(choices), p=probabilities))
    return choices[index], names[index]


def bicubic_degrade(hr, scale=4.0):
    height, width = hr.shape[:2]
    small = cv2.resize(hr, (max(1, round(width / scale)), max(1, round(height / scale))), interpolation=cv2.INTER_CUBIC)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)


def realistic_degrade(hr, rng):
    height, width = hr.shape[:2]
    params = {}

    kernel_size = int(rng.choice([7, 9, 11, 13, 15, 17]))
    sigma_x = float(rng.uniform(0.45, 2.2))
    sigma_y = float(rng.uniform(0.45, 2.2))
    angle = float(rng.uniform(0, math.pi))
    kernel = anisotropic_kernel(kernel_size, sigma_x, sigma_y, angle)
    degraded = cv2.filter2D(hr, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    params.update({
        "blur": "anisotropic_gaussian",
        "kernel_size": kernel_size,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "angle_degrees": math.degrees(angle),
    })

    scale = float(rng.uniform(1.7, 4.2))
    down_interpolation, down_name = resize_interpolation(rng, down=True)
    low_width, low_height = max(32, round(width / scale)), max(32, round(height / scale))
    degraded = cv2.resize(degraded, (low_width, low_height), interpolation=down_interpolation)
    params.update({"downsample_scale": scale, "downsample_interpolation": down_name})
    degraded = add_noise(degraded, rng, params)

    first_quality = int(rng.integers(48, 93))
    degraded = jpeg_roundtrip(degraded, first_quality)
    params["jpeg_quality_first"] = first_quality

    if rng.random() < 0.40:
        second_sigma = float(rng.uniform(0.25, 0.9))
        degraded = cv2.GaussianBlur(degraded, (5, 5), second_sigma)
        second_scale = float(rng.uniform(1.0, 1.45))
        second_width = max(24, round(degraded.shape[1] / second_scale))
        second_height = max(24, round(degraded.shape[0] / second_scale))
        degraded = cv2.resize(degraded, (second_width, second_height), interpolation=cv2.INTER_AREA)
        params.update({"second_degradation": True, "second_blur_sigma": second_sigma, "second_scale": second_scale})
    else:
        params["second_degradation"] = False

    up_interpolation, up_name = resize_interpolation(rng, down=False)
    degraded = cv2.resize(degraded, (width, height), interpolation=up_interpolation)
    final_quality = int(rng.integers(58, 96))
    degraded = jpeg_roundtrip(degraded, final_quality)
    params.update({"upsample_interpolation": up_name, "jpeg_quality_final": final_quality})
    return degraded, params


def psnr(reference, image):
    mse = float(np.mean((reference.astype(np.float32) - image.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10.0 * math.log10(255.0 ** 2 / mse)


def preview_panel(image, label, target_height=720):
    height, width = image.shape[:2]
    scale = min(1.0, target_height / height)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    bar = np.full((48, resized.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(bar, label, (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    return np.vstack([bar, resized])


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("num-samples must be positive")
    hr_dir = args.hr_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    files = sorted(path for path in hr_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(files) < args.num_samples:
        raise RuntimeError(f"Requested {args.num_samples} samples but found only {len(files)} HR images")

    rng = np.random.default_rng(args.seed)
    selected = sorted((files[index] for index in rng.choice(len(files), args.num_samples, replace=False)), key=lambda path: path.name)
    directories = {name: output_dir / name for name in ["HR", "LR_bicubic", "LR_realistic", "comparison"]}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    for index, source in enumerate(selected, 1):
        hr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if hr is None:
            raise RuntimeError(f"Cannot read {source}")
        bicubic = bicubic_degrade(hr)
        realistic, degradation = realistic_degrade(hr, rng)
        filename = f"{index:02d}_{source.stem}.png"
        shutil.copy2(source, directories["HR"] / filename)
        cv2.imwrite(str(directories["LR_bicubic"] / filename), bicubic, [cv2.IMWRITE_PNG_COMPRESSION, 2])
        cv2.imwrite(str(directories["LR_realistic"] / filename), realistic, [cv2.IMWRITE_PNG_COMPRESSION, 2])

        panels = [
            preview_panel(hr, "HR"),
            preview_panel(bicubic, "Bicubic x4 down-up"),
            preview_panel(realistic, "Realistic mixed degradation"),
        ]
        panel_height = max(panel.shape[0] for panel in panels)
        padded = []
        for panel in panels:
            if panel.shape[0] < panel_height:
                panel = cv2.copyMakeBorder(panel, 0, panel_height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            padded.append(panel)
        comparison = np.hstack(padded)
        cv2.imwrite(str(directories["comparison"] / filename), comparison, [cv2.IMWRITE_PNG_COMPRESSION, 2])

        record = {
            "index": index,
            "source": str(source),
            "filename": filename,
            "width": int(hr.shape[1]),
            "height": int(hr.shape[0]),
            "same_resolution": realistic.shape[:2] == hr.shape[:2],
            "psnr_bicubic": psnr(hr, bicubic),
            "psnr_realistic": psnr(hr, realistic),
            "degradation": degradation,
        }
        records.append(record)
        print(f"[{index:02d}/{args.num_samples}] {filename} {hr.shape[1]}x{hr.shape[0]} realistic_PSNR={record['psnr_realistic']:.2f}dB")

    (output_dir / "degradation_params.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "Each numbered sample has the same pixel dimensions in HR, LR_bicubic and LR_realistic.\n"
        "LR_bicubic: fixed x4 bicubic downsample followed by bicubic upsample.\n"
        "LR_realistic: random blur, non-integer resize, noise, JPEG and optional second degradation.\n"
        "comparison: HR | bicubic | realistic preview.\n",
        encoding="utf-8",
    )
    print("Saved samples to", output_dir)


if __name__ == "__main__":
    main()
