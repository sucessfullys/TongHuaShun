#!/usr/bin/env python3
"""Generate LR degradations without copying the HR images.

This keeps the existing HR directory read-only and writes only LR images plus
metadata/manifest files under the requested output root.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

cv2.setNumThreads(1)

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PROMPTS = [
    "Restore this degraded image to a clean, sharp, natural image while preserving all original content, colors, geometry, and composition.",
    "Remove blur, compression artifacts, and light noise while keeping the scene completely faithful to the input.",
    "Recover a faithful high-quality image without adding, removing, or changing any objects or structures.",
]
LEVELS = {
    "mild": dict(sig=(1.1, 1.8), kernels=[9, 11, 13, 15], scale=(2.2, 3.2), jpeg1=(70, 87), jpeg2=(76, 94), second=1.0),
    "medium": dict(sig=(1.2, 2.3), kernels=[11, 13, 15, 17], scale=(2.8, 4.0), jpeg1=(54, 80), jpeg2=(66, 91), second=1.0),
    "strong": dict(sig=(1.5, 3.0), kernels=[13, 15, 17, 19, 21], scale=(3.6, 5.0), jpeg1=(38, 66), jpeg2=(52, 84), second=1.0),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--noise-probability", type=float, default=0.35)
    parser.add_argument("--severity-probs", default="0.30,0.45,0.25", help="Comma-separated probabilities for mild,medium,strong.")
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def rng_for(seed, name):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")
    return np.random.default_rng(value)


def kernel(size, sx, sy, angle):
    radius = size // 2
    y, x = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
    c, s = math.cos(angle), math.sin(angle)
    xr, yr = c * x + s * y, -s * x + c * y
    k = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return (k / k.sum()).astype(np.float32)


def jpeg(image, quality):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def interpolation(rng, down):
    if down:
        values = [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR]
        names = ["area", "bicubic", "bilinear"]
        probs = [0.55, 0.25, 0.2]
    else:
        values = [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_LANCZOS4]
        names = ["bicubic", "bilinear", "lanczos"]
        probs = [0.5, 0.3, 0.2]
    idx = int(rng.choice(3, p=probs))
    return values[idx], names[idx]


def light_noise(image, rng, probability, record):
    if rng.random() >= probability:
        record["noise"] = "none"
        return image
    arr = image.astype(np.float32)
    if rng.random() < 0.85:
        sigma = float(rng.uniform(0.35, 2.5))
        gray = bool(rng.random() < 0.35)
        arr += rng.normal(0, sigma, (*arr.shape[:2], 1 if gray else 3))
        record.update(noise="light_gaussian", noise_sigma=sigma, gray_noise=gray)
    else:
        peak = float(rng.uniform(150, 450))
        arr = rng.poisson(np.clip(arr, 0, 255) / 255 * peak) / peak * 255
        record.update(noise="light_poisson", poisson_peak=peak)
    return np.clip(arr, 0, 255).astype(np.uint8)


def parse_severity_probs(value):
    probs = [float(x) for x in value.split(",")]
    if len(probs) != 3 or any(x < 0 for x in probs) or sum(probs) <= 0:
        raise ValueError("--severity-probs must be three non-negative values, for mild,medium,strong")
    total = sum(probs)
    return [x / total for x in probs]


def degrade(hr, rng, noise_probability, severity_probs):
    h, w = hr.shape[:2]
    level = str(rng.choice(["mild", "medium", "strong"], p=severity_probs))
    cfg = LEVELS[level]
    rec = {"severity": level}
    size = int(rng.choice(cfg["kernels"]))
    sx = float(rng.uniform(*cfg["sig"]))
    sy = max(0.15, sx * float(rng.uniform(0.65, 1.35)))
    angle = float(rng.uniform(0, math.pi))
    out = cv2.filter2D(hr, -1, kernel(size, sx, sy, angle), borderType=cv2.BORDER_REFLECT_101)
    rec.update(blur="anisotropic_gaussian", kernel_size=size, sigma_x=sx, sigma_y=sy, angle_degrees=math.degrees(angle))
    scale = float(rng.uniform(*cfg["scale"]))
    mode, mode_name = interpolation(rng, True)
    out = cv2.resize(out, (max(32, round(w / scale)), max(32, round(h / scale))), interpolation=mode)
    rec.update(downsample_scale=scale, downsample_interpolation=mode_name)
    out = light_noise(out, rng, noise_probability, rec)
    q1 = int(rng.integers(*cfg["jpeg1"]))
    out = jpeg(out, q1)
    rec["jpeg_quality_first"] = q1
    if rng.random() < cfg["second"]:
        sig2 = float(rng.uniform(0.2, 0.75))
        scale2 = float(rng.uniform(1, 1.3))
        out = cv2.GaussianBlur(out, (5, 5), sig2)
        out = cv2.resize(out, (max(24, round(out.shape[1] / scale2)), max(24, round(out.shape[0] / scale2))), interpolation=cv2.INTER_AREA)
        rec.update(second_degradation=True, second_blur_sigma=sig2, second_scale=scale2)
    else:
        rec["second_degradation"] = False
    mode, mode_name = interpolation(rng, False)
    out = cv2.resize(out, (w, h), interpolation=mode)
    q2 = int(rng.integers(*cfg["jpeg2"]))
    out = jpeg(out, q2)
    rec.update(upsample_interpolation=mode_name, jpeg_quality_final=q2)
    mse = float(np.mean((hr.astype(np.float32) - out.astype(np.float32)) ** 2))
    rec["psnr"] = float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    return out, rec


def scan(hr_dir, max_ratio):
    good, bad = [], []
    for path in sorted(p for p in hr_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS):
        with Image.open(path) as im:
            w, h = im.size
        ratio = max(w / h, h / w)
        (good if ratio <= max_ratio else bad).append((path, w, h, ratio))
    return good, bad


def process_one(task):
    src_s, lr_s, seed, noise_probability, severity_probs, w, h, ratio = task
    src = Path(src_s)
    lr_path = Path(lr_s)
    name = lr_path.name
    hr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if hr is None:
        raise RuntimeError(f"Read failed: {src}")
    lr, rec = degrade(hr, rng_for(seed, name), noise_probability, severity_probs)
    if lr.shape != hr.shape:
        raise AssertionError(f"HR/LR mismatch: {name}")
    if not cv2.imwrite(str(lr_path), lr, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
        raise RuntimeError(f"Write failed: {name}")
    return {
        "filename": name,
        "source": str(src),
        "width": w,
        "height": h,
        "aspect_ratio": ratio,
        "same_resolution": True,
        "degradation": rec,
    }


def main():
    args = parse_args()
    if not 0 <= args.noise_probability <= 1 or args.max_aspect_ratio < 1:
        raise ValueError("Invalid probability or aspect ratio")
    severity_probs = parse_severity_probs(args.severity_probs)
    hr_dir = args.hr_dir.resolve()
    out_dir = args.output_dir.resolve()
    lr_dir = out_dir / "LR"
    lr_dir.mkdir(parents=True, exist_ok=True)

    eligible, skipped = scan(hr_dir, args.max_aspect_ratio)
    if args.limit is not None:
        eligible = eligible[:args.limit]

    (out_dir / "skipped_aspect_ratio.json").write_text(json.dumps([
        {"source": str(p), "width": w, "height": h, "aspect_ratio": r}
        for p, w, h, r in skipped
    ], ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = out_dir / "degradation_params.jsonl"
    done = set()
    if manifest_path.is_file():
        done = {json.loads(x)["filename"] for x in manifest_path.read_text(encoding="utf-8").splitlines() if x.strip()}

    print(f"eligible={len(eligible)} skipped_ratio={len(skipped)} completed={len(done)}")
    made = 0
    tasks = []
    for src, w, h, ratio in eligible:
        name = src.with_suffix(".png").name
        if name in done and (lr_dir / name).is_file():
            continue
        tasks.append((str(src), str(lr_dir / name), args.seed, args.noise_probability, severity_probs, w, h, ratio))

    with manifest_path.open("a", encoding="utf-8") as manifest:
        if args.workers <= 1:
            for i, task in enumerate(tasks, 1):
                rec_item = process_one(task)
                manifest.write(json.dumps(rec_item, ensure_ascii=False) + "\n")
                manifest.flush()
                made += 1
                deg = rec_item["degradation"]
                if made <= 5 or made % 25 == 0:
                    print(f"generated={made} progress={i}/{len(tasks)} {rec_item['filename']} {rec_item['width']}x{rec_item['height']} {deg['severity']} noise={deg['noise']}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(process_one, task): i for i, task in enumerate(tasks, 1)}
                for future in as_completed(futures):
                    i = futures[future]
                    rec_item = future.result()
                    manifest.write(json.dumps(rec_item, ensure_ascii=False) + "\n")
                    manifest.flush()
                    made += 1
                    deg = rec_item["degradation"]
                    if made <= 5 or made % 25 == 0:
                        print(f"generated={made} progress={i}/{len(tasks)} {rec_item['filename']} {rec_item['width']}x{rec_item['height']} {deg['severity']} noise={deg['noise']}")

    records = [json.loads(x) for x in manifest_path.read_text(encoding="utf-8").splitlines() if x.strip()]

    source_by_filename = {}
    for record in records:
        filename = record.get("filename")
        source = record.get("source")
        if filename and source and Path(source).is_file() and (lr_dir / filename).is_file():
            source_by_filename[filename] = source
    pairs = sorted(source_by_filename)
    lines = []
    for i, name in enumerate(pairs):
        prompt = PROMPTS[i % len(PROMPTS)]
        lines.append(json.dumps({
            "prompt": prompt,
            "image": name,
            "source": source_by_filename[name],
            "template_inputs": {
                "image": str((lr_dir / name).resolve()),
                "prompt": prompt,
            },
        }, ensure_ascii=False))
    (out_dir / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "hr_dir": str(hr_dir),
        "lr_dir": str(lr_dir),
        "severity_probs": {
            "mild": severity_probs[0],
            "medium": severity_probs[1],
            "strong": severity_probs[2],
        },
        "noise_probability": args.noise_probability,
        "source_images": len(eligible) + len(skipped),
        "eligible_images": len(eligible),
        "skipped_aspect_ratio": len(skipped),
        "paired_images": len(pairs),
        "severity_counts": {k: sum(r["degradation"]["severity"] == k for r in records) for k in LEVELS},
        "noise_counts": {
            k: sum(r["degradation"]["noise"] == k for r in records)
            for k in ["none", "light_gaussian", "light_poisson"]
        },
    }
    (out_dir / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
