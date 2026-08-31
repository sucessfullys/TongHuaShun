#!/usr/bin/env python3
"""Generate a same-size realistic HR/LR training dataset."""

import argparse, hashlib, json, math, shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

SRC = Path("xxx")
OUT = Path("xxx")
EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PROMPTS = [
    "Restore this degraded image to a clean, sharp, natural image while preserving all original content, colors, geometry, and composition.",
    "Remove blur, compression artifacts, and light noise while keeping the scene completely faithful to the input.",
    "Recover a faithful high-quality image without adding, removing, or changing any objects or structures.",
]
LEVELS = {
    # Maximum PSNR values mirror reference samples 05, 03 and 08.
    "mild": dict(sig=(1.1, 1.8), kernels=[9, 11, 13, 15], scale=(2.2, 3.2), jpeg1=(70, 87), jpeg2=(76, 94), second=1.0),
    "medium": dict(sig=(1.2, 2.3), kernels=[11, 13, 15, 17], scale=(2.8, 4.0), jpeg1=(54, 80), jpeg2=(66, 91), second=1.0),
    "strong": dict(sig=(1.5, 3.0), kernels=[13, 15, 17, 19, 21], scale=(3.6, 5.0), jpeg1=(38, 66), jpeg2=(52, 84), second=1.0),
}


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, default=SRC)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--max-aspect-ratio", type=float, default=2.0)
    p.add_argument("--noise-probability", type=float, default=.35)
    p.add_argument("--seed", type=int, default=20260701)
    p.add_argument("--limit", type=int)
    return p.parse_args()


def rng_for(seed, name):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little")
    return np.random.default_rng(value)


def kernel(size, sx, sy, angle):
    r = size // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    c, s = math.cos(angle), math.sin(angle)
    xr, yr = c * x + s * y, -s * x + c * y
    k = np.exp(-.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return (k / k.sum()).astype(np.float32)


def jpeg(image, quality):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def interpolation(rng, down):
    if down:
        values, names, probs = [cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR], ["area", "bicubic", "bilinear"], [.55, .25, .2]
    else:
        values, names, probs = [cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_LANCZOS4], ["bicubic", "bilinear", "lanczos"], [.5, .3, .2]
    i = int(rng.choice(3, p=probs))
    return values[i], names[i]


def light_noise(image, rng, probability, record):
    if rng.random() >= probability:
        record["noise"] = "none"
        return image
    arr = image.astype(np.float32)
    if rng.random() < .85:
        sigma, gray = float(rng.uniform(.35, 2.5)), bool(rng.random() < .35)
        arr += rng.normal(0, sigma, (*arr.shape[:2], 1 if gray else 3))
        record.update(noise="light_gaussian", noise_sigma=sigma, gray_noise=gray)
    else:
        peak = float(rng.uniform(150, 450))
        arr = rng.poisson(np.clip(arr, 0, 255) / 255 * peak) / peak * 255
        record.update(noise="light_poisson", poisson_peak=peak)
    return np.clip(arr, 0, 255).astype(np.uint8)


def degrade(hr, rng, noise_probability):
    h, w = hr.shape[:2]
    level = str(rng.choice(["mild", "medium", "strong"], p=[.30, .45, .25]))
    cfg, rec = LEVELS[level], {"severity": level}
    size = int(rng.choice(cfg["kernels"]))
    sx = float(rng.uniform(*cfg["sig"])); sy = max(.15, sx * float(rng.uniform(.65, 1.35)))
    angle = float(rng.uniform(0, math.pi))
    out = cv2.filter2D(hr, -1, kernel(size, sx, sy, angle), borderType=cv2.BORDER_REFLECT_101)
    rec.update(blur="anisotropic_gaussian", kernel_size=size, sigma_x=sx, sigma_y=sy, angle_degrees=math.degrees(angle))
    scale = float(rng.uniform(*cfg["scale"])); mode, mode_name = interpolation(rng, True)
    out = cv2.resize(out, (max(32, round(w / scale)), max(32, round(h / scale))), interpolation=mode)
    rec.update(downsample_scale=scale, downsample_interpolation=mode_name)
    out = light_noise(out, rng, noise_probability, rec)
    q1 = int(rng.integers(*cfg["jpeg1"])); out = jpeg(out, q1); rec["jpeg_quality_first"] = q1
    if rng.random() < cfg["second"]:
        sig2, scale2 = float(rng.uniform(.2, .75)), float(rng.uniform(1, 1.3))
        out = cv2.GaussianBlur(out, (5, 5), sig2)
        out = cv2.resize(out, (max(24, round(out.shape[1] / scale2)), max(24, round(out.shape[0] / scale2))), interpolation=cv2.INTER_AREA)
        rec.update(second_degradation=True, second_blur_sigma=sig2, second_scale=scale2)
    else:
        rec["second_degradation"] = False
    mode, mode_name = interpolation(rng, False)
    out = cv2.resize(out, (w, h), interpolation=mode)
    q2 = int(rng.integers(*cfg["jpeg2"])); out = jpeg(out, q2)
    rec.update(upsample_interpolation=mode_name, jpeg_quality_final=q2)
    mse = float(np.mean((hr.astype(np.float32) - out.astype(np.float32)) ** 2))
    value = float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)
    rec.update(psnr=value)
    return out, rec


def scan(source, max_ratio):
    good, bad = [], []
    for path in sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in EXTS):
        with Image.open(path) as im:
            w, h = im.size
        ratio = max(w / h, h / w)
        (good if ratio <= max_ratio else bad).append((path, w, h, ratio))
    return good, bad


def main():
    a = args()
    if not 0 <= a.noise_probability <= 1 or a.max_aspect_ratio < 1:
        raise ValueError("Invalid probability or aspect ratio")
    source, output = a.source_dir.resolve(), a.output_dir.resolve()
    hr_dir, lr_dir = output / "HR", output / "LR"
    hr_dir.mkdir(parents=True, exist_ok=True); lr_dir.mkdir(parents=True, exist_ok=True)
    eligible, skipped = scan(source, a.max_aspect_ratio)
    if a.limit is not None:
        eligible = eligible[:a.limit]
    (output / "skipped_aspect_ratio.json").write_text(json.dumps([
        {"source": str(p), "width": w, "height": h, "aspect_ratio": r} for p, w, h, r in skipped
    ], ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = output / "degradation_params.jsonl"
    done = set()
    if manifest_path.is_file():
        done = {json.loads(x)["filename"] for x in manifest_path.read_text().splitlines() if x.strip()}
    print(f"eligible={len(eligible)} skipped_ratio={len(skipped)} completed={len(done)}")
    made = 0
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for i, (src, w, h, ratio) in enumerate(eligible, 1):
            name = src.with_suffix(".png").name
            if name in done and (hr_dir / name).is_file() and (lr_dir / name).is_file():
                continue
            hr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            lr, rec = degrade(hr, rng_for(a.seed, name), a.noise_probability)
            if lr.shape != hr.shape:
                raise AssertionError(f"HR/LR mismatch: {name}")
            shutil.copy2(src, hr_dir / name)
            if not cv2.imwrite(str(lr_dir / name), lr, [cv2.IMWRITE_PNG_COMPRESSION, 2]):
                raise RuntimeError(f"Write failed: {name}")
            manifest.write(json.dumps({"filename": name, "source": str(src), "width": w, "height": h,
                "aspect_ratio": ratio, "same_resolution": True, "degradation": rec}, ensure_ascii=False) + "\n")
            manifest.flush(); made += 1
            if made <= 5 or made % 25 == 0:
                print(f"generated={made} progress={i}/{len(eligible)} {name} {w}x{h} {rec['severity']} noise={rec['noise']}")
    pairs = sorted(p.name for p in hr_dir.glob("*.png") if (lr_dir / p.name).is_file())
    lines = []
    for i, name in enumerate(pairs):
        prompt = PROMPTS[i % len(PROMPTS)]
        lines.append(json.dumps({"prompt": prompt, "image": name,
            "template_inputs": {"image": str((lr_dir / name).resolve()), "prompt": prompt}}, ensure_ascii=False))
    (output / "metadata.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    records = [json.loads(x) for x in manifest_path.read_text().splitlines() if x.strip()]
    summary = {"source_images": len(eligible) + len(skipped), "eligible_images": len(eligible),
        "skipped_aspect_ratio": len(skipped), "paired_images": len(pairs),
        "severity_counts": {k: sum(r["degradation"]["severity"] == k for r in records) for k in LEVELS},
        "noise_counts": {k: sum(r["degradation"]["noise"] == k for r in records)
            for k in ["none", "light_gaussian", "light_poisson"]}}
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
