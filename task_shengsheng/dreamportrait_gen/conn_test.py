#!/usr/bin/env python3
"""Quick connection test for DreamFace HTTP service."""

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def collect_images(path):
    path = Path(path)
    if path.is_file():
        return [path]
    files = [item for item in sorted(path.iterdir()) if item.suffix.lower() in IMAGE_SUFFIXES]
    return files[:3]


def predict(image_paths, serving_url, save_path):
    pics = [encode_image(path) for path in image_paths]
    payload = {
        "prompt": os.environ.get("DREAMFACE_PROMPT", "make the person look natural, high quality"),
        "pics": pics,
        "seed": int(os.environ.get("DREAMFACE_SEED", "42")),
        "steps": int(os.environ.get("DREAMFACE_STEPS", "4")),
        "cfg": float(os.environ.get("DREAMFACE_CFG", "1.0")),
        "height": int(os.environ.get("DREAMFACE_HEIGHT", "1152")),
        "width": int(os.environ.get("DREAMFACE_WIDTH", "896")),
    }

    start_time = time.time()
    response = requests.post(serving_url, json=payload, timeout=600)
    cost_time = time.time() - start_time
    result = json.loads(response.text)

    print("-" * 70)
    print(f"files: {[str(path) for path in image_paths]}")
    print(f"cost time: {cost_time:.2f}s")
    print(f"code: {result.get('code')}, msg: {result.get('msg')}")

    if result.get("code") == 0:
        os.makedirs(save_path, exist_ok=True)
        image_data = base64.b64decode(result["data"]["img"])
        out_path = os.path.join(save_path, "result.png")
        with open(out_path, "wb") as f:
            f.write(image_data)
        print(f"saved: {out_path}")
    return result


if __name__ == "__main__":
    host = os.environ.get("DREAMFACE_HOST", "127.0.0.1")
    port = int(os.environ.get("DREAMFACE_PORT", "9001"))
    img_path = sys.argv[1] if len(sys.argv) > 1 else "test_imgs"
    save_path = sys.argv[2] if len(sys.argv) > 2 else "output"
    serving_url = f"http://{host}:{port}/image/dreamface"

    image_paths = collect_images(img_path)
    if not image_paths:
        raise RuntimeError(f"No test images found: {img_path}")
    predict(image_paths, serving_url, save_path)
