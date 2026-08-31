#!/usr/bin/env python3
"""
Download all reference images from CSV's home_img column to a local directory.
"""
import csv
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

CSV_PATH = "/mnt/data/image-edit/datasets/shensheng/code/stable/DreamFaceOmini/exp_out/csv-fluxout-1k/csv-fluxout.csv"
OUTPUT_DIR = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/Ref_image"

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Read CSV
rows = []
for encoding in ["utf-8-sig", "utf-8"]:
    try:
        with open(CSV_PATH, "r", encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        break
    except UnicodeDecodeError:
        continue

print(f"Total rows: {len(rows)}")

# Collect unique URLs
urls = {}
for row in rows:
    row_id = row.get("id", "")
    url = row.get("home_img", "").strip()
    if url:
        urls[row_id] = url

print(f"Unique URLs: {len(urls)}")

# Download
success = 0
failed = 0
for row_id, url in tqdm(urls.items(), desc="Downloading", unit="img"):
    # Determine file extension from URL
    parsed = urlparse(url)
    ext = os.path.splitext(parsed.path)[1] or ".png"
    out_path = os.path.join(OUTPUT_DIR, f"{row_id}{ext}")

    if os.path.exists(out_path):
        success += 1
        continue

    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            f.write(resp.content)
        success += 1
    except Exception as e:
        tqdm.write(f"  [{row_id}] FAILED: {e}")
        failed += 1

print(f"\nDone.  Success: {success}, Failed: {failed}")
print(f"Images saved to: {OUTPUT_DIR}")
