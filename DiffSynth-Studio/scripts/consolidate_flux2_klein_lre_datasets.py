#!/usr/bin/env python3
"""Consolidate FLUX2 Klein LRE training datasets into one HR/LR directory.

Default is dry-run. Use --execute to move files and write metadata.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hr_dir: Path
    lr_dir: Path
    metadata: Path


DATASETS = [
    DatasetSpec("face", Path("/mnt/image-edit/datasets/duanyufa/Face/HR"), Path("/mnt/image-edit/datasets/duanyufa/Face/LR"), Path("/mnt/image-edit/datasets/duanyufa/Face/metadata.jsonl")),
    DatasetSpec("instargram", Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram/HR"), Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram/LR"), Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram/metadata.jsonl")),
    DatasetSpec("xhs", Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/xhs/HR"), Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/xhs/LR"), Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/xhs/metadata.jsonl")),
    DatasetSpec("instargram_new1", Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram_new1/HR"), Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram_new1/LR"), Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/Instargram_new1/metadata.jsonl")),
    DatasetSpec("4klsdb", Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/4KLSDB/images/HR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/4KLSDB/images/LR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/4KLSDB/images/metadata.jsonl")),
    DatasetSpec("descan18k", Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/DESCAN-18K/DESCAN-18K/HR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/DESCAN-18K/DESCAN-18K/LR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/DESCAN-18K/DESCAN-18K/metadata.jsonl")),
    DatasetSpec("shhq", Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/SHHQ-1.0/SHHQ-1.0/HR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/SHHQ-1.0/SHHQ-1.0/LR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/SHHQ-1.0/SHHQ-1.0/metadata.jsonl")),
    DatasetSpec("doc", Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/HR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/LR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/文档图片/metadata.jsonl")),
    DatasetSpec("vitonhd", Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/VITON-HD/HR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/VITON-HD/LR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/VITON-HD/metadata.jsonl")),
    DatasetSpec("ours_v1", Path("/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/HR"), Path("/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/LR"), Path("/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/metadata.jsonl")),
    DatasetSpec("ffhq", Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/FFHQ/ffhq-dataset/HR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/FFHQ/ffhq-dataset/LR"), Path("/mnt/image-edit/datasets/duanyufa/SR_Dataset/FFHQ/ffhq-dataset/metadata.jsonl")),
    DatasetSpec("old_photo2", Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/Old_Photo_2/HR"), Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/Old_Photo_2/LR"), Path("/mnt/image-edit/datasets/duanyufa/Face/Other_data/Old_Photo_2/metadata.jsonl")),
]


def resolve_under(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def load_source_manifest(meta_path: Path) -> dict[str, str]:
    manifest = meta_path.parent / "degradation_params.jsonl"
    source_by_filename: dict[str, str] = {}
    if not manifest.is_file():
        return source_by_filename
    with manifest.open(encoding="utf-8") as f:
        for raw in f:
            if not raw.strip():
                continue
            item = json.loads(raw)
            if "filename" in item and "source" in item:
                source_by_filename[item["filename"]] = item["source"]
    return source_by_filename


def collect_records(output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_hr = output_root / "HR"
    out_lr = output_root / "LR"
    records: list[dict[str, Any]] = []
    missing_hr: list[str] = []
    missing_lr: list[str] = []
    target_collisions: list[str] = []
    seen_targets: set[str] = set()
    per_dataset: list[dict[str, Any]] = []

    for dataset_index, spec in enumerate(DATASETS):
        if not spec.metadata.is_file():
            raise FileNotFoundError(f"Missing metadata: {spec.metadata}")
        source_by_filename = load_source_manifest(spec.metadata)
        count = 0
        with spec.metadata.open(encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                if not raw.strip():
                    continue
                item = json.loads(raw)
                prompt = item.get("prompt", item.get("template_inputs", {}).get("prompt", ""))
                hr_src = resolve_under(spec.hr_dir, item["image"])
                if not hr_src.is_file() and item["image"] in source_by_filename:
                    hr_src = Path(source_by_filename[item["image"]])
                lr_value = item.get("template_inputs", {}).get("image")
                if not lr_value:
                    raise ValueError(f"{spec.metadata}:{line_no}: missing template_inputs.image")
                lr_src = resolve_under(spec.lr_dir, lr_value)

                if not hr_src.is_file():
                    missing_hr.append(str(hr_src))
                if not lr_src.is_file():
                    missing_lr.append(str(lr_src))

                prefix = f"{dataset_index:02d}_{spec.name}_"
                hr_dst = out_hr / f"{prefix}{hr_src.name}"
                lr_dst = out_lr / f"{prefix}{lr_src.name}"
                for target in [str(hr_dst), str(lr_dst)]:
                    if target in seen_targets:
                        target_collisions.append(target)
                    seen_targets.add(target)

                new_item = dict(item)
                new_item["image"] = hr_dst.name
                new_item["source"] = str(hr_dst)
                new_item["prompt"] = prompt
                template_inputs = dict(item.get("template_inputs", {}))
                template_inputs["image"] = str(lr_dst)
                template_inputs["prompt"] = prompt
                new_item["template_inputs"] = template_inputs
                new_item["_dataset_name"] = spec.name
                new_item["_original_hr"] = str(hr_src)
                new_item["_original_lr"] = str(lr_src)

                records.append({
                    "metadata": new_item,
                    "hr_src": str(hr_src),
                    "lr_src": str(lr_src),
                    "hr_dst": str(hr_dst),
                    "lr_dst": str(lr_dst),
                    "dataset_name": spec.name,
                    "dataset_index": dataset_index,
                    "line_no": line_no,
                })
                count += 1
        per_dataset.append({
            "dataset_index": dataset_index,
            "name": spec.name,
            "hr_dir": str(spec.hr_dir),
            "lr_dir": str(spec.lr_dir),
            "metadata": str(spec.metadata),
            "records": count,
        })

    existing_targets = [r["hr_dst"] for r in records if Path(r["hr_dst"]).exists()]
    existing_targets += [r["lr_dst"] for r in records if Path(r["lr_dst"]).exists()]
    summary = {
        "output_root": str(output_root),
        "total_records": len(records),
        "per_dataset": per_dataset,
        "missing_hr_count": len(missing_hr),
        "missing_lr_count": len(missing_lr),
        "missing_hr_examples": missing_hr[:10],
        "missing_lr_examples": missing_lr[:10],
        "target_collision_count": len(target_collisions),
        "target_collision_examples": target_collisions[:10],
        "existing_target_count": len(existing_targets),
        "existing_target_examples": existing_targets[:10],
    }
    return records, summary


def move_one(src: str, dst: str) -> tuple[str, str, str]:
    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.is_file():
        return (src, dst, "missing_src")
    if dst_p.exists():
        return (src, dst, "exists_dst")
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_p), str(dst_p))
    return (src, dst, "moved")


def write_outputs(records: list[dict[str, Any]], output_root: Path, summary: dict[str, Any]) -> None:
    meta_path = output_root / "metadata.jsonl"
    manifest_path = output_root / "move_manifest.jsonl"
    summary_path = output_root / "summary.json"
    with meta_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record["metadata"], ensure_ascii=False) + "\n")
    with manifest_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({k: v for k, v in record.items() if k != "metadata"}, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="/mnt/image-edit/datasets/duanyufa/交接/基模30W增强数据")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--execute", action="store_true", help="Actually move files and write merged metadata.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    records, summary = collect_records(output_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    blocking = any([
        summary["missing_hr_count"],
        summary["missing_lr_count"],
        summary["target_collision_count"],
        summary["existing_target_count"],
    ])
    if blocking:
        raise SystemExit("Preflight failed; no files were moved.")

    if not args.execute:
        print("Dry run only. Add --execute to move files and write metadata.jsonl.")
        return

    (output_root / "HR").mkdir(parents=True, exist_ok=True)
    (output_root / "LR").mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, str]] = []
    for record in records:
        jobs.append((record["hr_src"], record["hr_dst"]))
        jobs.append((record["lr_src"], record["lr_dst"]))

    moved = 0
    errors: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_map = [ex.submit(move_one, src, dst) for src, dst in jobs]
        for i, fut in enumerate(as_completed(future_map), 1):
            src, dst, status = fut.result()
            if status == "moved":
                moved += 1
            else:
                errors.append((src, dst, status))
            if i % 10000 == 0 or i == len(future_map):
                print(f"progress {i}/{len(future_map)} moved={moved} errors={len(errors)}", flush=True)

    summary["move_jobs"] = len(jobs)
    summary["moved_files"] = moved
    summary["move_error_count"] = len(errors)
    summary["move_error_examples"] = errors[:20]
    if errors:
        error_path = output_root / "move_errors.jsonl"
        with error_path.open("w", encoding="utf-8") as f:
            for src, dst, status in errors:
                f.write(json.dumps({"src": src, "dst": dst, "status": status}, ensure_ascii=False) + "\n")
        summary["move_errors_path"] = str(error_path)
        (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        raise SystemExit(f"Move finished with errors: {len(errors)}")

    write_outputs(records, output_root, summary)
    print(f"Done. metadata={output_root / 'metadata.jsonl'}")


if __name__ == "__main__":
    main()
