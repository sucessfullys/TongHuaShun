#!/usr/bin/env python3
"""Create fixed train/val splits for the v3 multi-hand reason dataset."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/body_deformity_multihand_reason_v3_train.jsonl",
    )
    parser.add_argument(
        "--train-output",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/body_deformity_multihand_reason_v3_train_fixed.jsonl",
    )
    parser.add_argument(
        "--val-output",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/body_deformity_multihand_reason_v3_val_fixed.jsonl",
    )
    parser.add_argument("--val-abnormal", type=int, default=60)
    parser.add_argument("--val-normal", type=int, default=100)
    parser.add_argument("--val-non-human", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    input_path = Path(args.input)
    by_label: dict[str, list[dict]] = defaultdict(list)
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            by_label[item["label"]].append(item)

    requested = {
        "abnormal": args.val_abnormal,
        "normal": args.val_normal,
        "non_human": args.val_non_human,
    }
    for label, count in requested.items():
        if count > len(by_label[label]):
            raise SystemExit(f"Not enough {label} records: requested={count}, available={len(by_label[label])}")

    rng = random.Random(args.seed)
    val_ids = set()
    for label, count in requested.items():
        label_records = list(by_label[label])
        rng.shuffle(label_records)
        val_ids.update(id(item) for item in label_records[:count])

    val_records = []
    train_records = []
    for label in sorted(by_label):
        for item in by_label[label]:
            if id(item) in val_ids:
                val_records.append(item)
            else:
                train_records.append(item)

    rng.shuffle(train_records)
    rng.shuffle(val_records)

    train_output = Path(args.train_output)
    val_output = Path(args.val_output)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    val_output.parent.mkdir(parents=True, exist_ok=True)

    with train_output.open("w", encoding="utf-8") as f:
        for i, item in enumerate(train_records, 1):
            item["id"] = f"multihand_v3_train_{i:06d}"
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    with val_output.open("w", encoding="utf-8") as f:
        for i, item in enumerate(val_records, 1):
            item["id"] = f"multihand_v3_val_{i:06d}"
            f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")

    stats = {
        "input": str(input_path),
        "train_output": str(train_output),
        "val_output": str(val_output),
        "seed": args.seed,
        "train_total": len(train_records),
        "val_total": len(val_records),
        "train_labels": dict(Counter(item["label"] for item in train_records)),
        "val_labels": dict(Counter(item["label"] for item in val_records)),
    }
    stats_path = val_output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
