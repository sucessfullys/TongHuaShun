#!/usr/bin/env python3
"""Evaluate multilimb reason v3 LoRA checkpoints.

This evaluator is intentionally separate from the grounding evaluator because
v3 does not train or score bbox output. It evaluates conclusion quality,
response format, and error cases from a folder test set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from multilimb_reason_v3_prompts import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT
USER_PROMPT = DEFAULT_USER_PROMPT

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_MAP = {
    "abnormal": "abnormal",
    "normal": "normal",
    "背景": "non_human",
    "background": "non_human",
    "non_human": "non_human",
    "real_no_human": "non_human",
}
LABELS = ["abnormal", "normal", "non_human"]

CONCLUSION_RE = re.compile(r"<conclusion>\s*(normal|abnormal|non_human)\s*</conclusion>", re.I)
EVIDENCE_RE = re.compile(r"<evidence>\s*(.*?)\s*</evidence>", re.I | re.S)
BBOX_KEY_RE = re.compile(r"\b(?:bbox|bbox_2d)\b|<bbox>", re.I)
FOUR_NUMBER_LIST_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Base Qwen3-VL model path.")
    parser.add_argument("--adapter", type=Path, default=None, help="Optional LoRA adapter checkpoint path.")
    parser.add_argument(
        "--test-root",
        type=Path,
        default=Path("/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/test"),
    )
    parser.add_argument("--eval-jsonl", type=Path, default=None, help="Optional JSONL test file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/test/"
            "multilimb_reason_v3_eval"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-max-token-num", default="2048")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--attn-impl", default=None)
    return parser.parse_args()


def parse_conclusion(text: str) -> str | None:
    match = CONCLUSION_RE.search(text)
    if match:
        return match.group(1).lower()
    lowered = text.lower()
    if "non_human" in lowered or "non-human" in lowered:
        return "non_human"
    if "abnormal" in lowered:
        return "abnormal"
    if "normal" in lowered:
        return "normal"
    return None


def has_bbox_leak(text: str) -> bool:
    if BBOX_KEY_RE.search(text):
        return True
    for match in FOUR_NUMBER_LIST_RE.finditer(text):
        nums = [float(x) for x in match.groups()]
        if all(0 <= x <= 2048 for x in nums) and nums[0] < nums[2] and nums[1] < nums[3]:
            return True
    return False


def make_record(image_path: Path, gt: str) -> dict[str, Any]:
    return {
        "id": image_path.stem,
        "image": str(image_path.resolve()),
        "group": image_path.parent.name,
        "gt_conclusion": gt,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    }


def load_records(test_root: Path, max_samples: int | None) -> list[dict[str, Any]]:
    records_by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for subdir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        gt = LABEL_MAP.get(subdir.name)
        if gt is None:
            continue
        for image_path in sorted(p for p in subdir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
            records_by_label[gt].append(make_record(image_path, gt))
    records = [r for label in LABELS for r in records_by_label[label]]
    if max_samples is None or max_samples >= len(records):
        return records

    selected: list[dict[str, Any]] = []
    cursors = {label: 0 for label in LABELS}
    active_labels = [label for label in LABELS if records_by_label[label]]
    while len(selected) < max_samples and active_labels:
        next_active = []
        for label in active_labels:
            idx = cursors[label]
            if idx < len(records_by_label[label]) and len(selected) < max_samples:
                selected.append(records_by_label[label][idx])
                cursors[label] += 1
            if cursors[label] < len(records_by_label[label]):
                next_active.append(label)
        active_labels = next_active
    return selected


def load_records_from_jsonl(eval_jsonl: Path, max_samples: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with eval_jsonl.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            data = json.loads(line)
            images = data.get("images") or []
            if isinstance(images, str):
                images = [images]
            if not images:
                raise ValueError(f"Missing images at {eval_jsonl}:{line_no}")
            gt = data.get("gt_conclusion") or data.get("conclusion")
            if gt is None:
                assistant = ""
                for message in data.get("messages", []):
                    if message.get("role") == "assistant":
                        assistant = message.get("content", "")
                gt = parse_conclusion(assistant)
            if gt not in LABELS:
                raise ValueError(f"Invalid gt conclusion at {eval_jsonl}:{line_no}: {gt}")
            image_path = Path(images[0])
            records.append(
                {
                    "id": data.get("id") or image_path.stem,
                    "image": str(image_path),
                    "group": data.get("group") or image_path.parent.name,
                    "gt_conclusion": gt,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT},
                    ],
                }
            )
    if max_samples is not None:
        records = records[:max_samples]
    return records


def infer_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": None,
    }
    engine_kwargs = {}
    if args.adapter is not None:
        engine_kwargs["adapters"] = [str(args.adapter)]
    engine = TransformersEngine(
        str(args.model),
        model_type="qwen3_vl",
        max_batch_size=args.batch_size,
        torch_dtype=dtype_map[args.torch_dtype],
        attn_impl=args.attn_impl,
        **engine_kwargs,
    )
    request_config = RequestConfig(max_tokens=args.max_new_tokens, temperature=args.temperature)
    results: list[dict[str, Any]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        batch_ids = ",".join(r["id"] for r in batch)
        print(
            f"[infer_start] {start + 1}-{start + len(batch)}/{len(records)} ids={batch_ids}",
            flush=True,
        )
        requests = [
            InferRequest(messages=r["messages"], images=[r["image"]])
            for r in batch
        ]
        responses = engine.infer(requests, request_config)
        for record, response in zip(batch, responses):
            text = response.choices[0].message.content
            pred = parse_conclusion(text)
            result = {
                "id": record["id"],
                "image": record["image"],
                "group": record["group"],
                "gt_conclusion": record["gt_conclusion"],
                "pred_conclusion": pred,
                "correct": pred == record["gt_conclusion"],
                "has_conclusion_tag": CONCLUSION_RE.search(text) is not None,
                "has_evidence_tag": EVIDENCE_RE.search(text) is not None,
                "bbox_leak": has_bbox_leak(text),
                "response": text,
            }
            results.append(result)
            print(
                f"[{len(results)}/{len(records)}] {record['id']} "
                f"gt={record['gt_conclusion']} pred={pred} correct={result['correct']}",
                flush=True,
            )
    return results


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def build_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = Counter(
        (r["gt_conclusion"], r["pred_conclusion"] or "invalid")
        for r in results
    )
    metrics: dict[str, Any] = {
        "num_samples": len(results),
        "valid_conclusion_rate": safe_div(sum(r["pred_conclusion"] is not None for r in results), len(results)),
        "conclusion_tag_rate": safe_div(sum(r["has_conclusion_tag"] for r in results), len(results)),
        "evidence_tag_rate": safe_div(sum(r["has_evidence_tag"] for r in results), len(results)),
        "bbox_leak_count": sum(r["bbox_leak"] for r in results),
        "accuracy_3class": safe_div(sum(r["correct"] for r in results), len(results)),
        "support_by_gt": dict(Counter(r["gt_conclusion"] for r in results)),
        "support_by_pred": dict(Counter(r["pred_conclusion"] or "invalid" for r in results)),
        "confusion": {f"{gt}->{pred}": n for (gt, pred), n in sorted(confusion.items())},
    }

    macro_f1 = 0.0
    weighted_f1 = 0.0
    per_class: dict[str, Any] = {}
    for label in LABELS:
        tp = confusion[(label, label)]
        fp = sum(confusion[(gt, label)] for gt in LABELS if gt != label)
        fn = sum(confusion[(label, pred)] for pred in LABELS + ["invalid"] if pred != label)
        support = sum(confusion[(label, pred)] for pred in LABELS + ["invalid"])
        scores = prf(tp, fp, fn)
        scores.update({"support": support, "accuracy_within_class": safe_div(tp, support)})
        per_class[label] = scores
        macro_f1 += scores["f1"]
        weighted_f1 += scores["f1"] * support
    metrics["per_class"] = per_class
    metrics["macro_f1_3class"] = macro_f1 / len(LABELS)
    metrics["weighted_f1_3class"] = safe_div(weighted_f1, len(results))

    tp = confusion[("abnormal", "abnormal")]
    fp = sum(confusion[(gt, "abnormal")] for gt in ["normal", "non_human"])
    fn = sum(confusion[("abnormal", pred)] for pred in ["normal", "non_human", "invalid"])
    tn = sum(
        confusion[(gt, pred)]
        for gt in ["normal", "non_human"]
        for pred in ["normal", "non_human", "invalid"]
    )
    metrics["binary_abnormal"] = {**prf(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}
    metrics["false_positive_abnormal_count"] = fp
    metrics["false_negative_abnormal_count"] = fn
    return metrics


def write_outputs(results: list[dict[str, Any]], metrics: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = output_dir / "predictions.jsonl"
    metric_path = output_dir / "metrics.json"
    mis_path = output_dir / "misclassified.jsonl"
    confusion_path = output_dir / "confusion.tsv"
    summary_path = output_dir / "summary.txt"

    with pred_path.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with mis_path.open("w", encoding="utf-8") as f:
        for result in results:
            if not result["correct"]:
                f.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
    with confusion_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["gt", "pred", "count"])
        confusion = Counter((r["gt_conclusion"], r["pred_conclusion"] or "invalid") for r in results)
        for gt in LABELS:
            for pred in LABELS + ["invalid"]:
                writer.writerow([gt, pred, confusion[(gt, pred)]])

    lines = [
        f"num_samples: {metrics['num_samples']}",
        f"accuracy_3class: {metrics['accuracy_3class']:.6f}",
        f"macro_f1_3class: {metrics['macro_f1_3class']:.6f}",
        f"weighted_f1_3class: {metrics['weighted_f1_3class']:.6f}",
        f"valid_conclusion_rate: {metrics['valid_conclusion_rate']:.6f}",
        f"conclusion_tag_rate: {metrics['conclusion_tag_rate']:.6f}",
        f"evidence_tag_rate: {metrics['evidence_tag_rate']:.6f}",
        f"bbox_leak_count: {metrics['bbox_leak_count']}",
        f"abnormal_precision: {metrics['binary_abnormal']['precision']:.6f}",
        f"abnormal_recall: {metrics['binary_abnormal']['recall']:.6f}",
        f"abnormal_f1: {metrics['binary_abnormal']['f1']:.6f}",
        f"abnormal_fp: {metrics['binary_abnormal']['fp']}",
        f"abnormal_fn: {metrics['binary_abnormal']['fn']}",
        "",
        "per_class:",
    ]
    for label in LABELS:
        item = metrics["per_class"][label]
        lines.append(
            f"  {label}: precision={item['precision']:.6f}, recall={item['recall']:.6f}, "
            f"f1={item['f1']:.6f}, support={item['support']}"
        )
    lines.append("")
    lines.append("confusion:")
    for key, value in metrics["confusion"].items():
        lines.append(f"  {key}: {value}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"predictions: {pred_path}")
    print(f"metrics: {metric_path}")
    print(f"misclassified: {mis_path}")
    print(f"confusion: {confusion_path}")
    print(f"summary: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", args.image_max_token_num)

    if args.eval_jsonl is not None:
        records = load_records_from_jsonl(args.eval_jsonl, args.max_samples)
    else:
        records = load_records(args.test_root, args.max_samples)
    if args.shard_count > 1:
        total = len(records)
        records = records[args.shard_index::args.shard_count]
        print(f"Shard {args.shard_index}/{args.shard_count}: {len(records)} records from {total}")

    results = infer_records(args, records)
    metrics = build_metrics(results)
    write_outputs(results, metrics, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
