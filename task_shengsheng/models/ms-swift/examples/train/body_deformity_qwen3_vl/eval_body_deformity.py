#!/usr/bin/env python3
"""Evaluate body deformity grounding SFT checkpoints.

The script supports:
1. single image inference;
2. JSONL dataset inference;
3. conclusion accuracy and bbox IoU metrics for labeled JSONL data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = (
    "你是人体结构异常检测助手。请判断图中是否存在明确的人体结构异常，并给出可见证据。"
    "只要图中存在可辨识的真实人体或人体部位，即使只出现手、脚、头部、面部、手臂、腿部或躯干，也不应判为 non_human。"
    "只有当身体部位数量、形态、比例、关节或连接结构存在明确错误，且不能由正常场景姿势、透视、遮挡、裁切、衣物、鞋袜、道具、模糊或多人重叠解释时，才判为 abnormal。"
    "裁切、遮挡、局部显示、正常姿态、运动动作、透视缩放、衣物遮挡或画面模糊，不得单独作为异常依据。"
    "无法确认异常时按 normal 处理。只有 abnormal 样本才输出 defects 和 bbox_2d；normal 和 non_human 不要输出 defects。"
    "最终结论只能是 normal、abnormal 或 non_human。"
)

USER_PROMPT = (
    "<image>请仅根据图中清晰可见的内容，判断人体或人体局部是否存在明确的结构错误。"
    "不要把裁切、遮挡、局部显示、正常姿态、透视变化、衣物、模糊或多人肢体重叠误判为异常。"
    "只要有可辨识的人体或人体部位，就不要判为 non_human；无法确认异常时按 normal 处理，并严格按规定格式输出。"
)
ABNORMAL_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8}
CLASS_TO_PART = {
    1: "head",
    2: "neck",
    3: "body",
    4: "arm",
    5: "hand",
    6: "leg",
    7: "foot",
    8: "multiple_parts",
}

CONCLUSION_RE = re.compile(r"<conclusion>\s*(normal|abnormal|non_human)\s*</conclusion>", re.I)
BBOX_FIELD_RE = re.compile(r'"?bbox_2d"?\s*:\s*\[([^\]]+)\]', re.I)
PART_FIELD_RE = re.compile(r'"?part"?\s*:\s*"([^"]+)"', re.I)
FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?")
DEFECTS_BLOCK_RE = re.compile(r"<defects>\s*(.*?)\s*</defects>", re.I | re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/output/"
            "body_deformity_qwen3_vl_grounding_v2_schemeA_full_sft/"
            "frombase_ep1_lr1e-5_gb16_img1536_len4096_dszero2_val0.02_20260716_174225/"
            "v0-20260716-174353/checkpoint-4199"
        ),
        help="Full model checkpoint, or base model path when --adapter is set.",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Optional LoRA/adapter checkpoint path. If set, --model should be the base model path.",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=Path,
        default=Path(
            "/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/"
            "examples/train/body_deformity_qwen3_vl/body_deformity_grounding_base_val.jsonl"
        ),
    )
    parser.add_argument("--image", type=Path, help="Run inference on one image and print the response.")
    parser.add_argument("--image-dir", type=Path, help="Run inference on all images in a folder.")
    parser.add_argument("--label-dir", type=Path, help="Optional YOLO label folder for --image-dir metrics.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSONL path for predictions.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-max-token-num", default="1536")
    parser.add_argument("--iou-thresholds", default="0.3,0.5")
    parser.add_argument("--attn-impl", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--no-metrics", action="store_true", help="Only save predictions; skip metric computation.")
    parser.add_argument("--draw-dir", type=Path, help="Optional directory to save GT and prediction visualization images.")
    return parser.parse_args()


def read_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
    return rows


def yolo_to_xyxy_norm(xc: float, yc: float, w: float, h: float) -> list[float]:
    return [
        round(max(0.0, min(1.0, xc - w / 2)), 6),
        round(max(0.0, min(1.0, yc - h / 2)), 6),
        round(max(0.0, min(1.0, xc + w / 2)), 6),
        round(max(0.0, min(1.0, yc + h / 2)), 6),
    ]


def read_yolo_label(path: Path) -> tuple[str, list[list[float]], list[str]]:
    rows = []
    parts_out = []
    if not path.exists():
        return "normal", rows, parts_out
    classes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}: expected YOLO rows with 5 columns")
        cls = int(float(parts[0]))
        classes.append(cls)
        if cls in ABNORMAL_CLASSES:
            rows.append(yolo_to_xyxy_norm(*map(float, parts[1:])))
            parts_out.append(CLASS_TO_PART[cls])
    if rows:
        return "abnormal", rows, parts_out
    if 9 in classes:
        return "non_human", [], []
    return "normal", [], []


def image_records_from_dir(image_dir: Path, label_dir: Path | None = None, max_samples: int | None = None) -> list[dict[str, Any]]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if max_samples is not None:
        paths = paths[:max_samples]
    records = []
    for path in paths:
        record = make_unlabeled_record(path)
        if label_dir is not None:
            conclusion, bboxes, parts = read_yolo_label(label_dir / f"{path.stem}.txt")
            record["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        "<normal_regions>[]</normal_regions>\n"
                        "<defects>[]</defects>\n"
                        f"<conclusion>{conclusion}</conclusion>"
                    ),
                }
            )
            if bboxes:
                record["objects"] = {"bbox": bboxes, "bbox_type": "norm1"}
                record["_gt_parts"] = parts
        records.append(record)
    return records


def make_unlabeled_record(image_path: Path) -> dict[str, Any]:
    return {
        "id": image_path.stem,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        "images": [str(image_path.resolve())],
    }


def request_from_record(record: dict[str, Any]) -> Any:
    from swift import InferRequest

    messages = [m for m in record["messages"] if m["role"] != "assistant"]
    return InferRequest(messages=messages, images=record["images"])


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


def normalize_bbox(nums: list[float]) -> list[float] | None:
    if len(nums) != 4:
        return None
    x1, y1, x2, y2 = nums
    if max(abs(v) for v in nums) > 1.5:
        # Qwen grounding commonly uses 0-1000 coordinates.
        x1, y1, x2, y2 = [v / 1000.0 for v in nums]
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(1.0, max(0.0, x1))
    y1 = min(1.0, max(0.0, y1))
    x2 = min(1.0, max(0.0, x2))
    y2 = min(1.0, max(0.0, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def parse_bboxes(text: str) -> list[list[float]]:
    block = DEFECTS_BLOCK_RE.search(text)
    if not block:
        return []
    defects_text = block.group(1)
    bboxes = []
    for match in BBOX_FIELD_RE.finditer(defects_text):
        nums = [float(x) for x in FLOAT_RE.findall(match.group(1))]
        bbox = normalize_bbox(nums[:4])
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def parse_parts(text: str) -> list[str]:
    block = DEFECTS_BLOCK_RE.search(text)
    if not block:
        return []
    return [match.group(1).strip() for match in PART_FIELD_RE.finditer(block.group(1))]


def gt_conclusion(record: dict[str, Any]) -> str | None:
    for message in record["messages"]:
        if message["role"] == "assistant":
            return parse_conclusion(message["content"])
    return None


def gt_bboxes(record: dict[str, Any]) -> list[list[float]]:
    return record.get("objects", {}).get("bbox", [])


def gt_parts(record: dict[str, Any]) -> list[str]:
    if "_gt_parts" in record:
        return record["_gt_parts"]
    for message in record["messages"]:
        if message["role"] == "assistant":
            parts = parse_parts(message["content"])
            if parts:
                return parts
    return ["unknown"] * len(gt_bboxes(record))


def binary_label(label: str | None) -> str | None:
    if label is None:
        return None
    return "abnormal" if label == "abnormal" else "non_abnormal"


def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def order_ious(gt: list[list[float]], pred: list[list[float]]) -> list[float]:
    return [iou(g, p) for g, p in zip(gt, pred)]


def greedy_matches(gt: list[list[float]], pred: list[list[float]], threshold: float) -> int:
    used_pred = set()
    matches = 0
    for g in gt:
        best_i = None
        best_iou = 0.0
        for idx, p in enumerate(pred):
            if idx in used_pred:
                continue
            score = iou(g, p)
            if score > best_iou:
                best_iou = score
                best_i = idx
        if best_i is not None and best_iou >= threshold:
            used_pred.add(best_i)
            matches += 1
    return matches


def greedy_part_matches(
    gt: list[list[float]],
    pred: list[list[float]],
    gt_parts_value: list[str],
    pred_parts_value: list[str],
    threshold: float,
) -> int:
    used_pred = set()
    matches = 0
    for gt_index, gt_box in enumerate(gt):
        gt_part = gt_parts_value[gt_index] if gt_index < len(gt_parts_value) else None
        best_i = None
        best_iou = 0.0
        for pred_index, pred_box in enumerate(pred):
            if pred_index in used_pred:
                continue
            pred_part = pred_parts_value[pred_index] if pred_index < len(pred_parts_value) else None
            if gt_part != pred_part:
                continue
            score = iou(gt_box, pred_box)
            if score > best_iou:
                best_iou = score
                best_i = pred_index
        if best_i is not None and best_iou >= threshold:
            used_pred.add(best_i)
            matches += 1
    return matches


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def build_metrics(results: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    labeled = [r for r in results if r.get("gt_conclusion")]
    metrics["num_samples"] = len(results)
    metrics["num_labeled"] = len(labeled)

    valid_pred = [r for r in labeled if r.get("pred_conclusion")]
    metrics["valid_conclusion_rate"] = safe_div(len(valid_pred), len(labeled))
    metrics["conclusion_accuracy_3class"] = safe_div(
        sum(r["gt_conclusion"] == r.get("pred_conclusion") for r in labeled), len(labeled)
    )
    metrics["conclusion_accuracy_binary_abnormal"] = safe_div(
        sum(binary_label(r["gt_conclusion"]) == binary_label(r.get("pred_conclusion")) for r in labeled),
        len(labeled),
    )

    confusion = Counter((r["gt_conclusion"], r.get("pred_conclusion") or "invalid") for r in labeled)
    metrics["conclusion_confusion"] = {f"{gt}->{pred}": n for (gt, pred), n in sorted(confusion.items())}

    gt_total = sum(len(r["gt_bboxes"]) for r in labeled)
    pred_total = sum(len(r["pred_bboxes"]) for r in labeled)
    metrics["gt_bbox_count"] = gt_total
    metrics["pred_bbox_count"] = pred_total

    ordered = []
    ordered_per_gt = []
    for r in labeled:
        ordered.extend(order_ious(r["gt_bboxes"], r["pred_bboxes"]))
        for index, gt_box in enumerate(r["gt_bboxes"]):
            if index < len(r["pred_bboxes"]):
                ordered_per_gt.append(iou(gt_box, r["pred_bboxes"][index]))
            else:
                ordered_per_gt.append(0.0)
    metrics["ordered_bbox_mean_iou"] = safe_div(sum(ordered), len(ordered))
    metrics["ordered_bbox_mean_iou_per_gt"] = safe_div(sum(ordered_per_gt), len(ordered_per_gt))

    for threshold in thresholds:
        matches = sum(greedy_matches(r["gt_bboxes"], r["pred_bboxes"], threshold) for r in labeled)
        precision = safe_div(matches, pred_total)
        recall = safe_div(matches, gt_total)
        metrics[f"bbox_iou@{threshold:g}_matches"] = matches
        metrics[f"bbox_iou@{threshold:g}_precision"] = precision
        metrics[f"bbox_iou@{threshold:g}_recall"] = recall
        metrics[f"bbox_iou@{threshold:g}_f1"] = safe_div(2 * precision * recall, precision + recall)

        part_matches = sum(
            greedy_part_matches(
                r["gt_bboxes"],
                r["pred_bboxes"],
                r.get("gt_parts", []),
                r.get("pred_parts", []),
                threshold,
            )
            for r in labeled
        )
        part_precision = safe_div(part_matches, pred_total)
        part_recall = safe_div(part_matches, gt_total)
        metrics[f"bbox_part_iou@{threshold:g}_matches"] = part_matches
        metrics[f"bbox_part_iou@{threshold:g}_precision"] = part_precision
        metrics[f"bbox_part_iou@{threshold:g}_recall"] = part_recall
        metrics[f"bbox_part_iou@{threshold:g}_f1"] = safe_div(
            2 * part_precision * part_recall, part_precision + part_recall
        )

    return metrics


def infer_records(args: argparse.Namespace, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import torch
    from swift import RequestConfig, TransformersEngine

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

    results = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        responses = engine.infer([request_from_record(r) for r in batch], request_config)
        for record, response in zip(batch, responses):
            text = response.choices[0].message.content
            result = {
                "id": record.get("id"),
                "image": record["images"][0],
                "response": text,
                "pred_conclusion": parse_conclusion(text),
                "pred_bboxes": parse_bboxes(text),
                "pred_parts": parse_parts(text),
            }
            gt_label = gt_conclusion(record)
            if gt_label is not None:
                result["gt_conclusion"] = gt_label
                result["gt_bboxes"] = gt_bboxes(record)
                result["gt_parts"] = gt_parts(record)
            results.append(result)
            print(f"[{len(results)}/{len(records)}] {result['id']} pred={result['pred_conclusion']} "
                  f"pred_boxes={len(result['pred_bboxes'])}")
    return results


def draw_one_image(image_path: Path, boxes: list[list[float]], parts: list[str], conclusion: str | None, output_path: Path,
                   *, color: tuple[int, int, int], prefix: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size

    title = f"{prefix} conclusion={conclusion or 'unknown'} boxes={len(boxes)}"
    draw.rectangle([0, 0, min(width, 8 * len(title) + 8), 18], fill=(0, 0, 0))
    draw.text((4, 4), title, fill=(255, 255, 255), font=font)

    for index, bbox in enumerate(boxes, start=1):
        x1, y1, x2, y2 = bbox
        xyxy = [round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)]
        label_part = parts[index - 1] if index - 1 < len(parts) else "unknown"
        label = f"{prefix} {index}:{label_part}"
        for offset in range(3):
            draw.rectangle(
                [xyxy[0] - offset, xyxy[1] - offset, xyxy[2] + offset, xyxy[3] + offset],
                outline=color,
            )
        text_w = max(50, 7 * len(label))
        text_box = [xyxy[0], max(0, xyxy[1] - 16), min(width, xyxy[0] + text_w), xyxy[1]]
        draw.rectangle(text_box, fill=color)
        draw.text((text_box[0] + 2, text_box[1] + 2), label, fill=(255, 255, 255), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def draw_results(results: list[dict[str, Any]], draw_dir: Path) -> None:
    gt_dir = draw_dir / "gt"
    pred_dir = draw_dir / "pred"
    for result in results:
        image_path = Path(result["image"])
        stem = f"{result.get('id') or image_path.stem}"
        pred_parts = result.get("pred_parts") or parse_parts(result.get("response", ""))
        gt_parts_value = result.get("gt_parts") or ["unknown"] * len(result.get("gt_bboxes", []))
        if result.get("gt_conclusion") is not None:
            draw_one_image(
                image_path,
                result.get("gt_bboxes", []),
                gt_parts_value,
                result.get("gt_conclusion"),
                gt_dir / f"{stem}_gt.jpg",
                color=(0, 180, 80),
                prefix="GT",
            )
        draw_one_image(
            image_path,
            result.get("pred_bboxes", []),
            pred_parts,
            result.get("pred_conclusion"),
            pred_dir / f"{stem}_pred.jpg",
            color=(220, 40, 40),
            prefix="PRED",
        )


def main() -> None:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < shard_count")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("IMAGE_MAX_TOKEN_NUM", args.image_max_token_num)
    os.environ.setdefault("QWENVL_BBOX_FORMAT", "new")

    if args.image:
        records = [make_unlabeled_record(args.image)]
    elif args.image_dir:
        records = image_records_from_dir(args.image_dir, args.label_dir, args.max_samples)
    else:
        records = read_jsonl(args.eval_jsonl, args.max_samples)

    if args.shard_count > 1:
        total_records = len(records)
        records = records[args.shard_index::args.shard_count]
        print(
            f"Shard {args.shard_index}/{args.shard_count}: "
            f"{len(records)} records from {total_records} selected records"
        )

    if args.output is None:
        args.output = Path(
            "/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/output/"
            "body_deformity_qwen3_vl_grounding_v2_schemeA_eval/predictions.jsonl"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = infer_records(args, records)
    with args.output.open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"\npredictions: {args.output}")

    if args.draw_dir is not None:
        draw_results(results, args.draw_dir)
        print(f"visualizations: {args.draw_dir}")

    if args.image:
        print("\nresponse:\n" + results[0]["response"])
        return

    if not args.no_metrics and any("gt_conclusion" in r for r in results):
        thresholds = [float(x) for x in args.iou_thresholds.split(",") if x.strip()]
        metrics = build_metrics(results, thresholds)
        metric_path = args.output.with_suffix(".metrics.json")
        metric_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"metrics: {metric_path}")
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
