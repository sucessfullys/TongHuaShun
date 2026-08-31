#!/usr/bin/env python3
"""Build HumanRefiner grounding SFT data for Qwen3-VL with ms-swift."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


SYSTEM_PROMPT = (
    "你是人体结构异常检测助手。判断图中人体是否存在结构异常，并给出可见证据。"
    "如存在异常，请在 defects 中输出 id、part、position、bbox_2d 和 reason。"
    "bbox_2d 为异常区域坐标框，最终结论只能是 normal、abnormal 或 non_human。"
)

USER_PROMPTS = [
    "<image>这张图片是否存在人体结构异常？请输出判断、异常类型、异常区域坐标和理由。",
    "<image>请判断图中是否有人体结构异常，并给出异常类型、位置、坐标框和理由。",
    "<image>请检查这张图的人体结构是否异常；如有异常，请指出部位、位置、bbox_2d 和原因。",
    "<image>图中人体是否存在结构异常？请用 defects 给出异常区域、坐标和简洁证据。",
    "<image>请分析图像中的人体结构状态，判断 normal、abnormal 或 non_human，并说明依据。",
]

ABNORMAL_CLASS_TO_PART = {
    1: "head",
    2: "neck",
    3: "body",
    4: "arm",
    5: "hand",
    6: "leg",
    7: "foot",
    8: "multiple_parts",
}

PART_CN = {
    "head": "头部",
    "neck": "颈部",
    "body": "躯干",
    "arm": "手臂",
    "hand": "手部",
    "leg": "腿部",
    "foot": "脚部",
    "multiple_parts": "多部位",
}

REASON_TEMPLATES = {
    "head": [
        "{position}的头部区域存在结构异常，整体形态与正常人体头部特征不一致。",
        "{position}可见头部区域表现不自然，该部位结构与常见人体头部形态不一致。",
        "异常位于{position}的头部区域，该区域头部外观与正常人体结构不协调。",
        "{position}的头部区域存在可见结构异常，局部形态不符合常见人体特征。",
        "{position}处的头部结构表现异常，整体外观与自然人体头部形态不一致。",
        "{position}的头部区域可见异常形态，结构表现与正常人体头部存在差异。",
        "该异常位于{position}的头部区域，局部结构不自然，属于头部结构异常。",
        "{position}附近的头部区域形态不协调，可作为人体结构异常证据。",
    ],
    "neck": [
        "{position}的颈部区域存在结构异常，整体形态与正常人体颈部特征不一致。",
        "{position}可见颈部区域表现不自然，该部位结构与常见人体颈部形态不一致。",
        "异常位于{position}的颈部区域，该区域颈部外观与正常人体结构不协调。",
        "{position}的颈部区域存在可见结构异常，局部形态不符合常见人体特征。",
        "{position}处的颈部结构表现异常，整体外观与自然人体颈部形态不一致。",
        "{position}的颈部区域可见异常形态，结构表现与正常人体颈部存在差异。",
        "该异常位于{position}的颈部区域，局部结构不自然，属于颈部结构异常。",
        "{position}附近的颈部区域形态不协调，可作为人体结构异常证据。",
    ],
    "body": [
        "{position}的躯干区域存在结构异常，整体形态与正常人体躯干特征不一致。",
        "{position}可见躯干区域表现不自然，该部位结构与常见人体躯干形态不一致。",
        "异常位于{position}的躯干区域，该区域躯干外观与正常人体结构不协调。",
        "{position}的躯干区域存在可见结构异常，局部形态不符合常见人体特征。",
        "{position}处的躯干结构表现异常，整体外观与自然人体躯干形态不一致。",
        "{position}的躯干区域可见异常形态，结构表现与正常人体躯干存在差异。",
        "该异常位于{position}的躯干区域，局部结构不自然，属于躯干结构异常。",
        "{position}附近的躯干区域形态不协调，可作为人体结构异常证据。",
    ],
    "arm": [
        "{position}的手臂区域存在结构异常，整体形态与正常人体手臂特征不一致。",
        "{position}可见手臂区域表现不自然，该部位结构与常见人体手臂形态不一致。",
        "异常位于{position}的手臂区域，该区域手臂外观与正常人体结构不协调。",
        "{position}的手臂区域存在可见结构异常，局部形态不符合常见人体特征。",
        "{position}处的手臂结构表现异常，整体外观与自然人体手臂形态不一致。",
        "{position}的手臂区域可见异常形态，结构表现与正常人体手臂存在差异。",
        "该异常位于{position}的手臂区域，局部结构不自然，属于手臂结构异常。",
        "{position}附近的手臂区域形态不协调，可作为人体结构异常证据。",
    ],
    "hand": [
        "{position}的手部区域存在结构异常，整体形态与正常人体手部特征不一致。",
        "{position}可见手部区域表现不自然，该部位结构与常见人体手部形态不一致。",
        "异常位于{position}的手部区域，该区域手部外观与正常人体结构不协调。",
        "{position}的手部区域存在可见结构异常，局部形态不符合常见人体特征。",
        "{position}处的手部结构表现异常，整体外观与自然人体手部形态不一致。",
        "{position}的手部区域可见异常形态，结构表现与正常人体手部存在差异。",
        "该异常位于{position}的手部区域，局部结构不自然，属于手部结构异常。",
        "{position}附近的手部区域形态不协调，可作为人体结构异常证据。",
    ],
    "leg": [
        "{position}的腿部区域存在结构异常，整体形态与正常人体腿部特征不一致。",
        "{position}可见腿部区域表现不自然，该部位结构与常见人体腿部形态不一致。",
        "异常位于{position}的腿部区域，该区域腿部外观与正常人体结构不协调。",
        "{position}的腿部区域存在可见结构异常，局部形态不符合常见人体特征。",
        "{position}处的腿部结构表现异常，整体外观与自然人体腿部形态不一致。",
        "{position}的腿部区域可见异常形态，结构表现与正常人体腿部存在差异。",
        "该异常位于{position}的腿部区域，局部结构不自然，属于腿部结构异常。",
        "{position}附近的腿部区域形态不协调，可作为人体结构异常证据。",
    ],
    "foot": [
        "{position}的脚部区域存在结构异常，整体形态与正常人体脚部特征不一致。",
        "{position}可见脚部区域表现不自然，该部位结构与常见人体脚部形态不一致。",
        "异常位于{position}的脚部区域，该区域脚部外观与正常人体结构不协调。",
        "{position}的脚部区域存在可见结构异常，局部形态不符合常见人体特征。",
        "{position}处的脚部结构表现异常，整体外观与自然人体脚部形态不一致。",
        "{position}的脚部区域可见异常形态，结构表现与正常人体脚部存在差异。",
        "该异常位于{position}的脚部区域，局部结构不自然，属于脚部结构异常。",
        "{position}附近的脚部区域形态不协调，可作为人体结构异常证据。",
    ],
    "multiple_parts": [
        "{position}存在多部位结构异常，整体形态与正常人体结构不一致。",
        "{position}可见多个人体部位表现不自然，整体结构与常见人体形态不协调。",
        "异常位于{position}的多部位区域，该区域包含复合性人体结构异常。",
        "{position}的多部位区域存在可见结构异常，整体表现不符合常见人体特征。",
        "{position}处可见多处人体结构异常，整体外观与自然人体形态不一致。",
        "{position}的区域呈现多部位异常形态，结构表现与正常人体存在差异。",
        "该异常位于{position}，涉及多个身体部位，属于复合型结构异常。",
        "{position}附近的人体多部位形态不协调，可作为人体结构异常证据。",
    ],
}

ABNORMAL_THINKS = [
    "图中存在人体结构异常，异常区域如下。",
    "图像中可见人体结构异常，具体异常区域如下。",
    "该图存在异常人体结构表现，相关区域如下。",
    "可观察到人体局部结构异常，异常信息如下。",
    "图中至少有一处人体结构不自然，标注如下。",
]

NORMAL_RESPONSES = [
    "<think>可见人体部位整体形态正常，没有明显结构异常。</think>\n<defects>[]</defects>\n<conclusion>normal</conclusion>",
    "<think>图中人体比例和结构较为协调，未观察到明确异常。</think>\n<defects>[]</defects>\n<conclusion>normal</conclusion>",
    "<think>画面中的人体结构整体自然，未发现需要定位的异常区域。</think>\n<defects>[]</defects>\n<conclusion>normal</conclusion>",
    "<think>图像中可见人体区域表现正常，没有明显人体结构异常证据。</think>\n<defects>[]</defects>\n<conclusion>normal</conclusion>",
    "<think>当前图像中的人体形态与常见人体结构一致，判断为正常。</think>\n<defects>[]</defects>\n<conclusion>normal</conclusion>",
]

NON_HUMAN_RESPONSES = [
    "<think>图中不存在明确的人类身体结构，因此不是人体结构异常样本。</think>\n<defects>[]</defects>\n<conclusion>non_human</conclusion>",
    "<think>画面中没有可辨识的人体主体，无法判断人体部位是否异常。</think>\n<defects>[]</defects>\n<conclusion>non_human</conclusion>",
    "<think>该图未呈现明确人体对象，不适合作为人体结构异常判断样本。</think>\n<defects>[]</defects>\n<conclusion>non_human</conclusion>",
    "<think>图像内容不包含清晰人体主体，因此结论为非人体样本。</think>\n<defects>[]</defects>\n<conclusion>non_human</conclusion>",
    "<think>没有观察到可用于人体结构分析的人体区域，判断为 non_human。</think>\n<defects>[]</defects>\n<conclusion>non_human</conclusion>",
]


def stable_index(key: str, size: int) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % size


def choose(items: list[str], key: str) -> str:
    return items[stable_index(key, len(items))]


def yolo_to_xyxy_norm(xc: float, yc: float, w: float, h: float) -> list[float]:
    x1 = max(0.0, min(1.0, xc - w / 2))
    y1 = max(0.0, min(1.0, yc - h / 2))
    x2 = max(0.0, min(1.0, xc + w / 2))
    y2 = max(0.0, min(1.0, yc + h / 2))
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def position_desc(bbox: list[float]) -> str:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    if cy < 1 / 3:
        y = "上方"
    elif cy < 2 / 3:
        y = "中部"
    else:
        y = "下方"
    if cx < 1 / 3:
        x = "左侧"
    elif cx < 2 / 3:
        x = ""
    else:
        x = "右侧"
    return f"图像{y}{x}" if x else f"图像{y}"


def parse_label_file(path: Path) -> list[tuple[int, list[float]]]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 columns, got {len(parts)}")
        cls = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        rows.append((cls, yolo_to_xyxy_norm(xc, yc, w, h)))
    return rows


def build_abnormal_response(image_id: str, abnormal_rows: list[tuple[int, list[float]]]) -> tuple[str, list[list[float]]]:
    defects = []
    bboxes = []
    for index, (cls, bbox) in enumerate(abnormal_rows, start=1):
        part = ABNORMAL_CLASS_TO_PART[cls]
        position = position_desc(bbox)
        reason = choose(REASON_TEMPLATES[part], f"{image_id}-{index}-{cls}").format(position=position)
        defects.append(
            f'  {{"id":"defect_{index}","part":"{part}","position":"{position}",'
            f'"bbox_2d":<bbox>,"reason":"{reason}"}}'
        )
        bboxes.append(bbox)
    think = choose(ABNORMAL_THINKS, f"{image_id}-think")
    content = f"<think>{think}</think>\n<defects>[\n" + ",\n".join(defects) + "\n]</defects>\n<conclusion>abnormal</conclusion>"
    return content, bboxes


def build_record(image_id: str, image_path: Path, rows: list[tuple[int, list[float]]]) -> dict:
    abnormal_rows = [(cls, bbox) for cls, bbox in rows if cls in ABNORMAL_CLASS_TO_PART]
    has_non_human = any(cls == 9 for cls, _ in rows)
    if abnormal_rows:
        assistant_content, bboxes = build_abnormal_response(image_id, abnormal_rows)
        conclusion = "abnormal"
    elif has_non_human:
        assistant_content = choose(NON_HUMAN_RESPONSES, f"{image_id}-non-human")
        bboxes = []
        conclusion = "non_human"
    else:
        assistant_content = choose(NORMAL_RESPONSES, f"{image_id}-normal")
        bboxes = []
        conclusion = "normal"

    record = {
        "id": image_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": choose(USER_PROMPTS, f"{image_id}-user")},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [str(image_path)],
    }
    if bboxes:
        record["objects"] = {"bbox": bboxes, "bbox_type": "norm1"}

    n_placeholders = assistant_content.count("<bbox>")
    if n_placeholders != len(bboxes):
        raise ValueError(f"{image_id}: <bbox> count {n_placeholders} != objects.bbox count {len(bboxes)}")
    if conclusion in {"normal", "non_human"} and bboxes:
        raise ValueError(f"{image_id}: {conclusion} sample should not contain bbox")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/HumanRefiner/Data"),
    )
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/"
            "examples/train/body_deformity_qwen3_vl/body_deformity_grounding_base_train.jsonl"
        ),
    )
    args = parser.parse_args()

    image_dir = args.data_root / args.split / "images"
    label_dir = args.data_root / args.split / "labels"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(label_dir.glob("*.txt"), key=lambda p: (0, int(p.stem)) if p.stem.isdigit() else (1, p.stem))
    stats = Counter()
    class_stats = Counter()

    with args.output.open("w", encoding="utf-8") as out:
        for label_path in label_paths:
            image_id = label_path.stem
            image_path = image_dir / f"{image_id}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(f"missing image for label {label_path}: {image_path}")
            rows = parse_label_file(label_path)
            for cls, _ in rows:
                class_stats[cls] += 1
            record = build_record(image_id, image_path.resolve(), rows)
            assistant = record["messages"][-1]["content"]
            conclusion = assistant.split("<conclusion>", 1)[1].split("</conclusion>", 1)[0]
            stats[conclusion] += 1
            stats["records"] += 1
            stats["bbox"] += len(record.get("objects", {}).get("bbox", []))
            out.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"output: {args.output}")
    print(f"records: {stats['records']}")
    print(f"abnormal: {stats['abnormal']}")
    print(f"normal: {stats['normal']}")
    print(f"non_human: {stats['non_human']}")
    print(f"abnormal_bboxes: {stats['bbox']}")
    print("class_counts:", dict(sorted(class_stats.items())))


if __name__ == "__main__":
    main()
