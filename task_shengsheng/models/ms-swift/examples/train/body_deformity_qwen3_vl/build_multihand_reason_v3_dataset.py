#!/usr/bin/env python3
"""Build the v3 multi-hand abnormality SFT dataset without bbox labels."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from multihand_reason_v3_prompts import SYSTEM_PROMPTS, USER_PROMPTS

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ABNORMAL_EVIDENCE = [
    "图中手部数量和连接关系不符合正常人体结构，可见手腕或前臂附近出现额外手部，超过正常双手数量。",
    "画面中的手部结构不自然，同一人体附近出现超过正常双手数量的手部区域，手与手臂的衔接关系异常。",
    "图中人体手部结构异常，可见额外手掌或手部从手腕附近延伸出来，整体不符合正常解剖连接。",
    "图中手部与前臂的连接关系异常，出现手臂附近额外分叉出手部结构的情况，因此属于多手异常。",
    "画面中人体手部区域存在异常增生或分叉表现，手部数量超过正常双手结构。",
    "图中可见额外手部与原有手臂区域混在一起，手部边界和连接方式不符合正常人体结构。",
    "人体手部附近出现不自然的重复或分叉结构，整体表现为超过正常双手数量的异常。",
    "图中手掌与手臂的对应关系异常，可见额外手部结构，属于明显的多手异常表现。",
    "画面里手部区域存在不合理扩展，额外手部与人体主体连接不自然，超过正常人体结构范围。",
    "该图的人体手部区域可见异常连接，手的数量和空间关系不符合正常双手结构。",
]

NORMAL_EVIDENCE = [
    "图中可见人体结构整体自然，未观察到额外手部、手腕分叉或超过正常双手数量的异常连接。",
    "图中没有明显多余手掌或手臂分叉现象，人体手部结构和连接关系看起来正常。",
    "画面中人体手部没有异常增加，未见同一前臂连接额外手部的结构，整体属于正常人体样本。",
    "可见人体手部与手臂的连接关系较自然，没有出现额外手部或异常分叉。",
    "图中人体手部区域没有超过正常双手数量的表现，整体结构符合常见人体特征。",
    "画面中的人体主体清晰，手部数量和位置关系自然，未见多手异常。",
    "人体手臂和手掌的对应关系正常，没有观察到重复手部或异常增生。",
    "图中可见人体姿态正常，手部结构没有出现不合理的额外连接。",
]

NON_HUMAN_EVIDENCE = [
    "图中没有明确可判断的人体主体或手部区域，因此不属于人体多手异常判断样本。",
    "画面中未出现清晰人体或手部结构，无法形成多手异常判断，因此应归为非人体样本。",
    "该图没有可见的人体主体，不能进行人体手部异常判断。",
    "图中缺少可识别的人体和手部区域，因此不属于正常或异常人体样本。",
    "画面主要为背景内容，没有明确人体手部结构，应判断为非人体样本。",
    "图片中没有清楚的人体目标，也没有可检查的手部连接关系。",
]


def iter_images(path: Path) -> list[Path]:
    return sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def build_record(
    idx: int,
    image: Path,
    label: str,
    source: str,
    evidence_pool: list[str],
    rng: random.Random,
) -> dict:
    return {
        "id": f"multihand_v3_{idx:06d}",
        "task_type": "multi_hand_abnormal_classification",
        "label": label,
        "source": source,
        "images": [str(image)],
        "messages": [
            {"role": "system", "content": rng.choice(SYSTEM_PROMPTS)},
            {"role": "user", "content": rng.choice(USER_PROMPTS)},
            {
                "role": "assistant",
                "content": f"<evidence>{rng.choice(evidence_pool)}</evidence>\n<conclusion>{label}</conclusion>",
            },
        ],
    }


def add_records(
    records: list[dict],
    images: Iterable[Path],
    label: str,
    source: str,
    evidence_pool: list[str],
    rng: random.Random,
) -> None:
    for image in images:
        records.append(build_record(len(records) + 1, image, label, source, evidence_pool, rng))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--abnormal-dir",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/abnormal_选择",
    )
    parser.add_argument(
        "--abnormal-bg-dir",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/abnormal_选择_bg_aug/base9b_bg_cfg4.0_step28_seed42",
    )
    parser.add_argument(
        "--normal-dir",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours_bad/hand_foder/normal_human",
    )
    parser.add_argument(
        "--non-human-dir",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/Open_dataset/Ours/Version1/背景",
    )
    parser.add_argument(
        "--output",
        default="/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/body_deformity_multihand_reason_v3_train.jsonl",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    abnormal = iter_images(Path(args.abnormal_dir))
    abnormal_bg = iter_images(Path(args.abnormal_bg_dir))
    normal = iter_images(Path(args.normal_dir))
    non_human = iter_images(Path(args.non_human_dir))

    records: list[dict] = []
    add_records(records, abnormal, "abnormal", "abnormal_selected", ABNORMAL_EVIDENCE, rng)
    add_records(records, abnormal_bg, "abnormal", "abnormal_bg_aug", ABNORMAL_EVIDENCE, rng)
    add_records(records, normal, "normal", "normal_human", NORMAL_EVIDENCE, rng)
    add_records(records, non_human, "non_human", "background_no_human", NON_HUMAN_EVIDENCE, rng)
    rng.shuffle(records)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for i, record in enumerate(records, 1):
            record["id"] = f"multihand_v3_{i:06d}"
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    stats = {
        "output": str(output),
        "total": len(records),
        "abnormal_selected": len(abnormal),
        "abnormal_bg_aug": len(abnormal_bg),
        "normal_human": len(normal),
        "background_no_human": len(non_human),
        "seed": args.seed,
    }
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
