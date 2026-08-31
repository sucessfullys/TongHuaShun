#!/usr/bin/env python3
"""Build v2 Scheme-A SFT JSONL for human structure abnormality detection.

Scheme A:
- Abnormal samples keep defect boxes from the v1 JSONL.
- Non-human samples keep empty defects.
- Real normal samples from Ours use no bbox/objects, to teach conservative normal decisions.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/mnt/image-edit/datasets/duanyufa")
EXAMPLE_DIR = ROOT / "task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl"
V1_JSONL = EXAMPLE_DIR / "body_deformity_grounding_base_train.jsonl"
OUT_JSONL = EXAMPLE_DIR / "body_deformity_grounding_v2_schemeA_train.jsonl"
SUMMARY_JSON = EXAMPLE_DIR / "body_deformity_grounding_v2_schemeA_train.summary.json"

HUMAN_REFINER_TRAIN = ROOT / "task_shengsheng/Open_dataset/HumanRefiner/Data/train"
HUMAN_REFINER_IMAGE_DIR = HUMAN_REFINER_TRAIN / "images"
HUMAN_REFINER_LABEL_DIR = HUMAN_REFINER_TRAIN / "labels"

OURS_NORMAL_DIRS = [
    ROOT / "task_shengsheng/Open_dataset/Ours/Version1/images",
    ROOT / "task_shengsheng/Open_dataset/Ours/Version1/images_1",
]

RANDOM_SEED = 20260716
MAX_ABNORMAL_BBOX = 9
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SYSTEM_PROMPT = (
    "你是人体结构异常检测助手。请判断图中是否存在明确的人体结构异常，并给出可见证据。"
    "只要图中存在可辨识的真实人体或人体部位，即使只出现手、脚、头部、面部、手臂、腿部或躯干，也不应判为 non_human。"
    "只有当身体部位数量、形态、比例、关节或连接结构存在明确错误，且不能由正常场景姿势、透视、遮挡、裁切、衣物、鞋袜、道具、模糊或多人重叠解释时，才判为 abnormal。"
    "裁切、遮挡、局部显示、正常姿态、运动动作、透视缩放、衣物遮挡或画面模糊，不得单独作为异常依据。"
    "无法确认异常时按 normal 处理。只有 abnormal 样本才输出 defects 和 bbox_2d；normal 和 non_human 不要输出 defects。"
    "最终结论只能是 normal、abnormal 或 non_human。"
)

USER_PROMPTS = [
    "<image>请仅根据图中清晰可见的内容，判断人体或人体局部是否存在明确的结构错误。不要把裁切、遮挡、局部显示、正常姿态、透视变化、衣物、模糊或多人肢体重叠误判为异常。只要有可辨识的人体或人体部位，就不要判为 non_human；无法确认异常时按 normal 处理，并严格按规定格式输出。",
    "<image>请判断图中是否存在明确人体结构异常。局部身体、遮挡、裁切、正常姿势或透视变化不直接视为异常；只有可确认的结构错误才输出异常区域、坐标和理由。",
    "<image>请检查画面中的人体或人体局部是否有明确结构错误。若图中只是局部显示、被遮挡、正常动作或衣物影响，请不要误判为 abnormal。",
    "<image>请基于可见证据判断 normal、abnormal 或 non_human。只要存在真实人体或人体部位，就不要判为 non_human；无法确认异常时输出 normal。",
    "<image>请分析图中人体结构状态。只有身体部位数量、形态、比例、关节或连接关系存在明确错误时，才输出 defects 和 bbox_2d。",
    "<image>请判断图中人体、手部、脚部、头部或其他局部身体区域是否存在真实结构异常；正常裁切、遮挡、姿态和场景透视不能单独作为异常依据。",
    "<image>请只依据图中可确认的人体结构证据进行判断。没有明确异常时不要猜测，不要输出 defects，并按 normal 处理。",
    "<image>请评估这张图是否存在人体结构异常。若只是画面裁切、局部人体、多人重叠、衣物遮挡、模糊或正常运动姿态，请保持保守判断。",
]

ABNORMAL_THINKS = [
    "图中存在明确人体结构异常，异常区域如下。",
    "可见人体局部结构与正常解剖形态不一致，异常信息如下。",
    "画面中至少有一处人体结构错误，相关区域如下。",
    "图中可观察到清晰的人体结构异常，异常区域如下。",
    "该图存在不能由正常姿势或遮挡解释的人体结构异常，标注如下。",
    "根据可见区域判断，图中存在明确异常人体部位，具体如下。",
    "图中有可确认的结构错误，异常部位和证据如下。",
    "画面中局部人体结构不符合正常人体特征，异常信息如下。",
    "可见人体部位存在明确形态或连接关系异常，区域如下。",
    "该图存在明显人体结构问题，以下区域需要判为异常。",
    "结合画面可见证据，图中存在确定的人体结构异常。",
    "异常并非单纯由裁切、姿态或透视造成，相关区域如下。",
]

NORMAL_THINKS = [
    "图中可见正常人体或局部人体结构，在当前场景和姿势下未见明确结构异常。",
    "画面中的人体区域虽然存在局部显示或裁切，但姿势关系和物理逻辑基本正常，未见异常畸形。",
    "该图包含可见人体或局部肢体区域，结合场景观察，未发现明显人体结构异常。",
    "图中人体在当前动作和构图下呈现正常结构关系，没有可确认的异常畸形表现。",
    "可见人体部位符合当前场景中的正常姿态和遮挡关系，未见明确结构错误。",
    "图中人体或局部人体区域没有显示出可确认的数量、比例、关节或连接异常。",
    "虽然画面只呈现部分人体区域，但可见结构符合正常人体表现，应按 normal 处理。",
    "当前图像中的人体姿势和局部显示具有合理场景解释，没有明确异常依据。",
    "图中可辨识的人体部位与场景动作相符，未观察到清晰结构异常。",
    "该图没有可确认的人体结构错误，局部显示或遮挡不足以判为异常。",
    "可见人体区域在形态和物理关系上基本合理，没有明显异常畸形。",
    "图中人体或人体局部表现为正常场景内容，不应因裁切、遮挡或姿态变化判为 abnormal。",
    "画面中的人体结构关系整体合理，未见不能由场景和姿势解释的异常。",
    "当前可见证据支持 normal，没有需要标注为 defects 的异常区域。",
    "图中存在可辨识人体或人体部位，但没有明确结构错误，因此判为 normal。",
    "可见局部人体结构符合正常人体逻辑，没有明显比例、数量或连接关系异常。",
]

NON_HUMAN_THINKS = [
    "图中没有可辨识的人体主体或人体部位，因此不属于人体结构异常判断样本。",
    "画面中未见明确真实人体区域，无法进行人体结构异常定位。",
    "该图不包含可判断的人体或人体局部，应归为 non_human。",
    "图中没有可辨识的人类身体结构，因此不输出人体异常区域。",
    "画面缺少真实人体或人体部位，不能进行人体结构异常判断。",
    "该图未呈现可识别的人体主体或局部身体区域，结论为 non_human。",
    "图中没有足够的人体内容用于判断结构状态，应按 non_human 处理。",
    "当前图像未包含可辨识人体部位，因此不生成 defects。",
]

PART_REASON_VARIANTS: dict[str, list[str]] = {
    "head": [
        "该头部区域存在明确结构异常，局部形态与正常人体头部特征不一致。",
        "该处头部结构表现不自然，可见与正常头部形态不符的异常。",
        "该区域的头部轮廓或结构关系异常，不能由普通姿态、遮挡或透视解释。",
        "该头部区域呈现清晰结构错误，形态比例与正常人体头部不符。",
        "该处头部外观存在明确异常，超出正常场景姿势造成的变化。",
        "该区域可见头部结构不协调，属于可确认的人体结构异常。",
        "该头部区域与正常人体头部特征存在明显差异，需标注为异常。",
        "该处头部轮廓和连接关系不合理，不能简单归因于裁切或模糊。",
        "该头部区域的可见形态缺乏正常人体头部应有的结构一致性。",
        "该处头部结构呈现异常外观，与正常头部比例或轮廓关系不符。",
        "该区域头部形态存在清晰异常，不能用画面角度或遮挡充分解释。",
        "该头部区域的结构表现明显偏离正常人体特征，属于异常证据。",
        "该处头部轮廓关系不自然，结合可见区域可判断为结构异常。",
        "该头部区域出现不合理的形态或连接表现，需要作为异常部位输出。",
        "该处头部结构与周围人体关系不协调，符合明确异常标注条件。",
        "该头部区域可见异常形态，和正常人体头部结构存在稳定差异。",
    ],
    "neck": [
        "该颈部区域存在明确结构异常，连接关系或形态与正常人体不一致。",
        "该处颈部结构表现不自然，可见异常的比例或连接关系。",
        "该颈部区域的结构关系异常，超出普通姿态或遮挡造成的变化。",
        "该处颈部形态与头部或躯干连接不合理，属于明确结构问题。",
        "该颈部区域可见异常结构关系，不能由正常场景姿势解释。",
        "该处颈部比例或连接表现异常，与正常人体特征不符。",
        "该颈部区域缺乏正常人体颈部应有的连接连续性。",
        "该处颈部轮廓和邻近部位关系不自然，可作为异常证据。",
        "该颈部区域呈现不合理形态，不能仅由衣物或姿势解释。",
        "该处颈部结构与正常头颈连接关系存在明显差异。",
        "该颈部区域可见清晰结构错误，需要在 defects 中标注。",
        "该处颈部外观不符合正常人体结构逻辑，属于异常表现。",
    ],
    "body": [
        "该躯干区域存在明确结构异常，整体形态与正常人体躯干不一致。",
        "该处躯干结构表现不自然，可见异常的比例或轮廓关系。",
        "该区域的身体结构与正常人体形态不符，属于可见结构异常。",
        "该躯干区域出现清晰形态错误，不能由衣物、姿态或遮挡解释。",
        "该处身体轮廓和结构关系异常，与正常人体躯干特征存在差异。",
        "该区域躯干比例或连接关系不合理，属于明确异常表现。",
        "该躯干区域的可见结构缺乏正常身体形态的一致性。",
        "该处身体区域呈现异常轮廓，与正常人体躯干逻辑不符。",
        "该区域身体形态存在明确异常，不应由裁切或场景透视解释。",
        "该躯干区域与相邻肢体或整体人体关系不协调，属于结构异常。",
        "该处身体区域可见不合理比例或轮廓，需要标注为异常。",
        "该躯干区域的形态表现明显偏离正常人体结构特征。",
        "该处身体结构不符合正常人体的物理和解剖关系。",
        "该区域可见躯干结构错误，异常依据较清晰。",
    ],
    "arm": [
        "该手臂区域存在明确结构异常，形态或连接关系与正常手臂不一致。",
        "该处手臂结构表现不自然，可见异常的比例或轮廓关系。",
        "该手臂区域的结构关系异常，不符合正常人体肢体特征。",
        "该处上肢形态或关节连接存在明确错误，不能由正常姿势解释。",
        "该手臂区域可见不合理的比例、形态或连接关系，属于结构异常。",
        "该处手臂外观与正常肢体结构不符，异常依据较明确。",
        "该手臂区域和相邻身体部位的连接关系不自然。",
        "该处上肢轮廓或结构走向异常，不符合正常人体逻辑。",
        "该手臂区域呈现异常形态，不能仅由遮挡或透视解释。",
        "该处手臂结构与正常上肢比例关系存在明显差异。",
        "该手臂区域可见清晰结构错误，需要作为异常部位输出。",
        "该处手臂形态缺乏正常肢体应有的连续性和协调性。",
        "该手臂区域的关节或连接表现不合理，属于可确认异常。",
        "该处上肢结构表现偏离正常人体特征，异常较明确。",
    ],
    "hand": [
        "该手部区域存在明确结构异常，局部形态与正常人体手部特征不一致。",
        "该处手部结构表现不自然，可见异常的形态、比例或连接关系。",
        "该手部区域与正常手部结构不符，异常并非单纯裁切或姿态造成。",
        "该区域的手部轮廓和结构关系异常，属于可见人体结构问题。",
        "该手部区域存在清晰结构错误，不能由遮挡、透视或衣物解释。",
        "该处手部形态与正常手部解剖关系不一致，需要标注为异常。",
        "该手部区域可见不合理的数量、比例、连接或轮廓表现。",
        "该处手部结构异常较明确，不属于正常动作或裁切造成的变化。",
        "该手部局部呈现异常形态，和正常人体手部特征存在明显差异。",
        "该区域手部关节或轮廓关系不自然，属于可确认的结构异常。",
        "该手部区域的可见结构缺乏正常手部应有的协调关系。",
        "该处手部轮廓与局部连接表现异常，不能由普通遮挡充分解释。",
        "该手部区域出现不合理结构关系，需要作为异常框输出。",
        "该处手部形态明显偏离正常手部特征，异常依据清晰。",
        "该手部区域在比例、形态或连接上存在明确结构错误。",
        "该处手部结构与周围人体关系不协调，属于可见异常表现。",
        "该手部区域的局部外观不符合正常人体结构逻辑。",
        "该处手部存在稳定可见的异常形态，不应按正常姿态处理。",
        "该手部区域的轮廓或连接关系不合理，符合 defects 标注条件。",
        "该处手部结构表现与正常手部差异明显，属于明确异常区域。",
    ],
    "leg": [
        "该腿部区域存在明确结构异常，形态或比例与正常人体腿部不一致。",
        "该处腿部结构表现不自然，可见异常的轮廓或连接关系。",
        "该腿部区域与正常下肢结构不符，属于可见结构异常。",
        "该处下肢比例、形态或关节关系异常，不能由正常姿势解释。",
        "该腿部区域可见不合理结构表现，与正常人体下肢特征存在差异。",
        "该处腿部轮廓和连接关系不协调，属于明确人体结构问题。",
        "该腿部区域的结构走向或比例关系不符合正常下肢特征。",
        "该处下肢形态存在清晰异常，不能单纯归因于遮挡或运动。",
        "该腿部区域与相邻身体部位连接不自然，需要标注为异常。",
        "该处腿部结构缺乏正常人体下肢应有的协调性。",
        "该腿部区域可见明确比例或轮廓错误，属于异常证据。",
        "该处下肢外观明显偏离正常人体结构逻辑。",
    ],
    "foot": [
        "该脚部区域存在明确结构异常，局部形态与正常人体脚部特征不一致。",
        "该处脚部结构表现不自然，可见异常的比例或轮廓关系。",
        "该脚部区域与正常足部结构不符，属于可见人体结构问题。",
        "该处足部形态或连接关系异常，不能由鞋袜、姿势或遮挡解释。",
        "该脚部区域呈现不合理轮廓或比例，与正常足部特征存在差异。",
        "该处脚部结构异常较明确，需要作为 defects 输出。",
        "该脚部区域的可见形态缺乏正常足部结构的协调性。",
        "该处足部轮廓或连接表现异常，不符合正常人体逻辑。",
        "该脚部区域存在清晰结构错误，不能仅由鞋袜或裁切解释。",
        "该处足部形态与正常脚部特征差异明显，属于异常表现。",
        "该脚部区域可见不合理比例或形态关系，需要标注为异常。",
        "该处足部结构与下肢连接关系不自然，异常依据较明确。",
    ],
    "multi": [
        "该区域存在多处人体结构异常，整体形态与正常人体特征不一致。",
        "该处包含复合结构异常，可见多个部位的形态或连接关系不自然。",
        "该区域呈现多部位异常表现，不能归因于普通姿态、遮挡或透视。",
        "该处人体结构问题涉及多个部位，属于明确异常区域。",
        "该区域可见复合异常，包含不合理的比例、形态或连接关系。",
        "该处多个人体部位同时出现结构不协调，需要标注为异常。",
        "该区域人体结构整体不符合正常人体逻辑，异常涉及多个可见部位。",
        "该处存在多重结构问题，不能由单一姿势、遮挡或裁切解释。",
        "该区域复合异常较明确，多个部位的轮廓或连接关系不自然。",
        "该处人体形态存在多方面结构错误，需要统一作为异常信息输出。",
        "该区域多个身体部位表现异常，整体结构关系明显不协调。",
        "该处可见多部位异常证据，符合 abnormal 标注条件。",
    ],
}

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def detect_conclusion(content: str) -> str:
    match = re.search(r"<conclusion>(.*?)</conclusion>", content)
    return match.group(1).strip() if match else ""


def parse_defects(content: str) -> list[dict[str, Any]]:
    match = re.search(r"<defects>\[(.*?)\]</defects>", content, flags=re.S)
    if not match:
        return []
    text = "[" + match.group(1).replace("<bbox>", '"__BBOX__"') + "]"
    defects = json.loads(text)
    for item in defects:
        if item.get("bbox_2d") == "__BBOX__":
            item["bbox_2d"] = "<bbox>"
    return defects


def image_exists_for_record(record: dict[str, Any]) -> bool:
    images = record.get("images") or []
    return bool(images) and Path(images[0]).exists()


def label_exists_for_record(record: dict[str, Any]) -> bool:
    images = record.get("images") or []
    if not images:
        return False
    stem = Path(images[0]).stem
    return (HUMAN_REFINER_LABEL_DIR / f"{stem}.txt").exists()


def format_defects(defects: list[dict[str, Any]], rng: random.Random) -> str:
    lines = []
    for idx, defect in enumerate(defects, 1):
        part = str(defect.get("part") or "multi")
        key = part if part in PART_REASON_VARIANTS else "multi"
        reason = rng.choice(PART_REASON_VARIANTS[key])
        row = {
            "id": defect.get("id") or f"defect_{idx}",
            "part": part,
            "position": defect.get("position") or "图像中部",
            "bbox_2d": "<bbox>",
            "reason": reason,
        }
        text = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
        text = text.replace('"<bbox>"', "<bbox>")
        lines.append("  " + text)
    return "[\n" + ",\n".join(lines) + "\n]"


def make_messages(user_prompt: str, assistant: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant},
    ]


def make_abnormal(record: dict[str, Any], rng: random.Random, index: int) -> dict[str, Any]:
    content = record["messages"][-1]["content"]
    defects = parse_defects(content)
    assistant = (
        f"<think>{rng.choice(ABNORMAL_THINKS)}</think>\n"
        "<normal_regions>[]</normal_regions>\n"
        f"<defects>{format_defects(defects, rng)}</defects>\n"
        "<conclusion>abnormal</conclusion>"
    )
    return {
        "id": f"v2_abnormal_{record['id']}",
        "messages": make_messages(USER_PROMPTS[index % len(USER_PROMPTS)], assistant),
        "images": record["images"],
        "objects": record["objects"],
    }


def make_non_human(record: dict[str, Any], rng: random.Random, index: int) -> dict[str, Any]:
    assistant = (
        f"<think>{rng.choice(NON_HUMAN_THINKS)}</think>\n"
        "<normal_regions>[]</normal_regions>\n"
        "<defects>[]</defects>\n"
        "<conclusion>non_human</conclusion>"
    )
    return {
        "id": f"v2_non_human_{record['id']}",
        "messages": make_messages(USER_PROMPTS[index % len(USER_PROMPTS)], assistant),
        "images": record["images"],
    }


def iter_normal_images() -> list[Path]:
    images: list[Path] = []
    for folder in OURS_NORMAL_DIRS:
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
    return images


def make_normal(path: Path, rng: random.Random, index: int) -> dict[str, Any]:
    assistant = (
        f"<think>{rng.choice(NORMAL_THINKS)}</think>\n"
        "<normal_regions>[]</normal_regions>\n"
        "<defects>[]</defects>\n"
        "<conclusion>normal</conclusion>"
    )
    subset = path.parent.name
    return {
        "id": f"v2_normal_{subset}_{path.stem}",
        "messages": make_messages(USER_PROMPTS[index % len(USER_PROMPTS)], assistant),
        "images": [str(path)],
    }


def validate(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    ids = set()
    for record in records:
        record_id = record["id"]
        if record_id in ids:
            raise ValueError(f"Duplicate id: {record_id}")
        ids.add(record_id)

        messages = record["messages"]
        assistant = messages[-1]["content"]
        conclusion = detect_conclusion(assistant)
        counts[conclusion] += 1
        if conclusion not in {"normal", "abnormal", "non_human"}:
            raise ValueError(f"Bad conclusion in {record_id}: {conclusion}")
        if not record.get("images") or not Path(record["images"][0]).exists():
            raise FileNotFoundError(f"Missing image in {record_id}: {record.get('images')}")
        if conclusion == "abnormal":
            objects = record.get("objects")
            if not objects or objects.get("bbox_type") != "norm1":
                raise ValueError(f"Bad objects in {record_id}")
            bbox_count = len(objects.get("bbox") or [])
            placeholder_count = assistant.count("<bbox>")
            if bbox_count != placeholder_count:
                raise ValueError(f"bbox mismatch in {record_id}: {bbox_count} vs {placeholder_count}")
            if bbox_count > MAX_ABNORMAL_BBOX:
                raise ValueError(f"Too many bbox in {record_id}: {bbox_count}")
        else:
            if "objects" in record:
                raise ValueError(f"Non-abnormal should not have objects: {record_id}")
            if "<bbox>" in assistant:
                raise ValueError(f"Non-abnormal should not have bbox placeholder: {record_id}")
            if "<defects>[]</defects>" not in assistant:
                raise ValueError(f"Non-abnormal should have empty defects: {record_id}")
    return counts


def main() -> None:
    rng = random.Random(RANDOM_SEED)
    v1_records = load_jsonl(V1_JSONL)

    output: list[dict[str, Any]] = []
    skipped = Counter()
    bbox_distribution = Counter()

    for record in v1_records:
        assistant = record["messages"][-1]["content"]
        conclusion = detect_conclusion(assistant)
        if not image_exists_for_record(record) or not label_exists_for_record(record):
            skipped["missing_current_train_file"] += 1
            continue
        if conclusion == "abnormal":
            bbox_count = len(record.get("objects", {}).get("bbox") or [])
            bbox_distribution[bbox_count] += 1
            if bbox_count == 0:
                skipped["abnormal_without_bbox"] += 1
                continue
            if bbox_count > MAX_ABNORMAL_BBOX:
                skipped["abnormal_bbox_gt_9"] += 1
                continue
            output.append(make_abnormal(record, rng, len(output)))
        elif conclusion == "non_human":
            output.append(make_non_human(record, rng, len(output)))
        else:
            skipped[f"unsupported_conclusion_{conclusion or 'empty'}"] += 1

    normal_images = iter_normal_images()
    for path in normal_images:
        output.append(make_normal(path, rng, len(output)))

    counts = validate(output)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "output": str(OUT_JSONL),
        "source_v1_jsonl": str(V1_JSONL),
        "scheme": "v2_schemeA_normal_no_bbox",
        "random_seed": RANDOM_SEED,
        "max_abnormal_bbox": MAX_ABNORMAL_BBOX,
        "records_total": len(output),
        "records_by_conclusion": dict(counts),
        "normal_image_sources": {str(folder): len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) for folder in OURS_NORMAL_DIRS},
        "skipped": dict(skipped),
        "v1_abnormal_bbox_distribution_seen": dict(sorted(bbox_distribution.items())),
        "validation": {
            "abnormal_bbox_placeholder_matches_objects": True,
            "normal_and_non_human_have_no_objects": True,
            "all_images_exist": True,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
