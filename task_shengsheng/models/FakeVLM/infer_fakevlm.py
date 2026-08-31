#!/usr/bin/env python3
"""FakeVLM 单图推理 —— 完全对齐 scripts/eval.py 逻辑。

用法:
    CUDA_VISIBLE_DEVICES=6 /root/miniconda3/envs/aigi-holmes/bin/python \
        infer_fakevlm.py --input /path/to/img.jpg
"""

import argparse
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor

MODEL_PATH = "/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/FakeVLM/checkpoints/lingcco--fakeVLM"
PROCESSOR_ID = "llava-hf/llava-1.5-7b-hf"   # 对齐 eval.py：用基础 LLaVA processor
MODEL = None
PROCESSOR = None

# QUERY = "Does the image looks real/fake?"
QUERY = "Is this image real or fake? Answer with exactly one word first: real or fake. Then explain the most visible image artifacts in one short sentence."

def get_model():
    global MODEL, PROCESSOR
    if MODEL is None:
        print(f"Loading model: {MODEL_PATH} ...")
        # 对齐 eval.py load_model()
        MODEL = LlavaForConditionalGeneration.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True,
        )
        # 对齐 eval.py validate()：用基础 LLaVA-1.5 processor + revision a272c74
        PROCESSOR = AutoProcessor.from_pretrained(PROCESSOR_ID, revision="a272c74")
        # 旧版 processor 缺 patch_size 属性，手动补上
        if getattr(PROCESSOR, "patch_size", None) is None:
            PROCESSOR.patch_size = 14
        if getattr(PROCESSOR, "num_additional_image_tokens", None) is None:
            PROCESSOR.num_additional_image_tokens = 0
        print("Model ready.")
    return MODEL, PROCESSOR


def detect(image_path):
    model, processor = get_model()
    from PIL import Image
    image = Image.open(image_path).convert("RGB")

    # 对齐 eval.py + FakeClue 数据集格式：<image> + 问题文本，无 USER/ASSISTANT 前缀
    prompt = f"<image>{QUERY}"
    inputs = processor(
        text=prompt, images=image, return_tensors="pt",
        padding="max_length", max_length=1024, truncation=True,
    ).to(model.device)
    # 对齐 eval.py validate()：squeeze batch 维度
    output = model.generate(**inputs, max_new_tokens=256)
    # 对齐 eval.py：全量 decode 后从 '?' 截取
    result = processor.decode(output[0], skip_special_tokens=True).split("?")[-1].strip()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()

    result = detect(args.input)
    print("\n" + "=" * 50)
    print(result)
    print("=" * 50)


if __name__ == "__main__":
    main()
