#!/usr/bin/env python3
"""DreamFace2.0 Gradio 推理界面 —— 支持切换模型、参考图预览、参数调节。

用法:
    /root/.venv/dreamface-omni/bin/python demo_gradio.py
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image
from PIL import Image

# ---------- 固定配置 ----------
OUTPUT_DIR = Path("/mnt/image-edit/datasets/duanyufa/task_shengsheng/dreamportrait_gen/output")
REFERENCE_IMAGE_AREA = 1024 * 1024

MODEL_CHOICES = {
    "DreamFace2.0 (人像)": "/mnt/image-edit/models/hithink-image-labs/DreamFace2.0",
    "FLUX.2-Klein-9B (通用)": "/mnt/image-edit/models/black-forest-labs/FLUX.2-klein-base-9B",
    "FLUX.2-Klein-4B (通用)": "/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B",
}
DEFAULT_MODEL = "DreamFace2.0 (人像)"


def prepare_reference_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    ratio = image.width / image.height
    tw = math.sqrt(REFERENCE_IMAGE_AREA * ratio)
    th = tw / ratio
    tw, th = round(tw / 32) * 32, round(th / 32) * 32
    scale = max(tw / image.width, th / image.height)
    image = image.resize(
        (round(image.width * scale), round(image.height * scale)),
        resample=Image.Resampling.BILINEAR,
    )
    left = max((image.width - tw) // 2, 0)
    top = max((image.height - th) // 2, 0)
    return image.crop((left, top, left + tw, top + th))


# ---------- 模型管理（切换模型时卸载旧的，节省显存） ----------
_current_key: tuple[str, str] | None = None
_current_pipe: Flux2KleinPipeline | None = None


def get_pipeline(model_path: str, device: str) -> Flux2KleinPipeline:
    global _current_key, _current_pipe
    key = (model_path, device)
    # 同模型同设备，直接用
    if _current_key == key and _current_pipe is not None:
        return _current_pipe
    # 切换模型/设备：卸载旧的
    if _current_pipe is not None:
        print(f"[DreamFace] 卸载旧模型: {_current_key}")
        del _current_pipe
        torch.cuda.empty_cache()
        _current_pipe = None
        _current_key = None
    # 加载新模型
    print(f"[DreamFace] 加载模型: {model_path} -> {device}")
    pipe = Flux2KleinPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.transformer.eval()
    _current_pipe = pipe
    _current_key = key
    print("[DreamFace] 模型加载完成")
    return pipe


@torch.inference_mode()
def generate(
    model_choice: str,
    device_id: int,
    prompt: str,
    ref_files: list[str] | None,
    seed: int,
    steps: int,
    cfg: float,
    height: int,
    width: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple[Image.Image, str]:
    device = f"cuda:{device_id}"
    model_path = MODEL_CHOICES[model_choice]
    pipe = get_pipeline(model_path, device)

    refs = []
    if ref_files:
        for f in ref_files:
            if f is not None:
                refs.append(prepare_reference_image(load_image(f)))
    refs = refs or None

    generator = torch.Generator(device=device).manual_seed(seed)

    progress(0, desc="推理中...")
    t0 = time.time()

    result = pipe(
        prompt=prompt,
        image=refs,
        height=height,
        width=width,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=generator,
    )

    elapsed = time.time() - t0
    final_image: Image.Image = result.images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_name = f"{timestamp}.png"
    image_path = OUTPUT_DIR / image_name
    final_image.save(image_path)

    # 追加标注到 meta.jsonl
    meta = {
        "image": image_name,
        "prompt": prompt,
        "height": height,
        "width": width,
    }
    meta_path = OUTPUT_DIR / "meta.jsonl"
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    log = (
        f"✅ 完成! 耗时 {elapsed:.2f}s\n"
        f"   模型: {model_choice}\n"
        f"   输出: {image_path}\n"
        f"   参数: steps={steps}, cfg={cfg}, seed={seed}\n"
        f"   分辨率: {width}x{height}, 参考图: {len(refs) if refs else 0} 张"
    )

    progress(1.0, desc="完成")
    return final_image, log


def files_to_gallery(files):
    if not files:
        return []
    result = []
    items = files if isinstance(files, list) else [files]
    for f in items:
        result.append((f, Path(f).name))
    return result


CSS = """
html, body, textarea, input, select, label, button, .prose, .markdown {
    font-family: -apple-system, BlinkMacSystemFont, 'Microsoft YaHei', 'PingFang SC', 'WenQuanYi Micro Hei', sans-serif !important;
}
footer {display:none !important}
"""


def build_ui():
    with gr.Blocks(title="DreamFace2.0 推理") as demo:
        gr.Markdown("# 🎨 图像生成推理\n上传参考图 + 提示词，可选不同模型。")

        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Dropdown(
                    label="模型选择",
                    choices=list(MODEL_CHOICES.keys()),
                    value=DEFAULT_MODEL,
                )

                device_id = gr.Dropdown(
                    label="CUDA 设备",
                    choices=[str(i) for i in range(7)],
                    value="1",
                )

                prompt = gr.Textbox(
                    label="提示词 (Prompt)",
                    lines=4,
                    value=(
                        "Maintain absolute consistency of the subject's facial features. "
                        "Medium shot portrait perspective, with the female subject occupying "
                        "two-thirds of the frame. She is elegantly sitting sideways on a pink "
                        "velvet stool, holding a strawberry near her lips, with a serene and "
                        "sweet expression of joy. ..."
                    ),
                )

                ref_images = gr.File(
                    label="上传参考图（可多张）",
                    file_count="multiple",
                    file_types=["image"],
                )

                ref_preview = gr.Gallery(
                    label="参考图预览",
                    columns=1, rows=1, height=400,
                    object_fit="scale-down",
                )
                ref_images.change(
                    fn=files_to_gallery,
                    inputs=[ref_images],
                    outputs=[ref_preview],
                )

                with gr.Accordion("⚙️ 参数", open=True):
                    seed = gr.Slider(-1, 2147483647, value=42, step=1, label="Seed")
                    steps = gr.Slider(1, 50, value=4, step=1, label="推理步数")
                    cfg = gr.Slider(0.5, 10.0, value=1.0, step=0.1, label="CFG")
                    height = gr.Slider(512, 2048, value=1024, step=64, label="高度")
                    width = gr.Slider(512, 2048, value=1024, step=64, label="宽度")

                run_btn = gr.Button("🚀 生成", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="生成结果", type="pil", height=500)
                log_box = gr.Textbox(label="日志", lines=6)

        run_btn.click(
            fn=generate,
            inputs=[model_choice, device_id, prompt, ref_images, seed, steps, cfg, height, width],
            outputs=[output_image, log_box],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7871, share=False, css=CSS, theme=gr.themes.Soft())
