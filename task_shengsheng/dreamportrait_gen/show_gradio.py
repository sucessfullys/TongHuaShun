#!/usr/bin/env python3
"""Gradio image cleaner: review 10 images per page and split accepted/rejected images."""

from __future__ import annotations

import argparse
import shutil
import threading
from pathlib import Path

import gradio as gr


DEFAULT_IMAGE_DIR = (
    "/mnt/image-edit/datasets/duanyufa/task_shengsheng/dreamportrait_gen/Outputs/"
    "steps28_cfg4.0_h1024_w1024_model9B_seed42_normal"
)
DEFAULT_REMOVED_DIR = (
    "/mnt/image-edit/datasets/duanyufa/task_shengsheng/dreamportrait_gen/Outputs/"
    "steps28_cfg4.0_h1024_w1024_model9B_seed42_normal_去除"
)
DEFAULT_SELECTED_DIR = (
    "/mnt/image-edit/datasets/duanyufa/task_shengsheng/dreamportrait_gen/Outputs/"
    "steps28_cfg4.0_h1024_w1024_model9B_seed42_normal_选择"
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".JPG", ".JPEG", ".PNG"}
PAGE_SIZE = 10


def natural_key(path: Path):
    text = path.stem
    parts = []
    current = ""
    is_digit = text[:1].isdigit()
    for ch in text:
        if ch.isdigit() == is_digit:
            current += ch
        else:
            parts.append(int(current) if is_digit else current)
            current = ch
            is_digit = ch.isdigit()
    if current:
        parts.append(int(current) if is_digit else current)
    return parts, path.name


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_SUFFIXES],
        key=natural_key,
    )


def unique_destination(dst_dir: Path, name: str) -> Path:
    dst = dst_dir / name
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    index = 1
    while True:
        candidate = dst_dir / f"{stem}_dup{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


class CleanerState:
    def __init__(self, image_dir: str, removed_dir: str, selected_dir: str):
        self.image_dir = Path(image_dir)
        self.removed_dir = Path(removed_dir)
        self.selected_dir = Path(selected_dir)
        self.removed_dir.mkdir(parents=True, exist_ok=True)
        self.selected_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.files = list_images(self.image_dir)
        self.page = 0
        self.page_items: list[Path | None] = []
        self._load_page(0)

    @property
    def total(self) -> int:
        return len(self.files)

    @property
    def max_page(self) -> int:
        return max((self.total - 1) // PAGE_SIZE, 0)

    def _load_page(self, page: int) -> None:
        self.page = min(max(page, 0), self.max_page)
        start = self.page * PAGE_SIZE
        items: list[Path | None] = self.files[start : start + PAGE_SIZE]
        while len(items) < PAGE_SIZE:
            items.append(None)
        self.page_items = items

    def page_payload(self):
        image_values = []
        label_values = []
        button_updates = []
        for slot, path in enumerate(self.page_items):
            if path is not None and path.exists():
                image_values.append(str(path))
                label_values.append(f"{self.page * PAGE_SIZE + slot + 1}. {path.name}")
                button_updates.append(gr.update(interactive=True, value="移动到去除"))
            else:
                image_values.append(None)
                label_values.append("空")
                button_updates.append(gr.update(interactive=False, value="已移除"))

        remaining_on_page = sum(1 for p in self.page_items if p is not None and p.exists())
        info = (
            f"当前页: {self.page + 1} / {self.max_page + 1}\n"
            f"源目录剩余: {self.total}\n"
            f"本页剩余: {remaining_on_page} / {PAGE_SIZE}\n"
            f"源目录: {self.image_dir}\n"
            f"选择目录: {self.selected_dir}\n"
            f"去除目录: {self.removed_dir}"
        )
        return image_values + label_values + button_updates + [info]

    def next_page(self):
        with self.lock:
            self._move_remaining_page_items(self.selected_dir)
            self.files = list_images(self.image_dir)
            self._load_page(self.page)
            return self.page_payload()

    def prev_page(self):
        with self.lock:
            self._load_page(self.page - 1)
            return self.page_payload()

    def reload_current(self):
        with self.lock:
            self.files = list_images(self.image_dir)
            self._load_page(self.page)
            return self.page_payload()

    def _move_remaining_page_items(self, dst_dir: Path) -> None:
        for slot, src in enumerate(self.page_items):
            if src is None or not src.exists():
                self.page_items[slot] = None
                continue

            dst = unique_destination(dst_dir, src.name)
            shutil.move(str(src), str(dst))
            self.page_items[slot] = None

    def remove_slot(self, slot: int):
        with self.lock:
            if not (0 <= slot < PAGE_SIZE):
                return self.page_payload()
            src = self.page_items[slot]
            if src is None or not src.exists():
                self.page_items[slot] = None
                return self.page_payload()

            dst = unique_destination(self.removed_dir, src.name)
            shutil.move(str(src), str(dst))
            self.page_items[slot] = None
            self.files = [p for p in self.files if p != src]
            return self.page_payload()


def make_ui(state: CleanerState):
    with gr.Blocks(title="DreamPortrait 图片清洗") as demo:
        gr.Markdown(
            "# DreamPortrait 图片清洗\n"
            "每页 10 张，点击单张按钮移动到“去除”；点击“下一页”会把本页剩余图片移动到“选择”，"
            "再加载下一批未清洗图片。"
        )

        info = gr.Textbox(label="状态", lines=5, interactive=False)

        image_components = []
        label_components = []
        delete_buttons = []
        for row_idx in range(2):
            with gr.Row():
                for col_idx in range(5):
                    slot = row_idx * 5 + col_idx
                    with gr.Column(scale=1):
                        image = gr.Image(label=f"图片 {slot + 1}", type="filepath", height=260)
                        label = gr.Textbox(label="文件名", lines=1, interactive=False)
                        button = gr.Button("移动到去除", variant="stop")
                        image_components.append(image)
                        label_components.append(label)
                        delete_buttons.append(button)

        with gr.Row():
            prev_btn = gr.Button("上一页")
            reload_btn = gr.Button("刷新当前页")
            next_btn = gr.Button("下一页（保留本页剩余）", variant="primary")

        outputs = image_components + label_components + delete_buttons + [info]

        def load_page():
            return state.page_payload()

        demo.load(fn=load_page, outputs=outputs)
        prev_btn.click(fn=state.prev_page, outputs=outputs)
        next_btn.click(fn=state.next_page, outputs=outputs)
        reload_btn.click(fn=state.reload_current, outputs=outputs)

        for slot, button in enumerate(delete_buttons):
            button.click(fn=lambda s=slot: state.remove_slot(s), outputs=outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--removed-dir", default=DEFAULT_REMOVED_DIR)
    parser.add_argument("--selected-dir", default=DEFAULT_SELECTED_DIR)
    parser.add_argument("--port", type=int, default=7872)
    args = parser.parse_args()

    state = CleanerState(args.image_dir, args.removed_dir, args.selected_dir)
    demo = make_ui(state)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        allowed_paths=[
            "/mnt/image-edit/datasets/duanyufa/task_shengsheng/dreamportrait_gen/Outputs/"
        ],
    )


if __name__ == "__main__":
    main()
