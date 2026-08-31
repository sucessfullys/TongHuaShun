#!/usr/bin/env python3
"""Gradio image viewer: show raw Unsplash images 10 per page."""

from __future__ import annotations

import argparse
from pathlib import Path

import gradio as gr


DEFAULT_IMAGE_DIR = "/mnt/image-edit-hdd/datasets/duanyufa/unsplash/清洗/real_human_hand/images"
IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
    ".avif",
}
PAGE_SIZE = 10


def natural_key(path: Path):
    text = path.relative_to(path.anchor).as_posix()
    parts = []
    current = ""
    is_digit = text[:1].isdigit()
    for ch in text:
        if ch.isdigit() == is_digit:
            current += ch
        else:
            parts.append(int(current) if is_digit else current.lower())
            current = ch
            is_digit = ch.isdigit()
    if current:
        parts.append(int(current) if is_digit else current.lower())
    return parts


def list_images(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        return []
    return sorted(
        [
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ],
        key=natural_key,
    )


class ViewerState:
    def __init__(self, image_dir: str):
        self.image_dir = Path(image_dir)
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
        for slot, path in enumerate(self.page_items):
            if path is not None and path.exists():
                image_values.append(str(path))
                try:
                    display_name = path.relative_to(self.image_dir).as_posix()
                except ValueError:
                    display_name = path.name
                label_values.append(f"{self.page * PAGE_SIZE + slot + 1}. {display_name}")
            else:
                image_values.append(None)
                label_values.append("空")

        start = self.page * PAGE_SIZE + 1 if self.total else 0
        end = min((self.page + 1) * PAGE_SIZE, self.total)
        info = (
            f"当前页: {self.page + 1} / {self.max_page + 1}\n"
            f"当前显示: {start} - {end}\n"
            f"图片总数: {self.total}\n"
            f"图片目录: {self.image_dir}"
        )
        return image_values + label_values + [info]

    def next_page(self):
        self._load_page(self.page + 1)
        return self.page_payload()

    def prev_page(self):
        self._load_page(self.page - 1)
        return self.page_payload()

    def reload_current(self):
        self.files = list_images(self.image_dir)
        self._load_page(self.page)
        return self.page_payload()


def make_ui(state: ViewerState):
    with gr.Blocks(title="Unsplash Raw 图片查看") as demo:
        gr.Markdown("# Unsplash Raw 图片查看\n每页展示 10 张，仅查看，不移动、不删除、不修改。")

        info = gr.Textbox(label="状态", lines=4, interactive=False)

        image_components = []
        label_components = []
        for row_idx in range(2):
            with gr.Row():
                for col_idx in range(5):
                    slot = row_idx * 5 + col_idx
                    with gr.Column(scale=1):
                        image = gr.Image(label=f"图片 {slot + 1}", type="filepath", height=260)
                        label = gr.Textbox(label="文件名", lines=1, interactive=False)
                        image_components.append(image)
                        label_components.append(label)

        with gr.Row():
            prev_btn = gr.Button("上一页")
            reload_btn = gr.Button("刷新当前页")
            next_btn = gr.Button("下一页", variant="primary")

        outputs = image_components + label_components + [info]

        def load_page():
            return state.page_payload()

        demo.load(fn=load_page, outputs=outputs)
        prev_btn.click(fn=state.prev_page, outputs=outputs)
        next_btn.click(fn=state.next_page, outputs=outputs)
        reload_btn.click(fn=state.reload_current, outputs=outputs)

    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--port", type=int, default=7873)
    args = parser.parse_args()

    state = ViewerState(args.image_dir)
    demo = make_ui(state)
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=False,
        allowed_paths=[str(Path(args.image_dir).resolve())],
    )


if __name__ == "__main__":
    main()
