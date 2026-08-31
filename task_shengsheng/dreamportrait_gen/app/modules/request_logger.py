#!/usr/bin/env python3
"""Request audit logging helpers for DreamFace inference."""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PIL import Image


logger = logging.getLogger("request_logger")


def create_request_id() -> str:
    return uuid.uuid4().hex


class RequestLogger:
    def __init__(
        self,
        *,
        enabled: bool = True,
        log_dir: str = "logs/requests",
        save_request_images: bool = True,
        save_result_images: bool = True,
    ):
        self.enabled = bool(enabled)
        self.log_dir = Path(log_dir)
        self.save_request_images = bool(save_request_images)
        self.save_result_images = bool(save_result_images)

    def _today(self) -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _request_image_dir(self, request_id: str) -> Path:
        return self.log_dir / "images" / self._today() / request_id

    def save_input_images(self, request_id: str, images: Iterable[Image.Image]) -> list[str]:
        if not self.enabled or not self.save_request_images:
            return []

        try:
            request_dir = self._request_image_dir(request_id)
            request_dir.mkdir(parents=True, exist_ok=True)

            paths = []
            for index, image in enumerate(images, start=1):
                path = request_dir / f"input_{index}.png"
                image.save(path, format="PNG")
                paths.append(str(path))
            return paths
        except Exception as e:
            logger.warning("Failed to save request images: %s", e)
            return []

    def save_output_image(self, request_id: str, image: Image.Image) -> str | None:
        if not self.enabled or not self.save_result_images:
            return None

        try:
            request_dir = self._request_image_dir(request_id)
            request_dir.mkdir(parents=True, exist_ok=True)

            path = request_dir / "output.png"
            image.save(path, format="PNG")
            return str(path)
        except Exception as e:
            logger.warning("Failed to save result image: %s", e)
            return None

    def write(self, record: dict):
        if not self.enabled:
            return

        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            path = self.log_dir / f"{self._today()}.jsonl"

            record = dict(record)
            record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))

            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("Failed to write request log: %s", e)
