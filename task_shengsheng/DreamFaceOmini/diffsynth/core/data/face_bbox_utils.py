"""GT face bbox helpers for preprocessing annotation and training-time loss weighting."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

IMAGE_REF_RE = re.compile(r"<image\s*(\d+)>", re.IGNORECASE)


def ref_key(index_1based: int) -> str:
    return f"image{index_1based}"


def parse_prompt_image_refs(prompt: str) -> set[str]:
    refs = {int(m.group(1)) for m in IMAGE_REF_RE.finditer(prompt or "")}
    return {ref_key(i) for i in refs}


def bbox_to_norm(bbox: tuple[float, float, float, float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = bbox
    w = max(width, 1)
    h = max(height, 1)
    return [
        float(np.clip(x1 / w, 0.0, 1.0)),
        float(np.clip(y1 / h, 0.0, 1.0)),
        float(np.clip(x2 / w, 0.0, 1.0)),
        float(np.clip(y2 / h, 0.0, 1.0)),
    ]


def norm_bbox_to_pixels(bbox_norm: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox_norm
    return x1 * width, y1 * height, x2 * width, y2 * height


def transform_bbox_norm_to_processed(
    bbox_norm: list[float],
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = norm_bbox_to_pixels(bbox_norm, orig_w, orig_h)
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    x1, x2 = x1 * scale, x2 * scale
    y1, y2 = y1 * scale, y2 * scale
    crop_x = (new_w - target_w) // 2
    crop_y = (new_h - target_h) // 2
    return x1 - crop_x, y1 - crop_y, x2 - crop_x, y2 - crop_y


def expand_bbox_pixels(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    expand: float,
    max_w: int,
    max_h: int,
) -> tuple[float, float, float, float]:
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = (x2 - x1) / 2 * expand
    half_h = (y2 - y1) / 2 * expand
    return (
        max(cx - half_w, 0.0),
        max(cy - half_h, 0.0),
        min(cx + half_w, float(max_w)),
        min(cy + half_h, float(max_h)),
    )


def build_latent_face_weight(
    bboxes_pixels: list[tuple[float, float, float, float]],
    processed_w: int,
    processed_h: int,
    *,
    face_mse_weight: float,
    face_mask_expand: float,
):
    import torch

    w16, h16 = processed_w // 16, processed_h // 16
    if w16 <= 0 or h16 <= 0:
        return None
    weight = torch.ones((h16, w16), dtype=torch.float32)
    applied = False
    for x1, y1, x2, y2 in bboxes_pixels:
        x1, y1, x2, y2 = expand_bbox_pixels(
            x1, y1, x2, y2,
            expand=face_mask_expand,
            max_w=processed_w,
            max_h=processed_h,
        )
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        half_w = (x2 - x1) / 2
        half_h = (y2 - y1) / 2
        tx1 = max(int((cx - half_w) / 16), 0)
        ty1 = max(int((cy - half_h) / 16), 0)
        tx2 = min(int((cx + half_w) / 16) + 1, w16)
        ty2 = min(int((cy + half_h) / 16) + 1, h16)
        if tx2 > tx1 and ty2 > ty1:
            weight[ty1:ty2, tx1:tx2] = face_mse_weight
            applied = True
    if not applied:
        return None
    weight = weight / weight.mean()
    return weight.reshape(1, -1, 1)


def collect_bbox_entries(
    gt_face_bboxes: dict[str, Any],
    *,
    prompt: str | None = None,
    min_match_score: float = 0.0,
    only_prompt_refs: bool = True,
) -> list[list[float]]:
    allowed = parse_prompt_image_refs(prompt or "") if only_prompt_refs and prompt else None
    out: list[list[float]] = []
    for key, entry in gt_face_bboxes.items():
        if allowed is not None and key not in allowed:
            continue
        if isinstance(entry, dict):
            score = float(entry.get("match_score", 1.0))
            bbox = entry.get("bbox_norm")
        else:
            score = 1.0
            bbox = entry
        if bbox is None or score < min_match_score:
            continue
        if isinstance(bbox[0], list):
            out.extend(bbox)
        else:
            out.append(bbox)
    return out
