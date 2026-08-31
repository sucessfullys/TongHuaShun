"""Image preprocessing utilities for the workflow pipeline.

Extracted from temp_wxy/utils_tryon.py. Handles model image resizing,
cloth image canvas fitting, and base64 conversion for VLM APIs.
"""

from __future__ import annotations

import base64
from io import BytesIO
from PIL import Image, ImageOps


def convert_transparent_to_white(image: Image.Image) -> Image.Image:
    """Convert RGBA/P mode image to RGB with white background."""
    if image.mode == "RGBA":
        white_bg = Image.new("RGB", image.size, (255, 255, 255))
        white_bg.paste(image, mask=image.split()[3])
        return white_bg
    elif image.mode == "P":
        rgba_img = image.convert("RGBA")
        white_bg = Image.new("RGB", rgba_img.size, (255, 255, 255))
        white_bg.paste(rgba_img, mask=rgba_img.split()[3])
        return white_bg
    return image.convert("RGB")


def fit_image_to_canvas(
    input_img: Image.Image,
    target_width: int = 1360,
    target_height: int = 2048,
) -> Image.Image:
    """Fit image into canvas without cropping (contain mode, white padding)."""
    original_width, original_height = input_img.size
    original_ratio = original_width / original_height
    target_ratio = target_width / target_height

    canvas = Image.new("RGB", (target_width, target_height), (255, 255, 255))

    if original_ratio > target_ratio:
        scale_factor = target_width / original_width
        new_width = target_width
        new_height = int(original_height * scale_factor)
        resized = input_img.resize((new_width, new_height), Image.LANCZOS)
        padding_top = (target_height - new_height) // 2
        canvas.paste(resized, (0, padding_top))
    else:
        scale_factor = target_height / original_height
        new_height = target_height
        new_width = int(original_width * scale_factor)
        resized = input_img.resize((new_width, new_height), Image.LANCZOS)
        padding_left = (target_width - new_width) // 2
        canvas.paste(resized, (padding_left, 0))

    return canvas


def _center_crop(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    """Center-crop or pad to exact target dimensions."""
    original_width, _ = img.size
    if original_width > target_width:
        x_start = (original_width - target_width) // 2
        return img.crop((x_start, 0, x_start + target_width, target_height))
    return fit_image_to_canvas(img, target_width, target_height)


def resize_model_image(
    img: Image.Image,
    target_height: int = 1376,
    target_width: int = 768,
) -> Image.Image:
    """Preprocess model image: scale height, center-crop width."""
    img = ImageOps.exif_transpose(img)
    if img.mode == "RGBA":
        img = convert_transparent_to_white(img)
    elif img.mode != "RGB":
        img = img.convert("RGB")

    width_percent = target_height / float(img.size[1])
    new_width = int(float(img.size[0]) * width_percent)
    img_resized = img.resize((new_width, target_height), Image.LANCZOS)
    return _center_crop(img_resized, target_width, target_height)


def resize_cloth_image(
    img: Image.Image,
    target_height: int = 1376,
    target_width: int = 768,
) -> Image.Image:
    """Preprocess cloth image: contain-fit with white padding."""
    img = ImageOps.exif_transpose(img)
    if img.mode == "RGBA":
        img = convert_transparent_to_white(img)
    elif img.mode != "RGB":
        img = img.convert("RGB")

    original_width, original_height = img.size
    ratio = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    img_resized = img.resize((new_width, new_height), Image.LANCZOS)
    background = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2
    background.paste(img_resized, (x_offset, y_offset))
    return background


def to_base64_url(pil_img: Image.Image) -> str:
    """Convert PIL image to base64 data URL for VLM APIs."""
    buffer = BytesIO()
    fmt = pil_img.format or "PNG"
    if fmt.upper() == "JPEG" and pil_img.mode in ("RGBA", "P"):
        pil_img = pil_img.convert("RGB")
    pil_img.save(buffer, format=fmt, quality=95)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{img_base64}"
