"""
Image editing model clients with support for both API calls and local pipeline inference.
"""
import base64
import io
import os
import sys
import torch
from abc import ABC, abstractmethod
from typing import Dict
from PIL import Image

import httpx
from diffusers import (
    Flux2KleinPipeline,
    QwenImageEditPlusPipeline,
    JoyImageEditPipeline,
    LongCatImageEditPipeline,
)
from transformers import AutoProcessor


# API Service ports
SERVICE_PORTS = {
    "FLUX2_klein_9b": 8899,
    "Qwen_Image_Edit_2511": 8890,
    "FireRed_Image_Edit": 8891,
    "JoyAI_Image_Edit": 8892,
}


def image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def base64_to_image(b64_str: str) -> Image.Image:
    img_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_data))


def resize_to_fit(img: Image.Image, target_pixels: int = 1024 * 1024) -> Image.Image:
    """Resize image to target pixel count while maintaining aspect ratio."""
    w, h = img.size
    if w * h <= target_pixels:
        return img
    scale = (target_pixels / (w * h)) ** 0.5
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def get_resized_dimensions(img: Image.Image, target_pixels: int = 1024 * 1024) -> tuple[int, int]:
    """Return target dimensions for resize_to_fit without modifying the image."""
    w, h = img.size
    if w * h <= target_pixels:
        return w, h
    scale = (target_pixels / (w * h)) ** 0.5
    return int(w * scale), int(h * scale)


def resize_to_divisible(img: Image.Image, divisor: int = 16) -> Image.Image:
    w, h = img.size
    new_w = (w // divisor) * divisor
    new_h = (h // divisor) * divisor
    if new_w != w or new_h != h:
        img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


class BaseImageEditor(ABC):
    """Base class for image editing models."""

    name: str = NotImplemented

    @abstractmethod
    def edit(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        raise NotImplementedError


# ============== API-based Editors ==============

class BaseAPIEditor(BaseImageEditor):
    """Base class for API-based image editing."""

    port: int = NotImplemented

    def edit(self, image: Image.Image, prompt: str, timeout: float = 300.0, **kwargs) -> Image.Image:
        img = image.convert("RGB")
        img_b64 = image_to_base64(img)

        response = httpx.post(
            f"http://localhost:{self.port}/edit",
            json={"images": [img_b64], "prompt": prompt},
            timeout=timeout,
        )
        response.raise_for_status()
        result_b64 = response.json()["results"][0]
        return base64_to_image(result_b64)


class FLUX2KleinAPIEditor(BaseAPIEditor):
    name = "FLUX2_klein_9b"
    port = SERVICE_PORTS["FLUX2_klein_9b"]


class QwenImageEditAPIEditor(BaseAPIEditor):
    name = "Qwen_Image_Edit_2511"
    port = SERVICE_PORTS["Qwen_Image_Edit_2511"]


class FireRedAPIEditor(BaseAPIEditor):
    name = "FireRed_Image_Edit"
    port = SERVICE_PORTS["FireRed_Image_Edit"]


class JoyAIAPIEditor(BaseAPIEditor):
    name = "JoyAI_Image_Edit"
    port = SERVICE_PORTS["JoyAI_Image_Edit"]


# ============== Pipeline-based Editors ==============

class FLUX2KleinPipelineEditor(BaseImageEditor):
    """FLUX.2-klein-9B image editing with local pipeline."""

    name = "FLUX2_klein_9b"

    def __init__(
        self,
        model_path: str = "/mnt/image-edit/datasets/dingbaojin/models/black-forest-labs/FLUX.2-klein-9B",
        device: str = "cuda:1",
        dtype: torch.dtype = torch.float16,
        max_image_size: int = 2048,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_image_size = max_image_size
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            self._pipeline = Flux2KleinPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
            )
            self._pipeline.to(self.device)

    def edit(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        self._load_pipeline()
        img = image.convert("RGB")

        if max(img.size) > self.max_image_size:
            scale = self.max_image_size / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        target_w, target_h = get_resized_dimensions(img)
        print("before size:", img.size, "target size:", (target_w, target_h))
        # img = resize_to_divisible(img)

        result = self._pipeline(
            image=img,
            prompt=prompt,
            height=img.height,
            width=img.width,
            guidance_scale=kwargs.get("guidance_scale", 1.0),
            num_inference_steps=kwargs.get("num_inference_steps", 4),
            generator=kwargs.get("generator", torch.Generator(device=self.device).manual_seed(42)),
        )
        return result.images[0]


class QwenImageEditPipelineEditor(BaseImageEditor):
    """Qwen-Image-Edit-2511 image editing with local pipeline."""

    name = "Qwen_Image_Edit_2511"

    def __init__(
        self,
        model_path: str = "/mnt/image-edit/datasets/dingbaojin/models/Qwen/Qwen-Image-Edit-2511",
        device: str = "cuda:1",
        dtype: torch.dtype = torch.bfloat16,
        max_image_size: int = 2048,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_image_size = max_image_size
        self._pipeline = None

    def _load_pipeline(self):
        print(f"loading {self.model_path} with torch_dtype={self.dtype} on device {self.device}")
        if self._pipeline is None:
            self._pipeline = QwenImageEditPlusPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
            )
            self._pipeline.to(self.device)

    def edit(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        self._load_pipeline()
        img = image.convert("RGB")

        if max(img.size) > self.max_image_size:
            scale = self.max_image_size / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # img = resize_to_divisible(img)
        # img = resize_to_fit(img, target_pixels=1024 * 1024)
        target_w, target_h = get_resized_dimensions(img)
        print("before size:", img.size, "target size:", (target_w, target_h))
        result = self._pipeline(
            image=img,
            prompt=prompt,
            height=target_h,
            width=target_w,
            generator=kwargs.get("generator", torch.Generator(device=self.device).manual_seed(42)),
            true_cfg_scale=kwargs.get("true_cfg_scale", 4.0),
            negative_prompt=kwargs.get("negative_prompt", " "),
            num_inference_steps=kwargs.get("num_inference_steps", 40),
            guidance_scale=kwargs.get("guidance_scale", 1.0),
            num_images_per_prompt=kwargs.get("num_images_per_prompt", 1),
        )
        print("after size:", result.images[0].size)
        return result.images[0]


class FireRedPipelineEditor(QwenImageEditPipelineEditor):
    """FireRed-Image-Edit-1.1 (same architecture as Qwen)."""

    name = "FireRed_Image_Edit_11"

    def __init__(
        self,
        model_path: str = "/mnt/image-edit/datasets/dingbaojin/models/FireRedTeam/FireRed-Image-Edit-1.1",
        device: str = "cuda:7",
        dtype: torch.dtype = torch.bfloat16,
        max_image_size: int = 2048,
    ):
        super().__init__(model_path, device, dtype, max_image_size)


class JoyAIPipelineEditor(BaseImageEditor):
    """JoyAI-Image-Edit with local pipeline."""

    name = "JoyAI_Image_Edit"

    def __init__(
        self,
        model_path: str = "/mnt/image-edit/datasets/dingbaojin/models/jdopensource/JoyAI-Image-Edit-Diffusers",
        device: str = "cuda:6",
        dtype: torch.dtype = torch.bfloat16,
        max_image_size: int = 2048,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_image_size = max_image_size
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            self._pipeline = JoyImageEditPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
            )
            self._pipeline.to(self.device)

    def edit(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        self._load_pipeline()
        img = image.convert("RGB")

        if max(img.size) > self.max_image_size:
            scale = self.max_image_size / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # img = resize_to_divisible(img)

        result = self._pipeline(
            image=img,
            prompt=prompt,
            num_inference_steps=kwargs.get("num_inference_steps", 30),
            guidance_scale=kwargs.get("guidance_scale", 4.0),
            generator=kwargs.get("generator", torch.Generator(device=self.device).manual_seed(42)),
        )
        return result.images[0]


# ============== HiDream Pipeline Editor ==============

class HiDreamPipelineEditor(BaseImageEditor):
    """HiDream-O1-Image editing with local pipeline."""

    name = "HiDream_O1_Image"

    def __init__(
        self,
        model_path: str = "/mnt/image-edit/datasets/dingbaojin/models/HiDream-ai/HiDream-O1-Image",
        device: str = "cuda:5",
        dtype: torch.dtype = torch.bfloat16,
        max_image_size: int = 2048,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_image_size = max_image_size
        self._model = None
        self._processor = None

    def _load_model(self):
        if self._model is None:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hidream_models"))
            from hidream_models.qwen3_vl_transformers import Qwen3VLForConditionalGeneration

            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path, torch_dtype=self.dtype, device_map=self.device
            ).eval()
            tokenizer = self._processor.tokenizer
            tokenizer.boi_token = "<|boi_token|>"
            tokenizer.bor_token = "<|bor_token|>"
            tokenizer.eor_token = "<|eor_token|>"
            tokenizer.bot_token = "<|bot_token|>"
            tokenizer.tms_token = "<|tms_token|>"

    def edit(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        self._load_model()

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "hidream_models"))
        from hidream_models.pipeline import generate_image, DEFAULT_TIMESTEPS

        img = image.convert("RGB")

        if max(img.size) > self.max_image_size:
            scale = self.max_image_size / max(img.size)
            new_size = (int(img.width * scale), int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        # target_w, target_h = get_resized_dimensions(img)
        print("before size:", img.size)
        print(kwargs)
        import tempfile
        ref_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            img.save(ref_path.name)
            ref_path.close()
            result = generate_image(
                model=self._model,
                processor=self._processor,
                prompt=prompt,
                ref_image_paths=[ref_path.name],
                # height=target_h,
                # width=target_w,
                num_inference_steps=kwargs.get("num_inference_steps", 50),
                guidance_scale=kwargs.get("guidance_scale", 5.0),
                shift=kwargs.get("shift", 3.0),
                timesteps_list=kwargs.get("timesteps_list", None),
                scheduler_name=kwargs.get("scheduler_name", "default"),
                seed=kwargs.get("seed", 32),
                keep_original_aspect=kwargs.get("keep_original_aspect", True),
            )
        finally:
            os.unlink(ref_path.name)

        return result


# ============== LongCat-image-edit Pipeline Editor ==============
class LongCatImageEditPipelineEditor(BaseImageEditor):
    """LongCat-Image-Edit with local diffusers pipeline."""

    name = "LongCat_Image_Edit"

    def __init__(
        self,
        model_path: str = "/mnt/image-edit/datasets/dingbaojin/models/meituan-longcat/LongCat-Image-Edit",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        max_prompt_tokens: int = 512,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.max_prompt_tokens = max_prompt_tokens
        self._pipeline = None

    def _load_pipeline(self):
        if self._pipeline is None:
            self._pipeline = LongCatImageEditPipeline.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                local_files_only=True,
            )
            self._pipeline.to(self.device, self.dtype)

    def _truncate_prompt(self, prompt: str) -> str:
        tokenizer = getattr(self._pipeline, "tokenizer", None)
        if tokenizer is None:
            text_processor = getattr(self._pipeline, "text_processor", None)
            tokenizer = getattr(text_processor, "tokenizer", None)

        if tokenizer is None:
            return prompt

        token_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(token_ids) <= self.max_prompt_tokens:
            return prompt
        return tokenizer.decode(token_ids[: self.max_prompt_tokens], skip_special_tokens=True)

    def edit(self, image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        self._load_pipeline()
        img = image.convert("RGB")
        prompt = self._truncate_prompt(prompt)

        result = self._pipeline(
            image=img,
            prompt=prompt,
            negative_prompt=kwargs.get("negative_prompt", ""),
            guidance_scale=kwargs.get("guidance_scale", 4.5),
            num_inference_steps=kwargs.get("num_inference_steps", 50),
            num_images_per_prompt=kwargs.get("num_images_per_prompt", 1),
            generator=kwargs.get("generator", torch.Generator("cpu").manual_seed(43)),
        )
        return result.images[0]

# ============== Registry ==============

# API-based editors
API_EDITORS: Dict[str, type] = {
    "FLUX2_klein_9b": FLUX2KleinAPIEditor,
    "Qwen_Image_Edit_2511": QwenImageEditAPIEditor,
    "FireRed_Image_Edit": FireRedAPIEditor,
    "JoyAI_Image_Edit": JoyAIAPIEditor,
}

# Pipeline-based editors
PIPELINE_EDITORS: Dict[str, type] = {
    "FLUX2_klein_9b": FLUX2KleinPipelineEditor,
    "FLUX2_klein_4b": FLUX2KleinPipelineEditor,
    "Qwen_Image_Edit_2511": QwenImageEditPipelineEditor,
    "FireRed_Image_Edit": FireRedPipelineEditor,
    # "FireRed_Image_Edit_11": FireRedPipelineEditor,
    "JoyAI_Image_Edit": JoyAIPipelineEditor,
    "HiDream_O1_Image": HiDreamPipelineEditor,
    "LongCat_Image_Edit": LongCatImageEditPipelineEditor,
}

# All available model names
ALL_MODELS = set(list(API_EDITORS.keys()) + list(PIPELINE_EDITORS.keys()))


def get_editor(model_name: str, use_api: bool = True, **kwargs) -> BaseImageEditor:
    """Get an image editor instance.

    Args:
        model_name: Name of the model (e.g., "FLUX2_klein_9b")
        use_api: If True, use API-based editor; if False, use local pipeline
        **kwargs: Additional arguments passed to the editor constructor

    Returns:
        An editor instance
    """
    if model_name not in ALL_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {sorted(ALL_MODELS)}")

    if use_api:
        if model_name not in API_EDITORS:
            raise ValueError(f"Model {model_name} does not support API mode")
        return API_EDITORS[model_name](**kwargs)
    else:
        if model_name not in PIPELINE_EDITORS:
            raise ValueError(f"Model {model_name} does not support pipeline mode")
        return PIPELINE_EDITORS[model_name](**kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test image editing models")
    parser.add_argument("--model", type=str, default="HiDream_O1_Image", choices=sorted(ALL_MODELS))
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Editing prompt")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., cuda:5)")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--keep-original-aspect", action="store_true", default=False, help="Keep original aspect ratio")
    args = parser.parse_args()

    device = args.device or "cuda:5"

    if args.model == "HiDream_O1_Image":
        editor = HiDreamPipelineEditor(device=device)
    else:
        editor = get_editor(args.model, use_api=False, device=device)

    img = Image.open(args.image)
    print(f"Input image: {img.size}")

    kwargs = {}
    if args.num_inference_steps is not None:
        kwargs["num_inference_steps"] = args.num_inference_steps
    if args.guidance_scale is not None:
        kwargs["guidance_scale"] = args.guidance_scale
    if args.shift is not None:
        kwargs["shift"] = args.shift
    kwargs["keep_original_aspect"] = args.keep_original_aspect

    print(f"Editing with {args.model}...")
    result = editor.edit(img, args.prompt, **kwargs)
    result.save(args.output)
    print(f"Output saved to: {args.output}")