#!/usr/bin/env python3
"""DreamFace2.0 Flux2 Klein HTTP inference wrapper using diffusers."""

import os
import math
from typing import Iterable

import torch
from diffusers import Flux2KleinPipeline
from PIL import Image


MODEL_BASE_PATH = os.environ.get(
    "DIFFUSERS_MODEL_BASE_PATH",
    os.environ.get("DIFFSYNTH_MODEL_BASE_PATH", "/mnt/data/image-edit/datasets/shensheng/models"),
)
REFERENCE_IMAGE_AREA = 1024 * 1024


def _resolve_path(path_or_id: str) -> str:
    if not path_or_id:
        return path_or_id
    if os.path.exists(path_or_id):
        return path_or_id
    candidate = os.path.join(MODEL_BASE_PATH, path_or_id)
    return candidate if os.path.exists(candidate) else path_or_id


def _normalize_lora_path(lora_path: str | None) -> str | None:
    if lora_path is None:
        return None
    lora_path = str(lora_path).strip()
    return lora_path or None


def _parse_max_memory(max_memory: str | None) -> dict[int | str, str] | None:
    if not max_memory:
        return None

    parsed = {}
    for item in str(max_memory).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "max_memory must use comma-separated key:value pairs, "
                "for example 0:22GiB,1:22GiB,cpu:60GiB"
            )

        key, value = item.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(
                "max_memory must use comma-separated key:value pairs, "
                "for example 0:22GiB,1:22GiB,cpu:60GiB"
            )

        parsed[int(key) if key.isdigit() else key] = value

    return parsed or None


def _resize_filter():
    return getattr(getattr(Image, "Resampling", Image), "BILINEAR")


def _calculate_reference_dimensions(target_area: int, ratio: float) -> tuple[int, int]:
    target_width = math.sqrt(target_area * ratio)
    target_height = target_width / ratio
    target_width = round(target_width / 32) * 32
    target_height = round(target_height / 32) * 32
    return target_width, target_height


class DreamFaceEditor:
    def __init__(
        self,
        device: str = "cuda:0",
        model_id: str = "hithink-image-labs/DreamFace2.0",
        lora_path: str | None = None,
        lora_alpha: float = 1.0,
        default_steps: int = 4,
        default_cfg: float = 1.0,
        default_height: int = 1152,
        default_width: int = 896,
        enable_cpu_offload: bool = False,
        device_map: str | None = None,
        max_memory: str | None = None,
    ):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but is not available.")

        self.device = device
        self.default_steps = int(default_steps)
        self.default_cfg = float(default_cfg)
        self.default_height = int(default_height)
        self.default_width = int(default_width)
        self.enable_cpu_offload = bool(enable_cpu_offload)
        self.device_map = str(device_map).strip() if device_map else ""
        self.max_memory = _parse_max_memory(max_memory)
        self.lora_path = _normalize_lora_path(lora_path)
        self.lora_alpha = float(lora_alpha)

        model_path = _resolve_path(model_id)
        print(f"[DreamFace] Loading Flux2 Klein pipeline: {model_path}")
        load_kwargs = {"torch_dtype": torch.bfloat16}
        if self.device_map:
            load_kwargs["device_map"] = self.device_map
            if self.max_memory:
                load_kwargs["max_memory"] = self.max_memory

        try:
            self.pipe = Flux2KleinPipeline.from_pretrained(model_path, **load_kwargs)
        except TypeError as e:
            if self.device_map:
                raise RuntimeError(
                    "Current diffusers/accelerate stack does not support device_map "
                    "for Flux2KleinPipeline. Upgrade diffusers and accelerate."
                ) from e
            raise

        component_names = list(getattr(self.pipe, "components", {}).keys())
        print(f"[DreamFace] Pipeline components: {', '.join(component_names)}")

        if self.device_map:
            if self.enable_cpu_offload:
                print("[DreamFace] enable_cpu_offload ignored because device_map is enabled.")
            print(f"[DreamFace] Accelerate device_map enabled: {self.device_map}")
            if self.max_memory:
                print(f"[DreamFace] Accelerate max_memory: {self.max_memory}")
            self._print_component_devices()
        elif self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            print("[DreamFace] CPU offload enabled.")
        else:
            self.pipe.to(self.device)
            print(f"[DreamFace] Pipeline moved to {self.device}.")
            self._print_component_devices()

        if self.lora_path:
            self.load_lora(self.lora_path, self.lora_alpha)

    def _get_component_device_info(self, component) -> str:
        if component is None:
            return "None"

        hf_device_map = getattr(component, "hf_device_map", None)
        if hf_device_map:
            return f"hf_device_map={hf_device_map}"

        device = getattr(component, "device", None)
        if device is not None:
            return str(device)

        if hasattr(component, "parameters"):
            try:
                return str(next(component.parameters()).device)
            except StopIteration:
                return "no parameters"

        return "unknown"

    def _print_component_devices(self):
        for component_name in ("text_encoder", "transformer", "vae"):
            component = getattr(self.pipe, component_name, None)
            print(f"[DreamFace] Component device: {component_name} -> {self._get_component_device_info(component)}")

    def _prepare_reference_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        target_width, target_height = _calculate_reference_dimensions(
            REFERENCE_IMAGE_AREA,
            image.size[0] / image.size[1],
        )

        src_width, src_height = image.size
        scale = max(target_width / src_width, target_height / src_height)
        resized_size = (round(src_width * scale), round(src_height * scale))
        image = image.resize(resized_size, resample=_resize_filter())

        left = max((image.width - target_width) // 2, 0)
        top = max((image.height - target_height) // 2, 0)
        return image.crop((left, top, left + target_width, top + target_height))

    def load_lora(self, lora_path: str, lora_alpha: float = 1.0):
        lora_path = _resolve_path(lora_path)
        if not os.path.isfile(lora_path) and not os.path.isdir(lora_path):
            raise FileNotFoundError(f"LoRA path does not exist: {lora_path}")

        print(f"[DreamFace] Loading LoRA: {lora_path} (alpha={lora_alpha})")
        try:
            self.pipe.load_lora_weights(lora_path, adapter_name="default")
        except TypeError:
            self.pipe.load_lora_weights(lora_path)
        except ValueError:
            self.pipe.load_lora_weights(lora_path, adapter_name="default", prefix=None)

        if hasattr(self.pipe, "set_adapters"):
            self.pipe.set_adapters(["default"], adapter_weights=[float(lora_alpha)])

        self.lora_path = lora_path
        self.lora_alpha = float(lora_alpha)

    def infer(
        self,
        prompt: str,
        images: Iterable[Image.Image] | None = None,
        *,
        seed: int = 42,
        steps: int | None = None,
        cfg: float | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> Image.Image:
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("prompt is required")

        ref_images = list(images or [])
        if ref_images:
            ref_images = [self._prepare_reference_image(img) for img in ref_images]

        steps = self.default_steps if steps is None else int(steps)
        cfg = self.default_cfg if cfg is None else float(cfg)
        height = self.default_height if height is None else int(height)
        width = self.default_width if width is None else int(width)
        generator = torch.Generator(device=self.device).manual_seed(int(seed))

        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                image=ref_images if ref_images else None,
                height=height,
                width=width,
                guidance_scale=cfg,
                num_inference_steps=steps,
                generator=generator,
            )

        if hasattr(result, "images"):
            return result.images[0]
        return result[0][0]
