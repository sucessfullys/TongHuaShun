#!/usr/bin/env python3
"""Build an importable ComfyUI workflow for FLUX.2 Klein + PiD 2k-to-4k.

The generated JSON is a ComfyUI UI workflow, not the backend API graph. That
means each node needs populated input/output slot metadata in addition to the
top-level links list, otherwise the frontend can show "Required input is
missing" even when the link list looks correct.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT_W, OUT_H = 3584, 4608
SCALE = 4
LDM_W, LDM_H = OUT_W // SCALE, OUT_H // SCALE
LDM_STEPS, LDM_CFG = 4, 1.0
PID_STEPS, PID_CFG = 4, 1.0

KLEIN_UNET = "flux-2-klein-9b.safetensors"
KLEIN_CLIP = "qwen_3_8b_fp8mixed.safetensors"
VAE = "flux2-vae.safetensors"
LORA = "dreamface_v21_comfyui.safetensors"
PID_UNET = "pid_flux2_1024_to_4096_4step_2606_bf16.safetensors"
PID_CLIP = "gemma_2_2b_it_elm_bf16.safetensors"

Slot = dict[str, Any]
Node = dict[str, Any]

NODE_SLOTS: dict[str, dict[str, list[Slot]]] = {
    "LoadImage": {
        "inputs": [{"name": "image", "type": "COMBO", "widget": True}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE"}, {"name": "MASK", "type": "MASK"}],
    },
    "ImageScaleToTotalPixels": {
        "inputs": [
            {"name": "image", "type": "IMAGE"},
            {"name": "upscale_method", "type": "COMBO", "widget": True},
            {"name": "megapixels", "type": "FLOAT", "widget": True},
            {"name": "resolution_steps", "type": "INT", "widget": True},
        ],
        "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
    },
    "VAELoader": {
        "inputs": [{"name": "vae_name", "type": "COMBO", "widget": True}],
        "outputs": [{"name": "VAE", "type": "VAE"}],
    },
    "VAEEncode": {
        "inputs": [{"name": "pixels", "type": "IMAGE"}, {"name": "vae", "type": "VAE"}],
        "outputs": [{"name": "LATENT", "type": "LATENT"}],
    },
    "UNETLoader": {
        "inputs": [
            {"name": "unet_name", "type": "COMBO", "widget": True},
            {"name": "weight_dtype", "type": "COMBO", "widget": True},
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL"}],
    },
    "CLIPLoader": {
        "inputs": [
            {"name": "clip_name", "type": "COMBO", "widget": True},
            {"name": "type", "type": "COMBO", "widget": True},
            {"name": "device", "type": "COMBO", "widget": True, "shape": 7},
        ],
        "outputs": [{"name": "CLIP", "type": "CLIP"}],
    },
    "LoraLoader": {
        "inputs": [
            {"name": "model", "type": "MODEL"},
            {"name": "clip", "type": "CLIP"},
            {"name": "lora_name", "type": "COMBO", "widget": True},
            {"name": "strength_model", "type": "FLOAT", "widget": True},
            {"name": "strength_clip", "type": "FLOAT", "widget": True},
        ],
        "outputs": [{"name": "MODEL", "type": "MODEL"}, {"name": "CLIP", "type": "CLIP"}],
    },
    "CLIPTextEncode": {
        "inputs": [{"name": "text", "type": "STRING", "widget": True}, {"name": "clip", "type": "CLIP"}],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING"}],
    },
    "ReferenceLatent": {
        "inputs": [
            {"name": "conditioning", "type": "CONDITIONING"},
            {"name": "latent", "type": "LATENT", "shape": 7},
        ],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING"}],
    },
    "EmptyFlux2LatentImage": {
        "inputs": [
            {"name": "width", "type": "INT", "widget": True},
            {"name": "height", "type": "INT", "widget": True},
            {"name": "batch_size", "type": "INT", "widget": True},
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT"}],
    },
    "Flux2Scheduler": {
        "inputs": [
            {"name": "steps", "type": "INT", "widget": True},
            {"name": "width", "type": "INT", "widget": True},
            {"name": "height", "type": "INT", "widget": True},
        ],
        "outputs": [{"name": "SIGMAS", "type": "SIGMAS"}],
    },
    "CFGGuider": {
        "inputs": [
            {"name": "model", "type": "MODEL"},
            {"name": "positive", "type": "CONDITIONING"},
            {"name": "negative", "type": "CONDITIONING"},
            {"name": "cfg", "type": "FLOAT", "widget": True},
        ],
        "outputs": [{"name": "GUIDER", "type": "GUIDER"}],
    },
    "KSamplerSelect": {
        "inputs": [{"name": "sampler_name", "type": "COMBO", "widget": True}],
        "outputs": [{"name": "SAMPLER", "type": "SAMPLER"}],
    },
    "RandomNoise": {
        "inputs": [{"name": "noise_seed", "type": "INT", "widget": True}],
        "outputs": [{"name": "NOISE", "type": "NOISE"}],
    },
    "SamplerCustomAdvanced": {
        "inputs": [
            {"name": "noise", "type": "NOISE"},
            {"name": "guider", "type": "GUIDER"},
            {"name": "sampler", "type": "SAMPLER"},
            {"name": "sigmas", "type": "SIGMAS"},
            {"name": "latent_image", "type": "LATENT"},
        ],
        "outputs": [
            {"name": "output", "type": "LATENT"},
            {"name": "denoised_output", "type": "LATENT"},
        ],
    },
    "PiDConditioning": {
        "inputs": [
            {"name": "positive", "type": "CONDITIONING"},
            {"name": "latent", "type": "LATENT"},
            {"name": "latent_format", "type": "COMBO", "widget": True},
            {"name": "degrade_sigma", "type": "FLOAT", "widget": True},
        ],
        "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING"}],
    },
    "EmptyChromaRadianceLatentImage": {
        "inputs": [
            {"name": "width", "type": "INT", "widget": True},
            {"name": "height", "type": "INT", "widget": True},
            {"name": "batch_size", "type": "INT", "widget": True},
        ],
        "outputs": [{"name": "LATENT", "type": "LATENT"}],
    },
    "PiDLatentToImage": {
        "inputs": [{"name": "samples", "type": "LATENT"}],
        "outputs": [{"name": "IMAGE", "type": "IMAGE"}],
    },
    "SaveImage": {
        "inputs": [
            {"name": "images", "type": "IMAGE"},
            {"name": "filename_prefix", "type": "STRING", "widget": True},
        ],
        "outputs": [{"name": "images", "type": "IMAGE"}],
    },
}


def _input_slot(spec: Slot) -> Slot:
    slot: Slot = {
        "localized_name": spec["name"],
        "name": spec["name"],
        "type": spec["type"],
        "link": None,
    }
    if spec.get("widget"):
        slot["widget"] = {"name": spec["name"]}
    if "shape" in spec:
        slot["shape"] = spec["shape"]
    return slot


def _output_slot(spec: Slot) -> Slot:
    return {
        "localized_name": spec["name"],
        "name": spec["name"],
        "type": spec["type"],
        "links": [],
    }


def node(nid: int, ntype: str, x: float, y: float, widgets: list[Any], title: str = "") -> Node:
    slots = NODE_SLOTS[ntype]
    return {
        "id": nid,
        "type": ntype,
        "pos": [x, y],
        "size": [320, 120],
        "flags": {},
        "order": nid,
        "mode": 0,
        "inputs": [_input_slot(s) for s in slots["inputs"]],
        "outputs": [_output_slot(s) for s in slots["outputs"]],
        "title": title,
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets,
    }


def slot_index(n: Node, slot_name: str, direction: str) -> int:
    slots = n[direction]
    for index, slot in enumerate(slots):
        if slot["name"] == slot_name:
            return index
    raise KeyError(f"{n['type']} has no {direction[:-1]} slot named {slot_name!r}")


def build() -> dict[str, Any]:
    nodes: list[Node] = []
    links: list[list[Any]] = []
    by_id: dict[int, Node] = {}
    lid = 1

    def add(n: Node) -> int:
        nodes.append(n)
        by_id[n["id"]] = n
        return n["id"]

    def wire(src: int, src_slot: str, dst: int, dst_slot: str, link_type: str) -> None:
        nonlocal lid
        src_node = by_id[src]
        dst_node = by_id[dst]
        src_idx = slot_index(src_node, src_slot, "outputs")
        dst_idx = slot_index(dst_node, dst_slot, "inputs")
        src_node["outputs"][src_idx]["links"].append(lid)
        dst_node["inputs"][dst_idx]["link"] = lid
        links.append([lid, src, src_idx, dst, dst_idx, link_type])
        lid += 1

    # Shared image input and Flux2 VAE encode.
    add(node(1, "LoadImage", 0, 0, ["example.png", "image"]))
    add(node(2, "ImageScaleToTotalPixels", 380, 0, ["nearest-exact", 1.0, 1]))
    add(node(3, "VAELoader", 0, 180, [VAE]))
    add(node(4, "VAEEncode", 760, 0, []))

    # LDM branch: FLUX.2 Klein 9B image edit to low-resolution latent.
    add(node(10, "UNETLoader", 0, 360, [KLEIN_UNET, "default"]))
    add(node(11, "CLIPLoader", 380, 360, [KLEIN_CLIP, "flux2", "default"]))
    add(node(12, "LoraLoader", 760, 360, [LORA, 1.0, 1.0]))
    add(node(13, "CLIPTextEncode", 0, 540, [""]))
    add(node(14, "CLIPTextEncode", 380, 540, [""]))
    add(node(15, "ReferenceLatent", 760, 500, []))
    add(node(16, "EmptyFlux2LatentImage", 1140, 360, [LDM_W, LDM_H, 1]))
    add(node(17, "Flux2Scheduler", 1140, 540, [LDM_STEPS, LDM_W, LDM_H]))
    add(node(18, "CFGGuider", 1520, 360, [LDM_CFG]))
    add(node(19, "KSamplerSelect", 1520, 540, ["euler"]))
    add(node(20, "RandomNoise", 1520, 720, [0, "randomize"]))
    add(node(21, "SamplerCustomAdvanced", 1900, 360, []))
    add(node(43, "ReferenceLatent", 760, 660, []))

    # PiD branch: PixelDiT decoder/upscaler conditioned by the LDM latent.
    add(node(30, "UNETLoader", 0, 900, [PID_UNET, "default"]))
    add(node(31, "CLIPLoader", 380, 900, [PID_CLIP, "pixeldit", "default"]))
    add(node(32, "CLIPTextEncode", 0, 1080, [""]))
    add(node(42, "CLIPTextEncode", 380, 1080, [""]))
    add(node(33, "PiDConditioning", 760, 900, ["flux", 0.0]))
    add(node(44, "PiDConditioning", 760, 1080, ["flux", 0.0]))
    add(node(34, "EmptyChromaRadianceLatentImage", 1140, 900, [OUT_W, OUT_H, 1]))
    add(node(35, "ManualSigmas", 1140, 1080, ["0.999, 0.866, 0.634, 0.342, 0.0"]))
    add(node(36, "CFGGuider", 1520, 900, [PID_CFG]))
    add(node(37, "KSamplerSelect", 1520, 1080, ["euler"]))
    add(node(38, "RandomNoise", 1520, 1260, [0, "randomize"]))
    add(node(39, "SamplerCustomAdvanced", 1900, 900, []))
    add(node(40, "PiDLatentToImage", 2280, 900, []))
    add(node(41, "SaveImage", 2660, 900, ["pid_comfy"]))

    wire(1, "IMAGE", 2, "image", "IMAGE")
    wire(2, "IMAGE", 4, "pixels", "IMAGE")
    wire(3, "VAE", 4, "vae", "VAE")

    wire(10, "MODEL", 12, "model", "MODEL")
    wire(11, "CLIP", 12, "clip", "CLIP")
    wire(12, "CLIP", 13, "clip", "CLIP")
    wire(12, "CLIP", 14, "clip", "CLIP")
    wire(13, "CONDITIONING", 15, "conditioning", "CONDITIONING")
    wire(4, "LATENT", 15, "latent", "LATENT")
    wire(14, "CONDITIONING", 43, "conditioning", "CONDITIONING")
    wire(4, "LATENT", 43, "latent", "LATENT")
    wire(12, "MODEL", 18, "model", "MODEL")
    wire(15, "CONDITIONING", 18, "positive", "CONDITIONING")
    wire(43, "CONDITIONING", 18, "negative", "CONDITIONING")
    wire(18, "GUIDER", 21, "guider", "GUIDER")
    wire(19, "SAMPLER", 21, "sampler", "SAMPLER")
    wire(17, "SIGMAS", 21, "sigmas", "SIGMAS")
    wire(16, "LATENT", 21, "latent_image", "LATENT")
    wire(20, "NOISE", 21, "noise", "NOISE")

    wire(31, "CLIP", 32, "clip", "CLIP")
    wire(31, "CLIP", 42, "clip", "CLIP")
    wire(32, "CONDITIONING", 33, "positive", "CONDITIONING")
    wire(21, "denoised_output", 33, "latent", "LATENT")
    wire(42, "CONDITIONING", 44, "positive", "CONDITIONING")
    wire(21, "denoised_output", 44, "latent", "LATENT")
    wire(30, "MODEL", 36, "model", "MODEL")
    wire(33, "CONDITIONING", 36, "positive", "CONDITIONING")
    wire(44, "CONDITIONING", 36, "negative", "CONDITIONING")
    wire(36, "GUIDER", 39, "guider", "GUIDER")
    wire(37, "SAMPLER", 39, "sampler", "SAMPLER")
    wire(35, "SIGMAS", 39, "sigmas", "SIGMAS")
    wire(34, "LATENT", 39, "latent_image", "LATENT")
    wire(38, "NOISE", 39, "noise", "NOISE")
    wire(39, "output", 40, "samples", "LATENT")
    wire(40, "IMAGE", 41, "images", "IMAGE")

    return {
        "last_node_id": 44,
        "last_link_id": lid - 1,
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "title": "FLUX.2 Klein 9B LDM (896x1152)",
                "bounding": [-40, -40, 2300, 820],
                "color": "#3f789e",
                "font_size": 24,
            },
            {
                "title": "PiD decode to 3584x4608 (2kto4k)",
                "bounding": [-40, 820, 3040, 620],
                "color": "#8A8",
                "font_size": 24,
            },
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.6, "offset": [0, 0]},
            "info": {
                "name": "flux2_klein_pid_2kto4k",
                "description": "FLUX.2 Klein 9B img2img + PiD 4x decode (ComfyUI PR #14103)",
            },
        },
        "version": 0.4,
    }


def main() -> None:
    out = Path(__file__).resolve().parent / "workflows" / "flux2_klein_pid_2kto4k.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(), indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
