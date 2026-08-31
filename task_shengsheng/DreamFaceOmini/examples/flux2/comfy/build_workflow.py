#!/usr/bin/env python3
"""Build ComfyUI workflow: FLUX.2 Klein 9B img2img + PiD 2k→4k decode.

Mirrors run_gradio_pid.sh defaults:
  LDM 896x1152 @ 4 steps, cfg=1 → PiD 3584x4608 @ 4 steps, cfg=1, scale=4.
"""

from __future__ import annotations

import json
from pathlib import Path

# Output pixel size (2kto4k preset, same as gradio_pid)
OUT_W, OUT_H = 3584, 4608
SCALE = 4
LDM_W, LDM_H = OUT_W // SCALE, OUT_H // SCALE
LDM_STEPS, LDM_CFG = 4, 1.0
PID_STEPS, PID_CFG = 4, 1.0

# Model filenames (must match setup_comfy_pid.sh links)
KLEIN_UNET = "flux-2-klein-9b.safetensors"
KLEIN_CLIP = "qwen_3_8b_fp8mixed.safetensors"
VAE = "flux2-vae.safetensors"
LORA = "dreamface_v21_comfyui.safetensors"
PID_UNET = "pid_flux2_1024_to_4096_4step_2606_bf16.safetensors"
PID_CLIP = "gemma_2_2b_it_elm_bf16.safetensors"


def node(nid: int, ntype: str, x: float, y: float, widgets: list, title: str = "") -> dict:
    return {
        "id": nid,
        "type": ntype,
        "pos": [x, y],
        "size": [320, 120],
        "flags": {},
        "order": nid,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        "title": title,
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets,
    }


def link(lid: int, src_node: int, src_slot: int, dst_node: int, dst_slot: int, ntype: str) -> list:
    return [lid, src_node, src_slot, dst_node, dst_slot, ntype]


def build() -> dict:
    nodes: list[dict] = []
    links: list[list] = []
    lid = 1

    def add(n: dict) -> int:
        nodes.append(n)
        return n["id"]

    def wire(src: int, ss: int, dst: int, ds: int, t: str) -> None:
        nonlocal lid
        links.append(link(lid, src, ss, dst, ds, t))
        lid += 1

    # --- inputs ---
    n_load = add(node(1, "LoadImage", 0, 0, ["example.png", "image"]))
    n_scale = add(node(2, "ImageScaleToTotalPixels", 380, 0, ["nearest-exact", 1.0, 1.0]))
    n_vae_ldm = add(node(3, "VAELoader", 0, 180, [VAE]))
    n_enc = add(node(4, "VAEEncode", 760, 0, []))

    n_k_unet = add(node(10, "UNETLoader", 0, 360, [KLEIN_UNET, "default"]))
    n_k_clip = add(node(11, "CLIPLoader", 380, 360, [KLEIN_CLIP, "flux2", "default"]))
    n_lora = add(node(12, "LoraLoader", 760, 360, [LORA, 1.0, 1.0]))
    n_pos = add(node(13, "CLIPTextEncode", 0, 540, [""]))
    n_neg = add(node(14, "CLIPTextEncode", 380, 540, [""]))
    n_ref = add(node(15, "ReferenceLatent", 760, 540, []))
    n_empty_ldm = add(node(16, "EmptyFlux2LatentImage", 1140, 360, [LDM_W, LDM_H, 1]))
    n_sched_ldm = add(node(17, "Flux2Scheduler", 1140, 540, [LDM_STEPS, LDM_W, LDM_H]))
    n_cfg_ldm = add(node(18, "CFGGuider", 1520, 360, [LDM_CFG]))
    n_samp_sel_ldm = add(node(19, "KSamplerSelect", 1520, 540, ["euler"]))
    n_noise_ldm = add(node(20, "RandomNoise", 1520, 720, [0, "randomize"]))
    n_samp_ldm = add(node(21, "SamplerCustomAdvanced", 1900, 360, []))

    # --- PiD decode branch ---
    n_p_unet = add(node(30, "UNETLoader", 0, 900, [PID_UNET, "default"]))
    n_p_clip = add(node(31, "CLIPLoader", 380, 900, [PID_CLIP, "pixeldit", "default"]))
    n_p_pos = add(node(32, "CLIPTextEncode", 0, 1080, [""]))
    n_pid_cond = add(node(33, "PiDConditioning", 760, 900, ["flux", 0.0]))
    n_empty_pid = add(node(34, "EmptyLatentImage", 1140, 900, [OUT_W, OUT_H, 1]))
    n_sched_pid = add(node(35, "Flux2Scheduler", 1140, 1080, [PID_STEPS, OUT_W, OUT_H]))
    n_cfg_pid = add(node(36, "CFGGuider", 1520, 900, [PID_CFG]))
    n_samp_sel_pid = add(node(37, "KSamplerSelect", 1520, 1080, ["euler"]))
    n_noise_pid = add(node(38, "RandomNoise", 1520, 1260, [0, "randomize"]))
    n_samp_pid = add(node(39, "SamplerCustomAdvanced", 1900, 900, []))
    n_decode = add(node(40, "VAEDecode", 2280, 900, []))
    n_save = add(node(41, "SaveImage", 2660, 900, ["pid_comfy"]))

    # LDM graph
    wire(1, 0, 2, 0, "IMAGE")
    wire(2, 0, 4, 0, "IMAGE")
    wire(3, 0, 4, 1, "VAE")
    wire(11, 0, 12, 1, "CLIP")
    wire(12, 1, 13, 0, "CLIP")
    wire(12, 1, 14, 0, "CLIP")
    wire(13, 0, 15, 0, "CONDITIONING")
    wire(4, 0, 15, 1, "LATENT")
    wire(15, 0, 18, 1, "CONDITIONING")
    wire(10, 0, 12, 0, "MODEL")
    wire(12, 0, 18, 0, "MODEL")
    wire(18, 0, 21, 1, "GUIDER")
    wire(19, 0, 21, 2, "SAMPLER")
    wire(17, 0, 21, 3, "SIGMAS")
    wire(16, 0, 21, 4, "LATENT")
    wire(20, 0, 21, 0, "NOISE")

    # PiD graph — use LDM denoised latent (output slot 1)
    wire(21, 1, 33, 1, "LATENT")
    wire(31, 0, 32, 0, "CLIP")
    wire(32, 0, 33, 0, "CONDITIONING")
    wire(33, 0, 36, 1, "CONDITIONING")
    wire(30, 0, 36, 0, "MODEL")
    wire(36, 0, 39, 1, "GUIDER")
    wire(37, 0, 39, 2, "SAMPLER")
    wire(35, 0, 39, 3, "SIGMAS")
    wire(34, 0, 39, 4, "LATENT")
    wire(38, 0, 39, 0, "NOISE")
    wire(39, 0, 40, 0, "LATENT")
    wire(3, 0, 40, 1, "VAE")
    wire(40, 0, 41, 0, "IMAGE")

    return {
        "last_node_id": 41,
        "last_link_id": lid - 1,
        "nodes": nodes,
        "links": links,
        "groups": [
            {
                "title": "FLUX.2 Klein 9B LDM (896×1152)",
                "bounding": [-40, -40, 2300, 820],
                "color": "#3f789e",
                "font_size": 24,
            },
            {
                "title": "PiD decode → 3584×4608 (2kto4k)",
                "bounding": [-40, 820, 2900, 620],
                "color": "#8A8",
                "font_size": 24,
            },
        ],
        "config": {},
        "extra": {
            "ds": {"scale": 0.6, "offset": [0, 0]},
            "info": {
                "name": "flux2_klein_pid_2kto4k",
                "description": "FLUX.2 Klein 9B img2img + PiD 4× decode (ComfyUI PR #14103)",
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
    from build_gradio_node_workflow import main as build_gradio_node_workflow_main

    build_gradio_node_workflow_main()
