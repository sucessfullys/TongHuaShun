#!/usr/bin/env python3

from tornado.options import define


global_variables = {
    "debug": {
        "default": False,
        "type": bool,
        "help": "",
    },
    "port": {
        "default": 9001,
        "type": int,
        "help": "",
    },
    "device": {
        "default": "cuda:0",
        "type": str,
        "help": "CUDA device for Flux2 Klein model",
    },
    "model_id": {
        "default": "hithink-image-labs/DreamFace2.0",
        "type": str,
        "help": "Hugging Face model id or local model path",
    },
    "lora_path": {
        "default": "",
        "type": str,
        "help": "Optional LoRA path",
    },
    "lora_alpha": {
        "default": 1.0,
        "type": float,
        "help": "LoRA alpha scale",
    },
    "default_steps": {
        "default": 4,
        "type": int,
        "help": "Default num_inference_steps",
    },
    "default_cfg": {
        "default": 1.0,
        "type": float,
        "help": "Default guidance_scale",
    },
    "default_height": {
        "default": 1152,
        "type": int,
        "help": "Default output height",
    },
    "default_width": {
        "default": 896,
        "type": int,
        "help": "Default output width",
    },
    "max_reference_images": {
        "default": 3,
        "type": int,
        "help": "Maximum number of reference images per request",
    },
    "max_image_pixels": {
        "default": 4096 * 4096,
        "type": int,
        "help": "Maximum pixels for each decoded reference image",
    },
    "enable_cpu_offload": {
        "default": False,
        "type": bool,
        "help": "Enable diffusers model CPU offload",
    },
    "device_map": {
        "default": "",
        "type": str,
        "help": "Optional Accelerate device_map, for example balanced or auto",
    },
    "max_memory": {
        "default": "",
        "type": str,
        "help": "Optional max memory map, for example 0:22GiB,1:22GiB,cpu:60GiB",
    },
    "enable_request_log": {
        "default": True,
        "type": bool,
        "help": "Enable DreamFace request audit logs",
    },
    "request_log_dir": {
        "default": "logs/requests",
        "type": str,
        "help": "Directory for DreamFace request audit logs",
    },
    "save_request_images": {
        "default": True,
        "type": bool,
        "help": "Save decoded request reference images",
    },
    "save_result_images": {
        "default": True,
        "type": bool,
        "help": "Save generated result images",
    },
    "max_thread_pool": {
        "default": 4,
        "type": int,
        "help": "",
    },
}


def define_global_variables():
    for key, value in global_variables.items():
        define(
            name=key,
            default=value["default"],
            type=value["type"],
            help=value["help"],
        )
