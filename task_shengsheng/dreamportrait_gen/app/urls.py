#!/usr/bin/env python3

from tornado.options import options

from app.handlers.index_handler import IndexHandler
from app.handlers.server_handler import DreamFaceHandler
from app.modules.dreamface_editor import DreamFaceEditor


dreamface_editor = DreamFaceEditor(
    device=options.device,
    model_id=options.model_id,
    lora_path=options.lora_path or None,
    lora_alpha=options.lora_alpha,
    default_steps=options.default_steps,
    default_cfg=options.default_cfg,
    default_height=options.default_height,
    default_width=options.default_width,
    enable_cpu_offload=options.enable_cpu_offload,
    device_map=options.device_map,
    max_memory=options.max_memory,
)


urls = [
    (r"/readiness$", IndexHandler),
    (r"/image/dreamface", DreamFaceHandler, dict(model=dreamface_editor)),
]
