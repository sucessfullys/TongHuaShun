import argparse
import gc
import json
import os
import queue
import random
import threading
import uuid
from datetime import datetime

import gradio as gr
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoTokenizer

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.utils.lora import GeneralLoRALoader


DEFAULT_TRANSFORMER_MODEL_ID = "black-forest-labs/FLUX.2-dev"
DEFAULT_VAE_MODEL_ID = "black-forest-labs/FLUX.2-dev"
DEFAULT_VAE_FILE = "vae/diffusion_pytorch_model.safetensors"
DEFAULT_LOG_DIR = "/mnt/data/image-edit/datasets/shensheng/outputs/gradio_dreamface_omini"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", default=None, help="Optional LoRA weights path (.safetensors)")
    parser.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha scale")
    parser.add_argument(
        "--transformer_model_id",
        default=DEFAULT_TRANSFORMER_MODEL_ID,
        help="Transformer model id",
    )
    parser.add_argument(
        "--vae_path",
        default=None,
        help="Optional custom VAE path (.safetensors file or vae/ directory)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="Single GPU index to use")
    parser.add_argument("--steps", type=int, default=4, help="Default inference steps")
    parser.add_argument("--cfg", type=float, default=1.0, help="Default CFG scale")
    parser.add_argument("--embedded_guidance", type=float, default=1.0, help="Embedded guidance scale (1.0 for wandb-aligned)")
    parser.add_argument("--s2_scale", type=float, default=0.0, help="S²-Guidance scale ω (0=off, paper default ~0.25)")
    parser.add_argument("--s2_drop_ratio", type=float, default=0.1, help="S²-Guidance single-block drop ratio")
    parser.add_argument("--s2_start", type=float, default=0.1, help="S²-Guidance start (fraction of steps, 0=first step)")
    parser.add_argument("--s2_end", type=float, default=0.9, help="S²-Guidance end (fraction of steps, 1=last step)")
    parser.add_argument("--sigma_mu", type=float, default=0.0, help="Sigma-schedule shift mu override (<=0 = auto, matches training)")
    parser.add_argument("--height", type=int, default=1152, help="Default output height")
    parser.add_argument("--width", type=int, default=896, help="Default output width")
    parser.add_argument("--seed", type=int, default=42, help="Default random seed")
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR, help="Directory to save every generation (image + params) for reproducibility")
    parser.add_argument("--no_log", action="store_true", help="Disable per-generation logging")
    parser.add_argument("--offload", action="store_true", help="Enable CPU offload")
    parser.add_argument("--server_name", default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link")
    return parser.parse_args()


def offload_config(device):
    return dict(
        offload_dtype=torch.bfloat16,
        offload_device="cpu",
        onload_dtype=torch.bfloat16,
        onload_device=device,
        preparing_dtype=torch.bfloat16,
        preparing_device=device,
        computation_dtype=torch.bfloat16,
        computation_device=device,
    )


def pinned_gpu_config(device):
    return dict(
        offload_dtype=torch.bfloat16,
        offload_device=device,
        onload_dtype=torch.bfloat16,
        onload_device=device,
        preparing_dtype=torch.bfloat16,
        preparing_device=device,
        computation_dtype=torch.bfloat16,
        computation_device=device,
    )


def is_32b_bundle(transformer_model_id: str) -> bool:
    tid = normalize_transformer_model_id(transformer_model_id).lower()
    if "32b" in tid:
        return True
    return bundle_base_model_id(transformer_model_id) == "black-forest-labs/FLUX.2-dev"


def resolve_vram_strategy(args, transformer_model_id: str) -> dict:
    primary = f"cuda:{args.gpu}"
    if args.offload:
        return {"mode": "cpu_offload", "primary": primary, "secondary": None}

    if not is_32b_bundle(transformer_model_id):
        return {"mode": "single", "primary": primary, "secondary": None}

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpu_count >= 2:
        return {
            "mode": "dual_gpu",
            "primary": "cuda:0",
            "secondary": "cuda:1",
        }

    print("[vram] 32B 模型仅 1 张可见 GPU，自动启用 CPU offload")
    return {"mode": "cpu_offload", "primary": primary, "secondary": None}


def component_vram_extra(component: str, strategy: dict) -> dict:
    mode = strategy["mode"]
    primary = strategy["primary"]
    if mode == "cpu_offload":
        return offload_config(primary)
    if mode == "dual_gpu":
        device = strategy["secondary"] if component == "text_encoder" else primary
        return pinned_gpu_config(device)
    return {}


def format_vram_strategy(strategy: dict) -> str:
    mode = strategy["mode"]
    if mode == "dual_gpu":
        return f"VRAM: 双卡 (DiT/VAE={strategy['primary']}, TE={strategy['secondary']})"
    if mode == "cpu_offload":
        return f"VRAM: CPU offload (compute={strategy['primary']})"
    return f"VRAM: 单卡 ({strategy['primary']})"


def normalize_lora_path(lora_path):
    if lora_path is None:
        return None
    lora_path = str(lora_path).strip()
    return lora_path or None


def normalize_transformer_model_id(transformer_model_id):
    if transformer_model_id is None:
        return DEFAULT_TRANSFORMER_MODEL_ID
    transformer_model_id = str(transformer_model_id).strip()
    transformer_model_id = transformer_model_id.strip("\"'")
    transformer_model_id = transformer_model_id.rstrip(":")
    transformer_model_id = transformer_model_id.replace("kelin", "klein")
    transformer_model_id = transformer_model_id.replace("Kelin", "Klein")
    return transformer_model_id or DEFAULT_TRANSFORMER_MODEL_ID


def bundle_base_model_id(transformer_model_id: str) -> str:
    """Return the HF/local bundle that owns text encoder + default VAE for this DiT."""
    tid = normalize_transformer_model_id(transformer_model_id).lower().replace("\\", "/")
    if "klein" in tid:
        if "4b" in tid:
            return "black-forest-labs/FLUX.2-klein-4B"
        return "black-forest-labs/FLUX.2-klein-9B"
    return "black-forest-labs/FLUX.2-dev"


def text_encoder_model_config(transformer_model_id, extra):
    base = bundle_base_model_id(transformer_model_id)
    return ModelConfig(
        model_id=base,
        origin_file_pattern="text_encoder/*.safetensors",
        **extra,
    )


def tokenizer_model_config(transformer_model_id):
    base = bundle_base_model_id(transformer_model_id)
    return ModelConfig(model_id=base, origin_file_pattern="tokenizer/")


def transformer_model_config(transformer_model_id, extra):
    transformer_model_id = normalize_transformer_model_id(transformer_model_id)
    if os.path.isfile(transformer_model_id):
        # 直接指向单个 .safetensors 文件（例如社区合并/量化的单文件权重，如 wikeeyang 的
        # Flux2-Klein-9B-True-Vx-bf16.safetensors）。跳过 model_id + origin_file_pattern
        # 的仓库下载/查找逻辑，交给 ModelPool.auto_load_model 按 state_dict 的 key/shape
        # 哈希自动识别模型类型（需要该哈希已在 diffsynth/configs/model_configs.py 中注册，
        # 否则会报 "Cannot detect the model type"）。
        return ModelConfig(path=transformer_model_id, **extra)
    return ModelConfig(
        model_id=transformer_model_id,
        origin_file_pattern="transformer/*.safetensors",
        **extra,
    )


def normalize_vae_path(vae_path):
    if vae_path is None:
        return None
    vae_path = str(vae_path).strip()
    if not vae_path:
        return None
    if os.path.isdir(vae_path):
        candidate = os.path.join(vae_path, "diffusion_pytorch_model.safetensors")
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(
            f"VAE 目录中找不到 diffusion_pytorch_model.safetensors: {vae_path}"
        )
    if not os.path.isfile(vae_path):
        raise FileNotFoundError(f"VAE 文件不存在: {vae_path}")
    return vae_path


def vae_model_config(vae_path, transformer_model_id=None, vram_extra=None):
    vram_extra = vram_extra or {}
    vae_path = normalize_vae_path(vae_path)
    if vae_path:
        return ModelConfig(path=vae_path, **vram_extra)
    default_id = (
        bundle_base_model_id(transformer_model_id)
        if transformer_model_id
        else DEFAULT_VAE_MODEL_ID
    )
    return ModelConfig(
        model_id=default_id,
        origin_file_pattern=DEFAULT_VAE_FILE,
        **vram_extra,
    )


def model_configs_for_bundle(transformer_model_id, vae_path, vram_strategy):
    te_extra = component_vram_extra("text_encoder", vram_strategy)
    dit_extra = component_vram_extra("transformer", vram_strategy)
    vae_extra = component_vram_extra("vae", vram_strategy)
    return [
        text_encoder_model_config(transformer_model_id, te_extra),
        transformer_model_config(transformer_model_id, dit_extra),
        vae_model_config(vae_path, transformer_model_id, vae_extra),
    ]


def build_pipeline(args, device, transformer_model_id=None, vae_path=None, vram_strategy=None):
    transformer_model_id = args.transformer_model_id if transformer_model_id is None else transformer_model_id
    vae_path = args.vae_path if vae_path is None else vae_path
    vram_strategy = vram_strategy or resolve_vram_strategy(args, transformer_model_id)
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs_for_bundle(transformer_model_id, vae_path, vram_strategy),
        tokenizer_config=tokenizer_model_config(transformer_model_id),
    )
    return pipe


def unload_module(module):
    if module is not None:
        del module
    gc.collect()
    torch.cuda.empty_cache()


def reload_tokenizer(pipe, transformer_model_id):
    tokenizer_config = tokenizer_model_config(transformer_model_id)
    tokenizer_config.download_if_necessary()
    pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)


def expected_text_encoder_kind(transformer_model_id: str) -> str:
    return "Qwen3" if "klein" in normalize_transformer_model_id(transformer_model_id).lower() else "Mistral"


def validate_transformer_bundle(pipe, transformer_model_id):
    transformer_model_id = normalize_transformer_model_id(transformer_model_id)
    if pipe.dit is None:
        raise RuntimeError("Transformer 加载失败。")

    te_kind = expected_text_encoder_kind(transformer_model_id)
    has_qwen = pipe.text_encoder_qwen3 is not None
    has_mistral = pipe.text_encoder is not None
    if te_kind == "Qwen3":
        if not has_qwen or has_mistral:
            raise RuntimeError(
                f"{transformer_model_id} 需要 Qwen3 text encoder，"
                f"当前加载状态 Mistral={has_mistral}, Qwen3={has_qwen}。"
            )
    elif not has_mistral or has_qwen:
        raise RuntimeError(
            f"{transformer_model_id} 需要 Mistral text encoder，"
            f"当前加载状态 Mistral={has_mistral}, Qwen3={has_qwen}。"
        )

    joint_dim = getattr(pipe.dit, "joint_attention_dim", None)
    if joint_dim is None and hasattr(pipe.dit, "context_embedder"):
        joint_dim = pipe.dit.context_embedder.in_features
    if te_kind == "Qwen3" and joint_dim not in (None, 12288, 7680):
        raise RuntimeError(
            f"DiT joint_attention_dim={joint_dim} 与 klein 期望的 12288/7680 不一致。"
        )
    if te_kind == "Mistral" and joint_dim not in (None, 15360):
        raise RuntimeError(
            f"DiT joint_attention_dim={joint_dim} 与 dev 期望的 15360 不一致。"
        )


def sync_vram_management_flag(pipe):
    pipe.vram_management_enabled = pipe.check_vram_management_state()


def should_reload_vae_on_transformer_switch(
    current_transformer_model_id,
    transformer_model_id,
    current_vae_path,
    old_vram_strategy,
    new_vram_strategy,
) -> bool:
    if old_vram_strategy["mode"] != new_vram_strategy["mode"]:
        return True
    if current_vae_path is not None:
        return False
    return bundle_base_model_id(current_transformer_model_id) != bundle_base_model_id(
        transformer_model_id
    )


def reload_transformer_module(
    pipe,
    args,
    device,
    transformer_model_id,
    vram_strategy=None,
    vae_path=None,
    reload_vae=False,
):
    transformer_model_id = normalize_transformer_model_id(transformer_model_id)
    vram_strategy = vram_strategy or resolve_vram_strategy(args, transformer_model_id)
    te_extra = component_vram_extra("text_encoder", vram_strategy)
    dit_extra = component_vram_extra("transformer", vram_strategy)
    vae_extra = component_vram_extra("vae", vram_strategy)

    old_dit = pipe.dit
    old_te = pipe.text_encoder
    old_qwen = pipe.text_encoder_qwen3
    old_vae = pipe.vae if reload_vae else None
    pipe.dit = None
    pipe.text_encoder = None
    pipe.text_encoder_qwen3 = None
    if reload_vae:
        pipe.vae = None
    unload_module(old_dit)
    unload_module(old_te)
    unload_module(old_qwen)
    if reload_vae:
        unload_module(old_vae)

    model_configs = [
        text_encoder_model_config(transformer_model_id, te_extra),
        transformer_model_config(transformer_model_id, dit_extra),
    ]
    if reload_vae:
        model_configs.append(vae_model_config(vae_path, transformer_model_id, vae_extra))

    model_pool = pipe.download_and_load_models(model_configs)
    pipe.text_encoder = model_pool.fetch_model("flux2_text_encoder")
    pipe.text_encoder_qwen3 = model_pool.fetch_model("z_image_text_encoder")
    pipe.dit = model_pool.fetch_model("flux2_dit")
    if reload_vae:
        pipe.vae = model_pool.fetch_model("flux2_vae")
        if pipe.vae is None:
            raise RuntimeError("VAE 加载失败。")

    validate_transformer_bundle(pipe, transformer_model_id)
    reload_tokenizer(pipe, transformer_model_id)
    sync_vram_management_flag(pipe)
    te_kind = "Qwen3" if pipe.text_encoder_qwen3 is not None else "Mistral"
    vae_note = " + VAE" if reload_vae else ""
    print(
        f"Text encoder bundle: {bundle_base_model_id(transformer_model_id)} ({te_kind}){vae_note}; "
        f"{format_vram_strategy(vram_strategy)}"
    )


def reload_vae_module(pipe, vae_path, transformer_model_id, vram_strategy):
    vae_extra = component_vram_extra("vae", vram_strategy)
    old_vae = pipe.vae
    pipe.vae = None
    unload_module(old_vae)

    model_pool = pipe.download_and_load_models(
        [vae_model_config(vae_path, transformer_model_id, vae_extra)]
    )
    new_vae = model_pool.fetch_model("flux2_vae")
    if new_vae is None:
        raise RuntimeError("VAE 加载失败。")
    pipe.vae = new_vae
    sync_vram_management_flag(pipe)


def load_lora_state_dict(lora_path):
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"LoRA file does not exist: {lora_path}")
    return load_file(lora_path, device="cpu")


def get_lora_layer_names(lora_state_dict):
    loader = GeneralLoRALoader()
    converted_state_dict = loader.convert_state_dict(lora_state_dict)
    return sorted(
        key[: -len(".lora_B.weight")]
        for key in converted_state_dict
        if key.endswith(".lora_B.weight")
    )


def get_uploaded_file_paths(uploaded_files):
    if uploaded_files is None:
        return []
    if isinstance(uploaded_files, (str, os.PathLike)):
        return [str(uploaded_files)]

    paths = []
    for uploaded_file in uploaded_files:
        if isinstance(uploaded_file, (str, os.PathLike)):
            paths.append(str(uploaded_file))
            continue

        path = getattr(uploaded_file, "path", None) or getattr(uploaded_file, "name", None)
        if path is None and isinstance(uploaded_file, dict):
            path = uploaded_file.get("path") or uploaded_file.get("name")
        if path:
            paths.append(str(path))
    return paths


def load_uploaded_images(uploaded_files):
    image_paths = get_uploaded_file_paths(uploaded_files)
    images = []
    for image_path in image_paths:
        if not os.path.isfile(image_path):
            raise gr.Error(f"输入图片不存在: {image_path}")
        with Image.open(image_path) as image:
            images.append(image.convert("RGB"))
    return images


def normalize_reference_images(reference_images):
    if not reference_images:
        return []
    return [image.convert("RGB") if isinstance(image, Image.Image) else image for image in reference_images]


def compute_auto_mu(steps, height, width):
    """What mu the scheduler would pick on its own (no override) for these settings."""
    try:
        dynamic_shift_len = (int(height) // 16) * (int(width) // 16)
        return FlowMatchScheduler.compute_empirical_mu(dynamic_shift_len, max(int(steps), 1))
    except Exception:
        return 0.8


def log_generation(log_dir, params, reference_images, result_image, step_previews=None):
    """Persist one generation (output + reference images + per-step previews + full params).

    Layout: <log_dir>/<YYYY-MM-DD>/<HHMMSS>_<shortid>/{output.png, ref_*.png, step_*.png, meta.json}
    Plus a flat <log_dir>/index.jsonl with one line per run for quick grepping/scanning.
    """
    now = datetime.now()
    day_dir = os.path.join(log_dir, now.strftime("%Y-%m-%d"))
    run_dir = os.path.join(day_dir, f"{now.strftime('%H%M%S')}_{uuid.uuid4().hex[:8]}")
    os.makedirs(run_dir, exist_ok=True)

    result_image.save(os.path.join(run_dir, "output.png"))
    for idx, ref_image in enumerate(reference_images):
        ref_image.save(os.path.join(run_dir, f"ref_{idx}.png"))
    step_previews = step_previews or []
    for idx, step_image in enumerate(step_previews):
        step_image.save(os.path.join(run_dir, f"step_{idx:02d}.png"))

    meta = dict(params)
    meta["timestamp"] = now.isoformat()
    meta["run_dir"] = run_dir
    meta["num_reference_images"] = len(reference_images)
    meta["num_step_previews"] = len(step_previews)

    with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(log_dir, "index.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    return run_dir


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Gradio demo.")

    device = f"cuda:{args.gpu}"
    pipeline_lock = threading.Lock()
    current_transformer_model_id = normalize_transformer_model_id(args.transformer_model_id)
    if args.vae_path:
        normalize_vae_path(args.vae_path)
    current_vae_path = normalize_vae_path(args.vae_path)
    current_lora_path = normalize_lora_path(args.lora)
    current_lora_alpha = float(args.lora_alpha)
    vram_strategy = resolve_vram_strategy(args, current_transformer_model_id)
    print(format_vram_strategy(vram_strategy))
    pipe = build_pipeline(
        args,
        device,
        current_transformer_model_id,
        current_vae_path,
        vram_strategy=vram_strategy,
    )
    base_weight_backup = {}

    def format_transformer_status():
        te_kind = "Qwen3" if pipe.text_encoder_qwen3 is not None else "Mistral"
        return (
            f"当前 Transformer model id: {current_transformer_model_id}\n"
            f"Text encoder bundle: {bundle_base_model_id(current_transformer_model_id)} ({te_kind})\n"
            f"{format_vram_strategy(vram_strategy)}"
        )

    def format_vae_status():
        if current_vae_path:
            return f"当前 VAE: {current_vae_path}"
        return f"当前 VAE: 默认 ({bundle_base_model_id(current_transformer_model_id)})"

    def format_lora_status():
        if current_lora_path:
            return f"当前 LoRA: {current_lora_path}\nalpha: {current_lora_alpha}"
        return "当前 LoRA: 未加载"

    def restore_lora_weights():
        if pipe is None or pipe.dit is None or not base_weight_backup:
            return
        module_dict = dict(pipe.dit.named_modules())
        for name, weight in base_weight_backup.items():
            module = module_dict.get(name)
            if module is None:
                continue
            state_dict = module.state_dict()
            state_dict["weight"] = weight.to(device=state_dict["weight"].device, dtype=state_dict["weight"].dtype)
            module.load_state_dict(state_dict)
        gc.collect()
        torch.cuda.empty_cache()

    def backup_lora_weights(layer_names):
        if pipe.dit is None:
            raise gr.Error("DiT 尚未加载，无法挂载 LoRA。")
        module_dict = dict(pipe.dit.named_modules())
        backed_up = 0
        missing_layers = []
        for name in layer_names:
            module = module_dict.get(name)
            if module is None or "weight" not in module.state_dict():
                missing_layers.append(name)
                continue
            if name not in base_weight_backup:
                base_weight_backup[name] = module.state_dict()["weight"].detach().cpu().clone()
                backed_up += 1
        return backed_up, missing_layers

    def apply_lora(lora_path, lora_alpha):
        nonlocal current_lora_path, current_lora_alpha
        lora_path = normalize_lora_path(lora_path)
        if not lora_path:
            raise gr.Error("请输入 LoRA 路径。")
        lora_alpha = float(lora_alpha)

        if current_lora_path and current_lora_path != lora_path:
            print(f"Unloading previous LoRA: {current_lora_path}")
            restore_lora_weights()
            current_lora_path = None

        try:
            lora_state_dict = load_lora_state_dict(lora_path)
        except FileNotFoundError as exc:
            raise gr.Error(str(exc)) from exc

        layer_names = get_lora_layer_names(lora_state_dict)
        backed_up, missing_layers = backup_lora_weights(layer_names)
        matched_layers = len(layer_names) - len(missing_layers)
        if matched_layers == 0:
            raise gr.Error("LoRA 没有匹配到任何 DiT 层，请确认使用的是 DiffSynth 格式 LoRA。")

        print(f"Loading LoRA: {lora_path} (alpha={lora_alpha})")
        if backed_up:
            print(f"Backed up {backed_up} base DiT weights for LoRA restore.")
        if missing_layers:
            print(f"[warning] {len(missing_layers)} LoRA layers were not found in DiT.")
        pipe.load_lora(pipe.dit, state_dict=lora_state_dict, alpha=lora_alpha)
        current_lora_path = lora_path
        current_lora_alpha = lora_alpha
        return format_lora_status()

    def load_lora_online(lora_path, lora_alpha):
        with pipeline_lock:
            restore_lora_weights()
            return apply_lora(lora_path, lora_alpha)

    def unload_lora_online(lora_alpha):
        nonlocal current_lora_path, current_lora_alpha
        with pipeline_lock:
            restore_lora_weights()
            current_lora_path = None
            current_lora_alpha = float(lora_alpha)
        return format_lora_status()

    def reload_transformer(transformer_model_id, reapply_lora=False):
        nonlocal current_transformer_model_id, current_lora_path, base_weight_backup, vram_strategy
        transformer_model_id = normalize_transformer_model_id(transformer_model_id)
        if transformer_model_id == current_transformer_model_id:
            return format_transformer_status(), format_vae_status(), format_lora_status()

        saved_lora_path = current_lora_path if reapply_lora else None
        saved_lora_alpha = current_lora_alpha if reapply_lora else current_lora_alpha

        old_vram_strategy = vram_strategy
        new_vram_strategy = resolve_vram_strategy(args, transformer_model_id)
        reload_vae = should_reload_vae_on_transformer_switch(
            current_transformer_model_id,
            transformer_model_id,
            current_vae_path,
            old_vram_strategy,
            new_vram_strategy,
        )
        print(f"Reloading transformer bundle: {transformer_model_id}")
        print(format_vram_strategy(new_vram_strategy))
        if reload_vae:
            print("同步重载 VAE（bundle 或 VRAM 策略已变化）")
        if current_lora_path and pipe.dit is not None:
            restore_lora_weights()
        base_weight_backup = {}
        current_lora_path = None
        try:
            reload_transformer_module(
                pipe,
                args,
                device,
                transformer_model_id,
                new_vram_strategy,
                vae_path=current_vae_path,
                reload_vae=reload_vae,
            )
        except Exception as exc:
            raise gr.Error(f"切换 Transformer 失败: {exc}") from exc
        vram_strategy = new_vram_strategy
        current_transformer_model_id = transformer_model_id

        if saved_lora_path:
            apply_lora(saved_lora_path, saved_lora_alpha)

        return format_transformer_status(), format_vae_status(), format_lora_status()

    def reload_vae(vae_path):
        nonlocal current_vae_path
        vae_path = normalize_vae_path(vae_path)
        if vae_path == current_vae_path:
            return format_transformer_status(), format_vae_status(), format_lora_status()

        print(f"Reloading VAE only: {vae_path or 'default'}")
        try:
            reload_vae_module(pipe, vae_path, current_transformer_model_id, vram_strategy)
        except Exception as exc:
            raise gr.Error(f"切换 VAE 失败: {exc}") from exc
        sync_vram_management_flag(pipe)
        current_vae_path = vae_path
        return format_transformer_status(), format_vae_status(), format_lora_status()

    def load_transformer_online(transformer_model_id):
        with pipeline_lock:
            return reload_transformer(transformer_model_id, reapply_lora=False)

    def reset_transformer_online():
        with pipeline_lock:
            return reload_transformer(DEFAULT_TRANSFORMER_MODEL_ID, reapply_lora=False)

    def load_vae_online(vae_path):
        try:
            vae_path = normalize_vae_path(vae_path)
        except FileNotFoundError as exc:
            raise gr.Error(str(exc)) from exc
        with pipeline_lock:
            return reload_vae(vae_path)

    def reset_vae_online():
        with pipeline_lock:
            return reload_vae(None)

    if current_lora_path:
        with pipeline_lock:
            apply_lora(current_lora_path, current_lora_alpha)

    def add_uploaded_reference_images(uploaded_files, reference_images):
        images = list(reference_images or [])
        images.extend(load_uploaded_images(uploaded_files))
        if not images:
            raise gr.Error("请先上传参考图。")
        return images, images

    def add_pasted_reference_image(pasted_image, reference_images):
        if pasted_image is None:
            raise gr.Error("请先粘贴或上传一张参考图。")
        images = list(reference_images or [])
        images.append(pasted_image.convert("RGB"))
        return images, images

    def clear_reference_images():
        return [], []

    def infer(
        prompt,
        reference_images,
        seed,
        steps,
        cfg,
        embedded_guidance,
        s2_scale,
        s2_drop_ratio,
        s2_start,
        s2_end,
        sigma_mu,
        height,
        width,
    ):
        if not prompt or not prompt.strip():
            raise gr.Error("请输入提示词。")

        seed = int(seed)
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)

        images = normalize_reference_images(reference_images)
        edit_image = images if images else None

        events = queue.Queue()
        outcome = {}

        def step_callback(progress_id, num_steps, preview_image):
            events.put(("step", progress_id, num_steps, preview_image))

        def worker():
            try:
                with pipeline_lock:
                    if pipe is None:
                        raise gr.Error("模型正在加载或尚未加载完成，请稍后重试。")
                    with torch.inference_mode():
                        result, step_previews = pipe(
                            prompt=prompt.strip(),
                            negative_prompt="",
                            cfg_scale=float(cfg),
                            embedded_guidance=float(embedded_guidance),
                            input_image=None,
                            edit_image=edit_image,
                            edit_image_auto_resize=True,
                            edit_image_scale=1.0,
                            height=int(height),
                            width=int(width),
                            seed=seed,
                            rand_device=device,
                            num_inference_steps=int(steps),
                            detail_amount=0.0,
                            s2_scale=float(s2_scale),
                            s2_drop_ratio=float(s2_drop_ratio),
                            s2_start=float(s2_start),
                            s2_end=float(s2_end),
                            sigma_mu=float(sigma_mu),
                            return_intermediate_steps=True,
                            step_callback=step_callback,
                        )
                outcome["result"] = result
                outcome["step_previews"] = step_previews
            except Exception as exc:
                outcome["error"] = exc
            finally:
                events.put(("done",))

        threading.Thread(target=worker, daemon=True).start()

        collected_previews = []
        while True:
            event = events.get()
            if event[0] == "step":
                _, progress_id, num_steps, preview_image = event
                collected_previews.append(preview_image)
                gallery_items = [(img, f"step {i}") for i, img in enumerate(collected_previews)]
                yield preview_image, f"生成中... 第 {progress_id + 1}/{num_steps} 步", gallery_items
            else:
                break

        if "error" in outcome:
            exc = outcome["error"]
            raise exc if isinstance(exc, gr.Error) else gr.Error(f"生成失败: {exc}")

        result = outcome["result"]
        step_previews = outcome["step_previews"]

        run_dir = None
        if not args.no_log:
            params = dict(
                prompt=prompt.strip(),
                seed=seed,
                steps=int(steps),
                cfg=float(cfg),
                embedded_guidance=float(embedded_guidance),
                s2_scale=float(s2_scale),
                s2_drop_ratio=float(s2_drop_ratio),
                s2_start=float(s2_start),
                s2_end=float(s2_end),
                sigma_mu=float(sigma_mu),
                sigma_mu_auto=compute_auto_mu(steps, height, width),
                height=int(height),
                width=int(width),
                transformer_model_id=current_transformer_model_id,
                lora_path=current_lora_path,
                lora_alpha=current_lora_alpha,
                vae_path=current_vae_path,
            )
            try:
                run_dir = log_generation(args.log_dir, params, images, result, step_previews)
            except Exception as exc:
                print(f"[warning] 保存生成日志失败: {exc}")

        auto_mu = compute_auto_mu(steps, height, width)
        mu_desc = f"{float(sigma_mu):.3f}(手动)" if sigma_mu and float(sigma_mu) > 0 else f"{auto_mu:.3f}(自动)"
        run_info = (
            f"seed={seed} | steps={int(steps)} cfg={float(cfg)} embedded_guidance={float(embedded_guidance)} | "
            f"s2_scale={float(s2_scale)} s2_drop_ratio={float(s2_drop_ratio)} s2_range=[{float(s2_start)}, {float(s2_end)}] | "
            f"sigma_mu={mu_desc}"
        )
        if run_dir:
            run_info += f"\n已保存到: {run_dir}"
        step_gallery_items = [(img, f"step {i}") for i, img in enumerate(step_previews)]
        yield result, run_info, step_gallery_items

    with gr.Blocks(title="DreamFaceOmini Gradio") as demo:
        gr.Markdown("# DreamFaceOmini 单卡图片编辑")
        gr.Markdown("输入提示词，可选添加一张或多张参考图，输出编辑后的结果图。")
        reference_images_state = gr.State([])

        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="提示词", lines=6)
                input_images = gr.File(
                    label="批量上传参考图（可多选，可留空）",
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                )
                with gr.Row():
                    add_uploaded_button = gr.Button("添加上传图")
                    clear_reference_button = gr.Button("清空参考图")
                pasted_image = gr.Image(
                    label="粘贴参考图（可多次粘贴后添加）",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                add_pasted_button = gr.Button("添加粘贴图")
                reference_gallery = gr.Gallery(
                    label="当前参考图",
                    columns=4,
                    rows=2,
                    height=260,
                    type="pil",
                )
                with gr.Accordion("Transformer 在线加载", open=False):
                    transformer_model_id = gr.Textbox(
                        label="Transformer model id（HF/ModelScope 仓库 id，或本地单个 .safetensors 文件路径）",
                        value=current_transformer_model_id,
                    )
                    with gr.Row():
                        load_transformer_button = gr.Button("加载/切换 Transformer")
                        reset_transformer_button = gr.Button("恢复默认 Transformer")
                    transformer_status = gr.Textbox(
                        label="Transformer 状态",
                        value=format_transformer_status(),
                        interactive=False,
                        lines=2,
                    )
                with gr.Accordion("VAE 在线加载", open=False):
                    vae_path_input = gr.Textbox(
                        label="VAE 路径（.safetensors 或 vae/ 目录，留空为默认）",
                        value=current_vae_path or "",
                    )
                    with gr.Row():
                        load_vae_button = gr.Button("加载/切换 VAE")
                        reset_vae_button = gr.Button("恢复默认 VAE")
                    vae_status = gr.Textbox(
                        label="VAE 状态",
                        value=format_vae_status(),
                        interactive=False,
                        lines=2,
                    )
                with gr.Accordion("LoRA 在线加载", open=False):
                    lora_path = gr.Textbox(label="LoRA 路径", value=current_lora_path or "")
                    lora_alpha = gr.Number(label="LoRA alpha", value=current_lora_alpha)
                    with gr.Row():
                        load_lora_button = gr.Button("加载/切换 LoRA")
                        unload_lora_button = gr.Button("卸载 LoRA")
                    lora_status = gr.Textbox(label="LoRA 状态", value=format_lora_status(), interactive=False, lines=3)
                with gr.Accordion("推理参数", open=False):
                    seed = gr.Number(label="Seed", value=args.seed, precision=0)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=args.steps, step=1)
                    cfg = gr.Slider(label="CFG", minimum=0.0, maximum=10.0, value=args.cfg, step=0.1)
                    embedded_guidance = gr.Slider(label="Embedded Guidance", minimum=0.0, maximum=10.0, value=args.embedded_guidance, step=0.1)
                    height = gr.Number(label="Height", value=args.height, precision=0)
                    width = gr.Number(label="Width", value=args.width, precision=0)
                with gr.Accordion("S²-Guidance（实验：随机丢 block 自我引导，缓解畸形/坏手）", open=True):
                    gr.Markdown(
                        "ω=0 等价于关闭（当前 wandb 对齐默认值）。开启后每个作用步会多算一次前向，"
                        "步数越少相对开销越大（4 步开满约等于多跑一倍算力）。"
                    )
                    s2_scale = gr.Slider(label="S² Scale (ω)", minimum=0.0, maximum=1.0, value=args.s2_scale, step=0.05)
                    s2_drop_ratio = gr.Slider(label="Block Drop Ratio", minimum=0.0, maximum=0.5, value=args.s2_drop_ratio, step=0.05)
                    s2_start = gr.Slider(label="S² Start（起始步比例）", minimum=0.0, maximum=1.0, value=args.s2_start, step=0.05)
                    s2_end = gr.Slider(label="S² End（结束步比例）", minimum=0.0, maximum=1.0, value=args.s2_end, step=0.05)
                with gr.Accordion("Sigma Schedule（实验：手动改 shift μ，谨慎用）", open=False):
                    gr.Markdown(
                        "μ 越大 → 前几步越贴近 σ=1（步子更保守，纯噪声阶段停留更久）；"
                        "μ 越小 → σ 分布更平均，但最后一步跳到 0 的幅度不会变小（σ_min=1/steps 是硬下限）。\n\n"
                        "**该模型是按自动 μ 蒸馏/训练的，偏离自动值＝离开训练分布，出的图可能更差而不是更好，仅用于诊断实验。**"
                        "设为 0 = 自动（不改变现有行为）。"
                    )
                    auto_mu_hint = gr.Markdown(f"当前设置下自动 μ ≈ **{compute_auto_mu(args.steps, args.height, args.width):.3f}**")
                    sigma_mu = gr.Slider(label="Sigma Shift μ override（0=自动）", minimum=0.0, maximum=4.0, value=args.sigma_mu, step=0.05)
                run_button = gr.Button("生成", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="输出图片（生成过程中会逐步刷新为每一步的预览，完成后定格为最终结果）", type="pil")
                run_info = gr.Textbox(
                    label="生成进度 / 本次实际使用的参数（复现、对照实验用）",
                    interactive=False,
                    lines=2,
                )
                step_gallery = gr.Gallery(
                    label="逐步回放（x0 estimate：从当前步一次外推到终点，不是原始噪声中间态；生成中实时填充，完成后仍可逐步查看）",
                    columns=4,
                    height=220,
                    type="pil",
                )

        run_button.click(
            fn=infer,
            inputs=[
                prompt,
                reference_images_state,
                seed,
                steps,
                cfg,
                embedded_guidance,
                s2_scale,
                s2_drop_ratio,
                s2_start,
                s2_end,
                sigma_mu,
                height,
                width,
            ],
            outputs=[output_image, run_info, step_gallery],
        )
        for control in (steps, height, width):
            control.change(
                fn=lambda s, h, w: f"当前设置下自动 μ ≈ **{compute_auto_mu(s, h, w):.3f}**",
                inputs=[steps, height, width],
                outputs=auto_mu_hint,
            )
        add_uploaded_button.click(
            fn=add_uploaded_reference_images,
            inputs=[input_images, reference_images_state],
            outputs=[reference_images_state, reference_gallery],
        )
        add_pasted_button.click(
            fn=add_pasted_reference_image,
            inputs=[pasted_image, reference_images_state],
            outputs=[reference_images_state, reference_gallery],
        )
        clear_reference_button.click(
            fn=clear_reference_images,
            outputs=[reference_images_state, reference_gallery],
        )
        load_transformer_button.click(
            fn=load_transformer_online,
            inputs=[transformer_model_id],
            outputs=[transformer_status, vae_status, lora_status],
        )
        reset_transformer_button.click(
            fn=reset_transformer_online,
            outputs=[transformer_status, vae_status, lora_status],
        )
        load_vae_button.click(
            fn=load_vae_online,
            inputs=[vae_path_input],
            outputs=[transformer_status, vae_status, lora_status],
        )
        reset_vae_button.click(
            fn=reset_vae_online,
            outputs=[transformer_status, vae_status, lora_status],
        )
        load_lora_button.click(
            fn=load_lora_online,
            inputs=[lora_path, lora_alpha],
            outputs=lora_status,
        )
        unload_lora_button.click(
            fn=unload_lora_online,
            inputs=[lora_alpha],
            outputs=lora_status,
        )

    demo.queue(max_size=8).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
