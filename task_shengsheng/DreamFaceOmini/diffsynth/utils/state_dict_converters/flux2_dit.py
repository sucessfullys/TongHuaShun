import torch

from diffsynth.core.loader.file import load_state_dict


def _swap_scale_shift(weight: torch.Tensor, dim: int = 0) -> torch.Tensor:
    shift, scale = weight.chunk(2, dim=dim)
    return torch.cat([scale, shift], dim=dim)


def _is_bfl_native_flux2_dit(state_dict) -> bool:
    for key in state_dict:
        if key.startswith("double_blocks.") or key.startswith("single_blocks."):
            return True
        if key in {"img_in.weight", "time_in.in_layer.weight", "double_stream_modulation_img.lin.weight"}:
            return True
    return False


def _is_rename_metadata_pass(state_dict) -> bool:
    from diffsynth.core.vram.disk_map import DiskMap

    if isinstance(state_dict, DiskMap):
        return False
    try:
        value = next(iter(state_dict.values()))
    except StopIteration:
        return False
    return isinstance(value, str)


def _build_bfl_native_rename_dict(state_dict) -> dict:
    """Build {model_key: file_key} for DiskMap.fetch_rename_dict (1:1 renames only)."""
    rename_dict = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
        "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "final_layer.linear": "proj_out",
    }
    block_key_map = {
        "img_attn.norm.query_norm": "attn.norm_q",
        "img_attn.norm.key_norm": "attn.norm_k",
        "img_attn.proj": "attn.to_out.0",
        "img_mlp.0": "ff.linear_in",
        "img_mlp.2": "ff.linear_out",
        "txt_attn.norm.query_norm": "attn.norm_added_q",
        "txt_attn.norm.key_norm": "attn.norm_added_k",
        "txt_attn.proj": "attn.to_add_out",
        "txt_mlp.0": "ff_context.linear_in",
        "txt_mlp.2": "ff_context.linear_out",
    }
    single_block_key_map = {
        "linear1": "attn.to_qkv_mlp_proj",
        "linear2": "attn.to_out",
        "norm.query_norm": "attn.norm_q",
        "norm.key_norm": "attn.norm_k",
    }

    converted = {}
    for file_key in state_dict:
        if "adaLN_modulation" in file_key:
            converted[file_key.replace("final_layer.adaLN_modulation.1", "norm_out.linear")] = file_key
            continue

        new_key = file_key
        for replace_key, target in rename_dict.items():
            new_key = new_key.replace(replace_key, target)
        if new_key != file_key:
            converted[new_key] = file_key
            continue

        parts = file_key.split(".")
        if parts[0] == "double_blocks" and "qkv" not in file_key:
            within_block_name = ".".join(parts[2:-1])
            param_type = parts[-1]
            if param_type == "scale":
                param_type = "weight"
            if within_block_name in block_key_map:
                model_key = f"transformer_blocks.{parts[1]}.{block_key_map[within_block_name]}.{param_type}"
                converted[model_key] = file_key
            continue

        if parts[0] == "single_blocks":
            within_block_name = ".".join(parts[2:-1])
            param_type = parts[-1]
            if param_type == "scale":
                param_type = "weight"
            if within_block_name in single_block_key_map:
                model_key = (
                    f"single_transformer_blocks.{parts[1]}.{single_block_key_map[within_block_name]}.{param_type}"
                )
                converted[model_key] = file_key

    return converted


def _convert_flux2_double_stream_blocks(key: str, state_dict: dict) -> None:
    if ".weight" not in key and ".bias" not in key and ".scale" not in key:
        return
    if not key.startswith("double_blocks."):
        return

    block_key_map = {
        "img_attn.norm.query_norm": "attn.norm_q",
        "img_attn.norm.key_norm": "attn.norm_k",
        "img_attn.proj": "attn.to_out.0",
        "img_mlp.0": "ff.linear_in",
        "img_mlp.2": "ff.linear_out",
        "txt_attn.norm.query_norm": "attn.norm_added_q",
        "txt_attn.norm.key_norm": "attn.norm_added_k",
        "txt_attn.proj": "attn.to_add_out",
        "txt_mlp.0": "ff_context.linear_in",
        "txt_mlp.2": "ff_context.linear_out",
    }

    parts = key.split(".")
    block_idx = parts[1]
    modality_block_name = parts[2]
    within_block_name = ".".join(parts[2:-1])
    param_type = parts[-1]
    if param_type == "scale":
        param_type = "weight"

    if "qkv" in within_block_name:
        fused_qkv_weight = state_dict.pop(key)
        to_q_weight, to_k_weight, to_v_weight = torch.chunk(fused_qkv_weight, 3, dim=0)
        if "img" in modality_block_name:
            q_name, k_name, v_name = "attn.to_q", "attn.to_k", "attn.to_v"
        elif "txt" in modality_block_name:
            q_name, k_name, v_name = "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"
        else:
            return
        prefix = f"transformer_blocks.{block_idx}"
        state_dict[f"{prefix}.{q_name}.{param_type}"] = to_q_weight
        state_dict[f"{prefix}.{k_name}.{param_type}"] = to_k_weight
        state_dict[f"{prefix}.{v_name}.{param_type}"] = to_v_weight
        return

    new_within_block_name = block_key_map[within_block_name]
    new_key = f"transformer_blocks.{block_idx}.{new_within_block_name}.{param_type}"
    state_dict[new_key] = state_dict.pop(key)


def _convert_flux2_single_stream_blocks(key: str, state_dict: dict) -> None:
    if ".weight" not in key and ".bias" not in key and ".scale" not in key:
        return
    if not key.startswith("single_blocks."):
        return

    block_key_map = {
        "linear1": "attn.to_qkv_mlp_proj",
        "linear2": "attn.to_out",
        "norm.query_norm": "attn.norm_q",
        "norm.key_norm": "attn.norm_k",
    }

    parts = key.split(".")
    block_idx = parts[1]
    within_block_name = ".".join(parts[2:-1])
    param_type = parts[-1]
    if param_type == "scale":
        param_type = "weight"

    new_within_block_name = block_key_map[within_block_name]
    new_key = f"single_transformer_blocks.{block_idx}.{new_within_block_name}.{param_type}"
    state_dict[new_key] = state_dict.pop(key)


def _convert_ada_layer_norm_weights(key: str, state_dict: dict) -> None:
    if ".weight" not in key or "adaLN_modulation" not in key:
        return
    key_without_param_type, param_type = key.rsplit(".", maxsplit=1)
    if key_without_param_type != "final_layer.adaLN_modulation.1":
        return
    new_key = f"norm_out.linear.{param_type}"
    state_dict[new_key] = _swap_scale_shift(state_dict.pop(key), dim=0)


def convert_flux2_dit_checkpoint_from_bfl_native(checkpoint: dict) -> dict:
    """Convert BFL native Flux2 DiT keys (single-file .safetensors) to diffsynth Flux2DiT layout."""
    rename_dict = {
        "img_in": "x_embedder",
        "txt_in": "context_embedder",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
        "guidance_in.in_layer": "time_guidance_embed.guidance_embedder.linear_1",
        "guidance_in.out_layer": "time_guidance_embed.guidance_embedder.linear_2",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "final_layer.linear": "proj_out",
    }

    converted = {key: checkpoint[key] for key in checkpoint}
    for key in list(converted.keys()):
        new_key = key
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        if new_key != key:
            converted[new_key] = converted.pop(key)

    for key in list(converted.keys()):
        if "adaLN_modulation" in key:
            _convert_ada_layer_norm_weights(key, converted)
        elif key.startswith("double_blocks."):
            _convert_flux2_double_stream_blocks(key, converted)
        elif key.startswith("single_blocks."):
            _convert_flux2_single_stream_blocks(key, converted)

    return converted


def _materialize_state_dict(state_dict):
    from diffsynth.core.vram.disk_map import DiskMap

    if isinstance(state_dict, DiskMap):
        return load_state_dict(
            state_dict.path,
            torch_dtype=state_dict.torch_dtype,
            device=state_dict.device,
        )
    return state_dict


def Flux2DiTStateDictConverterFromBFLNative(state_dict):
    if _is_rename_metadata_pass(state_dict):
        if not _is_bfl_native_flux2_dit(state_dict):
            return state_dict
        return _build_bfl_native_rename_dict(state_dict)

    tensor_state_dict = _materialize_state_dict(state_dict)
    if not _is_bfl_native_flux2_dit(tensor_state_dict):
        return tensor_state_dict
    return convert_flux2_dit_checkpoint_from_bfl_native(tensor_state_dict)
