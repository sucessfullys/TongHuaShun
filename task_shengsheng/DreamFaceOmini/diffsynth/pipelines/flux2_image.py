import torch, math, torchvision, random
from PIL import Image
from tqdm import tqdm
from einops import rearrange
import numpy as np
from typing import Any, Callable, Union, List, Optional, Tuple

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput

from transformers import AutoProcessor, AutoTokenizer
from ..models.flux2_text_encoder import Flux2TextEncoder
from ..models.flux2_dit import Flux2DiT
from ..models.flux2_vae import Flux2VAE
from ..models.z_image_text_encoder import ZImageTextEncoder


def module_computation_device(module, fallback):
    if module is None:
        return fallback
    computation_device = getattr(module, "computation_device", None)
    if computation_device is not None:
        return computation_device
    try:
        return next(module.parameters()).device
    except StopIteration:
        return getattr(module, "device", fallback)


def _make_detail_daemon_schedule(steps, amount=0.1, start=0.2, end=0.8, bias=0.5, exponent=1.0):
    """Build a bell-shaped sigma-adjustment schedule (Detail Daemon).
    Returns an array of length `steps` with peak value `amount` in the middle."""
    mid = start + bias * (end - start)
    schedule = np.zeros(steps)
    si, mi, ei = (int(round(x * (steps - 1))) for x in (start, mid, end))
    if mi > si:
        ramp = np.linspace(0, 1, mi - si + 1)
        ramp = 0.5 * (1 - np.cos(ramp * np.pi))
        schedule[si:mi + 1] = (ramp ** exponent) * amount
    if ei > mi:
        ramp = np.linspace(1, 0, ei - mi + 1)
        ramp = 0.5 * (1 - np.cos(ramp * np.pi))
        schedule[mi:ei + 1] = (ramp ** exponent) * amount
    return schedule


class Flux2ImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("FLUX.2")
        self.text_encoder: Flux2TextEncoder = None
        self.text_encoder_qwen3: ZImageTextEncoder = None
        self.dit: Flux2DiT = None
        self.vae: Flux2VAE = None
        self.tokenizer: AutoProcessor = None
        self.in_iteration_models = ("dit",)
        self.units = [
            Flux2Unit_ShapeChecker(),
            Flux2Unit_PromptEmbedder(),
            Flux2Unit_Qwen3PromptEmbedder(),
            Flux2Unit_NoiseInitializer(),
            Flux2Unit_InputImageEmbedder(),
            Flux2Unit_EditImageEmbedder(),
            Flux2Unit_ImageIDs(),
        ]
        self.model_fn = model_fn_flux2
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
    ):
        # Initialize pipeline
        pipe = Flux2ImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        pipe.text_encoder = model_pool.fetch_model("flux2_text_encoder")
        pipe.text_encoder_qwen3 = model_pool.fetch_model("z_image_text_encoder")
        pipe.dit = model_pool.fetch_model("flux2_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")
        if tokenizer_config is not None:
            tokenizer_config.download_if_necessary()
            pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Edit
        edit_image: Union[Image.Image, List[Image.Image]] = None,
        edit_image_auto_resize: bool = True,
        edit_image_scale: float = 1.0,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        initial_noise: torch.Tensor = None,
        # Steps
        num_inference_steps: int = 30,
        # Sigma schedule: override the auto-computed shift mu (None/<=0 = auto, matches training)
        sigma_mu: Optional[float] = None,
        # Detail enhancement (Detail Daemon)
        detail_amount: float = 0.1,
        # S²-Guidance (stochastic self-guidance)
        s2_scale: float = 0.0,
        s2_drop_ratio: float = 0.1,
        s2_start: float = 0.1,
        s2_end: float = 0.9,
        attention_probe: Any = None,
        # Debug: decode a per-step "predicted final image" preview (x0 estimate), for diagnosing at which step an artifact first appears
        return_intermediate_steps: bool = False,
        step_callback: Optional[Callable[[int, int, Image.Image], None]] = None,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        self.scheduler.set_timesteps(
            num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=height//16*width//16,
            mu_override=sigma_mu if sigma_mu and sigma_mu > 0 else None,
        )

        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize, "edit_image_scale": edit_image_scale,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device, "initial_noise": initial_noise,
            "num_inference_steps": num_inference_steps,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        if attention_probe is not None:
            prompt_meta = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "prompt_token_records": inputs_posi.get("prompt_token_records", []),
                "target_grid": {
                    "height": height // 16,
                    "width": width // 16,
                },
                "edit_ref_ranges": inputs_shared.get("edit_ref_ranges", []),
            }
            set_sample_metadata = getattr(attention_probe, "set_sample_metadata", None)
            if set_sample_metadata is not None:
                set_sample_metadata(prompt_meta)
            inputs_posi["attention_probe"] = attention_probe
            inputs_posi["attention_probe_metadata"] = prompt_meta

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        num_steps = len(self.scheduler.timesteps)
        dd_schedule = _make_detail_daemon_schedule(num_steps, amount=detail_amount) if detail_amount > 0 else None
        want_step_previews = return_intermediate_steps or step_callback is not None
        step_previews = []

        # S²-Guidance: precompute step boundaries and RNG
        s2_active = s2_scale > 0.0 and self.dit is not None
        if s2_active:
            s2_rng = random.Random(seed)
            n_single = len(self.dit.single_transformer_blocks)
            n_drop = max(1, round(n_single * s2_drop_ratio))
            s2_start_step = round(s2_start * max(num_steps - 1, 1))
            s2_end_step = round(s2_end * max(num_steps - 1, 1))

        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            model_timestep = timestep
            if dd_schedule is not None:
                dd_adj = float(dd_schedule[progress_id]) * 0.1
                model_timestep = timestep * max(1e-6, 1.0 - dd_adj)

            apply_s2 = s2_active and s2_start_step <= progress_id <= s2_end_step

            if not apply_s2:
                noise_pred = self.cfg_guided_model_fn(
                    self.model_fn, cfg_scale,
                    inputs_shared, inputs_posi, inputs_nega,
                    **models, timestep=model_timestep, progress_id=progress_id
                )
            else:
                model_kwargs = dict(**models, timestep=model_timestep, progress_id=progress_id)
                noise_pred_posi = self.model_fn(**inputs_posi, **inputs_shared, **model_kwargs)
                if cfg_scale != 1.0:
                    noise_pred_nega = self.model_fn(**inputs_nega, **inputs_shared, **model_kwargs)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi
                drop_indices = set(s2_rng.sample(range(1, n_single), min(n_drop, n_single - 1)))
                noise_pred_sub = self.model_fn(**inputs_posi, **inputs_shared, **model_kwargs, drop_block_indices=drop_indices)
                noise_pred = noise_pred - s2_scale * (noise_pred_sub - noise_pred_posi)

            if want_step_previews:
                # One-shot Euler extrapolation straight to sigma=0 with the *current* velocity prediction,
                # i.e. "what the model currently thinks the final image looks like" - decodable even at step 0,
                # unlike the raw (still mostly noise) latent x_t. Decoded immediately (not batched at the
                # end) so `step_callback` can drive a live/animated preview in the caller's UI.
                x0_pred = self.scheduler.step(
                    noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"], to_final=True
                )
                _W_lat = inputs_shared["width"] // 16
                _H_lat = x0_pred.shape[1] // _W_lat
                preview_latents = rearrange(x0_pred, "B (H W) C -> B C H W", H=_H_lat, W=_W_lat)
                self.load_models_to_device(['vae'])
                preview_image = self.vae_output_to_image(self.vae.decode(preview_latents))
                self.load_models_to_device(self.in_iteration_models)
                step_previews.append(preview_image)
                if step_callback is not None:
                    step_callback(progress_id, num_steps, preview_image)

            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae'])
        _final_W = inputs_shared["width"] // 16
        _final_H = inputs_shared["latents"].shape[1] // _final_W
        latents = rearrange(inputs_shared["latents"], "B (H W) C -> B C H W", H=_final_H, W=_final_W)
        image = self.vae.decode(latents)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        if return_intermediate_steps:
            return image, step_previews
        return image


class Flux2Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: Flux2ImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


class Flux2Unit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder",)
        )
        self.system_message = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."

    def format_text_input(self, prompts: List[str], system_message: str = None):
        # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
        # when truncation is enabled. The processor counts [IMG] tokens and fails
        # if the count changes after truncation.
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in prompts]

        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned_txt
        ]

    def get_mistral_3_small_prompt_embeds(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
        # fmt: off
        system_message: str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation.",
        # fmt: on
        hidden_states_layers: List[int] = (10, 20, 30),
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        # Format input messages
        messages_batch = self.format_text_input(prompts=prompt, system_message=system_message)

        # Process all messages at once
        inputs = tokenizer.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        # Move to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds
    
    def prepare_text_ids(
        self,
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def encode_prompt(
        self,
        text_encoder,
        tokenizer,
        prompt: Union[str, List[str]],
        dtype = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
        text_encoder_out_layers: Tuple[int] = (10, 20, 30),
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self.get_mistral_3_small_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
                system_message=self.system_message,
                hidden_states_layers=text_encoder_out_layers,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def process(self, pipe: Flux2ImagePipeline, prompt):
        # Skip if Qwen3 text encoder is available (handled by Qwen3PromptEmbedder)
        if pipe.text_encoder_qwen3 is not None:
            return {}
        
        pipe.load_models_to_device(self.onload_model_names)
        te_device = module_computation_device(pipe.text_encoder, pipe.device)
        prompt_embeds, text_ids = self.encode_prompt(
            pipe.text_encoder, pipe.tokenizer, prompt,
            dtype=pipe.torch_dtype, device=te_device,
        )
        prompt_embeds = prompt_embeds.to(device=pipe.device, dtype=pipe.torch_dtype)
        text_ids = text_ids.to(device=pipe.device)
        return {"prompt_embeds": prompt_embeds, "text_ids": text_ids}


class Flux2Unit_Qwen3PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder_qwen3",)
        )
        self.hidden_states_layers = (9, 18, 27)  # Qwen3 layers

    def get_qwen3_prompt_embeds(
        self,
        text_encoder: ZImageTextEncoder,
        tokenizer: AutoTokenizer,
        prompt: Union[str, List[str]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 512,
    ):
        dtype = text_encoder.dtype if dtype is None else dtype
        device = text_encoder.device if device is None else device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )

            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        # Forward pass through the model
        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Only use outputs from intermediate layers and stack them
        out = torch.stack([output.hidden_states[k] for k in self.hidden_states_layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
        return prompt_embeds

    def prepare_text_ids(
        self,
        x: torch.Tensor,  # (B, L, D) or (L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def encode_prompt(
        self,
        text_encoder: ZImageTextEncoder,
        tokenizer: AutoTokenizer,
        prompt: Union[str, List[str]],
        dtype = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_embeds = self.get_qwen3_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.prepare_text_ids(prompt_embeds)
        text_ids = text_ids.to(device)
        return prompt_embeds, text_ids

    def get_prompt_token_records(
        self,
        tokenizer: AutoTokenizer,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 512,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        all_records = []
        for single_prompt in prompt:
            messages = [{"role": "user", "content": single_prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_sequence_length,
            )
            input_ids = inputs["input_ids"][0].tolist()
            attention_mask = inputs["attention_mask"][0].tolist()
            records = []
            for idx, (token_id, active) in enumerate(zip(input_ids, attention_mask)):
                token_text = tokenizer.decode([token_id], skip_special_tokens=False)
                records.append({
                    "index": idx,
                    "id": int(token_id),
                    "text": token_text,
                    "active": bool(active),
                })
            all_records.append(records)
        return all_records

    def process(self, pipe: Flux2ImagePipeline, prompt):
        # Check if Qwen3 text encoder is available
        if pipe.text_encoder_qwen3 is None:
            return {}
        
        pipe.load_models_to_device(self.onload_model_names)
        te_device = module_computation_device(pipe.text_encoder_qwen3, pipe.device)
        prompt_embeds, text_ids = self.encode_prompt(
            pipe.text_encoder_qwen3, pipe.tokenizer, prompt,
            dtype=pipe.torch_dtype, device=te_device,
        )
        prompt_embeds = prompt_embeds.to(device=pipe.device, dtype=pipe.torch_dtype)
        text_ids = text_ids.to(device=pipe.device)
        prompt_token_records = self.get_prompt_token_records(pipe.tokenizer, prompt)
        return {"prompt_embeds": prompt_embeds, "text_ids": text_ids, "prompt_token_records": prompt_token_records}


class Flux2Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device", "initial_noise"),
            output_params=("noise",),
        )

    def process(self, pipe: Flux2ImagePipeline, height, width, seed, rand_device, initial_noise):
        if initial_noise is not None:
            noise = initial_noise.clone()
        else:
            noise = pipe.generate_noise((1, 128, height//16, width//16), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        noise = noise.reshape(1, 128, height//16 * width//16).permute(0, 2, 1)
        return {"noise": noise}


class Flux2Unit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: Flux2ImagePipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae.encode(image)
        input_latents = rearrange(input_latents, "B C H W -> B (H W) C")
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class Flux2Unit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "edit_image_auto_resize"),
            output_params=("edit_latents", "edit_image_ids", "edit_ref_ranges"),
            onload_model_names=("vae",)
        )

    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def edit_image_auto_resize(self, edit_image):
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, edit_image.size[0] / edit_image.size[1])
        return self.crop_and_resize(edit_image, calculated_height, calculated_width)
    
    def process_image_ids(self, image_latents, scale=10):
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    def process(self, pipe: Flux2ImagePipeline, edit_image, edit_image_auto_resize):
        
        if edit_image is None:
            print(f"[warning] Flux2Unit_EditImageEmbedder.process: edit_image is None")
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_image, Image.Image):
            edit_image = [edit_image]
        resized_edit_image, edit_latents, edit_ref_ranges = [], [], []
        token_start = 0
        for ref_idx, image in enumerate(edit_image):
            # Preprocess
            if edit_image_auto_resize is None or edit_image_auto_resize:
                image = self.edit_image_auto_resize(image)
            resized_edit_image.append(image)
            # Encode
            image = pipe.preprocess_image(image)
            latents = pipe.vae.encode(image)
            edit_latents.append(latents)
            _, _, latent_h, latent_w = latents.shape
            token_count = latent_h * latent_w
            edit_ref_ranges.append({
                "ref_index": ref_idx,
                "start": token_start,
                "end": token_start + token_count,
                "token_count": token_count,
                "grid_height": latent_h,
                "grid_width": latent_w,
                "resized_width": resized_edit_image[-1].width,
                "resized_height": resized_edit_image[-1].height,
                "image_id_t": 10 + 10 * ref_idx,
            })
            token_start += token_count
        edit_image_ids = self.process_image_ids(edit_latents).to(pipe.device)
        edit_latents = torch.concat([rearrange(latents, "B C H W -> B (H W) C") for latents in edit_latents], dim=1)
        return {"edit_latents": edit_latents, "edit_image_ids": edit_image_ids, "edit_ref_ranges": edit_ref_ranges}


class Flux2Unit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("image_ids",),
        )

    def prepare_latent_ids(self, height, width):
        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(1, -1, -1)

        return latent_ids

    def process(self, pipe: Flux2ImagePipeline, height, width):
        image_ids = self.prepare_latent_ids(height // 16, width // 16).to(pipe.device)
        return {"image_ids": image_ids}


def model_fn_flux2(
    dit: Flux2DiT,
    latents=None,
    timestep=None,
    embedded_guidance=None,
    prompt_embeds=None,
    text_ids=None,
    image_ids=None,
    edit_latents=None,
    edit_image_ids=None,
    edit_image_scale=1.0,
    attention_probe=None,
    attention_probe_metadata=None,
    progress_id=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    image_seq_len = latents.shape[1]
    if edit_latents is not None:
        image_seq_len = latents.shape[1]
        latents = torch.concat([latents, edit_latents], dim=1)
        image_ids = torch.concat([image_ids, edit_image_ids], dim=1)

    joint_attention_kwargs = None
    if edit_latents is not None and edit_image_scale != 1.0:
        text_len = prompt_embeds.shape[1]
        target_len = image_seq_len
        edit_len = edit_latents.shape[1]
        total_len = text_len + target_len + edit_len
        attn_mask = torch.zeros(
            1, 1, total_len, total_len,
            device=latents.device, dtype=latents.dtype,
        )
        log_s = torch.log(
            torch.tensor(edit_image_scale, device=latents.device, dtype=latents.dtype).clamp(min=1e-6)
        )
        attn_mask[..., text_len:text_len + target_len, text_len + target_len:] = log_s
        joint_attention_kwargs = {"attention_mask": attn_mask}

    if attention_probe is not None:
        joint_attention_kwargs = dict(joint_attention_kwargs or {})
        attention_probe_context = dict(attention_probe_metadata or {})
        attention_probe_context.update({
            "step": int(progress_id) if progress_id is not None else None,
            "text_len": int(prompt_embeds.shape[1]),
            "target_len": int(image_seq_len),
            "image_len": int(latents.shape[1]),
            "total_len": int(prompt_embeds.shape[1] + latents.shape[1]),
        })
        joint_attention_kwargs["attention_probe"] = attention_probe
        joint_attention_kwargs["attention_probe_context"] = attention_probe_context

    drop_block_indices = kwargs.pop("drop_block_indices", None)
    embedded_guidance = torch.tensor([embedded_guidance], device=latents.device)
    model_output = dit(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=embedded_guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_ids,
        joint_attention_kwargs=joint_attention_kwargs,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        drop_block_indices=drop_block_indices,
    )
    model_output = model_output[:, :image_seq_len]
    return model_output
