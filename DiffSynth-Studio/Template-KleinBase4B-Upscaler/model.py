from typing import Any, Dict, List, Optional, Tuple, Union
import torch, math
import torch.nn as nn
from einops import rearrange
from diffsynth.core.attention import attention_forward
from diffsynth.core.gradient import gradient_checkpoint_forward
from diffsynth.models.flux2_dit import apply_rotary_emb, Flux2PosEmbed
from diffsynth.models.general_modules import get_timestep_embedding


class AdaLayerNormContinuous(nn.Module):
    def __init__(self, dim_in, dim_out, eps=1e-6):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out * 2, bias=False)
        self.norm = nn.LayerNorm(dim_in, eps=eps, elementwise_affine=False, bias=False)

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(torch.nn.functional.silu(conditioning_embedding)).chunk(2, dim=1)
        x = self.norm(x) * (1 + scale) + shift
        return x


class Flux2FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_in = nn.Linear(dim, dim*3*2, bias=False)
        self.linear_out = nn.Linear(dim*3, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.linear_in(x).chunk(2, dim=-1)
        x = torch.nn.functional.silu(x1) * x2
        x = self.linear_out(x)
        return x


class Flux2TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, eps=1e-6):
        super().__init__()
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_ff = Flux2FeedForward(dim)
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_ff = Flux2FeedForward(dim)

        self.num_heads = num_heads
        self.img_to_qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.img_norm_q = torch.nn.RMSNorm(dim // num_heads, eps=eps)
        self.img_norm_k = torch.nn.RMSNorm(dim // num_heads, eps=eps)
        self.img_to_out = torch.nn.Linear(dim, dim, bias=False)
        self.txt_to_qkv = torch.nn.Linear(dim, 3 * dim, bias=False)
        self.txt_norm_q = torch.nn.RMSNorm(dim // num_heads, eps=eps)
        self.txt_norm_k = torch.nn.RMSNorm(dim // num_heads, eps=eps)
        self.txt_to_out = torch.nn.Linear(dim, dim, bias=False)

    def attention(self, img: torch.Tensor, txt: torch.Tensor, image_rotary_emb: torch.Tensor, **kwargs) -> torch.Tensor:
        img_q, img_k, img_v = self.img_to_qkv(img).chunk(3, dim=-1)
        txt_q, txt_k, txt_v = self.txt_to_qkv(txt).chunk(3, dim=-1)
        img_q, img_k, img_v, txt_q, txt_k, txt_v = tuple(map(lambda x: x.unflatten(-1, (self.num_heads, -1)), (img_q, img_k, img_v, txt_q, txt_k, txt_v)))
        img_q = self.img_norm_q(img_q)
        img_k = self.img_norm_k(img_k)
        txt_q = self.txt_norm_q(txt_q)
        txt_k = self.txt_norm_k(txt_k)

        q = torch.cat([txt_q, img_q], dim=1)
        k = torch.cat([txt_k, img_k], dim=1)
        v = torch.cat([txt_v, img_v], dim=1)
        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        img = attention_forward(q, k, v, q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s (n d)")
        txt, img = img.split_with_sizes([txt.shape[1], img.shape[1] - txt.shape[1]], dim=1)
        txt = self.txt_to_out(txt)
        img = self.img_to_out(img)
        return img, txt, (k, v)

    def forward(self, img, txt, temb_mod_params_img, temb_mod_params_txt, image_rotary_emb):
        (img_shift_msa, img_scale_msa, img_gate_msa), (img_shift_mlp, img_scale_mlp, img_gate_mlp) = temb_mod_params_img
        (txt_shift_msa, txt_scale_msa, txt_gate_msa), (txt_shift_mlp, txt_scale_mlp, txt_gate_mlp) = temb_mod_params_txt

        norm_img = (1 + img_scale_msa) * self.img_norm1(img) + img_shift_msa
        norm_txt = (1 + txt_scale_msa) * self.txt_norm1(txt) + txt_shift_msa
        img_attn_out, txt_attn_out, kv_cache = self.attention(norm_img, norm_txt, image_rotary_emb)

        img = img + img_gate_msa * img_attn_out
        norm_img = self.img_norm2(img) * (1 + img_scale_mlp) + img_shift_mlp
        img = img + img_gate_mlp * self.img_ff(norm_img)

        txt = txt + txt_gate_msa * txt_attn_out
        norm_txt = self.txt_norm2(txt) * (1 + txt_scale_mlp) + txt_shift_mlp
        txt = txt + txt_gate_mlp * self.txt_ff(norm_txt)
        return txt, img, kv_cache


class Flux2SingleTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.dim = dim
        self.num_heads = num_heads
        self.norm_q = torch.nn.RMSNorm(dim // num_heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim // num_heads, eps=eps, elementwise_affine=True)
        self.to_qkv_mlp_proj = torch.nn.Linear(dim, dim * 3 + dim * 3 * 2, bias=False)
        self.to_out = torch.nn.Linear(dim + dim * 3, dim, bias=False)

    def attention(self, x: torch.Tensor, image_rotary_emb: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x = self.to_qkv_mlp_proj(x)
        qkv, mlp_x = torch.split(x, [3 * self.dim, self.dim * 3 * 2], dim=-1)
        q, k, v = tuple(map(lambda x: x.unflatten(-1, (self.num_heads, -1)), qkv.chunk(3, dim=-1)))

        q = self.norm_q(q)
        k = self.norm_k(k)
        q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        x = attention_forward(q, k, v, q_pattern="b s n d", k_pattern="b s n d", v_pattern="b s n d", out_pattern="b s (n d)")

        x1, x2 = mlp_x.chunk(2, dim=-1)
        x = torch.cat([x, torch.nn.functional.silu(x1) * x2], dim=-1)
        x = self.to_out(x)
        return x, (k, v)

    def forward(self, x, temb_mod_params, image_rotary_emb):
        mod_shift, mod_scale, mod_gate = temb_mod_params
        norm_x = (1 + mod_scale) * self.norm(x) + mod_shift
        attn_output, kv_cache = self.attention(x=norm_x, image_rotary_emb=image_rotary_emb,)
        x = x + mod_gate * attn_output
        return x, kv_cache


class Flux2TimestepGuidanceEmbeddings(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.dim_in = dim_in
        self.timestep_embedder = torch.nn.Sequential(nn.Linear(dim_in, dim_out, bias=False), nn.SiLU(), nn.Linear(dim_out, dim_out, bias=False))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timesteps_proj = get_timestep_embedding(timestep, self.dim_in, flip_sin_to_cos=True, downscale_freq_shift=0)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(timestep.dtype))
        return timesteps_emb


class Flux2Modulation(nn.Module):
    def __init__(self, dim: int, mod_param_sets: int = 2, bias: bool = False):
        super().__init__()
        self.mod_param_sets = mod_param_sets
        self.linear = nn.Linear(dim, dim * 3 * self.mod_param_sets, bias=bias)

    def forward(self, temb: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
        mod = torch.nn.functional.silu(temb)
        mod = self.linear(mod)
        mod = mod.unsqueeze(1)
        mod_params = torch.chunk(mod, 3 * self.mod_param_sets, dim=-1)
        return tuple(mod_params[3 * i : 3 * (i + 1)] for i in range(self.mod_param_sets))


class Flux2DiTVariantModel(torch.nn.Module):
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 128,
        out_channels: Optional[int] = None,
        num_layers: int = 5,
        num_single_layers: int = 20,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 7680,
        timestep_guidance_channels: int = 256,
        axes_dims_rope: Tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: int = 2000,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # 1. Sinusoidal positional embedding for RoPE on image and text tokens
        self.pos_embed = Flux2PosEmbed(theta=rope_theta, axes_dim=axes_dims_rope)

        # 2. Combined timestep + guidance embedding
        self.time_guidance_embed = Flux2TimestepGuidanceEmbeddings(
            dim_in=timestep_guidance_channels,
            dim_out=self.inner_dim,
        )

        # 3. Modulation (double stream and single stream blocks share modulation parameters, resp.)
        # Two sets of shift/scale/gate modulation parameters for the double stream attn and FF sub-blocks
        self.double_stream_modulation_img = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        self.double_stream_modulation_txt = Flux2Modulation(self.inner_dim, mod_param_sets=2, bias=False)
        # Only one set of modulation parameters as the attn and FF sub-blocks are run in parallel for single stream
        self.single_stream_modulation = Flux2Modulation(self.inner_dim, mod_param_sets=1, bias=False)

        # 4. Input projections
        self.img_embedder = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.txt_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)

        # 5. Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList([Flux2TransformerBlock(dim=self.inner_dim, num_heads=num_attention_heads) for _ in range(num_layers)])

        # 6. Single Stream Transformer Blocks
        self.single_transformer_blocks = nn.ModuleList([Flux2SingleTransformerBlock(dim=self.inner_dim, num_heads=num_attention_heads) for _ in range(num_single_layers)])

        # 7. Output layers
        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=False)

    def prepare_static_parameters(self, img, txt):
        timestep = torch.zeros((1,), dtype=txt.dtype, device=txt.device)
        img_ids = []
        for latent_id, latent in enumerate(img):
            _, _, height, width = latent.shape
            x_ids = torch.cartesian_prod(torch.tensor([(latent_id + 1) * 10]), torch.arange(height), torch.arange(width), torch.arange(1))
            img_ids.append(x_ids)
        img_ids = torch.cat(img_ids, dim=0).to(txt.device)
        txt_ids = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(txt.shape[1])).to(txt.device)
        return timestep, img_ids, txt_ids
    
    def patchify(self, img):
        img_ = []
        for latent in img:
            latent = rearrange(latent, "B C H W -> B (H W) C")
            img_.append(latent)
        img_ = torch.concat(img_, dim=1)
        return img_
    
    @torch.no_grad()
    def process_inputs(
        self,
        pipe, image, prompt,
        **kwargs
    ):
        images = image
        if not isinstance(images, list):
            images = [images]
        pipe.load_models_to_device(["vae"])
        kv_cache_input_latents = [pipe.vae.encode(pipe.preprocess_image(image)) for image in images]
        prompt_emb_unit = [unit for unit in pipe.units if unit.__class__.__name__ == "Flux2Unit_Qwen3PromptEmbedder"][0]
        kv_cache_prompt_emb = prompt_emb_unit.process(pipe, prompt)["prompt_embeds"]
        pipe.load_models_to_device([])
        return {
            "kv_cache_input_latents": kv_cache_input_latents,
            "kv_cache_prompt_emb": kv_cache_prompt_emb,
        }

    def forward(
        self,
        kv_cache_input_latents,
        kv_cache_prompt_emb,
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        **kwargs,
    ):
        img = kv_cache_input_latents
        txt = kv_cache_prompt_emb
        num_txt_tokens = txt.shape[1]

        # 1. Calculate timestep embedding and modulation parameters
        timestep, img_ids, txt_ids = self.prepare_static_parameters(img, txt)
        img = self.patchify(img)

        temb = self.time_guidance_embed(timestep)
        double_stream_mod_img = self.double_stream_modulation_img(temb)
        double_stream_mod_txt = self.double_stream_modulation_txt(temb)
        single_stream_mod = self.single_stream_modulation(temb)[0]

        # 2. Input projection for image (img) and conditioning text (txt)
        img = self.img_embedder(img)
        txt = self.txt_embedder(txt)

        # 3. Calculate RoPE embeddings from image and text tokens
        image_rotary_emb = self.pos_embed(img_ids)
        text_rotary_emb = self.pos_embed(txt_ids)
        concat_rotary_emb = (
            torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
            torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
        )

        # 4. Double Stream Transformer Blocks
        kv_cache = {}
        for block_id, block in enumerate(self.transformer_blocks):
            txt, img, kv_cache_ = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                img=img,
                txt=txt,
                temb_mod_params_img=double_stream_mod_img,
                temb_mod_params_txt=double_stream_mod_txt,
                image_rotary_emb=concat_rotary_emb,
            )
            kv_cache[f"double_{block_id}"] = kv_cache_
        # Concatenate text and image streams for single-block inference
        img = torch.cat([txt, img], dim=1)

        # 5. Single Stream Transformer Blocks
        for block_id, block in enumerate(self.single_transformer_blocks):
            img, kv_cache_ = gradient_checkpoint_forward(
                block,
                use_gradient_checkpointing=use_gradient_checkpointing,
                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
                x=img,
                temb_mod_params=single_stream_mod,
                image_rotary_emb=concat_rotary_emb,
            )
            kv_cache[f"single_{block_id}"] = kv_cache_
        # # Remove text tokens from concatenated stream
        # img = img[:, num_txt_tokens:, ...]

        # # 6. Output layers
        # img = self.norm_out(img, temb)
        # output = self.proj_out(img)

        return {"kv_cache": kv_cache}


class TrainDataProcessor:
    def __init__(self):
        import os
        from diffsynth.core import UnifiedDataset
        max_pixels = int(os.environ.get("TEMPLATE_MAX_PIXELS", 1024*1024))
        self.image_oparator = UnifiedDataset.default_image_operator(
            base_path="", # If your dataset contains relative paths, please specify the root path here.
            max_pixels=max_pixels,
            height_division_factor=16,
            width_division_factor=16,
        )

    def __call__(self, image, prompt, **kwargs):
        return {
            "image": self.image_oparator(image),
            "prompt": prompt,
        }

TEMPLATE_MODEL = Flux2DiTVariantModel
TEMPLATE_MODEL_PATH = "model.safetensors"
TEMPLATE_DATA_PROCESSOR = TrainDataProcessor
