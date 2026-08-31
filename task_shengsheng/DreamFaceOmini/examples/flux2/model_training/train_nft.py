"""
train_nft.py — DiffusionNFT RL training for DreamFaceOmini.

Trains on reward-annotated data from collect_rewards.py using the
DiffusionNFT forward-process RL algorithm.

Usage:
    accelerate launch --num_processes 7 train_nft.py \
        --reward_jsonl /path/to/rewards_with_advantage.jsonl \
        --lora_checkpoint /path/to/current_lora.safetensors \
        --output_path /path/to/nft_output \
        --nft_beta 0.5 --kl_beta 0.01 \
        --num_epochs 3 --learning_rate 5e-5
"""

import torch, os, json, argparse, copy, random
import accelerate
from PIL import Image
from tqdm import tqdm
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig
from diffsynth.diffusion import *
from diffsynth.diffusion.loss import FlowMatchNFTLoss


class NFTDataset(torch.utils.data.Dataset):
    """Load reward-annotated JSONL: each record has generated image, edit_image, advantage."""

    def __init__(self, jsonl_path: str, max_pixels: int = 2073600):
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    if "advantage" in r and os.path.exists(r.get("generated_image", "")):
                        self.records.append(r)
        print(f"[NFTDataset] loaded {len(self.records)} records with advantage")
        self.max_pixels = max_pixels
        self.load_from_cache = False

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        gen_img = Image.open(r["generated_image"]).convert("RGB")

        w, h = gen_img.size
        if w * h > self.max_pixels:
            scale = (self.max_pixels / (w * h)) ** 0.5
            w, h = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
            gen_img = gen_img.resize((w, h), Image.LANCZOS)

        edit_paths = r.get("edit_image", [])
        if isinstance(edit_paths, str):
            edit_paths = [edit_paths]
        edit_images = []
        for p in edit_paths:
            if os.path.exists(p):
                edit_images.append(Image.open(p).convert("RGB"))

        return {
            "image": gen_img,
            "prompt": r.get("prompt", ""),
            "edit_image": edit_images,
            "advantage": r["advantage"],
        }


class Flux2NFTTrainingModule(DiffusionTrainingModule):
    """DiffusionNFT training module with dual LoRA adapters."""

    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32,
        lora_checkpoint=None,
        use_gradient_checkpointing=True,
        extra_inputs=None,
        device="cpu",
        nft_beta=0.5,
        kl_beta=0.01,
        adv_clip_max=5.0,
        ema_decay=0.999,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, device=device)
        tokenizer_config = self.parse_path_or_model_id(
            tokenizer_path,
            default_value=ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"),
        )
        self.pipe = Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=model_configs, tokenizer_config=tokenizer_config,
        )
        self.pipe = self.split_pipeline_units("sft", self.pipe, trainable_models, lora_base_model)

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            task="sft",
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.extra_inputs = extra_inputs.split(",") if extra_inputs else []
        self.nft_beta = nft_beta
        self.kl_beta = kl_beta
        self.adv_clip_max = adv_clip_max
        self.ema_decay = ema_decay

        self._old_state = self._snapshot_lora_params()
        self._ref_state = self._snapshot_lora_params(zero_lora=True)

    def _snapshot_lora_params(self, zero_lora=False):
        """Snapshot LoRA parameters for old/ref model."""
        state = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                if zero_lora:
                    state[name] = torch.zeros_like(param.data)
                else:
                    state[name] = param.data.clone()
        return state

    def _apply_lora_state(self, state_dict):
        """Temporarily swap LoRA params, run model, swap back."""
        originals = {}
        for name, param in self.named_parameters():
            if name in state_dict:
                originals[name] = param.data.clone()
                param.data.copy_(state_dict[name])
        return originals

    def _restore_lora_state(self, originals):
        for name, param in self.named_parameters():
            if name in originals:
                param.data.copy_(originals[name])

    def ema_update_old(self):
        """EMA update: old_params = decay * old_params + (1-decay) * current_params."""
        for name, param in self.named_parameters():
            if name in self._old_state and param.requires_grad:
                self._old_state[name] = (
                    self.ema_decay * self._old_state[name] +
                    (1 - self.ema_decay) * param.data.clone()
                )

    def _model_fn_with_state(self, state_dict, inputs, timestep):
        """Run model_fn with a different set of LoRA params (no grad)."""
        originals = self._apply_lora_state(state_dict)
        try:
            models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
            with torch.no_grad():
                output = self.pipe.model_fn(**models, **inputs, timestep=timestep)
        finally:
            self._restore_lora_state(originals)
        return output

    def get_pipeline_inputs(self, data):
        prompt = data["prompt"]
        if random.random() < 0.1:
            prompt = ""
        inputs_posi = {"prompt": prompt}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            "embedded_guidance": 1.0,
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        advantage = data.get("advantage", 0.0)

        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)

        inputs_shared, inputs_posi, inputs_nega = inputs
        merged = {**inputs_shared, **inputs_posi}

        def model_fn_old(**kwargs):
            ts = kwargs.pop("timestep")
            return self._model_fn_with_state(self._old_state, kwargs, ts)

        def model_fn_ref(**kwargs):
            ts = kwargs.pop("timestep")
            return self._model_fn_with_state(self._ref_state, kwargs, ts)

        result = FlowMatchNFTLoss(
            self.pipe,
            model_fn_old=model_fn_old,
            model_fn_ref=model_fn_ref,
            advantage=advantage,
            nft_beta=self.nft_beta,
            kl_beta=self.kl_beta,
            adv_clip_max=self.adv_clip_max,
            **merged,
        )

        if isinstance(result, dict):
            self._last_metrics = {k: v for k, v in result.items() if k != "loss"}
            return result["loss"]
        self._last_metrics = {}
        return result


def launch_nft_training(accelerator, dataset, model, model_logger, args):
    """NFT training loop: standard flow matching training + EMA update on old adapter."""
    optimizer = torch.optim.AdamW(
        model.trainable_modules(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=args.num_workers,
    )
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    log_interval = 10
    for epoch_id in range(args.num_epochs):
        for data in tqdm(dataloader, desc=f"NFT epoch {epoch_id}"):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model(data)
                accelerator.backward(loss)
                if args.max_grad_norm > 0 and accelerator.sync_gradients:
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()

                unwrapped = accelerator.unwrap_model(model)
                unwrapped.ema_update_old()

                model_logger.on_step_end(accelerator, model, args.save_steps, loss=loss)
                if accelerator.is_main_process and model_logger.num_steps % log_interval == 0:
                    metrics = getattr(unwrapped, "_last_metrics", {})
                    if metrics:
                        parts = [f"step={model_logger.num_steps}", f"epoch={epoch_id}"]
                        for k, v in metrics.items():
                            val = v.item() if isinstance(v, torch.Tensor) else v
                            parts.append(f"{k}={val:.4f}")
                        tqdm.write(" | ".join(parts))

        if args.save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, args.save_steps)


def main():
    parser = argparse.ArgumentParser(description="DiffusionNFT RL training for DreamFaceOmini")
    parser = add_general_config(parser)
    parser.add_argument("--reward_jsonl", required=True, help="Path to rewards_with_advantage.jsonl")
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--nft_beta", type=float, default=0.5,
                        help="Interpolation coefficient for positive/negative velocity mixing")
    parser.add_argument("--kl_beta", type=float, default=0.01,
                        help="KL regularization weight to base model")
    parser.add_argument("--adv_clip_max", type=float, default=5.0,
                        help="Advantage clipping range")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay for old adapter update")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)],
    )

    dataset = NFTDataset(args.reward_jsonl, max_pixels=args.max_pixels)

    model = Flux2NFTTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        extra_inputs=args.extra_inputs,
        device="cpu",
        nft_beta=args.nft_beta,
        kl_beta=args.kl_beta,
        adv_clip_max=args.adv_clip_max,
        ema_decay=args.ema_decay,
    )

    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )

    launch_nft_training(accelerator, dataset, model, model_logger, args)


if __name__ == "__main__":
    main()
