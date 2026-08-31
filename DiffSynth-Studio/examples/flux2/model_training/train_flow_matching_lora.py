import argparse
import json
import os
from pathlib import Path

import accelerate
import torch
import yaml
from peft import LoraConfig, inject_adapter_in_model
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from train_flow_matching import (
    Flux2FlowMatchingTrainingModule,
    build_lr_scheduler,
    build_pipeline,
)
from train_self_flow import (
    build_dataset,
    collate_single_sample,
    initialize_deepspeed_gradient_checkpointing,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


FLUX2_KLEIN_LORA_TARGETS = (
    "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,"
    "linear_in,linear_out,to_qkv_mlp_proj,"
    "single_transformer_blocks.0.attn.to_out,"
    "single_transformer_blocks.1.attn.to_out,"
    "single_transformer_blocks.2.attn.to_out,"
    "single_transformer_blocks.3.attn.to_out,"
    "single_transformer_blocks.4.attn.to_out,"
    "single_transformer_blocks.5.attn.to_out,"
    "single_transformer_blocks.6.attn.to_out,"
    "single_transformer_blocks.7.attn.to_out,"
    "single_transformer_blocks.8.attn.to_out,"
    "single_transformer_blocks.9.attn.to_out,"
    "single_transformer_blocks.10.attn.to_out,"
    "single_transformer_blocks.11.attn.to_out,"
    "single_transformer_blocks.12.attn.to_out,"
    "single_transformer_blocks.13.attn.to_out,"
    "single_transformer_blocks.14.attn.to_out,"
    "single_transformer_blocks.15.attn.to_out,"
    "single_transformer_blocks.16.attn.to_out,"
    "single_transformer_blocks.17.attn.to_out,"
    "single_transformer_blocks.18.attn.to_out,"
    "single_transformer_blocks.19.attn.to_out"
)


class Flux2LoRATrainingModule(Flux2FlowMatchingTrainingModule):
    def __init__(
        self,
        pipe,
        lora_rank=32,
        lora_alpha=None,
        lora_target_modules=FLUX2_KLEIN_LORA_TARGETS,
        use_gradient_checkpointing=True,
    ):
        torch.nn.Module.__init__(self)
        pipe.freeze_except([])
        targets = [item for item in lora_target_modules.split(",") if item]
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=targets,
        )
        pipe.dit = inject_adapter_in_model(lora_config, pipe.dit)
        for name, parameter in pipe.dit.named_parameters():
            parameter.requires_grad = "lora_" in name
            if parameter.requires_grad:
                parameter.data = parameter.data.to(pipe.torch_dtype)
        self.pipe = pipe
        self.use_gradient_checkpointing = use_gradient_checkpointing


def load_config_defaults(parser):
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with open(preliminary.config, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        parser.set_defaults(**config)
    return parser.parse_args()


def build_parser():
    parser = argparse.ArgumentParser(
        description="FLUX.2 Klein LoRA training with metadata/tar datasets."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--base_model", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument(
        "--dataset_type",
        choices=["dummy", "metadata", "metadata_tar"],
        default="dummy",
    )
    parser.add_argument("--metadata_path", type=str, default=None)
    parser.add_argument("--image_root", type=str, default="")
    parser.add_argument("--image_column", type=str, default="image")
    parser.add_argument("--caption_column", type=str, default="prompt")
    parser.add_argument("--tar_column", type=str, default="tar_file")
    parser.add_argument("--tar_cache_size", type=int, default=8)
    parser.add_argument("--dummy_length", type=int, default=8)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=1024 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler",
        choices=["constant", "warmup_cosine"],
        default="constant",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=7813)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default=FLUX2_KLEIN_LORA_TARGETS,
    )
    return parser


def lora_state_dict(state_dict):
    result = {}
    for key, value in state_dict.items():
        if "lora_A" not in key and "lora_B" not in key:
            continue
        if key.startswith("pipe.dit."):
            key = key[len("pipe.dit."):]
        result[key] = value.detach().cpu().contiguous()
    return result


def save_training_checkpoint(accelerator, model, output_dir, global_step):
    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(str(checkpoint_dir))
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_file(lora_state_dict(state_dict), str(checkpoint_dir / "lora.safetensors"))
        with (checkpoint_dir / "trainer_state.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump({"global_step": global_step}, file)
    accelerator.wait_for_everyone()


def main():
    args = load_config_defaults(build_parser())
    if not args.base_model or not args.output_dir:
        raise ValueError("base_model and output_dir must be set by config or CLI.")
    if args.train_batch_size != 1:
        raise ValueError("Set train_batch_size=1 and scale with gradient accumulation.")

    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(
        args.gradient_accumulation_steps
    )
    accelerator_kwargs = {
        "step_scheduler_with_optimizer": False,
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    }
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() != "true":
        accelerator_kwargs.update(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    accelerate.utils.set_seed(args.seed, device_specific=True)

    model = Flux2LoRATrainingModule(
        pipe=build_pipeline(args.base_model, accelerator.device),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        scheduler_type=args.lr_scheduler,
        max_steps=args.max_steps,
        warmup_steps=args.lr_warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )
    dataset = build_dataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_single_sample,
    )
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        dataloader,
        lr_scheduler,
    )
    if accelerator.gradient_accumulation_steps != args.gradient_accumulation_steps:
        raise RuntimeError(
            "Gradient accumulation mismatch after distributed initialization: "
            f"requested={args.gradient_accumulation_steps}, "
            f"actual={accelerator.gradient_accumulation_steps}."
        )
    initialize_deepspeed_gradient_checkpointing(accelerator)

    global_step = 0
    if args.resume_from_checkpoint:
        accelerator.load_state(args.resume_from_checkpoint)
        state_path = Path(args.resume_from_checkpoint) / "trainer_state.json"
        if state_path.exists():
            with state_path.open("r", encoding="utf-8") as file:
                global_step = int(json.load(file)["global_step"])

    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        trainable = sum(
            parameter.numel()
            for parameter in unwrapped.parameters()
            if parameter.requires_grad
        )
        effective_batch_size = (
            accelerator.num_processes
            * args.train_batch_size
            * accelerator.gradient_accumulation_steps
        )
        print(
            f"Objective: standard FLUX.2 flow matching LoRA. "
            f"rank={args.lora_rank}, trainable_params={trainable}"
        )
        print(
            "Distributed training: "
            f"processes={accelerator.num_processes}, "
            f"micro_batch={args.train_batch_size}, "
            f"gradient_accumulation={accelerator.gradient_accumulation_steps}, "
            f"effective_batch={effective_batch_size}"
        )

    model.train()
    while global_step < args.max_steps:
        for sample in dataloader:
            with accelerator.accumulate(model):
                losses = model(sample)
                accelerator.backward(losses["loss"])
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unwrapped.trainable_parameters(),
                        args.max_grad_norm,
                    )
                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    if (
                        accelerator.is_main_process
                        and global_step % args.log_every == 0
                    ):
                        print(
                            f"step={global_step} "
                            f"loss={losses['loss'].item():.6f} "
                            f"loss_gen={losses['loss'].item():.6f} "
                            f"lr={lr_scheduler.get_last_lr()[0]:.3e}"
                        )
                    if (
                        args.checkpointing_steps > 0
                        and global_step % args.checkpointing_steps == 0
                    ):
                        save_training_checkpoint(
                            accelerator,
                            model,
                            args.output_dir,
                            global_step,
                        )
            if global_step >= args.max_steps:
                break

    if (
        args.checkpointing_steps <= 0
        or global_step % args.checkpointing_steps != 0
    ):
        save_training_checkpoint(
            accelerator,
            model,
            args.output_dir,
            global_step,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
