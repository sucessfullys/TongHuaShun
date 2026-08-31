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

from train_flow_matching import build_lr_scheduler
from train_flow_matching_lora import (
    FLUX2_KLEIN_LORA_TARGETS,
    lora_state_dict,
)
from train_self_flow import (
    Flux2SelfFlowTrainingModule,
    SelfFlowProjectionHead,
    build_dataset,
    build_pipeline_and_teacher,
    collate_single_sample,
    initialize_deepspeed_gradient_checkpointing,
    layer_index_from_ratio,
    state_dict_subset,
    update_ema_teacher,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Flux2SelfFlowLoRATrainingModule(Flux2SelfFlowTrainingModule):
    """Self-Flow training where only DiT LoRA adapters and projector are trainable."""

    def __init__(
        self,
        pipe,
        teacher_dit,
        gamma=0.8,
        ema_decay=0.9999,
        mask_ratio=0.25,
        student_layer_ratio=0.3,
        teacher_layer_ratio=0.7,
        lora_rank=32,
        lora_alpha=None,
        lora_target_modules=FLUX2_KLEIN_LORA_TARGETS,
        use_gradient_checkpointing=True,
        spatial_norm=False,  # 借鉴 iREPA：是否对 teacher 隐藏状态做空间 z-score 归一化
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
        teacher_dit = inject_adapter_in_model(lora_config, teacher_dit)
        teacher_dit.load_state_dict(pipe.dit.state_dict(), strict=True)

        for name, parameter in pipe.dit.named_parameters():
            parameter.requires_grad = "lora_" in name
            if parameter.requires_grad:
                parameter.data = parameter.data.to(pipe.torch_dtype)
        teacher_dit.eval().requires_grad_(False)

        self.pipe = pipe
        self.teacher_dit = teacher_dit
        self.projector = SelfFlowProjectionHead(self.pipe.dit.inner_dim).to(
            dtype=self.pipe.torch_dtype,
            device=self.pipe.device,
        )
        self.gamma = gamma
        self.ema_decay = ema_decay
        self.mask_ratio = mask_ratio
        self.student_layer = layer_index_from_ratio(
            self.pipe.dit,
            student_layer_ratio,
        )
        self.teacher_layer = layer_index_from_ratio(
            self.teacher_dit,
            teacher_layer_ratio,
        )
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.spatial_norm = spatial_norm  # 借鉴 iREPA：对 teacher 特征做空间归一化


def load_config_defaults(parser):
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with open(preliminary.config, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        parser.set_defaults(**config)
    return parser.parse_args()


def build_parser():
    parser = argparse.ArgumentParser(
        description="LoRA Self-Flow training for FLUX.2 Klein."
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
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--student_layer_ratio", type=float, default=0.3)
    parser.add_argument("--teacher_layer_ratio", type=float, default=0.7)
    parser.add_argument(
        "--spatial_norm", action="store_true",
        help="借鉴 iREPA：对 teacher 隐藏状态做空间 z-score 归一化"
    )
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--save_ema_teacher", action="store_true")
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


def prefixed_lora_state_dict(state_dict, prefix):
    subset = {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    return lora_state_dict(subset)


def save_training_checkpoint(
    accelerator,
    model,
    output_dir,
    global_step,
    save_ema_teacher,
):
    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(str(checkpoint_dir))
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        save_file(
            prefixed_lora_state_dict(state_dict, "pipe.dit."),
            str(checkpoint_dir / "lora.safetensors"),
        )
        projector = state_dict_subset(state_dict, "projector.")
        save_file(projector, str(checkpoint_dir / "self_flow_projector.safetensors"))
        if save_ema_teacher:
            save_file(
                prefixed_lora_state_dict(state_dict, "teacher_dit."),
                str(checkpoint_dir / "ema_teacher_lora.safetensors"),
            )
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

    pipe, teacher_dit = build_pipeline_and_teacher(
        args.base_model,
        accelerator.device,
    )
    model = Flux2SelfFlowLoRATrainingModule(
        pipe=pipe,
        teacher_dit=teacher_dit,
        gamma=args.gamma,
        ema_decay=args.ema_decay,
        mask_ratio=args.mask_ratio,
        student_layer_ratio=args.student_layer_ratio,
        teacher_layer_ratio=args.teacher_layer_ratio,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        spatial_norm=args.spatial_norm,  # 借鉴 iREPA：空间归一化开关
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
            "LoRA Self-Flow logical layers: "
            f"student={unwrapped.student_layer}, teacher={unwrapped.teacher_layer}; "
            f"gamma={args.gamma}, mask_ratio={args.mask_ratio}, "
            f"ema={args.ema_decay}, rank={args.lora_rank}, "
            f"spatial_norm={args.spatial_norm}, "
            f"trainable_params={trainable}"
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
                    update_ema_teacher(
                        unwrapped.pipe.dit,
                        unwrapped.teacher_dit,
                        args.ema_decay,
                    )
                    global_step += 1
                    if (
                        accelerator.is_main_process
                        and global_step % args.log_every == 0
                    ):
                        print(
                            f"step={global_step} "
                            f"loss={losses['loss'].item():.6f} "
                            f"loss_gen={losses['loss_gen'].item():.6f} "
                            f"loss_rep={losses['loss_rep'].item():.6f} "
                            f"lr={lr_scheduler.get_last_lr()[0]:.3e} "
                            f"mask={losses['mask_ratio'].item():.4f} "
                            f"t={losses['t_mean'].item():.1f} "
                            f"s={losses['s_mean'].item():.1f}"
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
                            args.save_ema_teacher,
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
            args.save_ema_teacher,
        )
    accelerator.end_training()


if __name__ == "__main__":
    main()
