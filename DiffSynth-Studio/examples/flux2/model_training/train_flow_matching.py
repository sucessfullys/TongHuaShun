import argparse
import json
import os
from pathlib import Path

import accelerate
import torch
import yaml
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from train_self_flow import (
    build_dataset,
    build_lr_scheduler,
    collate_single_sample,
    initialize_deepspeed_gradient_checkpointing,
)

from diffsynth.core import ModelConfig
from diffsynth.diffusion.loss import FlowMatchSFTLoss
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Flux2FlowMatchingTrainingModule(torch.nn.Module):
    """Full-parameter FLUX.2 DiT training with the standard flow-matching loss."""

    def __init__(self, pipe, use_gradient_checkpointing=True):
        super().__init__()
        self.pipe = pipe
        self.pipe.freeze_except(["dit"])
        self.use_gradient_checkpointing = use_gradient_checkpointing

    def train(self, mode=True):
        super().train(mode)
        self.pipe.eval()
        self.pipe.dit.train(mode)
        return self

    def trainable_parameters(self):
        return (
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _pipeline_inputs(self, sample):
        inputs_shared = {
            "input_image": sample["image"],
            "height": sample["image"].size[1],
            "width": sample["image"].size[0],
            "embedded_guidance": 1.0,
            "cfg_scale": 1.0,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": False,
        }
        inputs_posi = {"prompt": sample["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit,
                self.pipe,
                inputs_shared,
                inputs_posi,
                inputs_nega,
            )
        return {**inputs_shared, **inputs_posi}

    def forward(self, sample):
        inputs = self._pipeline_inputs(sample)
        loss = FlowMatchSFTLoss(self.pipe, **inputs)
        return {"loss": loss}


def load_config_defaults(parser):
    preliminary, _ = parser.parse_known_args()
    if preliminary.config:
        with open(preliminary.config, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        parser.set_defaults(**config)
    return parser.parse_args()


def build_parser():
    parser = argparse.ArgumentParser(
        description="Full-parameter standard flow-matching training for FLUX.2 Klein."
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
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--lr_scheduler",
        choices=["constant", "warmup_cosine"],
        default="constant",
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--use_gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument("--log_every", type=int, default=1)
    return parser


def build_pipeline(base_model, device):
    base = Path(base_model)
    text_encoder_files = sorted(
        str(path) for path in (base / "text_encoder").glob("*.safetensors")
    )
    if not text_encoder_files:
        raise FileNotFoundError(
            f"No text encoder weights found under {base / 'text_encoder'}"
        )
    transformer_files = sorted(
        str(path)
        for path in (base / "transformer").glob("*.safetensors")
        if not path.name.endswith(".index.json")
    )
    if not transformer_files:
        raise FileNotFoundError(
            f"No transformer weights found under {base / 'transformer'}"
        )
    transformer_path = (
        transformer_files[0]
        if len(transformer_files) == 1
        else transformer_files
    )
    model_configs = [
        ModelConfig(path=text_encoder_files),
        ModelConfig(path=transformer_path),
        ModelConfig(
            path=str(base / "vae" / "diffusion_pytorch_model.safetensors")
        ),
    ]
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=model_configs,
        tokenizer_config=ModelConfig(path=str(base / "tokenizer")),
    )
    pipe.scheduler.set_timesteps(1000, training=True)
    return pipe


def state_dict_subset(state_dict, prefix):
    return {
        key[len(prefix):]: value.detach().cpu().contiguous()
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def save_training_checkpoint(accelerator, model, output_dir, global_step):
    checkpoint_dir = Path(output_dir) / f"checkpoint-{global_step}"
    accelerator.save_state(str(checkpoint_dir))
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        student = state_dict_subset(state_dict, "pipe.dit.")
        save_file(student, str(checkpoint_dir / "student.safetensors"))
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
        raise ValueError(
            "Set train_batch_size=1 and scale with gradient accumulation."
        )

    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(
        args.gradient_accumulation_steps
    )
    accelerator_kwargs = {
        "step_scheduler_with_optimizer": False,
        "kwargs_handlers": [
            accelerate.DistributedDataParallelKwargs(
                find_unused_parameters=False
            )
        ],
    }
    if os.environ.get("ACCELERATE_USE_DEEPSPEED", "false").lower() != "true":
        accelerator_kwargs.update(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
        )
    accelerator = accelerate.Accelerator(**accelerator_kwargs)
    accelerate.utils.set_seed(args.seed, device_specific=True)

    model = Flux2FlowMatchingTrainingModule(
        pipe=build_pipeline(args.base_model, accelerator.device),
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
        effective_batch_size = (
            accelerator.num_processes
            * args.train_batch_size
            * accelerator.gradient_accumulation_steps
        )
        print("Objective: standard FLUX.2 flow matching (no Self-Flow).")
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
