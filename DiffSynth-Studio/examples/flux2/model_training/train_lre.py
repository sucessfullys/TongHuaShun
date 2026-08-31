import argparse
import os

import accelerate
import torch
from PIL import Image

from diffsynth.core import UnifiedDataset
from diffsynth.diffusion import *
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def FlowMatchLRESFTLoss(pipe, lre_strength=0.8, **inputs):
    if "lora" in inputs:
        pipe.clear_lora(verbose=0)
        pipe.load_lora(pipe.dit, state_dict=inputs["lora"], hotload=True, verbose=0)

    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))
    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)

    input_latents = inputs["input_latents"]
    noise = torch.randn_like(input_latents) * inputs.get("noise_scale", 1.0)
    lr_latents = inputs.get("lre_lr_latents")
    if lr_latents is not None:
        if lr_latents.shape != noise.shape:
            raise ValueError(f"lre_lr_latents shape {lr_latents.shape} != noise shape {noise.shape}")
        noise = (1.0 - lre_strength) * lr_latents + lre_strength * noise

    inputs["latents"] = pipe.scheduler.add_noise(input_latents, noise, timestep)
    training_target = pipe.scheduler.training_target(input_latents, noise, timestep)

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


class Flux2ImageLRETrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None,
        model_id_with_origin_paths=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None,
        lora_target_modules="",
        lora_rank=32,
        lora_checkpoint=None,
        preset_lora_path=None,
        preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        template_model_id_or_path=None,
        resume_from_checkpoint=None,
        remove_prefix_in_ckpt=None,
        enable_lora_hot_loading=False,
        device="cpu",
        task="sft",
        lre_strength=0.8,
    ):
        super().__init__()
        model_configs = self.parse_model_configs(
            model_paths, model_id_with_origin_paths,
            fp8_models=fp8_models, offload_models=offload_models, device=device,
        )
        tokenizer_config = self.parse_path_or_model_id(
            tokenizer_path,
            default_value=ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"),
        )
        self.pipe = Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16, device=device,
            model_configs=model_configs, tokenizer_config=tokenizer_config,
        )
        self.pipe = self.load_training_template_model(
            self.pipe, template_model_id_or_path,
            use_gradient_checkpointing, use_gradient_checkpointing_offload,
        )
        self.pipe = self.split_pipeline_units(
            task, self.pipe, trainable_models, lora_base_model,
            remove_unnecessary_params=True,
        )
        self.resume_from_checkpoint(resume_from_checkpoint, remove_prefix_in_ckpt)
        if enable_lora_hot_loading:
            self.pipe.dit = self.pipe.enable_lora_hot_loading(self.pipe.dit)

        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.task = task
        self.lre_strength = float(lre_strength)
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchLRESFTLoss(
                pipe, lre_strength=self.lre_strength, **inputs_shared, **inputs_posi
            ),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchLRESFTLoss(
                pipe, lre_strength=self.lre_strength, **inputs_shared, **inputs_posi
            ),
        }

    @staticmethod
    def load_lre_image(image, target_size):
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        else:
            image = Image.open(image).convert("RGB")
        if image.size != target_size:
            image = image.resize(target_size, Image.BICUBIC)
        return image

    @torch.no_grad()
    def encode_lre_lr_latents(self, template_inputs, target_size):
        if template_inputs is None or "image" not in template_inputs:
            return None
        image = self.load_lre_image(template_inputs["image"], target_size)
        pipe = self.pipe
        pipe.load_models_to_device(["vae"])
        lr_tensor = pipe.preprocess_image(image)
        lr_latents = pipe.vae.encode(lr_tensor)
        return lr_latents.reshape(1, 128, -1).permute(0, 2, 1)

    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            "input_image": data["image"],
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            "embedded_guidance": 1.0,
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        if "template_inputs" in inputs_shared:
            inputs_shared["lre_lr_latents"] = self.encode_lre_lr_latents(
                inputs_shared["template_inputs"],
                target_size=data["image"].size,
            )
        return inputs_shared, inputs_posi, inputs_nega

    def forward(self, data, inputs=None):
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        return self.task_to_loss[self.task](self.pipe, *inputs)


def parser():
    p = argparse.ArgumentParser(description="FLUX.2 Template LRE training.")
    p = add_general_config(p)
    p = add_image_size_config(p)
    p.add_argument("--tokenizer_path", type=str, default=None)
    p.add_argument("--initialize_model_on_cpu", default=False, action="store_true")
    p.add_argument("--lre_strength", type=float, default=0.8)
    return p


if __name__ == "__main__":
    p = parser()
    args = p.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        ),
    )
    model = Flux2ImageLRETrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        template_model_id_or_path=args.template_model_id_or_path,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_lora_hot_loading=args.enable_lora_hot_loading,
        task=args.task,
        lre_strength=args.lre_strength,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_total_limit=args.save_total_limit,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
