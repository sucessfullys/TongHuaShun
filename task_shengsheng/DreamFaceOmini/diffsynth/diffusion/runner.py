import os, torch
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def _linear_decay(initial_value, final_value, total_steps, current_step):
    """Linearly interpolate from initial_value to final_value over total_steps."""
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    return initial_value + (final_value - initial_value) * current_step / total_steps


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
    
    max_grad_norm = getattr(args, "max_grad_norm", 0.0) if args is not None else 0.0
    initial_grad_norm_ratio = getattr(args, "initial_grad_norm_ratio", 5.0) if args is not None else 5.0
    grad_clip_warmup_steps = getattr(args, "grad_clip_warmup_steps", 1000) if args is not None else 1000
    abnormal_grad_ratio = getattr(args, "abnormal_grad_ratio", 5.0) if args is not None else 5.0
    use_grad_clip = max_grad_norm > 0

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    trainable_params = [p for p in model.parameters() if p.requires_grad] if use_grad_clip else []
    
    log_interval = 10
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)

                if use_grad_clip and accelerator.sync_gradients:
                    step = model_logger.num_steps
                    ceiling = _linear_decay(
                        max_grad_norm * initial_grad_norm_ratio,
                        max_grad_norm,
                        grad_clip_warmup_steps,
                        step,
                    )
                    grads = [p.grad for p in trainable_params if p.grad is not None]
                    if grads:
                        total_norm = torch.norm(torch.stack([g.detach().norm(2) for g in grads]), 2).item()
                        if step > grad_clip_warmup_steps and total_norm / ceiling > abnormal_grad_ratio:
                            ceiling = ceiling / min(total_norm / ceiling, 10.0)
                    accelerator.clip_grad_norm_(trainable_params, ceiling)

                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                scheduler.step()

                if accelerator.is_main_process and model_logger.num_steps % log_interval == 0:
                    unwrapped = accelerator.unwrap_model(model)
                    metrics = getattr(unwrapped, "_last_metrics", {})
                    if metrics:
                        parts = [f"step={model_logger.num_steps}", f"epoch={epoch_id}"]
                        for k, v in metrics.items():
                            val = v.item() if isinstance(v, torch.Tensor) else v
                            parts.append(f"{k}={val:.4f}")
                        tqdm.write(" | ".join(parts))

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
