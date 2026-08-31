# FLUX.2 KleinBase4B Self-Flow 训练权重交接说明

本目录用于交接 FLUX.2 KleinBase4B Self-Flow 训练得到的模型权重，以及对应的训练脚本和验证脚本路径说明。

## 权重目录

当前交接权重：

```text
/mnt/image-edit/datasets/duanyufa/交接/self-flow训练模型权重文件/checkpoint-7813
```

关键文件：

```text
checkpoint-7813/student.safetensors              # Self-Flow student 权重
checkpoint-7813/self_flow_projector.safetensors  # Self-Flow projector 权重
checkpoint-7813/trainer_state.json               # 训练状态
checkpoint-7813/zero_to_fp32.py                  # DeepSpeed ZeRO 权重转换脚本
checkpoint-7813/pytorch_model/                  # ZeRO 分片状态，包含模型和优化器状态
checkpoint-7813/random_states_*.pkl             # 多卡随机状态
checkpoint-7813/scheduler.bin                   # scheduler 状态
```

该 checkpoint 是训练中保存的完整断点目录，既包含模型权重，也包含继续训练需要的优化器、scheduler 和随机状态。

## 运行环境

当前机器入口示例：

```text
root@interactive-vgp5mjjztgvu-6fbcb646f4-9zxvd:/mnt/image-edit/datasets/duanyufa#
```

环境备份目录：

```text
/mnt/image-edit/datasets/duanyufa/conda_env_backup
```

训练脚本默认使用当前 shell 中的 Python/accelerate/deepspeed 环境。因此运行前建议先确认：

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
python -V
which python
python -c "import torch, accelerate, deepspeed; print('env ok')"
```

如果环境中缺少 `deepspeed` 或 `peft`，训练脚本会直接报错提示。

## 工程目录

DiffSynth-Studio 工程目录：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
```

基础模型路径：

```text
/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B
```

## 全参数 Self-Flow 训练脚本

脚本路径：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_self_flow.sh
```

底层训练代码：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/examples/flux2/model_training/train_self_flow.py
```

配置文件：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/configs/train/flux2_klein_base_4b_self_flow.yaml
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/configs/train/accelerate_flux2_klein_base_4b_self_flow_zero3.yaml
```

正式训练命令：

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_self_flow.sh
```

冒烟测试命令：

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_self_flow.sh smoke
```

默认输出目录：

```text
/mnt/image-edit/datasets/duanyufa/outputs/flux2_klein_base_4b_self_flow_gamma1
```

如需指定输出目录：

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
OUTPUT_DIR=/mnt/image-edit/datasets/duanyufa/outputs/flux2_klein_base_4b_self_flow_custom \
  bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_self_flow.sh
```

## LoRA Self-Flow 训练脚本

脚本路径：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_self_flow_lora.sh
```

底层训练代码：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/examples/flux2/model_training/train_self_flow_lora.py
```

配置文件：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/configs/train/flux2_klein_base_4b_self_flow_lora.yaml
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/configs/train/accelerate_flux2_klein_base_4b_self_flow_zero3.yaml
```

正式训练命令：

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_self_flow_lora.sh
```

冒烟测试命令：

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_self_flow_lora.sh smoke
```

默认输出目录：

```text
/mnt/image-edit/datasets/duanyufa/outputs/flux2_klein_base_4b_self_flow_lora
```

如需指定输出目录：

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
OUTPUT_DIR=/mnt/image-edit/datasets/duanyufa/outputs/flux2_klein_base_4b_self_flow_lora_custom \
  bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_self_flow_lora.sh
```

## 主要训练配置

全参数版本默认配置来自：

```text
configs/train/flux2_klein_base_4b_self_flow.yaml
```

关键参数：

```text
base_model: /mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B
output_dir: /mnt/image-edit/datasets/duanyufa/outputs/flux2_klein_base_4b_self_flow_gamma1

dataset_type: metadata_tar
metadata_path: /mnt/zixuan_workspace/caption_scripts/vllm_caption_gemma/caption_splits/all_id_1person_caption_end_analysis/sample_250k.jsonl
image_column: image
caption_column: caption
tar_column: tar_file
max_pixels: 1048576

mixed_precision: bf16
train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-5
lr_scheduler: constant
max_steps: 7813

mask_ratio: 0.25
gamma: 1.0
ema_decay: 0.9999
student_layer_ratio: 0.3
teacher_layer_ratio: 0.7
checkpointing_steps: 500
seed: 42
```

LoRA 版本默认配置来自：

```text
configs/train/flux2_klein_base_4b_self_flow_lora.yaml
```

关键区别：

```text
learning_rate: 1.0e-4
lora_rank: 32
gamma: 0.8
use_gradient_checkpointing: false
```

## 断点继续训练

如果要从当前交接 checkpoint 继续训练，可以传入 `--resume_from_checkpoint`：

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
bash /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_self_flow.sh \
  --resume_from_checkpoint /mnt/image-edit/datasets/duanyufa/交接/self-flow训练模型权重文件/checkpoint-7813
```

如果底层脚本使用的是 argparse 参数名 `--resume-from-checkpoint`，请以底层训练代码实际参数为准。当前配置文件中也有：

```text
resume_from_checkpoint: null
```

可以改成 checkpoint 路径后再启动。

## 验证 / 测试脚本

测试脚本路径：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/tests/test_flux2_self_flow.py
```

这个文件是 pytest 测试脚本，用于验证 Self-Flow 训练相关逻辑，不是单图生成推理脚本。

它覆盖的内容包括：

```text
双时间步加噪和 token mask
Flux2DiT per-token timestep 前向和反向传播
EMA teacher 更新
warmup cosine 学习率调度
tar 图像 caption 数据集读取
动态分辨率保持长宽比
```

运行命令：

```bash
cd /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
python -m pytest /mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/tests/test_flux2_self_flow.py -q
```

## 权重完整性检查

```bash
cd /mnt/image-edit/datasets/duanyufa/交接/self-flow训练模型权重文件/checkpoint-7813
ls -lh student.safetensors self_flow_projector.safetensors trainer_state.json
find pytorch_model -type f | wc -l
```

## 注意事项

1. 当前交接的 `checkpoint-7813` 是训练断点，不是一个单文件推理包。
2. `tests/test_flux2_self_flow.py` 是逻辑验证脚本，不负责读取一张图片做生成推理。
3. 训练默认依赖 8 卡 DeepSpeed ZeRO-3 配置。
4. 配置中的训练数据路径包含 `/mnt/zixuan_workspace/.../sample_250k.jsonl`，换机器时需要确认该路径是否存在。
5. 如果只需要模型权重本身，重点关注 `student.safetensors` 和 `self_flow_projector.safetensors`；如果要继续训练，需要保留整个 `checkpoint-7813` 目录。
