# 多模态畸形检测 Reward Model 单图推理说明

本目录提供一个最简洁的单图推理入口，用于加载 Qwen3-VL-8B-Instruct 基座模型和已训练 LoRA 权重，对一张图片判断是否存在人体结构异常。

## 文件

```text
infer_single_body_deformity.py  # Python 单图推理脚本
run_infer_single.sh             # Bash 启动脚本
README.md                       # 当前说明文件
```

## 默认模型路径

```text
基座模型：
/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct

LoRA 权重：
/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model/checkpoint-3450

ms-swift 工程：
/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift

Python 环境：
/mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python
```

## 固定提示词

system prompt：

```text
你是人体结构异常检测助手。请判断图中是否存在多手、多腿、肢体分叉或异常连接等人体肢体异常，并给出简洁、可见的判断理由。最终只能用 <conclusion>normal</conclusion>、<conclusion>abnormal</conclusion> 或 <conclusion>non_human</conclusion> 输出结论。
```

user prompt：

```text
<image>请判断画面中是否有多手、多腿、肢体分叉或异常连接现象，并给出理由和结论。
```

## 输出格式

模型输出应包含：

```xml
<evidence>自然语言理由</evidence>
<conclusion>normal/abnormal/non_human</conclusion>
```

含义：

```text
normal     图中人体结构正常
abnormal   图中存在多手、多腿、肢体分叉或异常连接等异常
non_human  图中没有明确人体主体，或不是人体结构检测对象
```

## 运行命令

```bash
bash "/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model/推理/run_infer_single.sh" \
  /path/to/input_image.png \
  "/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model/推理/output_single.json"
```

示例：

```bash
bash "/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model/推理/run_infer_single.sh" \
  "/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model/数据集文件夹/multi_hand/multilimb_v8_000006.png" \
  "/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model/推理/output_single.json"
```

## 可选参数

可以通过环境变量覆盖默认配置：

```bash
CUDA_VISIBLE_DEVICES=0
DEVICE=0
MAX_NEW_TOKENS=256
IMAGE_MAX_TOKEN_NUM=2048
TEMPERATURE=0
PYTHON_BIN=/mnt/image-edit/datasets/duanyufa/conda_envs/miniconda3/envs/ms-swift/bin/python
MODEL_PATH=/mnt/image-edit/datasets/duanyufa/models/Qwen3-VL-8B-Instruct
ADAPTER_PATH="/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model/checkpoint-3450"
PROJECT_DIR=/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift
```

例如指定物理 GPU 1：

```bash
CUDA_VISIBLE_DEVICES=1 DEVICE=0 bash "/mnt/image-edit/datasets/duanyufa/交接/多模态畸形检测reward model/推理/run_infer_single.sh" \
  /path/to/input_image.png \
  /path/to/output_single.json
```

这里 `CUDA_VISIBLE_DEVICES=1` 表示只让程序看到物理 GPU 1，脚本内部 `DEVICE=0` 表示使用当前可见 GPU 中的第 0 张。

## 注意事项

1. 该推理脚本只处理单张图片。
2. 该模型输出分类和理由，不输出 bbox 坐标框。
3. 输出 JSON 中会保存图片路径、模型路径、固定提示词、完整模型回答和解析出的 `conclusion`。
4. 如果迁移到其他机器，需要同步修改基座模型、LoRA 权重、ms-swift 工程和 Python 环境路径。
5. 环境直接使用ms-swift环境，同时必须使用ms-swift框架
6.目前支持手部异常的畸形检测
7.训练脚本路径/mnt/image-edit/datasets/duanyufa/task_shengsheng/models/ms-swift/examples/train/body_deformity_qwen3_vl/train_sft_lora.sh
8.环境通过root@interactive-vgp5mjjztgvu-6fbcb646f4-9zxvd:，conda activate ms-swift也可以通过/mnt/image-edit/datasets/duanyufa/conda_env_backup/ms-swift.tar.gz解压激活
