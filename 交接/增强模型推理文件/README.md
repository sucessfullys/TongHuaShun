# FLUX.2 KleinBase4B Deblur LRE 交接推理包

## 文件

- `step-10000.safetensors`: 训练得到的 Template 权重
- `infer_single_lre.py`: 单图推理脚本
- `run_infer_single.sh`: 最简运行入口
- `Template-KleinBase4B-Enhance`: Template 结构代码和初始权重，原名 Template-KleinBase4B-Upscaler

## 输入

1. 一张退化/LR 图片
2. 一个标注文件，支持：
   - `.txt`: 文件内容直接作为 prompt
   - `.json/.jsonl`: 读取 `prompt` 或 `template_inputs.prompt`

## 运行

正式推理命令示例：

```bash
DEVICE=cuda:0 bash /mnt/image-edit/datasets/duanyufa/交接/增强模型文件/run_infer_single.sh \
  /path/to/input.png \
  /path/to/annotation.txt \
  /path/to/output.png
```

其中：

```text
/path/to/input.png       输入的退化/LR 图片
/path/to/annotation.txt  文本标注文件，内容作为 prompt
/path/to/output.png      推理后保存的增强结果图片
```

不指定 `DEVICE` 时，也可以直接运行：

```bash
bash /mnt/image-edit/datasets/duanyufa/交接/增强模型文件/run_infer_single.sh \
  /path/to/input.png \
  /path/to/annotation.txt \
  /path/to/output.png
```

## 默认依赖路径

脚本默认使用当前机器上的 DiffSynth 环境和基础模型：

```text
PYTHON_BIN=/mnt/image-edit/datasets/duanyufa/conda_envs/DiffSynth/bin/python
BASE_MODEL=/mnt/image-edit/datasets/duanyufa/FLUX.2-klein-base-4B
TEMPLATE_MODEL=/mnt/image-edit/datasets/duanyufa/交接/增强模型文件/Template-KleinBase4B-Enhance
DIFFSYNTH_ROOT=/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
```

如果换机器，这三个路径需要对应改成新机器上的路径。

## 训练脚本

模型训练工程目录：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio
```

本模型对应的历史训练封装脚本：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_deblur_multi_dataset_all_lre.sh
```

底层实际训练脚本：

```text
/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/examples/flux2/model_training/train_lre.py
```
环境：root@interactive-vgp5mjjztgvu-6fbcb646f4-9zxvd:/mnt/image-edit/datasets/duanyufa#直接运行
最新交接训练数据集：

```text
/mnt/image-edit/datasets/duanyufa/交接/基模30W增强数据
├── HR
├── LR
└── metadata.jsonl
```

如果要基于最新交接数据集复训，参考/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio/scripts/train_flux2_klein_base_4b_deblur_multi_dataset_all_lre.sh

## 可调参数

```bash
DEVICE=cuda:0
NUM_INFERENCE_STEPS=50
CFG_SCALE=4.0
EMBEDDED_GUIDANCE=4.0
LRE_STRENGTH=0.8
```

## 纯文本标注示例

```text
Restore this degraded image to a clean, sharp, natural image while preserving all original content, colors, geometry, and composition.
```
