# 基模 30W 增强数据说明

本目录是用于图像增强/去模糊/修复模型训练的成对数据集交接目录。数据已经统一整理到一个目录下，包含高清目标图、退化输入图和训练标注文件。

## 目录结构

```text
/mnt/image-edit/datasets/duanyufa/交接/基模30W增强数据/
├── HR/              # 高清目标图，也就是训练时希望模型恢复到的图像
├── LR/              # 低质/退化输入图，也就是训练时喂给模型的条件图像
└── metadata.jsonl   # 训练标注文件，一行对应一对 HR/LR 样本
```

## HR 是什么

`HR/` 表示 High Resolution / High Quality 图像。

它是训练的目标图，也就是模型最终应该生成或恢复出来的清晰图像。训练时通常作为监督信号使用。

例如：

```text
HR/00_face_04344.png
```

表示该样本的高清目标图。

## LR 是什么

`LR/` 表示 Low Resolution / Low Quality 图像。

它是训练的输入图，通常包含模糊、噪声、压缩损伤、清晰度下降等退化。模型需要根据 LR 图像和 prompt，恢复出对应的 HR 图像。

例如：

```text
LR/00_face_04344.png
```

与下面这个 HR 文件是一对：

```text
HR/00_face_04344.png
```

也就是说，同名文件在 `HR/` 和 `LR/` 中应当一一对应。

## metadata.jsonl 是什么

`metadata.jsonl` 是训练用标注文件。每一行是一个 JSON 对象，对应一条训练样本。

示例：

```json
{"prompt":"Deblur and denoise this image while enhancing sharpness. Reconstruct fine details and textures faithfully from the remaining image information. Do not change people, facial features, clothing, colors, proportions, composition, or background. Do not introduce new objects or remove existing ones. Keep the image visually unchanged except for improved clarity.","image":"00_face_04344.png","template_inputs":{"image":"/mnt/image-edit/datasets/duanyufa/交接/基模30W增强数据/LR/00_face_04344.png","prompt":"Deblur and denoise this image while enhancing sharpness. Reconstruct fine details and textures faithfully from the remaining image information. Do not change people, facial features, clothing, colors, proportions, composition, or background. Do not introduce new objects or remove existing ones. Keep the image visually unchanged except for improved clarity."},"source":"/mnt/image-edit/datasets/duanyufa/交接/基模30W增强数据/HR/00_face_04344.png","_dataset_name":"face","_original_hr":"/mnt/image-edit/datasets/duanyufa/Face/HR/04344.png","_original_lr":"/mnt/image-edit/datasets/duanyufa/Face/LR/04344.png"}
```

字段说明：

```text
prompt
  当前样本的图像增强任务描述。训练时告诉模型要进行去模糊、去噪、增强清晰度或修复等操作，并要求尽量保持人物、颜色、构图和背景不变。

image
  HR 目标图的文件名。它对应 HR/ 目录下的同名图片。
  例如 image = "00_face_04344.png"，目标图就是 HR/00_face_04344.png。

template_inputs.image
  LR 输入图的绝对路径。训练时模型读取这个图作为输入条件图。

template_inputs.prompt
  输入侧使用的 prompt，通常和外层 prompt 保持一致。

source
  当前交接目录中的 HR 目标图绝对路径。

_dataset_name
  样本来源数据集名称，用于追踪该样本来自哪个子数据集。

_original_hr
  整理合并前的原始 HR 路径，仅用于溯源。

_original_lr
  整理合并前的原始 LR 路径，仅用于溯源。
```

## 一条样本如何对应

以 `00_face_04344.png` 为例：

```text
输入图：/mnt/image-edit/datasets/duanyufa/交接/基模30W增强数据/LR/00_face_04344.png
目标图：/mnt/image-edit/datasets/duanyufa/交接/基模30W增强数据/HR/00_face_04344.png
标注行：metadata.jsonl 中 image 字段为 "00_face_04344.png" 的那一行
```

训练逻辑可以理解为：

```text
给模型：LR 图像 + prompt
让模型学习输出：HR 图像
```

## 数据使用建议

训练时推荐以 `metadata.jsonl` 为入口读取数据：

1. 从每行 JSON 中读取 `template_inputs.image`，得到 LR 输入图。
2. 从 `image` 字段得到 HR 文件名。
3. 拼接 `HR/` 目录，得到 HR 目标图路径。
4. 使用 `prompt` 或 `template_inputs.prompt` 作为文本条件。

伪代码：

```python
hr_path = os.path.join(dataset_root, "HR", item["image"])
lr_path = item["template_inputs"]["image"]
prompt = item["prompt"]
```

## 注意事项

1. `HR/` 和 `LR/` 中同名文件是一对训练样本，不建议单独移动其中一个目录。
2. `metadata.jsonl` 里的 `template_inputs.image` 是绝对路径。如果把整个数据集移动到其他机器或其他目录，需要批量替换该字段路径，或者在数据加载代码里重新按文件名拼接 LR 路径。
3. `_original_hr` 和 `_original_lr` 是合并前路径，只用于溯源，不建议训练时依赖。
4. 如果训练框架要求目标图使用绝对路径，可以使用 `source` 字段；如果要求相对文件名，可以使用 `image` 字段并指定 `HR/` 作为目标图根目录。
5. 该数据集是多来源合并后的增强训练数据，文件名前缀用于减少不同来源之间的重名冲突。

## 快速检查命令

```bash
cd /mnt/image-edit/datasets/duanyufa/交接/基模30W增强数据

find HR -maxdepth 1 -type f | wc -l
find LR -maxdepth 1 -type f | wc -l
wc -l metadata.jsonl
head -1 metadata.jsonl
```

正常情况下，`HR` 图片数量、`LR` 图片数量和 `metadata.jsonl` 行数应该一致。
