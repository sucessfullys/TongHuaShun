# 真实退化 LR/HR 数据集构建脚本说明

本目录用于交接“由高清原图生成真实退化 LR 图像”的数据构建方法。

## 文件

```text
generate_realistic_lr_dataset.py
README.md
```

## 脚本作用

`generate_realistic_lr_dataset.py` 会从一个原始高清图片目录读取图片，生成成对训练数据：

```text
output_dir/
├── HR/                         # 高清目标图，直接从 source-dir 复制而来
├── LR/                         # 退化输入图，由 HR 经过真实退化流程生成
├── metadata.jsonl              # 训练使用的标注文件
├── degradation_params.jsonl    # 每张图的退化参数记录
├── skipped_aspect_ratio.json   # 因长宽比过大被跳过的图片
└── generation_summary.json     # 数据集统计信息
```

生成的 HR 和 LR **保持同分辨率**，适合训练“输入退化图，输出清晰图”的图像增强、去模糊、超分/修复模型。

## 构建原理

对每张输入图片，脚本会按固定随机种子生成确定性的退化参数。相同 `--seed` 和相同文件名会得到相同退化结果，便于复现。

退化流程如下：

1. 读取原始图片作为 HR。
2. 根据严重程度随机选择 `mild`、`medium`、`strong`。
3. 使用各向异性高斯模糊模拟镜头模糊/运动模糊倾向。
4. 按随机倍率下采样，模拟低分辨率采集。
5. 按概率加入轻微高斯噪声或泊松噪声。
6. 第一次 JPEG 压缩，模拟压缩损伤。
7. 执行第二阶段轻退化：轻微高斯模糊 + 再次下采样。
8. 上采样回原图尺寸，保证 LR 和 HR 尺寸一致。
9. 第二次 JPEG 压缩。
10. 计算 HR/LR 的 PSNR，并将所有退化参数写入 `degradation_params.jsonl`。
11. 生成训练用 `metadata.jsonl`。

## 严重程度配置

脚本内置三档退化强度：

```text
mild    轻度退化，占比约 30%
medium  中度退化，占比约 45%
strong  强退化，占比约 25%
```

不同档位会影响：

```text
模糊 sigma 范围
卷积核大小
下采样倍率
JPEG 压缩质量
二阶段退化强度
```

## metadata.jsonl 格式

每行是一条训练样本，例如：

```json
{"prompt":"Restore this degraded image to a clean, sharp, natural image while preserving all original content, colors, geometry, and composition.","image":"000001.png","template_inputs":{"image":"/abs/path/output/LR/000001.png","prompt":"Restore this degraded image to a clean, sharp, natural image while preserving all original content, colors, geometry, and composition."}}
```

字段含义：

```text
prompt                  模型生成/增强目标描述
image                   HR 图像文件名，训练时通常配合 dataset_base_path 或 HR 目录使用
template_inputs.image   LR 输入图像的绝对路径
template_inputs.prompt  LR 输入对应的 prompt
```

脚本内置 3 条 prompt，会按样本顺序轮换使用。

## degradation_params.jsonl 格式

每行记录一张图的退化细节，例如：

```json
{
  "filename": "000001.png",
  "source": "/path/to/source/000001.jpg",
  "width": 1024,
  "height": 1024,
  "aspect_ratio": 1.0,
  "same_resolution": true,
  "degradation": {
    "severity": "medium",
    "blur": "anisotropic_gaussian",
    "kernel_size": 15,
    "sigma_x": 1.8,
    "sigma_y": 1.4,
    "angle_degrees": 42.3,
    "downsample_scale": 3.2,
    "downsample_interpolation": "area",
    "noise": "light_gaussian",
    "jpeg_quality_first": 65,
    "second_degradation": true,
    "upsample_interpolation": "bicubic",
    "jpeg_quality_final": 78,
    "psnr": 28.5
  }
}
```

这个文件可用于追溯每张 LR 图的生成方式。

## 参数说明

### `--source-dir`

原始高清图片目录。

```bash
--source-dir /path/to/source_images
```

脚本只扫描该目录第一层文件，不递归子目录。支持格式：

```text
.png .jpg .jpeg .webp .bmp .tif .tiff
```

### `--output-dir`

输出数据集目录。

```bash
--output-dir /path/to/output_dataset
```

脚本会在该目录下创建：

```text
HR/
LR/
metadata.jsonl
degradation_params.jsonl
skipped_aspect_ratio.json
generation_summary.json
```

### `--max-aspect-ratio`

最大允许长宽比，默认 `2.0`。

```bash
--max-aspect-ratio 2.0
```

如果图片长边/短边大于该值，会跳过并写入 `skipped_aspect_ratio.json`。

### `--noise-probability`

加入噪声的概率，默认 `0.35`。

```bash
--noise-probability 0.35
```

取值范围 `[0, 1]`。如果触发噪声：

```text
85% 概率加入轻微高斯噪声
15% 概率加入轻微泊松噪声
```

### `--seed`

随机种子，默认 `20260701`。

```bash
--seed 20260701
```

脚本会结合 `seed + 文件名` 生成每张图独立的随机数，因此结果可复现。

### `--limit`

限制处理图片数量，默认不限制。

```bash
--limit 100
```

适合先做小规模冒烟测试。

## 推荐运行方式

### 冒烟测试

```bash
python generate_realistic_lr_dataset.py \
  --source-dir /path/to/source_images \
  --output-dir /path/to/output_dataset_test \
  --limit 20
```

检查输出：

```bash
find /path/to/output_dataset_test/HR -maxdepth 1 -type f | wc -l
find /path/to/output_dataset_test/LR -maxdepth 1 -type f | wc -l
wc -l /path/to/output_dataset_test/metadata.jsonl
cat /path/to/output_dataset_test/generation_summary.json
```

### 正式生成

```bash
python generate_realistic_lr_dataset.py \
  --source-dir /path/to/source_images \
  --output-dir /path/to/output_dataset \
  --max-aspect-ratio 2.0 \
  --noise-probability 0.35 \
  --seed 20260701
```

## 断点续跑

脚本支持简单断点续跑：

- 如果 `degradation_params.jsonl` 已存在，会读取其中的 `filename` 作为已完成样本。
- 当对应 `HR/filename` 和 `LR/filename` 都存在时，会跳过该样本。
- 新样本会追加到 `degradation_params.jsonl`。
- 最后会重新生成完整的 `metadata.jsonl` 和 `generation_summary.json`。

## 注意事项

1. 输出 HR 文件统一保存为 `.png` 文件名，但 HR 内容是从原图复制来的。如果原图不是 PNG，文件名会变成 `.png`，但 `shutil.copy2` 不会转码。这一点如果后续工具严格依赖扩展名解析，建议把 HR 复制逻辑改成真正 PNG 保存。
2. LR 图会真正编码保存为 PNG。
3. 输入目录只扫描第一层，不递归。
4. 文件名相同会被视为同一输出名，正式生成前建议确保 source-dir 内没有同 stem 的不同扩展图片。
5. `metadata.jsonl` 中 `template_inputs.image` 是 LR 绝对路径，移动数据集目录后需要重新生成或替换路径。
6. 如果训练脚本要求 `image` 是 HR 相对路径，通常需要设置 HR 目录作为 `dataset_base_path`，或使用支持 `HR:LR:metadata.jsonl` 的数据集加载逻辑。

## 依赖

需要 Python 环境中安装：

```text
opencv-python
numpy
Pillow
```

常见安装命令：

```bash
pip install opencv-python numpy Pillow
```
