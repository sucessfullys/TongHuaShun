# 手部正常/异常分类：CLIP ViT-L/14 + MLP

## 方案与可行性

这是一个适合当前小数据集的基线方案：冻结预训练 CLIP ViT-L/14，只提取图像特征，再训练一个 `768 → 256 → 64 → 2` 的轻量 MLP 做二分类。MLP 在 256 维隐藏层后使用一次 `Dropout(0.35)`。

- 类别：`good = 0`（正常手），`bad = 1`（异常手）。
- 当前数据共有 26 个 UUID，每个 UUID 在 `good/`、`bad/` 各有一张图，共 52 张。
- 相同 UUID 的正常图和异常图会作为一个组，同时进入 train、val 或 test，避免同源样本跨集合造成数据泄漏。
- 训练集默认为每张图缓存 4 个轻度增强视图的 CLIP 特征。CLIP 只运行一次，之后 100 个 epoch 只训练 MLP，速度和显存开销都较低。

该方案可作为第一版验证，但 26 组数据很少，测试指标波动会比较大，不能据此判断生产效果。CLIP 会把整图缩放/裁剪到 224 像素，小手或局部畸形可能丢失。后续优先增加数据，再考虑手部检测裁剪、多尺度特征或分组五折交叉验证。

## 文件

- `hand_clip.py`：数据扫描、UUID 分组切分、图像增强。
- `modeling.py`：MLP 分类头、CLIP 特征提取和评估指标。
- `train.py`：训练、验证、测试、早停和模型保存。
- `infer.py`：单张图片或目录批量推理。

## 训练

推荐使用已有的 `flow_grpo` 环境：

```bash
PY=/mnt/image-edit/datasets/duanyufa/conda_envs/flow_grpo/bin/python
CUDA_VISIBLE_DEVICES=0 "$PY" \
  /mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/train.py
```

训练默认使用：

- 数据集：`/mnt/image-edit/datasets/duanyufa/task_shengsheng/手部异常数据集/hand`
- CLIP 权重：`/mnt/image-edit/datasets/duanyufa/RAR/checkpoints/RAR_modelzoo/CLIP/ViT-L-14.pt`
- 输出目录：`/mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/outputs/clip_vitl14_mlp`

首次运行会生成并保存 CLIP 特征。若参数和数据切分未改变，可跳过重复特征提取：

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" \
  /mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/train.py \
  --reuse-feature-cache
```

主要输出：

- `best_model.pt`：最佳 MLP 权重以及 CLIP 配置。
- `feature_cache.pt`：冻结 CLIP 提取的特征。
- `split.json`：train/val/test 的 UUID 分组和文件清单。
- `metrics.json`：训练历史、验证和测试指标。
- `val_predictions.json`、`test_predictions.json`：逐张预测结果。

## 推理

单张图片：

```bash
"$PY" /mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/infer.py \
  --checkpoint /mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/outputs/clip_vitl14_mlp/best_model.pt \
  --input /path/to/image.png
```

目录批量推理：

```bash
"$PY" /mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/infer.py \
  --checkpoint /mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/outputs/clip_vitl14_mlp/best_model.pt \
  --input /path/to/images \
  --recursive
```

输出为 JSONL，每行包含文件路径、预测类别、`prob_good` 和 `prob_bad`。默认以 `prob_bad >= 0.5` 判为异常，可用 `--bad-threshold` 调整阈值。

## 可解释性输出

单张图片遮挡热力图：

```bash
CUDA_VISIBLE_DEVICES=0 \
/mnt/image-edit/datasets/duanyufa/conda_envs/flow_grpo/bin/python \
/mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/explain.py \
--input /path/to/image.png
```

红色区域支持异常判断，蓝色区域支持正常判断。输出包含 `original_clip_input.png`、`heatmap.png`、`overlay.png` 和 `explanation.json`。

CLIP 特征 PCA 分布图：

```bash
/mnt/image-edit/datasets/duanyufa/conda_envs/flow_grpo/bin/python \
/mnt/image-edit/datasets/duanyufa/task_shengsheng/project/分类/visualize_features.py
```

蓝点为正常手、红点为异常手；圆形、方形、三角形分别表示训练、验证、测试集。

## 使用注意

- 切勿按图片随机切分；必须按 UUID 分组，否则同一对图可能同时出现在训练集和测试集。
- `feature_cache.pt` 只应在 CLIP 权重、随机种子、增强设置、训练视图数和数据均未变化时复用。
- 当前增强只采用水平翻转和轻微颜色扰动，避免旋转、强裁剪等改变手部异常语义的增强。
- 正式比较模型时，建议固定独立测试集，或报告 5 折分组交叉验证的均值和标准差。
