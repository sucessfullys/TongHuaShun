# 实验追踪表

> 每次跑实验前加一行，出结果后补"结果"列。花 30 秒，省 3 天回忆。

---

## 1. 进行中 & 最近实验

| ID | 日期 | 服务器 | GPU | 基座ckpt | 关键变化 | 评估epoch | 结果 |
|----|------|--------|-----|---------|---------|----------|------|
| 0413_1_firered_w0.05_cap0.9 | 4/13 | fgly | 7卡 | v2.151e20 | firered η=0.05 cap=0.9 | e0 已评估 | 见下方 |
| v2.17_0413_1_firered_w0.05 | 4/13~14 | fgly | 7卡 | v2.151e20 | firered η=0.05 cap=0.9 + 增强 | e1、e6 已评估 | 画质正常不崩坏，但一致性仍然不稳定，有的时候脸不像 |


### 0413_1 训练日志观察（epoch 0 前 30 步）

| step | sigma | sigma_weight(σ²) | loss_diff | loss_id | face_cos_sim | 有效ID loss (η×σ²×L_id) |
|------|-------|-----------------|-----------|---------|-------------|------------------------|
| 10 | 0.996 | 0.809 | 0.019 | 0.909 | 0.091 | 0.037 |
| 20 | 0.225 | 0.050 | 0.756 | 0.041 | 0.959 | 0.0001 |
| 30 | 0.142 | 0.020 | 0.620 | 0.196 | 0.804 | 0.0002 |

**分析**：η=0.05 下有效 ID loss 最大约 0.037（高噪声时），与 loss_diff 同量级但不压过它，行为健康。低噪声时（σ<0.3）ID loss 几乎为零，不干扰细节生成，符合 FireRed 设计意图。对比 η=0.5 时有效 ID loss 高达 0.385 导致崩坏，现在已大幅改善。

### v2.17_0413_1 评估观察（e6，2026-04-14）

**现象**：推理结果画质正常，无崩坏；但一致性不稳定，部分案例生成的脸和 reference 不像同一人。

**训练日志 epoch 7 分析**（30 个采样点，按 σ 区间分组）：

| σ 区间 | 采样次数 | 平均 face_cos_sim | 典型表现 |
|---|---|---|---|
| σ < 0.3 | 5次 | **0.96** | 模型低噪声区身份保持好 |
| 0.3~0.6 | 7次 | **0.86** | 波动较大（0.58~0.92） |
| 0.6~0.8 | 7次 | **0.72** | 继续下降 |
| σ ≥ 0.8 | 11次 | **0.55** | 最大波动（0.03~0.81） |

**根因诊断**：

1. **η=0.05 信号被淹没**：在典型 σ=0.5 时，有效 ID loss = η × σ² × L_id ≈ 0.05 × 0.25 × 0.3 = **0.00375**，而 loss_diff 典型值为 0.3~1.0，ID 信号比扩散 loss 小 50-200 倍。LoRA 梯度几乎完全由扩散 loss 主导，ID loss 的修正信号太弱。

2. **数据量不足导致信号噪声无法平均**：4349 条数据 + 7 卡训练，每条样本被 ID loss 有效影响的次数有限，高 σ 区 ArcFace 梯度方向不稳定，噪声无法被大量样本平均。FireRed 用 100M+ 数据让这个问题消失了。

3. **η 矫枉过正**：从 η=0.5（崩坏）直接降到 η=0.05，可能跨过了甜区。

**η 有效 ID loss 量级对比**：

| η | σ=0.7 时有效 ID loss | 与 loss_diff 的比值 | 预期效果 |
|---|---|---|---|
| 0.5 | ~0.185 | ~30-60% | 主导训练，崩坏 |
| 0.15 | ~0.055 | ~5-15% | **可能甜区** |
| 0.05 | ~0.018 | ~1-5% | 被淹没，一致性不足 |
| 0.01 | ~0.003 | <1% | 几乎无效 |

**结论**：η=0.05 不是 ID loss 本身无效，而是权重太低、信号比太小。建议下一步实验 **η=0.1~0.15**，达到 loss_diff 的 5-15%。

### v2.151e20 基线对比（2026-04-14 补充）

对 v2.151e20 checkpoint（无 ID loss）跑了相同 benchmark（cfg=1.0），结果：

- **一致性显著好于 v2.17 η=0.05**（直观对比：成龙、刘亦菲、胡歌 case 都更像）
- 画质自然、无崩坏

**关键发现：v2.151e20 的"一致性好"是虚假的 copy-paste，不是真正的 identity preservation。**

v2.151e20 训练数据（~4万条）的特征：ref 和 target **姿势几乎完全一样**，只换了背景。模型走了捷径——直接把 ref 的脸部区域 paste 到新背景，根本没学会"在不同角度下重建同一身份"。这解释了为什么：
- v2.151e20 一致性好 → copy-paste 的天然结果
- v2.151e20 有 copy-paste 感 → 就是因为模型在做 copy-paste

**v2.17 的数据设计方向是对的**——混入了 2000 条多角度数据（ref 和 target 姿势、角度不同），打破了 copy-paste 捷径。模型被迫学习真正的跨姿势 identity preservation，但任务难度大幅上升，4000 条数据不够让这个更难的任务收敛。

**一致性下降不是退化，是模型面对了更难任务但数据量不足。**

### 数据瓶颈分析

| 数据集 | 条数 | 多角度占比 | 模型学到的能力 | 一致性来源 |
|---|---|---|---|---|
| v2.151 训练集 | ~4万 | ~0% | copy-paste | 假一致性（同姿势） |
| merged_train_4349 | 4349 | ~46%（2000条） | 跨姿势 identity | 真一致性（但数据不足） |
| 目标数据集 | 1~2万+ | 50%+ 多角度 | 稳健的跨姿势 identity | 真一致性（数据充分） |

**数据量是当前最核心瓶颈**。ID loss 调参在数据充足后才能发挥作用；在 2000 条多角度数据上调 η，效果上限有限。


---

## 2. 完整版本演进历史

### 第一阶段：基础管线搭建 & Bug 修复（3月第2~3周）

#### v2.1 初始版本

基于 FLUX.2-Klein-Base-9B，LoRA fine-tuning，多任务人像编辑（单人换脸 + 双人合照 + 场景生成）。

**训练数据**：10万+ 多任务人像数据（单人、换脸、双人合照）

**提示词风格**（v2.1 初始）：
> The same person from image 1, standing upright with hands by her sides, wearing a dark teal patterned suit...

| 问题 | 表现 | 根因 |
|------|------|------|
| 画质差 | 模糊、细节丢失 | 模型未收敛（非数据问题） |
| 提示词遵循度低 | 不按指令生成 | 多任务解空间复杂 |
| 一致性不稳定 | ID 忽好忽坏 | 同上 |

**关键排查过程**：
1. 训练 ~5 epoch 后效果不理想
2. 缩小数据量实验（只选 3 组不同任务 case）→ 模型可正常收敛
3. **结论：问题不在数据管线，而是多任务联合训练尚未充分收敛**

#### 代码 Bug 修复（3月第3周）

**重大发现**：代码存在命名不匹配 —— `edit_images` vs `edit_image`，导致**条件图像从未传入模型**。模型一直在做无条件生成。

**修复后效果**：显著提升。另外发现 cfg=1 改成 cfg=4 解决了"坏手问题"（因为 FLUX-Klein 没有蒸馏的 guidance）。

### 第二阶段：提示词 & 画质优化（3月第4周 ~ 4月初）

#### v2.11 提升训练可见分辨率

- 尝试提高训练时图像分辨率
- **结果**：效果没有明显提升

#### v2.12 Caption 优化

- 更详细的提示词风格：
> A realistic photo of the same woman, with blonde hair styled in a sleek updo, facing forward with a warm smile, standing upright with her arms at her sides, holding a small clutch in her right hand. She wears a vibrant coral-pink sleeveless gown...

- **结论**：正向提示词可以提升质量，但必须是正向描述

#### v2.13 仿真降质（参考 Real-ESRGAN）

- 对 edit_image 施加仿真降质
- **结果**：过于饱和，有点假
- **结论**：降质概率不应设置 0.5，太高了

#### v2.14 仿真降质 + Dropout

- 降质 + 条件 dropout
- **结果**：过于饱和 + 一致性下降
- **结论**：dropout 概率 0.8 太高
- 尝试加正向后缀 `authentic, high quality, high resolution, clean composition, rich details, low ISO, pristine quality, front-facing, subject in the center of the frame`

### 第三阶段：Seedream 蒸馏（4月初）

#### v2.151 Seedream 蒸馏（2000 + mix2000）

- 混合训练：2000 条 Seedream 生成数据 + 2000 条原始数据
- **结果**：初期画风接近原模型但更多噪点，后期朝 Seedream 风格拟合
- 越往后期，一致性和画质接近 Seedream
- 人头方向改善不明显（可能数据还没改过来）
- **策略调整**：降低原数据比重 mix2000 → mix400

#### v2.152 Seedream 蒸馏（2000 + mix400）

- 混合训练：2000 条 Seedream 数据 + 400 条原始数据
- 这是 v2.151 的跟进实验
- **v2.151e20 成为后续实验的最优基线 checkpoint**

### 第四阶段：ID Loss 探索（4月中旬）

#### v2.16 ID loss 初探（2000 + mix400 + id loss）

- 基于 v2.152 的数据配比，加入 ID loss
- **结果**：画质有点假，一致性也没有提升多少
- **结论**：ID loss 参数可能不对，需要进一步调优

#### v2.161 FireRed ID loss（η=0.5, cap=0.9）

基于 v2.151e20 checkpoint，使用 merged_train_4349.jsonl 训练。

训练配置：lr=1e-4, lora_rank=32, max_pixels=2073600, firered mode

| epoch | 整体质量 | 脸部 | 典型问题 |
|-------|---------|------|---------|
| e0 | 尚可 | 尚可 | 刚开始训，idloss 影响不大 |
| e3 | 开始下降 | 出现异常 | 脸部开始有不自然感 |
| e10 | **严重崩坏** | 鬼脸/重影 | 00003: 胸口出现重影脸；00005: 双脸叠加 |
| e20 | 更差 | 人像消失 | 画质变差 |

**根因**：η=0.5 过大（详见"重要发现"第 2 条）

---

## 3. 策略验证总结（3月第4周）

| 策略 | 结论 |
|------|------|
| 正向提示词 | **有效**，可提升画质 |
| Scale 提升 | **有效**，可进一步提升相似性 |
| cfg=4 替代 cfg=1 | **有效**，解决坏手问题 |
| 空提示词训练 | **已实现** |
| 仿真降质（高概率）| **无效/有害**，过于饱和，概率需降低 |
| Dropout（高概率）| **无效/有害**，一致性下降 |
| 提升训练分辨率 | **无效**，没有明显提升 |

---

## 4. 重要发现 & 实验结论

### 4.1 多任务联合训练需要充分收敛

- 单人换脸任务简单（提示词单一），3 张图迭代 300 次即可完美拟合
- 双人合照 + 场景生成较难（提示词涉及生成），需要更多数据和迭代
- 混合数据 3 张图也可收敛（e1 差 → e10 好 → e31 很好），证明管线无问题
- **结论：多场景混合任务解空间复杂，需要更多数据和更长训练**

### 4.2 ID loss weight=0.5 导致鬼脸崩坏（v2.161 实验）

- η=0.5 过大，有效 ID loss 在 σ=0.9 时高达 0.5 × 0.81 × 0.95 ≈ 0.385，几乎主导训练
- LoRA rank=32 容量有限，被迫优先满足 ID loss，牺牲基础生成质量
- 结果：模型在各处"画脸"以讨好 ID loss，出现鬼脸、重影、双脸叠加
- **根因是 η 太大，不是 σ² 加权策略本身的问题**

### 4.3 x̂₀ 身份保持实验（experiment_x0hat_robustness）

用 FLUX.2-Klein-9B 基础模型（未经 ID loss 训练），30 张人脸图，测量不同 σ 下 x̂₀ 的 ArcFace cos_sim：

| σ | x̂₀ cos_sim | x_t cos_sim | 说明 |
|---|-----------|-----------|------|
| 0.05 | 0.907 | 0.907 | 身份完全保持 |
| 0.20 | 0.790 | 0.753 | 身份基本保持 |
| 0.40 | 0.505 | 0.396 | cos>0.5 的边界 |
| 0.50 | 0.369 | 0.239 | 身份开始漂移 |
| 0.70 | 0.168 | 0.044 | 严重漂移 |
| 0.90 | 0.053 | 0.040 | 几乎完全漂移 |

**如何正确理解这个数据（结合 FireRed 论文 Section 3.7）：**

- x̂₀ 在高 σ 时虽然身份漂移严重，但图像视觉上仍然清晰（不是随机噪声）
- **低 cos_sim 不代表梯度方向无用**——ArcFace 对 x̂₀ 的降质具有鲁棒性，即使 cos_sim 较低，提取的 embedding 仍然包含身份相关的语义信息，梯度方向仍然指向"让 x̂₀ 更像目标身份"
- 这个实验测的是**训练前的基线**——ID loss 的目的恰恰是让模型学会在高 σ 时也预测出正确身份
- **高噪声区是身份形成的关键窗口**（FireRed 原文："The early stage (high-noise regime) is pivotal for identity formulation"），σ² 加权在此处施加强监督是正确的
- **低噪声区身份已"锁定"**，额外的 ID loss 反而会和细节生成竞争，产生视觉伪影

### 4.4 FireRed 论文 vs 我们的设置——关键差异

| | FireRed | 我们 |
|---|---|---|
| 训练数据 | **100M+** 高质量样本 | **4349** 样本（差 4 个数量级）|
| 训练方式 | 全模型多阶段（PT → SFT → RL）| LoRA fine-tuning |
| 模型容量 | 完整 DiT 参数 | LoRA rank=32 |
| η 值 | 未公开（推测很小）| 0.5（第一次）/ 0.05（当前）|
| σ² + cap=0.9 | 论文原始设定 | 应保留 |

数据量差 4 个数量级 + LoRA 容量有限 → η 需要相应大幅缩小。

### 4.5 v2.151e20 一致性的本质（2026-04-14 新增）

v2.151e20 基线在 benchmark 上一致性看起来好，**实质是 copy-paste 捷径，不是真正的 identity preservation**。

**训练数据特征**：~4万条，ref 和 target **姿势几乎完全相同**，只换背景。模型没有动力学习"跨姿势身份重建"，直接走了"把 ref 的脸 paste 到新背景"的捷径。

**证据链**：
- v2.151e20 一致性好 → copy-paste 的天然结果
- v2.151e20 有 copy-paste 感 → 训练数据就是同姿势换背景
- v2.17 换多角度数据后一致性下降 → 不是退化，而是任务难度真实上升，模型无法再走捷径

**数据设计对比**：

| 数据集 | 条数 | 多角度占比 | 模型能力 | 一致性来源 |
|---|---|---|---|---|
| v2.151 训练集 | ~4万 | ~0% | copy-paste | 假一致性（同姿势） |
| merged_train_4349 | 4349 | ~46%（2000条多角度） | 跨姿势 identity（学习中） | 真一致性（数据不足，未收敛） |
| 目标数据集 | 1~2万+ | 50%+ 多角度 | 稳健的跨姿势 identity | 真一致性（数据充分） |

**结论：数据瓶颈是当前最核心问题**，优先于任何超参调整。在 2000 条多角度数据上调 η、改 loss，效果上限有限。

### 4.6 参数建议（修正版）

| 参数 | 建议值 | 理由 |
|------|--------|------|
| face_id_weight (η) | **0.1 ~ 0.15** | η=0.05 信号被 loss_diff 淹没（比值仅 1-5%）；η=0.15 约为 5-15%，不至于崩坏，信号有效 |
| face_sigma_cap | **0.9**（保持论文值）| σ² 高噪声监督是正确设计，不应截断 |
| face_id_mode | **firered** | σ² 的论据成立；(1-σ) 在低噪声区加力反而可能有害 |

> ⚠️ 参数建议从原 0.01~0.05 更新为 0.1~0.15，依据：e6 评估确认 η=0.05 信号太弱，但 η=0.5 崩坏，甜区在中间。更根本的问题是多角度数据不足，应优先扩充。

---

## 5. 待验证的实验方向

### P0：数据扩充（最高优先级）

- [ ] **扩充多角度 character consistency 数据到 1~2万条**
  - 当前：2000条多角度 + 2000条原始 = 4000条，任务太难数据不足
  - 目标：同一人的不同角度、不同场景配对，打破 copy-paste 捷径
  - 方案：用 Seedream 等高质量生图模型生成 target，VLM 生成 caption（spectrum 模式）

### P1：ID Loss 调参（可与数据扩充并行）

- [x] firered η=0.5 cap=0.9 → **崩坏**（v2.161，e10 鬼脸重影）
- [x] firered η=0.05 cap=0.9 → **一致性不足**（v2.17，e6，信号被淹没）
- [ ] **firered η=0.15 cap=0.9** ← 当前数据量下的下一个实验（预期甜区）
- [ ] firered η=0.1 cap=0.9（备选）
- [ ] 数据扩充后重新评估 η 的合适范围

### P2：画质 & 其他

- [ ] 加速蒸馏（DMD2 / FLUX-Klein KV-Cache）
- [ ] RLHF / reward model 引导的美学质量优化
- [ ] Detail Daemon 提升细节（中间步骤传入略低 timestep）
- [ ] Benchmark 完善：推进人像评测体系

---

## 6. 关键文件路径

| 名称 | 路径 |
|------|------|
| 基线 ckpt (v2.151e20) | `/mnt/data/image-edit/datasets/shensheng/v2.151e20.safetensors` |
| v2.161 idloss0.5 训练输出 | `models/train/FLUX.2-klein-base-9B_lora_double_person_results_captions_merged/` |
| 当前训练 (w=0.1) 输出 | `models/train/FLUX.2-klein-base-9B_lora_double_person_results_captions_merged-face_id_weight0.1/` |
| 评估 benchmark | `/mnt/data/image-edit/datasets/shensheng/datasets/benchmark/明星/demo/02-2.json` |
| 训练数据 | `/mnt/data/image-edit/datasets/shensheng/datasets/merged_train_4349.jsonl` |
| ArcFace 权重 | `/mnt/data/image-edit/models/arcface/weights/arcface-r100-glint360k.pth` |

---

## 7. 代码变更记录

### 2026-04-13 鲁棒 MSE + 动态梯度裁剪

**背景**：参考 FireRed-Image-Edit 训练代码（`train/src/forward_step.py` & `sft.py`），补全两项对 firered ID loss 场景特别重要的工程优化。

#### 鲁棒 MSE（离群值截断）

**文件**：`diffsynth/diffusion/loss.py`

新增 `robust_mse_loss`，对 `|pred - target| > 50` 的位置做硬 mask，不参与 loss 计算。替换 `FlowMatchSFTLoss`、`FlowMatchSFTLossWithFaceID`、`FlowMatchSFTLossFireRedID` 三处。

**动机**：ID loss 的长梯度链（DiT → x̂₀ → VAE decode → ArcFace）遇到异常样本时容易产生极端残差，硬截断以最小代价防御。

#### 动态梯度裁剪

**文件**：`diffsynth/diffusion/runner.py`、`diffsynth/diffusion/parsers.py`、`run_train.sh`

前 N 步上限从 `max_grad_norm × initial_grad_norm_ratio` 线性衰减到 `max_grad_norm`；warmup 后若梯度异常则额外收紧。通过 `accelerator.clip_grad_norm_` 执行，与 DDP 兼容。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_grad_norm` | `1.0` | 目标上限；`0` 禁用 |
| `--initial_grad_norm_ratio` | `5.0` | 初始放大倍率 |
| `--grad_clip_warmup_steps` | `500` | 衰减步数 |
| `--abnormal_grad_ratio` | `5.0` | 异常触发倍率 |

### 2026-04-14 Timestep Importance Sampling（方案 B：零浪费采样）

**背景**：原方案 uniform 采样所有 σ + Gaussian loss weight 加权，导致 ~15-20% 的训练 step 落在 weight≈0 的两端区间，forward + backward 计算量白白浪费。对 LoRA 微调尤其浪费——训练步数本来就少。

**方案 B 核心思想**：把 Gaussian weight 从"loss 乘子"变成"采样概率"。数学上完全等价，但每个 step 的梯度都非零——零浪费。

**改动 1**：`diffsynth/diffusion/flow_match.py` — `set_training_weight()` + `sample_timestep_id()`

```
set_training_weight():
  # 原有 Gaussian bell curve 作为 linear_timesteps_weights（保留用于向后兼容）
  # 新增：同一条 bell curve 归一化为概率分布 timestep_sample_probs

sample_timestep_id(min_boundary, max_boundary):
  # 旧：torch.randint(min, max, (1,))   — 均匀采样
  # 新：torch.multinomial(probs, 1)      — 按 Gaussian 概率采样
```

**改动 2**：`diffsynth/diffusion/loss.py` — `FlowMatchSFTLossFireRedID()`

```
# 旧：timestep_id = torch.randint(min, max, (1,))
#     loss_diff = robust_mse_loss(...) * pipe.scheduler.training_weight(timestep)
#     → 均匀采样 + 后乘 Gaussian weight → 两端浪费

# 新：timestep_id = pipe.scheduler.sample_timestep_id(min, max)
#     loss_diff = robust_mse_loss(...)  # 不再乘 training_weight
#     → Gaussian 概率采样 + 无额外 weight → 零浪费，数学等价
```

注意：`FlowMatchSFTLoss`（无 ID loss 版本）已经在之前正确使用了 `sample_timestep_id` 且没有乘 `training_weight`，无需改动。

**为什么不用 logit-normal？** 见分析：FLUX.2-klein 基座在某种分布下预训练，LoRA 改变采样分布会导致部分 σ 区间 LoRA 没学到东西但推理时仍会经过。方案 B 不改变有效分布形状（仍然是 Gaussian bell），只是消除计算浪费，对基座匹配最安全。

---

## 8. 关键经验沉淀

1. **别没看完代码就跑实验**——edit_images vs edit_image 的 bug 浪费了大量时间
2. **跑大数据前先做小规模验证**——3 组 case 验证管线，防止无必要的回撤
3. **先看训练集，再看测试集**——训练集都拟合不了，说明模型没收敛而非数据问题
4. **从因果链推导，不要看到现象就提方案**——一致性差不一定要加 arcface，可能只是没收敛
5. **超参要和训练规模匹配**——FireRed 的 η 在 100M 数据上 work，不代表 4K 数据上也 work
6. **调超参要看信号比，不能只看绝对值**——η=0.05 的有效 ID loss 与 loss_diff 比值仅 1-5%，在 loss 层面几乎无效，即使绝对值看似"合理"
7. **区分"假一致性"和"真一致性"**——同姿势数据训出来的一致性是 copy-paste，换角度就失效。真正的 identity preservation 需要多角度数据，而且任务更难，需要更多数据量才能收敛
8. **一致性下降不一定是退化**——可能是数据设计更合理（加入多角度）导致任务变难，模型面对更难任务数据不足时，表面指标会下降，但方向是对的
