# 实验工作流速查手册

> 忘了就来看这里。全部操作加起来每次实验多花 1 分钟。

---

## 一、跑训练实验（最常用）

### 步骤

```bash
# 1. 改代码（如果需要改 loss.py 之类的）
vim diffsynth/diffusion/loss.py

# 2. 编辑 run_train.sh 顶部的参数区域
vim run_train.sh
#    - 改 EXP_NAME：格式 "MMDD_序号_关键词"，如 "0413_2_sigma_bell_curve"
#    - 改 EXP_NOTE：一句话说你在试什么
#    - 改需要调的超参数

# 3. 跑（脚本会自动 git commit + 保存配置 + tee 日志）
bash run_train.sh
```

就这三步。脚本会自动帮你：
- 把当前代码 git commit（绑定到实验名）
- 保存完整配置到 `exp_logs/{EXP_NAME}/config.json`
- 复制本次使用的脚本到日志目录
- 训练日志同时输出到屏幕和文件

### 命名规范

```
{月日}_{当天序号}_{关键词}
```

例子：
- `0413_1_baseline_no_idloss`
- `0413_2_firered_w0.1_cap0.45`
- `0414_1_bell_curve_sigma_weight`

关键词不用长，能让你认出来就行。

---

## 二、跑推理评估

```bash
# 编辑 run_eval.sh 顶部
vim run_eval.sh
#    - EXP_NAME 改成你要评估的训练实验名
#    - EPOCH 改成你要评估的 epoch 编号
#    - EVAL_NOTE 写一句你关心什么

# 跑
bash run_eval.sh
```

结果保存在 `exp_out/{EXP_NAME}_e{EPOCH}/`，配置保存在 `exp_logs/{EXP_NAME}/eval_e{EPOCH}.json`。

---

## 三、记录实验结果

跑完看了结果，打开 `EXPERIMENTS.md`，补上结果那一列：

```bash
vim EXPERIMENTS.md
# 在表格里加一行或补结果
```

然后 commit：

```bash
git add EXPERIMENTS.md && git commit -m "更新实验结果"
```

---

## 四、找回历史版本（最容易忘的）

### "那个效果好的版本，代码长什么样？"

```bash
# 查看所有实验对应的代码版本
git log --oneline

# 输出类似：
# d4e5f6a 0414_1_bell_curve_sigma_weight: 试bell curve加权
# b2c3d4e 0413_2_firered_w0.1_cap0.45: 降低sigma_cap到0.45
# a1b2c3d 0413_1_baseline: 基线实验

# 看某个版本的 loss.py
git show b2c3d4e:diffsynth/diffusion/loss.py

# 对比两个版本之间改了什么
git diff a1b2c3d b2c3d4e -- diffsynth/diffusion/loss.py
```

### "我想回到那个版本的代码重新跑"

```bash
# 方法1：只是看看（不影响当前代码）
git stash                          # 暂存当前修改
git checkout b2c3d4e               # 切到那个版本
# ... 看完之后 ...
git checkout master                # 回来
git stash pop                      # 恢复暂存的修改

# 方法2：基于那个版本创建分支继续改
git checkout -b try_old_version b2c3d4e
```

### "这个 checkpoint 是哪个代码版本训出来的？"

```bash
# 查看实验的 config.json
cat exp_logs/0413_2_firered_w0.1_cap0.45/config.json
# 里面有 "git_commit": "b2c3d4e"

# 然后
git show b2c3d4e:diffsynth/diffusion/loss.py
```

---

## 五、两台服务器同步代码

### 从服务器 A 同步到服务器 B

```bash
# 在服务器 A 上执行（改成你的实际路径和地址）
rsync -avz \
  --exclude='models/' \
  --exclude='exp_out/' \
  --exclude='exp_logs/*/train.log' \
  --exclude='__pycache__/' \
  /mnt/data/image-edit/datasets/shensheng/code/stable/Dream/ \
  服务器B地址:/对应路径/Dream/
```

这会同步代码 + git 历史 + 配置文件，但跳过大文件（模型、输出图、日志）。

---

## 六、目录结构说明

```
Dream/
├── run_train.sh          ← 训练启动入口（每次改顶部参数）
├── run_eval.sh           ← 评估启动入口（每次改顶部参数）
├── EXPERIMENTS.md        ← 实验总表（人工维护）
├── WORKFLOW.md           ← 本文件
├── .gitignore            ← git 忽略规则
│
├── diffsynth/            ← 核心代码（git 跟踪）
│   └── diffusion/
│       └── loss.py       ← ID loss 实现
│
├── examples/             ← 训练/推理脚本（git 跟踪）
│
├── exp_logs/             ← 实验配置记录（git 跟踪 config.json，忽略 train.log）
│   ├── 0413_1_.../
│   │   ├── config.json   ← 自动保存的完整超参
│   │   ├── run_train.sh  ← 本次使用的启动脚本副本
│   │   ├── eval_e3.json  ← 评估配置
│   │   └── train.log     ← 训练日志（不进 git，太大）
│   └── ...
│
├── models/train/         ← 训练输出的 checkpoint（不进 git，太大）
│   ├── 0413_1_.../
│   │   ├── epoch-0.safetensors
│   │   ├── epoch-1.safetensors
│   │   └── ...
│   └── ...
│
└── exp_out/              ← 推理输出的图片（不进 git）
    └── 0413_1_..._e3/
        ├── 00001/result.png
        └── ...
```

---

## 七、Git 常用命令速查

| 场景 | 命令 |
|------|------|
| 看历史 | `git log --oneline` |
| 看某版本的文件 | `git show HASH:路径` |
| 对比两个版本 | `git diff HASH1 HASH2 -- 文件路径` |
| 看当前改了什么 | `git diff` |
| 手动提交 | `git add -A && git commit -m "说明"` |
| 暂存当前修改 | `git stash` |
| 恢复暂存 | `git stash pop` |
| 切到旧版本看看 | `git checkout HASH` |
| 回到最新 | `git checkout master` |

---

## 八、最重要的三件事（贴在显示器上）

1. **跑实验前**：改 `run_train.sh` 顶部 → `bash run_train.sh`（自动 commit）
2. **看完结果**：在 `EXPERIMENTS.md` 补一句结论 → `git add -A && git commit -m "更新结果"`
3. **改了代码**：不用额外操作，`run_train.sh` 会自动帮你 commit
