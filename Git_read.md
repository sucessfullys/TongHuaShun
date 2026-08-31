# 使用 PAT 上传本地项目到 Git

## 绝对红线

1. **一定不用对本地源目录进行任何的更改**，只是 Git 上传即可，所有操作都不能越过这个红线。
2. **一定不能删除 Git 远端仓库上的任何文件**，只允许新增（A）和修改（M），绝不允许删除（D）。
3. 每次操作前默念：只增不改删，只增不改删。

## 给 Codex 的执行说明

当用户要求 Git 上传时，先读取本文档，然后只向用户索取以下三项必需信息：

1. 本地文件夹的绝对路径。
2. 目标 Git 仓库的 HTTPS 地址。
3. 具有 `read_repository` 和 `write_repository` 权限的 PAT Token。

如无法从仓库地址判断，再询问 Git 用户名。分支默认使用远端默认分支；用户有指定时按用户要求执行。

可直接这样询问用户：

> 请提供：①本地文件夹绝对路径；②目标 Git HTTPS 地址；③PAT Token。默认不上传大文件、图片、权重、数据集、输出目录和缓存。

## 默认上传范围

上传源码、配置、脚本、文本说明和必要的小型项目文件，同时包含本地尚未提交的有效代码修改。

默认排除：

- 原仓库和嵌套仓库的 `.git/`。
- 权重及模型文件：`*.safetensors`、`*.ckpt`、`*.pt`、`*.pth`、`*.bin`、`*.pkl` 等。
- 图片：`*.png`、`*.jpg`、`*.jpeg`、`*.gif`、`*.webp`、`*.bmp`、`*.tif`、`*.tiff` 等。
- 大文件：默认排除单文件大于或等于 50MB 的文件；发现不明的大文件时不要自行上传。
- 常见大目录：`outputs/`、`data/`、`models/`、下载目录、数据集目录和图片结果目录。
- 缓存及构建产物：`__pycache__/`、`*.pyc`、`.pytest_cache/`、`*.egg-info/`、日志、临时文件和编辑器缓存。
- Token、密码、私钥、`.env` 及其他疑似凭据文件。

若项目自己的 `.gitignore` 意外忽略了应上传的源码或脚本，应在临时上传副本中明确纳入；不要因此修改用户的源目录。

## ⚠️ 排除规则的路径陷阱（血的教训 ×5）

### 陷阱一：短模式名误匹配深层子目录

使用 tar/rsync/find 排除目录时，**短模式名会误匹配深层子目录**：

| 错误写法 | 匹配结果 | 正确写法 |
|----------|----------|----------|
| `--exclude=models` | `models/`、**`diffsynth/models/`**（误杀！） | `--exclude=./models` |
| `--exclude=data` | `data/`、**`diffsynth/core/data/`**（误杀！） | `--exclude=./data` |
| `--exclude=outputs` | `outputs/`、`diffsynth/outputs/` | `--exclude=./outputs` |

**规则**：所有排除目录的模式必须加 `./` 前缀，只匹配项目根目录下的对应目录，绝不能误伤子目录中的同名文件夹。

### 陷阱二：嵌套 `.git` 目录被当作子模块（后果严重）

如果源项目的子目录中存在 `.git`（比如从别处复制来的项目带有自己的 git 历史），tar 默认不会排除它。打包后这些目录会被 Git 当成**子模块引用**（显示为 `folder @ commithash`），点不进去，文件全部丢失。

**症状**：GitLab 网页上目录显示为 `DreamFaceOmini @ 73d738b3` 而非正常文件夹。

**修复**：打包时显式排除所有嵌套 `.git`：
```bash
tar -cf /tmp/project.tar \
  --exclude=./.git \
  --exclude=./subdir1/.git \
  --exclude=./subdir2/.git \
  --exclude=./subdir3/.git \
  ... \
  .
```

**打包前排查**：
```bash
find . -name “.git” -type d 2>/dev/null
```
只保留项目根目录的 `.git`，其他全部加入 `--exclude`。

> 注意：`tar --exclude='.git'` 不会排除子目录的 `.git`（因为 tar 的路由匹配机制）。必须用 `--exclude=./path/.git` 精确指定每个嵌套 `.git` 的路径。

### 陷阱三：`git init` 默认分支名不匹配

`git init` 默认分支名可能是 `master`，但远端目标分支是 `main`，导致 `git push ... main` 报 `src refspec main does not match any`。

**修复**：`git init` 后立即 `git branch -m main`。

### 陷阱四：远端 main 分支被保护

GitLab 的 `main` 分支默认是受保护的（protected），`git push --force` 会被拒绝。

**症状**：`[remote rejected] main -> main (pre-receive hook declined)`

**修复**：去 GitLab → Settings → Repository → Protected branches → 临时 Unprotect `main`，force push 完成后再重新保护。

### 陷阱五：新增脚本被 `.gitignore` 忽略，导致看似同步但没有上传

有些仓库会在 `.gitignore` 里忽略整类目录，例如：

```text
/scripts/*
```

这种情况下，`rsync` 已经把本地新增脚本同步到了 `/tmp/upload/scripts/xxx.sh`，但 `git status` 默认不会显示它，`git add -A` 也不会把它纳入提交。结果就是：本地文件存在、临时目录文件也存在，但 GitLab 上看不到。

**症状**：

```bash
git ls-files scripts/new_script.sh
# 无输出，表示没有被 Git 跟踪

git check-ignore -v scripts/new_script.sh
# .gitignore:3:/scripts/* scripts/new_script.sh
```

**修复**：对确认应该上传的源码、脚本、配置、文本文件，在临时上传副本中使用 `git add -f` 强制纳入；不要修改用户本地源目录的 `.gitignore`。

```bash
cd /tmp/upload
git add -f scripts/new_script.sh
git status --short
```

**必须补做的兜底检查**：同步后，从本地源目录生成“符合上传规则的文件清单”，和 `/tmp/upload` 的 `git ls-files` 做比对，确认没有 eligible 文件漏传。

```bash
python - <<'PY'
import os, subprocess
from pathlib import Path

src = Path('/abs/local/project')
repo = Path('/tmp/upload')
tracked = set(subprocess.check_output(['git', '-C', str(repo), 'ls-files'], text=True).splitlines())

exclude_top = {'.git', 'outputs', 'data', 'datasets', 'downloads', 'models'}
exclude_ext = {
    '.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.pkl',
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tif', '.tiff',
    '.pyc', '.log', '.tmp',
}
exclude_names = {'.env'}
missing = []

for root, dirs, files in os.walk(src):
    rootp = Path(root)
    rel_root = rootp.relative_to(src)
    if rel_root.parts and rel_root.parts[0] in exclude_top:
        dirs[:] = []
        continue
    dirs[:] = [
        d for d in dirs
        if d not in {'.git', '__pycache__', '.pytest_cache'} and not d.endswith('.egg-info')
    ]
    for name in files:
        p = rootp / name
        rel = str(p.relative_to(src))
        if name in exclude_names or p.suffix.lower() in exclude_ext:
            continue
        if p.stat().st_size >= 50 * 1024 * 1024:
            continue
        if rel not in tracked:
            missing.append(rel)

print('eligible_missing_from_git=', len(missing))
for item in missing[:200]:
    print(item)
PY
```

只有 `eligible_missing_from_git=0`，才算“新增该加的都加了”。

如果使用 tar，排除大目录的正确写法：
```bash
tar -cf /tmp/project.tar \
  --exclude=./.git \
  --exclude=./subdir_with_git/.git \
  --exclude=./outputs \
  --exclude=./models \
  --exclude=./data \
  --exclude='./*.safetensors' \
  --exclude='./*.png' \
  .
```

创建 tar 后，必须检查：
```bash
# 确认 diffsynth 等源码目录下的文件都在 tar 中
tar -tf /tmp/project.tar | grep “diffsynth/models/” | head -5
tar -tf /tmp/project.tar | grep “diffsynth/core/” | head -5
# 确认无嵌套 .git
tar -tf /tmp/project.tar | grep “\.git/” && echo “⚠️ 还有嵌套 .git！” || echo “✅ 无嵌套 .git”
```

## 标准执行流程（修订版 v2——防删除版）

### 方案 A：远端已有内容（推荐，最安全）

核心思路：**从远端克隆 → 用本地源文件覆盖 → 只产生 A/M 变更，永远不会 D**。

1. **只读检查源目录**：Git 状态、文件体积及类型，确认排除项。
2. **在 `/tmp` 克隆远端仓库**（而非 init 空仓库）：
   ```bash
   git clone --depth=5 “https://oauth2:TOKEN@host/path.git” /tmp/upload
   ```
3. **切换到目标分支**（默认 `main`），确保和远端同步。
4. **用 rsync/tar/cp 从源目录同步文件到克隆目录**（排除 `.git`、权重、图片、大文件、outputs、顶层的 models/data）。**排除模式必须加 `./` 前缀**。
5. **关键检查：确认无删除**：
   ```bash
   cd /tmp/upload && git status
   ```
   只应看到 M（修改）和 ??（新增未跟踪），**不应有任何 D（删除）**。
   ```bash
   # 如果有 D，立即停下排查！
   git diff --name-status | grep “^D”
   ```
6. **再次验证被排除的目录**：确认 `diffsynth/models/`、`diffsynth/core/data/` 等源码目录未被误排。
7. **检查是否有新增文件被 `.gitignore` 漏掉**：
   ```bash
   git ls-files --others --ignored --exclude-standard
   git check-ignore -v path/to/new_script.sh
   ```
   对确认应该上传的源码、脚本、配置、文本文件使用 `git add -f path/to/file`。然后做一次 eligible 清单比对，确认 `eligible_missing_from_git=0`。
8. **提交并推送**：
   ```bash
   git add -A
   git commit -m “dsw-${THSCC_TRAIN_DSW_TASK_ID}: update ...”
   ```
9. **推送前最后确认**：
   ```bash
   # 确认本次提交中没有删除任何文件
   git diff --name-status HEAD~1 HEAD | grep “^D” && echo “⚠️ 有删除！停止推送！” || echo “✅ 无删除，可以推送”
   ```
   只有看到 `✅ 无删除` 才能执行推送。
10. **推送**：`git push origin main`
11. **核对结果**，向用户报告。

### 方案 B：远端空仓库（仅首次上传）

核心思路：**从源目录打包 → 排除不需要的文件 → init 新仓库 → 推送**。

1. **只读检查源目录**：Git 状态、文件体积及类型，确认排除项（同方案 A 步骤 1）。
2. **确认远端为空仓库**：用 `git ls-remote` 检查，或直接询问用户。
3. **在 `/tmp` 创建 tar 包**，排除规则同上（`./` 前缀用于目录，不加前缀用于文件扩展名）：
   ```bash
   tar -cf /tmp/project.tar \
     --exclude=./.git \
     --exclude=./outputs \
     --exclude=./models \
     --exclude=./data \
     --exclude='*.safetensors' \
     --exclude='*.ckpt' \
     --exclude='*.pt' \
     --exclude='*.pth' \
     --exclude='*.bin' \
     --exclude='*.pkl' \
     --exclude='*.png' \
     --exclude='*.jpg' \
     --exclude='*.jpeg' \
     --exclude='*.gif' \
     --exclude='*.webp' \
     --exclude='*.bmp' \
     --exclude='*.pdf' \
     --exclude='__pycache__' \
     --exclude='*.pyc' \
     --exclude='.pytest_cache' \
     .
   ```
4. **验证 tar 包**：
   ```bash
   # 确认无被误排除的源码
   tar -tf /tmp/project.tar | grep -E '\.(png|jpg|safetensors|pt|pth)$'  # 应为空
   tar -tf /tmp/project.tar | grep 'outputs/'  # 应为空
   tar -tf /tmp/project.tar | grep '/models/'  # 确认源码中的 models 子目录没有被误杀
   ```
5. **解压并初始化**：
   ```bash
   mkdir -p /tmp/upload && cd /tmp/upload
   tar -xf /tmp/project.tar
   git init
   ```
6. **注意分支名**：`git init` 默认分支可能是 `master`，需要重命名为 `main`（以匹配远端默认分支）：
   ```bash
   git branch -m main
   ```
7. **提交**：
   ```bash
   git add -A
   git commit -m "dsw-${THSCC_TRAIN_DSW_TASK_ID}: feat initial upload"
   ```
8. **推送**：
   ```bash
   git push "https://oauth2:TOKEN@host/path.git" main
   ```
9. **清理**：`rm -rf /tmp/upload /tmp/project.tar`

> ⚠️ 常见坑：`git init` 后直接 `git push ... main` 报 `src refspec main does not match any`，因为默认分支叫 `master`。解决：先 `git branch -m main`。

### 合并策略（仅方案 A）

| 场景 | 做法 |
|------|------|
| 用源文件覆盖克隆副本 | `cp -r src/* /tmp/upload/`（排除 `.git`、权重等） |
| 出现冲突 | 优先保留本地源文件 |
| 绝对禁止 | `git push --force`、`git reset --soft` 叠加在远端历史上 |

**为什么禁用 `git reset --soft FETCH_HEAD`**：如果本地文件集不全（比如被 tar 误排除了一部分），会导致那些文件在远端历史上被”删除”。宁可基于远端克隆做增量提交，也不要凭空构造提交。

## 安全分类器阻塞时的回退策略

当模型的安全分类器（deepseek-v4-flash safety classifier）频繁拒绝执行 Bash 命令（报错 `temporarily unavailable`）时，**不应反复重试**，而是：

1. 将当前已完成的操作状态告知用户。
2. 把剩余需要执行的完整命令序列输出给用户。
3. 让用户在终端直接执行，然后由 Codex 验证结果。

示例回退输出格式：

> 安全分类器又阻塞了。我把完整命令给你，在终端依次执行：
>
> ```bash
> # 1. 解压 + 初始化
> mkdir -p /tmp/upload && cd /tmp/upload
> tar -xf /tmp/project.tar
> git init && git branch -m main
>
> # 2. 提交
> git add -A
> git commit -m "dsw-81834: feat initial upload"
>
> # 3. 推送
> git push https://oauth2:TOKEN@host/path.git main
> ```

**注意**：输出命令时 PAT Token 仍需保留在 URL 中（用户已提供且仅用于单次认证）。命令执行完成后提醒清理临时文件。

## DSW 提交规范

此 GitLab 的提交钩子要求 commit 标题包含当前 DSW 实例执行历史 ID，格式为：

```text
dsw-{执行历史ID}: {commit type} {commit title}
```

在 DSW 实例中优先读取环境变量 `THSCC_TRAIN_DSW_TASK_ID`，无需让用户手动查找。例如：

```bash
git commit -m “dsw-${THSCC_TRAIN_DSW_TASK_ID}: fix update training script”
```

若该环境变量为空，再请用户从当前 DSW 实例详情页的”执行历史 ID”处提供编号。不要使用虚构或示例任务号。

## PAT 安全规则

- PAT 仅用于本次认证，不写入本文档、Git remote URL、Git 配置、脚本或项目文件。
- 不启用明文的 `credential-store`，不长期保存 PAT。
- 执行命令和输出中尽量避免回显 Token。
- 用户若把 PAT 发到聊天中，完成上传后提醒用户撤销该 Token；下次上传重新创建或提供新的 PAT。
- 任何时候都不得把 PAT 一并提交到仓库。

## 操作边界

- 未经用户明确允许，不上传图片、权重或大文件。
- 不使用 `git push --force`，除非用户明确要求且已说明风险。
- 不覆盖用户本地修改，不删除用户文件，不改动原仓库远端配置。
- 远端已有内容、分支冲突、权限不足或筛选范围不明确时，先停止并向用户说明。
