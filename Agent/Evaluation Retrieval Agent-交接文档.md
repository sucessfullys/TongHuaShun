# ERA 项目交接文档

## 1. 项目简介

ERA，全称 Evaluation Retrieval Agent，是一个基于 Claude Code CLI 运行的 AIGC 评测方案检索与验证 Agent。项目目标不是固定写死某一种评测方法，而是根据 `/era:init` 时输入的任务描述，自动调研、设计、运行并通过人工反馈迭代，最终检索出最贴近人工判断且成本较低的评测协议。

公司内部 Git 地址：

```text
https://git-cc.myhexin.com:6443/10jqka/llm/aigc-01-06-06/evaluation-retrieval-agent.git
```

当前 README 标注版本为 v0.1.7，已实现 Stage 0 到 Stage 9 的端到端流程：

- Stage 0：任务初始化，命令为 `/era:init`。
- Stage 1：文献与开源方案调研。
- Stages 2-4：候选评测方案生成、多人设评审、方案决策。
- Stages 5-6：实验计划生成与实验执行。
- Stages 7-8：实验结果预比较与人工反馈 Web app。
- Stage 9：ReAct 迭代门控，决定继续推进还是修订下一轮。
- Stage 10：`final_report` 目前仍是 stub。

这个项目主要面向图像生成、图像编辑等 AIGC 任务的评测方案探索，也可以扩展到其他生成任务。任务本身由操作者在初始化时输入，不在代码里硬编码。

## 2. 代码结构

仓库根目录的关键内容如下：

```text
plugin/                 Claude Code 插件目录
  commands/             /era:init、/era:start、/era:status 等 slash commands
  scripts/              启动前检查脚本 preflight.sh
  skills/               各阶段被模型调用的技能说明
  agents/               literature-scout、era-heavy/standard/light 等子代理

era/                    Python 包主体
  probe/                GPU、数据、checkpoint、credential 探测逻辑
  orchestration/        工作区、生命周期、Ralph loop、实验调度、Web app 后端编排
  webapp/               Stage 8 人工反馈 Web app，FastAPI + React
  annotate/             /era:annotate 标注 Web app

docs/prompts/           各阶段运行提示词模板
docs/mcp-servers.md     MCP 配置说明
knowledge/              开发过程中的任务计划和提示词，不是运行时入口
tests/                  pytest 测试
config.example.yaml     config.yaml 配置示例
CLAUDE.md               Claude Code 在本仓库工作时的项目说明
README.md               项目主说明
```

工作区默认生成在：

```text
workspaces/{project}/
```

其中每个项目工作区会包含 `config.yaml`、`spec.md`、`status.json`、`.mcp.json`、`.claude/ralph-prompt.txt`、`iter_NNN/` 等文件和目录。

## 3. 环境要求

基础要求：

- Python >= 3.11
- Claude Code CLI
- 可用的 shell 环境
- 建议安装 `jq`，Ralph loop 插件的 Stop hook 依赖它
- 如果要跑 GPU/VLM 实验，需要可用 GPU、模型 checkpoint 和数据集路径

可选要求：

- `codex` CLI：当 `config.yaml` 里 `experiment.codex_reviewer: true` 时，用于 Stage 6 runner 的独立代码审查。
- GitHub Personal Access Token：Stage 1 使用 GitHub MCP 搜索开源实现时需要。

不要把真实 token 写进文档或提交到仓库。建议放在 `.env` 中，例如：

```bash
GITHUB_PERSONAL_ACCESS_TOKEN=ghp_xxx
```

启动 Claude Code 前需要把 `.env` 导入当前 shell：

```bash
set -a
source .env
set +a
```

## 4. 安装步骤

在 ERA 仓库根目录执行：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

这一步会安装 ERA Python 包和运行依赖，包括：

- `pyyaml`
- `arxiv-mcp-server`
- `fastapi`
- `uvicorn`

如果需要测试依赖，也可以使用：

```bash
.venv/bin/pip install -r requirements.txt
```

然后生成仓库根目录的 MCP 配置：

```bash
.venv/bin/python3 -m era.cli write-mcp-config
```

如果仓库移动过位置，或者 `.venv` 被重建过，需要重新运行上面的 `write-mcp-config`，因为 `.mcp.json` 中会写入本机 `.venv/bin/arxiv-mcp-server` 的绝对路径。

## 5. Claude Code 启动方式

首次使用前建议安装 Ralph loop 插件：

```bash
claude plugin marketplace add anthropics/claude-plugins-official
claude plugin install ralph-loop@claude-plugins-official
```

在 ERA 仓库根目录启动 Claude Code：

```bash
claude --plugin-dir ./plugin --dangerously-skip-permissions
```

虚拟试衣评测任务初始化示例，可在 Claude Code 中直接作为 `/era:init` 输入：

```text
/era:init Evaluate the transformed try-on images in:

/mnt/image-edit/datasets/xywang/dataset/tryon_results

Dataset: Full try-on results dataset. Each sample should include input_cloth.png, input_model.png, tryon_result.png, annotation.json, and metadata.json. Use metadata.json only for source tracing and bookkeeping.
Never use metadata.json as evaluation ground truth.

Goal:
Quantitatively score each tryon_result.png against its corresponding input_cloth.png and input_model.png on:

1. Cloth detail preservation
- Color
- Texture/material
- Pattern
- Logo/text
- Garment shape
- Distinctive design details
- Collar/sleeve/hem structure
- Overall fidelity to input_cloth.png

2. Model preservation
- Face/identity
- Pose
- Body shape
- Hair
- Background
- Lighting
- Non-clothing regions
- Overall fidelity to input_model.png

3. Visual quality
- Artifacts
- Warping
- Unnatural blending
- Missing or distorted body parts
- Garment-body alignment
- Boundary quality
- Overall realism

4. Outfit aesthetics and appropriateness
- Visual appeal
- Natural garment placement
- Suitability for the model's pose/context
- Overall outfit coherence

5. Other useful evaluation methods
- Add any additional objective or VLM-based metrics that help evaluate try-on quality.

For Family A VLM and hybrid scoring:
- Ask the VLM to inspect the input cloth, input model, and try-on result before scoring.
- Ask direct atomic questions where possible.
- Express concrete checks as binary 0/1 judgments (just output 0 for No or 1 for Yes), such as:
    - Was the clothing changed successfully?
    - Was the posture maintained?
    - Was the cloth color preserved?
    - Were cloth details preserved?
    - Was the model identity preserved?
    - Are there obvious artifacts?

Pairwise comparison metrics are not needed.
And try the larger model first, like Qwen3.5-397B-A17B-FP8, Qwen3.5-122B-A10B, Qwen3-VL-235B-A22B-Instruct for Family A.

Hardware:
- Use GPUs 0-5.

Models:
Prefer RAM-backed checkpoints in /dev/shm/models/ because they are much faster than network-mounted models.

Already staged:
- /dev/shm/models/Qwen2.5-VL-72B-Instruct/
- /dev/shm/models/Qwen3.5-397B-A17B-FP8/
- /dev/shm/models/Qwen3.5-122B-A10B/
- /mnt/image-edit/models/Google/Gemma-4-31B-it
- /mnt/model/Qwen/Qwen3.6-35B-A3B
- /mnt/image-edit/models/Qwen/Qwen3-VL-235B-A22B-Instruct
For any other model:
- Fall back to /mnt/model/, /mnt/model/Qwen, or /mnt/model/DeepSeek.
- These network-mounted paths are slow.
- If a model will be reused and /dev/shm/ has enough free space, copy it to /dev/shm/models/ first.
- Leave at least 100 GB free in /dev/shm/ for safety.

Expected output:
Produce per-sample quantitative scores and aggregate summaries.

For each sample, include:
- sample path
- output group or method name, if applicable
- cloth detail preservation score
- model preservation score
- visual quality score
- outfit aesthetics score
- any additional metric scores
- concise failure notes
- metadata-derived source tracing, if useful

Also produce:
- per-group or per-method averages
- overall averages
- ranked failure cases
- common error patterns
- recommendation of the most reliable evaluation metric set

Note:
use KV cache and thinking mode when using VLM model
```

仓库根目录和每个 `/era:init` 生成的 workspace 都会通过 `.claude/settings.json` 启用：

- `ralph-loop@claude-plugins-official`
- `enableAllProjectMcpServers: true`

如果 `jq` 或 Ralph loop 插件不可用，`/era:start` 会进入 manual fallback 模式。manual fallback 仍然能推进流水线，只是没有 Ralph loop 插件带来的 Stop hook 体验。

## 6. 标准运行流程

### 6.1 初始化项目

在仓库根目录的 Claude Code 会话中运行：

```text
/era:init <你的评测任务描述>
```

示例：

```text
/era:init Evaluate the edited images under /path/to/edits, comparing two editing methods. Each sample directory holds source.png and edited.png. Use GPUs 0-1. Pretrained models are in /path/to/models.
```

`/era:init` 会做这些事：

- 解析任务描述。
- 探测 GPU、数据路径、checkpoint、`.env` 凭据等环境信息。
- 对不明确的信息向操作者确认。
- 生成 `workspaces/{project}/` 工作区。
- 写入 `config.yaml`、`spec.md`、`status.json`、workspace 的 `.mcp.json` 和 Claude 设置。

初始化完成后，命令输出会提示新 workspace 路径。

### 6.2 进入 workspace 并启动流水线

进入 `/era:init` 生成的 workspace：

```bash
cd workspaces/{project}
```

在这个 workspace 中重新启动 Claude Code，然后运行：

```text
/era:start
```

也可以从仓库根目录指定项目名：

```text
/era:start {project}
```

`/era:start` 会：

- 运行 `plugin/scripts/preflight.sh` 做启动前检查。
- 将 `status.json` 的 `run_state` 更新为 `running`。
- 编译 `.claude/ralph-prompt.txt`。
- 通过 Ralph loop 或 manual fallback 逐阶段运行流水线。

从 Stage 1 到 Stage 8 人工反馈前，流水线应自动运行，不会中途要求操作者决策。

### 6.3 人工反馈阶段

到 Stage 8 时，ERA 会启动人工反馈 Web app，并把状态置为：

```text
run_state: awaiting_human
```

Web app 默认绑定在远端机器的 localhost。README 中给出的访问方式是：

```bash
ssh -N -L 8731:127.0.0.1:8731 <user>@<gpu-box>
```

然后在本地浏览器打开：

```text
http://localhost:8731/
```

在 Web app 中完成标注并点击 Finalize 后，会写入：

```text
iter_NNN/human/feedback.json
iter_NNN/human/human_labels.json
```

之后回到 Claude Code 运行：

```text
/era:resume
```

流水线会继续进入 Stage 9，根据人工反馈决定 ADVANCE 或 REVISE。

## 7. Slash Commands

### `/era:init <mission>`

Stage 0 初始化命令。用于探测环境、确认任务信息并创建 workspace。首次运行一个评测任务必须先执行它。

### `/era:start [project]`

启动已初始化项目的自动流水线。通常在 workspace 目录下运行，无参数即可；在仓库根目录可以传项目名。

### `/era:status [project]`

查看一个或所有项目的状态。会展示项目名、当前 stage、stage index、iteration、`run_state`、更新时间以及是否已有文献调研结果。

`run_state` 常见值：

- `idle`：已初始化但未启动。
- `running`：正在运行。
- `awaiting_human`：等待人工反馈。
- `blocked`：阻塞。
- `done`：完成。
- `stopped`：被暂停。

### `/era:stop [project]`

暂停正在运行的流水线。它会把 `run_state` 设置为 `stopped`，后续可以用 `/era:resume` 继续。

### `/era:resume [project]`

恢复被暂停或中断的流水线。它会从 `status.json` 记录的位置继续，而不是从头开始。

如果当前是 `awaiting_human`，`/era:resume` 会先检查人工反馈是否已经 Finalize。未完成时不会强行继续。

### `/era:annotate <dataset_root>`

启动独立的数据集标注 Web app。这个命令不依赖某个 ERA workspace，直接针对数据集路径工作。

标注结果会保存到：

```text
<dataset>/annotations/<sample_key>.json
```

这些标注后续会被 Stage 6 的 pass/recall 自动验证门控读取，用于判断候选评测配置是否与人工标注一致。

## 8. 核心功能说明

### 8.1 Stage 1：文献调研

Stage 1 会通过多个 `literature-scout` 子代理调研评测方法、指标和 benchmark。数据来源包括：

- arXiv MCP
- GitHub MCP
- Claude Code 内置 WebSearch / WebFetch

如果 MCP 不可用，会降级到 WebSearch / WebFetch，但调研质量可能下降。

### 8.2 Stages 2-4：方案生成、评审与决策

Stage 2 `plan_brainstorm` 会生成候选评测方案，使用的人设包括：

- judge-advocate
- metrics-advocate
- cost-pragmatist
- hybrid-innovator

Stage 3 `multi_review` 会用 alignment、feasibility、rigor 等批评视角审查候选方案。

Stage 4 `plan_decision` 会综合决策，产出后续实验需要的 handoff 文件：

```text
iter_NNN/design/plan.md
iter_NNN/design/experiment_brief.json
iter_NNN/design/hypotheses.md
iter_NNN/design/decision.json
```

其中 `experiment_brief.json` 会经过 `era.cli check-experiment-brief` 做确定性校验。

### 8.3 Stages 5-6：实验计划与实验执行

Stage 5 `experiment_plan` 会把 Stage 4 的 `experiment_brief.json` 扩展成可执行任务 DAG：

```text
iter_NNN/experiments/plans/task_plan.json
```

任务类型包括 serve、eval、aggregate、compare 等。

Stage 6 `full_experiment` 会执行任务 DAG：

- 编写 evaluator runner。
- 审查 runner。
- 根据 GPU 空闲情况调度任务。
- 用 marker files 恢复和跟踪任务状态。
- 失败时进行有限自动修复。
- 先跑 pilot，再跑 annotated/full round。
- 汇总结果到 `experiments/results/`。

GPU 调度使用跨 workspace 的 `fcntl` GPU leases，并遵守 Rule 6：同一时刻只允许一个 Family-A VLM judge resident，占用完整 GPU pool。

### 8.4 自动验证门控

v0.1.7 增加了 pass/recall auto-validation gate。Stage 6 在 pilot 和 full round 之间会先基于 `/era:annotate` 的人工标注跑 annotated round：

- 如果候选配置通过自动验证，才进入 full N=50 round。
- 如果全部失败，则触发 auto-revise，进入下一轮迭代。
- full round 的样本窗口由 `era.cli sample-window` 随机选择，但同一 iteration 内所有方法和配置使用同一批样本，保证可比性。

### 8.5 Stages 7-8：人工比较与反馈

Stage 7 `pre_human_comparison` 会把实验结果整理为：

```text
iter_NNN/comparison/comparison.json
```

Stage 8 `human_feedback` 启动人工反馈 Web app。操作者需要检查：

- Family-A / hybrid 判断是否错误。
- Family-B 相对排序是否错误。
- 未标记为错误的内容会被记录为 endorsed/correct。

Finalize 后生成 `feedback.json` 和 `human_labels.json`，供后续迭代使用。

### 8.6 Stage 9：ReAct 迭代

Stage 9 会读取当前 iteration 的实验结果和人工反馈，决定：

- `ADVANCE`：继续推进。
- `REVISE`：进入下一轮 iteration，重新从 Stages 2-8 改进方案。
- `REVISE_SKIP_STAGE1`：跳过 Stage 1，直接基于已有调研和反馈进入新一轮设计。

最大迭代次数由 `config.yaml` 中的 `react.max_iterations` 或相关默认值控制。

## 9. MCP 配置

ERA 的 MCP 配置由 `era/orchestration/mcp.py` 统一生成，并写入 `.mcp.json`。

默认包含：

- `arxiv-mcp-server`：stdio，本地 `.venv/bin/arxiv-mcp-server`。
- `github`：HTTP，使用 GitHub hosted MCP endpoint。
- `codex`：stdio，调用 `codex mcp`。

仓库根目录可以手动生成：

```bash
.venv/bin/python3 -m era.cli write-mcp-config
```

Claude Code 只读取启动目录下的 `.mcp.json`，不会自动向父目录查找。所以：

- 在仓库根目录启动时，需要根目录 `.mcp.json`。
- 在 workspace 中启动时，需要 workspace 自己的 `.mcp.json`。
- `/era:init` 会自动给每个 workspace 生成 `.mcp.json`。

验证 MCP：

```bash
claude mcp list
```

`arxiv-mcp-server` 和 `github` 正常时应显示 connected。GitHub 未导出 `GITHUB_PERSONAL_ACCESS_TOKEN` 时会连接失败，但 Stage 1 会降级到 web search。

## 10. 配置文件说明

每个 workspace 的主要配置文件是：

```text
workspaces/{project}/config.yaml
```

示例和字段解释见：

```text
config.example.yaml
```

重点字段：

- `project_name`：项目名。
- `mission`：操作者输入的任务描述。
- `task_family`：例如 generation 或 editing。
- `task_adapter`：例如 virtual_tryon、object_replacement、style_transfer、generic。
- `hardware`：GPU、显存、预留 GPU 等信息。
- `checkpoints`：本地模型目录和 checkpoint 信息。
- `serving`：模型服务后端和端口范围。
- `data`：数据根目录、方法目录、样本匹配规则、输入输出文件名。
- `credentials`：只记录凭据是否存在，不保存密钥。
- `budget`：API 成本和 wallclock 预算。
- `agent_modes`：不同阶段使用 heavy、standard、light 哪个模型 tier。
- `debate.max_rounds`：Stage 4 方案辩论最大轮数。
- `experiment`：GPU 调度、重试、pilot、Codex reviewer 等实验策略。

流水线状态不保存在 `config.yaml`，而是保存在：

```text
workspaces/{project}/status.json
```

## 11. 常用维护命令

运行测试：

```bash
.venv/bin/python3 -m pytest
```

查看 workspace 状态：

```bash
.venv/bin/python3 -m era.cli status <<JSON
{}
JSON
```

重新生成 MCP：

```bash
.venv/bin/python3 -m era.cli write-mcp-config
```

查看 Claude Code MCP：

```bash
claude mcp list
```

启动独立 Stage 8 feedback app，通常用于复查已完成 iteration：

```bash
.venv/bin/python3 -m era.cli serve-feedback <<JSON
{"workspace_path": "/abs/path/to/workspaces/project"}
JSON
```

查看 feedback app 状态：

```bash
.venv/bin/python3 -m era.cli feedback-status <<JSON
{"workspace_path": "/abs/path/to/workspaces/project"}
JSON
```

停止 feedback app：

```bash
.venv/bin/python3 -m era.cli stop-feedback <<JSON
{"workspace_path": "/abs/path/to/workspaces/project"}
JSON
```

## 12. 前端 Web app

项目有两个 React 前端：

```text
era/webapp/frontend      Stage 8 human-feedback review frontend
era/annotate/frontend    /era:annotate image-annotation frontend
```

两个前端都使用 Vite，脚本一致：

```bash
npm install
npm run dev
npm run build
npm run preview
```

正常业务运行不需要手动进入前端目录启动，`era.cli serve-feedback` 和 `era.cli serve-annotate` 会由后端编排启动服务。只有开发前端时才需要单独使用 `npm run dev`。

## 13. 常见问题与排障

### 13.1 `/era:start` 之前 preflight 失败

先看 `plugin/scripts/preflight.sh` 输出。常见原因：

- `.venv` 不存在或依赖未安装。
- 仓库处于 merge/conflict 状态。
- 运行目录不对。

修复后再运行 `/era:start` 或 `/era:resume`。

### 13.2 Ralph loop 插件不可用

确认插件已安装：

```bash
claude plugin marketplace add anthropics/claude-plugins-official
claude plugin install ralph-loop@claude-plugins-official
```

确认系统有 `jq`：

```bash
jq --version
```

如果仍不可用，ERA 会进入 manual fallback 模式，流水线仍可继续。

### 13.3 GitHub MCP 连接失败

确认 token 已放入 `.env`，并且在启动 Claude Code 前已导出：

```bash
set -a
source .env
set +a
```

然后重新启动 Claude Code，并运行：

```bash
claude mcp list
```

### 13.4 人工反馈 Web app 打不开

确认远端服务是否启动，并使用 SSH tunnel 访问。服务绑定在远端 `127.0.0.1`，从本地电脑直接访问远端端口通常不通。

示例：

```bash
ssh -N -L 8731:127.0.0.1:8731 <user>@<gpu-box>
```

本地浏览器打开：

```text
http://localhost:8731/
```

### 13.5 `/era:annotate` 没有图片

`/era:annotate` 会先 probe 数据集布局。若 probe 显示 sample 数为 0 或图片角色无法解析，不要强行启动 server。需要检查数据集是否符合 `<root>/<method>/<sample>/<images>` 这类结构，或在命令交互中提供 output/input 文件名 override。

### 13.6 README 和 command 文档版本不一致

当前 `README.md` 写明 v0.1.7 已实现 Stages 0-9，Stage 10 仍是 stub。但 `plugin/commands/start.md` 尾部仍有旧的 v0.1.4 描述，提到 Stages 7-11 是 stub。接手时以 `README.md` 和实际代码为准，不要被旧尾注误导。

## 14. 接手建议

接手后建议按下面顺序熟悉项目：

1. 阅读 `README.md` 和本文档，先跑通安装。
2. 阅读 `config.example.yaml`，理解 workspace 配置结构。
3. 阅读 `plugin/commands/*.md`，理解 Claude Code slash command 的行为。
4. 阅读 `docs/prompts/stage*.md`，理解每个阶段如何驱动模型。
5. 阅读 `era/orchestration/lifecycle.py`、`era/orchestration/ralph.py`、`era/cli.py`，理解流水线状态推进。
6. 用一个小数据集跑 `/era:init` 和 `/era:start`，观察 `workspaces/{project}/iter_001/` 的产物。

如果只做日常运行维护，重点掌握 `/era:init`、`/era:start`、`/era:status`、`/era:stop`、`/era:resume`、`/era:annotate` 和 MCP/token 配置即可。
