# System-Prompt-Retrieval-Agent 交接文档

## 1. 项目简介

`System-Prompt-Retrieval-Agent` 是一个用于虚拟试衣场景的系统提示词检索与评估项目。整体目标是自动生成、测试、评估并迭代适合虚拟试衣任务的 system prompt / negative prompt 组合，最终筛选出效果最好的 prompt pair。

项目由本地代理和远端 GPU 推理服务两部分组成：

- 本地代理：负责 prompt pair 生成、轮次调度、远端调用、本地 API 评估、评分聚合、记忆管理、断点恢复和结果沉淀。
- 远端服务：部署在 3H100 机器上，负责 Gemma、FLUX、Qwen 三阶段模型推理。
- MCP 工具设计：提供一组后续可封装为 MCP server 的 JSON Schema，目前是设计参考，不是已经上线的 MCP server。
- 评估脚本：包含 V0.1 全量评估脚本和若干 smoke/calibration 工具。

代码仓库地址：

```bash
https://git-cc.myhexin.com:6443/10jqka/llm/aigc-01-06-06/system-prompt-retrieval-agent
```

当前本地仓库 remote 配置为：

```bash
origin https://git-cc.myhexin.com:6443/10jqka/llm/aigc-01-06-06/system-prompt-retrieval-agent.git
```

远端 GPU 机器约定：

```text
SSH alias: 3h100
远端项目根目录: /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent
远端服务目录: /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote
远端数据集根目录: /mnt/image-edit/datasets/xywang/dataset
```

## 2. 代码目录结构

```text
.
├── System-Prompt-Retrieval-Agent/      # 本地 Python 代理主项目
├── Image-Generater-Remote/             # 远端 GPU 推理服务
├── eval_run/                           # V0.1 全量评估脚本
├── CLAUDE.md                           # 项目操作约定和固定路径说明
├── project proposal.md                 # 项目原始 proposal
└── .env.example                        # 环境变量示例
```

重点目录说明：

- `System-Prompt-Retrieval-Agent/src/system_prompt_retrieval_agent/`：本地代理源码。
- `System-Prompt-Retrieval-Agent/config.yaml.example`：本地代理配置模板。
- `System-Prompt-Retrieval-Agent/mcp_tools/`：MCP 工具候选 schema 和设计说明。
- `System-Prompt-Retrieval-Agent/skills/`：项目内置技能说明，描述 prompt 生成、远端阶段、评分聚合等工作流。
- `Image-Generater-Remote/server/`：单 GPU supervisor 服务，提供 `/load`、`/unload`、`/infer/*` 等接口。
- `Image-Generater-Remote/workflow/`：远端 workflow controller，提供 `/stage/gemma`、`/stage/flux`、`/stage/qwen`、`/v022/stage/{stage}`。
- `Image-Generater-Remote/scripts/`：部署、启动、停止、smoke test 和模型资产检查脚本。
- `Image-Generater-Remote/agent_sim/`：V0.1 模拟 agent，全链路驱动 Gemma -> FLUX -> Qwen。
- `eval_run/`：旧版全量评估流水线，适合参考或复现实验。

## 3. 环境依赖

### 本地代理

本地代理要求 Python 3.10 及以上，核心依赖包括：

- `openai`
- `pydantic`
- `pyyaml`
- `python-dotenv`
- `httpx`
- `jsonschema`
- `pillow`
- `pandas`
- `numpy`
- `fastapi`
- `uvicorn`
- `pytest`

依赖文件：

```text
System-Prompt-Retrieval-Agent/requirements.txt
System-Prompt-Retrieval-Agent/pyproject.toml
```

### 远端 GPU 服务

远端服务要求 GPU/CUDA 环境，当前依赖是按 3H100 机器调试过的版本固定的。核心栈包括：

- Python FastAPI 服务：`fastapi`、`uvicorn`、`pydantic`、`httpx`
- 深度学习栈：`torch==2.10.0`、`torchvision==0.25.0`、`torchaudio==2.10.0`
- 推理框架：`vllm==0.19.1`、`diffusers==0.38.0.dev0`
- 模型相关：`transformers`、`accelerate`、`peft`、`safetensors`、`sentencepiece`
- Qwen fallback：`ms-swift`

依赖文件：

```text
Image-Generater-Remote/requirements.txt
```

注意：项目约定不要自动下载模型 checkpoint。模型资产必须由接手人确认已存在，并在配置中指向正确路径。

## 4. 配置说明

### 本地代理配置

复制模板：

```bash
cd System-Prompt-Retrieval-Agent
cp config.yaml.example config.yaml
```

关键配置项：

- `paths.local_project_root`：本地项目根目录。
- `paths.local_agent_root`：本地代理目录。
- `paths.remote_image_service_root`：远端 GPU 服务目录。
- `paths.dataset_root`：数据集目录。
- `paths.output_root`：本地输出目录。
- `paths.memory_root`：记忆文件目录。
- `paths.env_file`：`.env` 文件路径。
- `remote.ssh_alias`：默认是 `3h100`。
- `remote.controller_base_url`：默认是 `http://127.0.0.1:17700`。
- `api.openai_api_key_env`：默认读取 `OPENAI_API_KEY`。
- `api.google_api_key_env`：默认读取 `Google_API_KEY`。
- `workflow.max_rounds`：最大迭代轮数。
- `workflow.score_threshold`：达到该分数后可停止。
- `evaluation.run_local_api_eval`：是否运行本地 API 评估。
- `budget.daily_usd_cap` / `budget.per_round_usd_cap`：API 预算保护。

### 远端服务配置

复制模板：

```bash
cd Image-Generater-Remote
cp config.yaml.example config.yaml
```

关键配置项：

- `sync.ssh_alias`：默认是 `3h100`。
- `sync.local_root`：本地远端服务源码目录。
- `sync.remote_root`：远端部署目录。
- `sync.remote_venv`：远端 venv 路径。
- `ports.host_gpu0`、`ports.host_gpu1`、`ports.host_gpu2`：三个 supervisor 端口，默认 `17610`、`17611`、`17612`。
- `ports.agent_callback`：controller/callback 端口，默认 `17700`。
- `gpu.visible_devices`：默认 `0,1,2`。
- `models.gemma4.path`：Gemma 模型路径。
- `models.flux2_klein.path`：FLUX.2 Klein 模型路径。
- `models.qwen3_vl.path`：Qwen3-VL 模型路径。
- `models.qwen3_vl.adapter_path`：可选 LoRA adapter 路径。
- `paths.dataset_root`：远端数据集根目录。
- `paths.output_root`：远端输出根目录。
- `callback.default_url`：默认 `http://127.0.0.1:17700/status`。

### 环境变量

根目录有 `.env.example`。实际运行前准备 `.env`，至少确认：

```bash
OPENAI_API_KEY=...
Google_API_KEY=...
```

不要提交 `.env`、`config.yaml`、日志、输出和模型权重。

## 5. 安装步骤

### 5.1 克隆代码

```bash
git clone https://git-cc.myhexin.com:6443/10jqka/llm/aigc-01-06-06/system-prompt-retrieval-agent.git
cd system-prompt-retrieval-agent
```

如果在共享挂载目录中运行 `git status` 报 `dubious ownership`，需要接手人在自己的环境里显式信任该目录：

```bash
git config --global --add safe.directory /path/to/system-prompt-retrieval-agent
```

### 5.2 安装本地代理

```bash
cd System-Prompt-Retrieval-Agent
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .[dev]
cp config.yaml.example config.yaml
```

安装后命令行入口：

```bash
system-prompt-retrieval-agent --help
```

### 5.3 安装远端服务

在远端 3H100 机器上执行：

```bash
ssh 3h100
cd /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote
python -m venv .venv
source .venv/bin/activate
pip install -U pip
```

先按 PyTorch CUDA 版本安装 torch：

```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

再安装其他依赖：

```bash
pip install -r requirements.txt
pip install "git+https://github.com/huggingface/diffusers.git"
cp config.yaml.example config.yaml
```

安装完成后，建议先检查模型资产：

```bash
python scripts/check_model_assets.py --config config.yaml
```

## 6. 启动与运行

### 6.1 部署远端服务代码

在本地 `Image-Generater-Remote/` 目录执行。先 dry-run：

```bash
cd Image-Generater-Remote
bash scripts/sync_to_remote.sh
```

确认同步内容无误后再真正同步：

```bash
bash scripts/sync_to_remote.sh --apply
```

### 6.2 启动远端 GPU supervisors

在远端执行：

```bash
ssh 3h100
cd /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote
source .venv/bin/activate
bash scripts/start_hosts.sh
```

该脚本会启动三个 GPU-pinned supervisor：

```text
GPU 0 -> port 17610
GPU 1 -> port 17611
GPU 2 -> port 17612
```

健康检查：

```bash
curl http://127.0.0.1:17610/health
curl http://127.0.0.1:17611/health
curl http://127.0.0.1:17612/health
```

停止服务：

```bash
bash scripts/stop_hosts.sh
```

日志位置：

```text
Image-Generater-Remote/logs/
```

### 6.3 启动 workflow controller

controller 默认监听 `17700`，提供 staged pipeline 接口。远端执行：

```bash
cd /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote
source .venv/bin/activate
python -m workflow.controller --config config.yaml --dataset /mnt/image-edit/datasets/xywang/dataset
```

健康检查：

```bash
curl http://127.0.0.1:17700/health
curl http://127.0.0.1:17700/status
```

本地代理默认通过 SSH tunnel 或本机转发访问：

```bash
ssh -N -L 17700:127.0.0.1:17700 3h100
```

### 6.4 运行本地代理

```bash
cd System-Prompt-Retrieval-Agent
source .venv/bin/activate
system-prompt-retrieval-agent run --config config.yaml --limit 5 --max-rounds 1
```

常用参数：

- `--limit N`：限制样本数量，调试时建议先用小样本。
- `--max-rounds N`：限制最大迭代轮数。
- `--dry-run`：走旧版 dry-run 路径，不触发生产 runner。
- `--resume-from-run-id RUN_ID`：复用指定 run id，启动前会校验 config、user prompt corpus、prompt pair corpus、sample corpus 哈希。

查看状态：

```bash
system-prompt-retrieval-agent status --config config.yaml
```

恢复最后一次或指定 run：

```bash
system-prompt-retrieval-agent resume --config config.yaml
system-prompt-retrieval-agent resume --config config.yaml --run-id RUN_ID
```

查看当前 best pair：

```bash
system-prompt-retrieval-agent best-pair --config config.yaml
```

### 6.5 运行远端 pilot pipeline

远端服务启动后，可以用模拟 agent 跑 pilot：

```bash
ssh 3h100
cd /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote
source .venv/bin/activate
python agent_sim/run_pipeline.py --config config.yaml --limit 30
```

### 6.6 运行 V0.1 全量评估

本地执行：

```bash
bash eval_run/run_full_eval.sh all 0
```

也可以限制样本数：

```bash
bash eval_run/run_full_eval.sh all 10
```

脚本会把 `eval_run/eval_pipeline.py` 和 `eval_run/prompts_data.py` 同步到远端，执行后把结果拉回本地 `eval_results/`。

## 7. 核心功能说明

### 7.1 Prompt pair 生成

本地代理根据当前 round、历史记忆和已有结果调用 OpenAI 模型生成新的 prompt pair。配置入口在：

```text
System-Prompt-Retrieval-Agent/config.yaml.example
```

相关模块：

```text
system_prompt_retrieval_agent/prompt_generation/
system_prompt_retrieval_agent/agent_loop.py
system_prompt_retrieval_agent/agent_loop_v022.py
system_prompt_retrieval_agent/runner_v022.py
```

默认每轮生成数量由 `prompt_generation.prompt_pairs_per_round` 控制。

### 7.2 远端三阶段推理

远端服务按阶段执行：

1. Gemma：输入 model/cloth 图片和 system prompt，生成中间 prompt。
2. FLUX：根据中间 prompt 生成虚拟试衣图。
3. Qwen：对生成图做视觉评估，输出结构化结果。

主要接口：

```text
POST /stage/gemma
POST /stage/flux
POST /stage/qwen
POST /v022/stage/{stage}
```

V0.2.2 生产路径主要使用 `/v022/stage/{stage}`，它按 cell 维度调度，并通过 manifest 保证恢复和一致性。

### 7.3 本地 API 评估

本地代理可以在 Qwen 评估之外调用 OpenAI VLM 做补充评估。相关配置：

```yaml
evaluation:
  run_local_api_eval: true
  max_concurrent: 4
  api_concurrency: 4
  api_rps_limit: null
```

注意：项目约定 OpenAI 在线 VLM 调用不能超过 3 req/s，必须通过项目内 rate limiter，不能绕过。

### 7.4 评分聚合

评分聚合综合多个维度：

- `qwen_pass_rate`
- `edit_correctness`
- `garment_transfer_correctness`
- `preservation`
- `artifact_penalty`
- `category_balance_bonus`

权重在 `scoring.weights` 中配置。相关模块：

```text
system_prompt_retrieval_agent/scoring/
system_prompt_retrieval_agent/scoring_v022.py
```

最终会生成 round 级别结果，并写出当前最优 prompt pair。

### 7.5 Memory 管理

项目维护长期记忆和轮次记忆，用于避免重复探索、复用高分经验、剪枝低价值 prompt。相关配置：

```yaml
paths:
  memory_root: ...
```

相关模块：

```text
system_prompt_retrieval_agent/memory/
system_prompt_retrieval_agent/memory_v022.py
```

主要能力：

- 加载历史上下文用于生成。
- 写入每轮 prompt pair。
- 把高价值 prompt pair 写入长期记忆。
- 对长期记忆做 top-k 保留和低分剪枝。

### 7.6 Resume 和一致性校验

生产路径支持通过 run id 恢复。恢复时会校验：

- config hash
- user prompt corpus hash
- prompt pair corpus hash
- sample corpus hash
- prior stage manifest

如果当前配置、prompt corpus 或样本集发生漂移，会在 dispatch/copy-back 前终止，避免污染已有 run。

相关模块：

```text
system_prompt_retrieval_agent/resume_from_run_id.py
system_prompt_retrieval_agent/survivor_resume_v022.py
system_prompt_retrieval_agent/barrier_v022.py
```

### 7.7 Copy-back 和 artifact 管理

远端输出会通过 copy-back 逻辑同步到本地输出目录，并校验 required files。相关模块：

```text
system_prompt_retrieval_agent/copy_back_v022.py
system_prompt_retrieval_agent/remote/
```

常见产物包括：

- stage manifest
- input manifest
- generated images
- intermediate prompts
- qwen outputs
- local API eval results
- aggregate scores
- best pair

### 7.8 MCP 工具设计

`System-Prompt-Retrieval-Agent/mcp_tools/README.md` 定义了 8 个候选 MCP 工具：

- `remote_prompt_pair_pipeline`
- `remote_gemma_stage`
- `remote_flux_stage`
- `remote_qwen_stage`
- `artifact_manager`
- `memory_manager`
- `evaluation_runner`
- `score_aggregator`

当前该目录只包含设计文档和 JSON Schema，不包含 MCP server 实现。Python 模块仍是运行时契约的 source of truth。

## 8. 输出产物与日志

### 本地代理输出

由 `paths.output_root` 控制，默认示例是：

```text
System-Prompt-Retrieval-Agent/outputs/v02
```

重点关注：

- `best_pair.yaml`：当前最佳 prompt pair。
- run/round 目录：每轮生成、评估、聚合结果。
- manifest 文件：恢复和一致性校验依据。
- tracing/logging 文件：排查失败原因。

### 远端服务输出

由远端配置 `paths.output_root` 控制，默认示例是：

```text
/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote/outputs/v01
```

远端日志：

```text
Image-Generater-Remote/logs/supervisor_17610.log
Image-Generater-Remote/logs/supervisor_17611.log
Image-Generater-Remote/logs/supervisor_17612.log
Image-Generater-Remote/logs/ports_manifest.json
```

## 9. 测试与验收

### 本地代理测试

```bash
cd System-Prompt-Retrieval-Agent
source .venv/bin/activate
pytest
```

重点测试覆盖：

- CLI
- config 加载和校验
- remote client
- memory manager
- scoring
- resume
- V0.2.2 runner
- visualization
- MCP schema 映射

验证 MCP schema：

```bash
.venv/bin/python scripts/verify_mcp_schemas.py
```

### 远端服务测试

```bash
cd Image-Generater-Remote
source .venv/bin/activate
pytest
```

模型 smoke test：

```bash
python scripts/smoke_gemma.py --config config.yaml
python scripts/smoke_flux.py --config config.yaml
python scripts/smoke_qwen.py --config config.yaml
```

模型资产检查：

```bash
python scripts/check_model_assets.py --config config.yaml
```

### 全链路验收建议

新接手或改动配置后，按以下顺序验收：

1. 本地 `pytest` 通过。
2. MCP schema 校验通过。
3. 远端 `check_model_assets.py` 通过。
4. 三个 supervisor `/health` 正常。
5. Gemma、FLUX、Qwen smoke test 分别通过。
6. 用 `--limit 5 --max-rounds 1` 跑一轮本地代理。
7. 检查 output、manifest、score、best pair 是否生成。

## 10. 常见问题与排障

### Git 报 dubious ownership

共享目录上可能出现：

```text
fatal: detected dubious ownership in repository
```

在自己的机器上执行：

```bash
git config --global --add safe.directory /path/to/system-prompt-retrieval-agent
```

### 远端端口不通

检查：

```bash
ssh 3h100
ps aux | grep server.app
curl http://127.0.0.1:17610/health
curl http://127.0.0.1:17611/health
curl http://127.0.0.1:17612/health
tail -f Image-Generater-Remote/logs/supervisor_17610.log
```

如需重启：

```bash
bash scripts/stop_hosts.sh
bash scripts/start_hosts.sh
```

### GPU 显存不足

先查看：

```bash
nvidia-smi
```

`start_hosts.sh` 会尝试清理 `NoGPUAlarmNew.py`。如果仍然 OOM，检查是否有残留 worker、vLLM 或 diffusers 进程。

### 模型路径错误

运行：

```bash
python scripts/check_model_assets.py --config config.yaml
```

不要在代码里硬编码新模型路径，也不要让程序自动下载模型。应先确认模型资产已由人工上传，再更新配置。

### 本地代理无法连接 controller

确认 tunnel：

```bash
ssh -N -L 17700:127.0.0.1:17700 3h100
```

另开终端检查：

```bash
curl http://127.0.0.1:17700/health
```

再确认 `config.yaml`：

```yaml
remote:
  controller_base_url: http://127.0.0.1:17700
```

### resume 失败

常见原因：

- 当前 config 和历史 run 的 hash 不一致。
- user prompt corpus 变化。
- prompt pair corpus 变化。
- sample manifest 变化。
- 历史 run 目录缺少必要 stage manifest。

处理方式是先确认是否真的要续跑旧 run。如果输入或配置已经变化，建议新开 run，不要强行复用旧 run id。

### API 限流或预算中断

检查：

```yaml
rate_limits:
  requests_per_second: 3
  requests_per_minute: 120

budget:
  daily_usd_cap: 50.0
  per_round_usd_cap: 10.0
```

不能绕过 `rate_limiter.py`。如果需要提高限额，先确认 API 配额和成本预算。

## 11. 交接注意事项

- 代码仓库以 `master` 为当前主分支。
- `config.yaml`、`.env`、模型权重、outputs、logs 都不应提交。
- 远端部署必须先 dry-run，再 `--apply`。
- SSH/rsync 使用 alias `3h100`，不要直接写裸 IP。
- 本地代理和远端服务职责要分开：本地目录改本地代理，远端目录改 GPU 服务，不要混写。
- `System-Prompt-Retrieval-Agent/mcp_tools/` 是 schema/design reference，不代表 MCP server 已实现。
- 配置里的路径大多带有历史机器路径，新接手时第一步应该核对本地路径、远端路径、数据集路径、模型路径。
- 生产运行前先用小样本 `--limit 5 --max-rounds 1` 验证链路。
- OpenAI API 调用成本受预算配置保护，跑全量前确认预算。
- 如需改模型、数据集、评分权重或 prompt corpus，应同步更新配置模板和相关说明，避免后续恢复时出现 hash drift。
