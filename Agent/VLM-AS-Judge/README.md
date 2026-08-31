# VLM-AS-Judge / GEditBench-v2 评测代码交接说明

本目录只整理代码、配置和脚本文件，不包含模型权重、数据集图片、候选结果图片、运行日志和评测输出大文件。

## 目录结构

```text
/mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/
├── image-agent/          # 生成候选图片的代码，对应 generate_for_benchmark_multi_gpu_new.py
├── GEditBench_v2_eval/   # GEditBench-v2 pairwise judge 评测代码，对应 eval_new.sh / eval.sh
├── GEditBench_v2_elo/    # ELO 子任务统计和 HTML/MD 后处理代码
└── README.md             # 当前说明文件
```

## 原始代码来源

```text
image-agent 原始路径：
/mnt/image-edit/datasets/dingbaojin/project/tmp_nizuxuan/image-agent

GEditBench_v2_eval 原始路径：
/mnt/image-edit/datasets/dingjianbiao/agent/benchmark/GEditBench_v2

GEditBench_v2_elo 原始路径：
/mnt/image-edit/datasets/dingbaojin/project/tmp_nizuxuan/benchmark/GEditBench_v2
```

## 外部数据和模型路径

数据集：

```text
/mnt/zixuan_workspace/caption_scripts/vllm_caption_gemma/caption_splits/all_id_1person_caption_end.jsonl
```

Qwen-Image-Edit-2511 模型：

```text
/mnt/image-edit/datasets/dingbaojin/models/Qwen/Qwen-Image-Edit-2511
```

生成评测工程运行环境：

```text
/mnt/image-edit/datasets/dingbaojin/conda_envs/IE_agent_custom
```

注意：这些外部数据和模型没有复制到本目录，需要在运行机器上保持可访问。

## 1. 生成候选图片

原始运行目录：

```text
/mnt/image-edit/datasets/dingbaojin/project/tmp_nizuxuan/image-agent
```

交接代码目录：

```text
/mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/image-agent
```

运行命令：

```bash
cd /mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/image-agent
conda activate /mnt/image-edit/datasets/dingbaojin/conda_envs/IE_agent_custom

python generate_for_benchmark_multi_gpu_new.py \
  --model-name Qwen_Image_Edit_2511 \
  --model-path /mnt/image-edit/datasets/dingbaojin/models/Qwen/Qwen-Image-Edit-2511 \
  --gpus 0-7
```

生成结果默认包括：

```text
GEditBench-v2-CandidatesGallery/<output-model-name>/
GEditBench-v2-CandidatesGallery/<output-model-name>_generation_results.jsonl
GEditBench-v2-CandidatesGallery/metadata_时间戳.jsonl
```

其中 `metadata_时间戳.jsonl` 后续会作为 GEditBench-v2 评测清单使用。

## 2. 跑 pairwise judge 评测

原始脚本：

```text
/mnt/image-edit/datasets/dingjianbiao/agent/benchmark/GEditBench_v2/eval_new.sh
```

交接目录：

```text
/mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/GEditBench_v2_eval
```

运行前需要在脚本中确认或修改：

```bash
GEDITV2_METADATA_FILE="$BASE_DIR/datasets/GEditBench-v2-CandidatesGallery/metadata_20260604_030607.jsonl"
SAVE_PATH="$BASE_DIR/data/e_geditv2_pair_res_qwen_image_edit_2511_0603"
```

运行命令：

```bash
cd /mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/GEditBench_v2_eval
bash eval_new.sh
```

如果当前目录只有 `eval.sh`，则运行：

```bash
cd /mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/GEditBench_v2_eval
bash eval.sh
```

裁判可见的比较模型由下面配置决定：

```text
/mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/GEditBench_v2_eval/configs/datasets/bmk.json
```

原始路径对应：

```text
/mnt/image-edit/datasets/dingjianbiao/agent/benchmark/GEditBench_v2/configs/datasets/bmk.json
```

## 3. 跑 ELO 子任务统计

原始脚本：

```text
/mnt/image-edit/datasets/dingbaojin/project/tmp_nizuxuan/benchmark/GEditBench_v2/scripts/elo_subtask_new.sh
```

交接脚本：

```text
/mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/GEditBench_v2_elo/scripts/elo_subtask_new.sh
```

运行前需要确认脚本中的结果目录，例如：

```bash
RESULTS_ROOT="/mnt/image-edit/datasets/dingjianbiao/agent/benchmark/GEditBench_v2/data/e_geditv2_pair_res_firered_image_edit_0603/openedit"
--table-output "${REPO_ROOT}/tmp_elo_subtask_firered_image_edit_table.html"
```

运行命令：

```bash
cd /mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/GEditBench_v2_elo
bash scripts/elo_subtask_new.sh
```

## 4. ELO HTML 转 Markdown

原始脚本：

```text
/mnt/image-edit/datasets/dingbaojin/project/tmp_nizuxuan/benchmark/GEditBench_v2/process_elo_subtask_html.py
```

交接脚本：

```text
/mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/GEditBench_v2_elo/process_elo_subtask_html.py
```

运行命令示例：

```bash
cd /mnt/image-edit/datasets/duanyufa/Agent/VLM-AS-Judge/GEditBench_v2_elo

python process_elo_subtask_html.py \
  --html tmp_elo_subtask_firered_image_edit_table.html \
  --model FireRed_Image_Edit \
  --output tmp_elo_subtask_firered_image_edit_table.md
```

## 本次复制排除内容

为了保证交接目录轻量、只包含代码，本次复制时排除了：

```text
.git
__pycache__
.venv
node_modules
GEditBench-v2-CandidatesGallery
outputs / output / runs / logs / tmp / data / datasets
models / checkpoints / weights / merged_PVC_Judge
*.safetensors / *.pt / *.pth / *.ckpt / *.bin / *.onnx / *.engine
*.zip / *.tar / *.tar.gz / *.tgz / *.parquet / *.jsonl / *.csv / *.tsv
图片、视频和 numpy 数据文件
```

因此，如果要完整跑通评测，需要确保外部模型、数据集、候选图片目录和评测结果目录仍然存在，或按脚本中的路径重新生成。
