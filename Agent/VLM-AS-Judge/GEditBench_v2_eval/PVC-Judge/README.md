---
license: mit
base_model:
- Qwen/Qwen3-VL-8B-Instruct
---

<h1 align="center">PVC-Judge is a state-of-the-art 8B assessment model for evaluating image editing models in visual consistency.</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2603.28547"><img src="https://img.shields.io/badge/Paper-arXiv%3A2603.28547-b31b1b?logo=arxiv&logoColor=red"></a>
  <a href="https://zhangqijiang07.github.io/gedit2_web/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project%20Page-Website-8A2BE2"></a>
  <a href="https://huggingface.co/datasets/GEditBench-v2/GEditBench-v2"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-GEditBench v2-blue"></a>
  <a href="https://huggingface.co/datasets/GEditBench-v2/VCReward-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HF-VCReward Bench-blue"></a>


## 🚀 Quick Start!
### Clone github repo
```bash
git clone https://github.com/ZhangqiJiang07/GEditBench_v2.git
cd GEditBench_v2
```

### Option 1: Packaged as an online client
- Merge LoRA weights to models, required env `torch/peft/transformers`
```bash
python ./scripts/merge_lora.py \
  --base-model-path /path/to/Qwen3/VL/8B/Instruct \
  --lora-weights-path /path/to/LoRA/Weights \
  --model-save-dir /path/to/save/PVC/Judge/model
```

- Implement online server via vLLM
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/save/PVC/Judge/model \
  --served-model-name PVC-Judge \
  --tensor-parallel-size 1 \
  --mm-encoder-tp-mode data \
  --limit-mm-per-prompt.video 0 \
  --host 0.0.0.0 \
  --port 25930 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.80 \
  --max_num_seqs 32 \
  --max-model-len 48000 \
  --distributed-executor-backend mp
```

- Use `autopipeline` for inference.

See our [repo](https://github.com/ZhangqiJiang07/GEditBench_v2/tree/main) for detailed usage!


### Option 2: Offline Inference

```bash
# For local judge inference
conda env create -f environments/pvc_judge.yml
conda activate pvc_judge
# or:
python3.12 -m venv .venvs/pvc_judge
source .venvs/pvc_judge/bin/activate
python -m pip install -r environments/requirements/pvc_judge.lock.txt


# Run
bash ./scripts/local_eval.sh vc_reward
```