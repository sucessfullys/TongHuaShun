# Image-Generater-Remote — V0.1 Remote Model Service

Remote inference layer for the System-Prompt-Retrieval-Agent project.

## Scope

V0.1 delivers three GPU-pinned FastAPI supervisor hosts on the 3H100 server,
each managing one backend worker child process at a time. Supported models:

- **Gemma-4-31B-it** — intermediate prompt generation (vLLM)
- **FLUX.2-klein-9B** — clothing-transfer image generation (diffusers)
- **Qwen3-VL-8B-Instruct** — visual evaluation (vLLM + optional LoRA adapter)

A simulated agent (`agent_sim/run_pipeline.py`) drives the full pipeline
end-to-end on a 30-image pilot dataset.

## Quick Start

```bash
# Copy and edit config
cp config.yaml.example config.yaml

# Deploy to remote
bash scripts/sync_to_remote.sh          # dry-run by default
bash scripts/sync_to_remote.sh --apply  # real rsync

# Start supervisors on remote
ssh 3h100 'cd /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote && bash scripts/start_hosts.sh'

# Run pilot
ssh 3h100 'cd /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote && python agent_sim/run_pipeline.py --config config.yaml'
```

## Reference-Only Warning

The directory `/Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/Image-Generater-Remote`
is a **reference-only** codebase. Do not write into it or deploy from it.
All V0.1 code lives in this directory under `System-Prompt-Retrieval-Agent/Image-Generater-Remote/`.
