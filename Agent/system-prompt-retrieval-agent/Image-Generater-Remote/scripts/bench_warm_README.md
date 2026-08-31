# B8 CPU-warm lifecycle benchmark

**This is an OPERATOR-RUN benchmark.**
It is NOT part of CI and MUST NOT be called by the production controller.

---

## Purpose

Measures model-load wall-clock times for Gemma-31B, FLUX-9B, and Qwen-VL-8B
across two lifecycle modes:

- **cold**: disk → GPU (full load from storage each time).
- **warm**: disk → CPU prefetch once, then CPU → GPU on demand.

The script records host RAM and GPU VRAM deltas at each transition and
produces a CSV with one row per model.  The parent then uses the decision rule
below to decide whether to flip `lifecycle_mode` from `cold` to `warm` in
`config.yaml`.

---

## How to run on 3h100

```bash
ssh 3h100

cd /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/Image-Generater-Remote

python3 scripts/bench_warm_mode.py \
    --gemma-path  /mnt/image-edit/datasets/xywang/models/gemma-3-27b-it \
    --flux-path   /mnt/image-edit/datasets/xywang/models/FLUX.1-dev \
    --qwen-path   /mnt/image-edit/datasets/xywang/models/Qwen2.5-VL-7B-Instruct \
    --gpu-ids     0,1,2 \
    --output-csv  /mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/knowledge/research/V0.2.1/cpu_warm_lifecycle_data.csv \
    --tmp-dir     /tmp/bench_warm
```

Optional flags:

| Flag | Effect |
|------|--------|
| `--skip-gemma` | Skip Gemma measurement |
| `--skip-flux`  | Skip FLUX measurement |
| `--skip-qwen`  | Skip Qwen measurement |

Estimated runtime: 30–90 minutes total (dominated by model loading).

The script writes per-server logs to `--tmp-dir` for debugging.

---

## Output CSV columns

| Column | Type | Description |
|--------|------|-------------|
| `model` | string | Model label: `Gemma-31B`, `FLUX-9B`, or `Qwen-VL-8B` |
| `cold_load_s` | float | Wall-clock seconds: disk → GPU, server ready |
| `warm_to_gpu_s` | float | Wall-clock seconds: CPU-prefetched state → GPU ready |
| `gpu_unload_s` | float | Wall-clock seconds: GPU → fully freed (SIGTERM → VRAM clear) |
| `gpu_vram_peak_gib` | float | Peak GPU VRAM allocated per GPU during load (GiB) |
| `host_ram_peak_gib` | float | Peak host RAM increase during CPU prefetch phase (GiB) |
| `recommendation` | string | `warm` or `cold (reason)` per the decision rule below |

---

## Decision rule (S10b.02)

For each model, the script applies:

```
if (warm_to_gpu_s < 0.5 * cold_load_s) AND (host_ram_peak_gib < 64):
    recommendation = "warm"
else:
    recommendation = "cold (<reason>)"
```

Interpretation:

- The 0.5 threshold means warm reload must be at least **2x faster** than a
  cold load to justify the host-RAM cost.
- The 64 GiB RAM cap ensures the 3h100 host retains headroom for OS and
  data-loader buffers (`remote.post_stage_host_ram_free_gib` default).

**Change `lifecycle_mode` to `warm` in config.yaml only when ALL three models
return `recommendation=warm`.**  Partial warm results mean the pipeline would
fall back to `disk_unloaded` for at least one stage anyway, eliminating the
benefit.

---

## Plan reference

- Plan §6.6 — Model lifecycle states (`disk_unloaded`, `cpu_prefetched`,
  `gpu_loaded`, `gpu_unloaded_cpu_retained`).
- Plan §18 / B8 — Full recommendation criteria:
  1. Per-round wall-clock saving >= 90 s.
  2. Host-RAM headroom >= 64 GiB sustained.
  3. No measurable GPU-step regression.

This script covers criteria 1 and 2.  Criterion 3 (GPU-step regression) must
be assessed separately via a pipeline smoke run after enabling warm mode.

---

## What this script does NOT do

- Does NOT modify `workflow/controller.py` or any production workflow file.
- Does NOT start or stop the production Flask controller.
- Does NOT push results anywhere automatically; the operator copies the CSV to
  `knowledge/research/V0.2.1/cpu_warm_lifecycle_data.csv` and updates the wiki.
- Does NOT run as part of `pytest` or any CI pipeline.
