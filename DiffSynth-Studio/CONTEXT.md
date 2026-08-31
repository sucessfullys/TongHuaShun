# Codex Context

## Purpose

This file stores recoverable context for future Codex sessions. Update it at the end of each meaningful task so a new chat can resume quickly.

## Current Focus

- Project: `DiffSynth-Studio`
- Project root: `/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio`
- Current goal: prepare and run training using the DiffSynth-Studio codebase.

## Session Notes

- Created this context file so future work can be resumed from repository state instead of relying only on chat history.
- Created conda environment `DiffSynth` at `/root/anaconda3/envs/DiffSynth`.
- Verified `DiffSynth` uses Python 3.10.20.
- Started `conda run -n DiffSynth pip install -e .`, but stopped it after it hung on downloading `torch-2.12.0` from the Aliyun PyPI mirror.
- Current `DiffSynth` environment only has base packages: `pip`, `setuptools`, `wheel`, and `packaging`.
- Packed the `DiffSynth` conda environment to `/mnt/image-edit/datasets/duanyufa/conda_env_backup/DiffSynth.tar.gz`.
- Verified the backup archive with `gzip -t`; the archive passed integrity validation.
- User is now preparing to start training with `/mnt/image-edit/datasets/duanyufa/DiffSynth-Studio`.
- Future training setup, commands, environment changes, failures, fixes, and results should be recorded in this file.
- Later environment inspection showed `DiffSynth` now contains the core project dependencies, including `diffsynth 2.0.12`, `torch 2.12.0`, `accelerate 1.13.0`, and `transformers 5.9.0`.
- Training planning exploration identified usable training entrypoints: `examples/flux2/model_training/train.py` and `examples/qwen_image/model_training/train.py`.
- Reviewed example training scripts: `examples/flux2/model_training/lora/FLUX.2-klein-4B.sh`, `examples/qwen_image/model_training/lora/FireRed-Image-Edit-1.1.sh`, and `examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh`.

## Training Log

- 2026-06-05: Training preparation started. No training command has been finalized or run yet in this session.
- 2026-06-05: Paused before end of day. Completed pre-training planning exploration; no training was started and no training scripts were modified.

## Decisions

- Use `CONTEXT.md` as the main persistent task summary file for this project.
- At the end of each task, ask Codex to summarize progress and update this file.
- Use conda environment name `DiffSynth` for this project.

## Open Tasks

- After restoring `DiffSynth.tar.gz` on a fresh server, run `conda-unpack` inside the extracted environment before use.
- Confirm the target model; current candidates include FLUX.2 LoRA, FireRed image-edit LoRA, and Qwen image-edit LoRA.
- User currently prefers custom training data.
- First training pass should be a small smoke run before any full LoRA run.
- Confirm custom dataset path and metadata format.
- Identify the final training script, output directory, GPU/VRAM plan, and required config changes.
- Record every executed training/setup command and its result under `Training Log`.

## Resume Checklist

- Read this file first.
- Inspect the files mentioned in `Current Focus` and `Open Tasks`.
- Run relevant tests or validation commands before making further changes.
- Activate with `conda activate DiffSynth`.
- Confirm with `python --version` and `pip list`.
- Confirm custom dataset path and metadata format.
- Confirm target model, output directory, GPU/VRAM strategy, and exact training command.
- Before training, document the intended command, then run environment/data checks.
