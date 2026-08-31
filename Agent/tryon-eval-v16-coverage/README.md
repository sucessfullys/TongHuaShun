# Standalone try-on evaluator — v16-coverage

> **`hybrid-editscope-zoomgate-122b-v16-coverage-1read`** — the champion
> evaluator. One consolidated *coverage* read on the reference-anchored
> edit-scope spine (Qwen3.5-122B-A10B judge).
>
> Full-dataset vs operator marks: **90.9% agreement · precision 0.818 ·
> recall 0.847 · accuracy 0.908**.

This is a fully self-contained, single-method virtual try-on evaluator. It
emits, per generated image, a plain **PASS / NOT PASS + reason** verdict. It has
**no dependency on the `evaluation-retrieval-agent` (ERA) repo** — the byte-frozen
scoring recipe, the judge launcher, and the review web app all live under this
directory and run with any `python3` ≥ 3.9.

## What this evaluator does

For each (sample, try-on method) it runs the byte-frozen iter_038
`score_v16_coverage` recipe against a Qwen3.5-122B-A10B VLM judge:

- a reference-anchored **edit-scope spine** (did a non-target region change?)
- one **consolidated coverage read** over the full spec surface — cloth
  preservation (color / pattern / belt-cuffs / material), model preservation
  (pose / underlayer), and visual quality — in a single judge call
- report-only target-region color/DINO metrics for corroboration (they never
  gate the verdict, so they run on CPU and degrade gracefully if absent).

The judge's fused `defect` flag becomes the verdict: **no defect → PASS**,
**defect → NOT PASS** with a plain-English reason (e.g. *edit-scope: a non-target
region changed*, *target garment not faithfully reproduced*, *color changed*).

## Setup — create this method's virtualenv

Each package owns its **own** `.venv` (never shared). One command builds it:

```bash
cd deliver/v16-coverage
bash setup.sh            # -> creates deliver/v16-coverage/.venv + installs deps
```

`setup.sh` runs `python3 run.py --bootstrap-only`, which creates
`v16-coverage/.venv` via `python -m venv --system-site-packages`, hashes
`requirements*.txt` into a `.bootstrapped` sentinel (re-runs are fast no-ops),
and `pip install`s into it. The venv inherits system site-packages, so
already-present `torch` / `vllm` / `numpy` are not re-downloaded. You can skip
`setup.sh` entirely — any `python3 run.py …` auto-creates and re-execs inside the
venv on first use. `--light` skips `requirements-metrics.txt` (torch/torchvision/
timm) for a judge-only venv (metrics are report-only here, so verdicts still hold).

## Quick start

```bash
# score the full configured dataset (CES ces_old + WHQ whq_new), serve the judge
python3 run.py

# reuse an already-running judge instead of launching vLLM
python3 run.py --endpoint http://127.0.0.1:8011/v1 --served Qwen3.5-122B-A10B
```

The run serves (or attaches to) the judge on 4 free H100s, scores every
(sample, method), writes the outputs, then opens the review web app (unless
`--no-webapp`).

## Output

Everything lands under `runs/<TIMESTAMP>/`:

- **`REPORT.md`** — the primary human-readable deliverable: a pass-rate summary
  table per try-on method, then a per-sample list of `PASS` / `NOT PASS — <reason>`.
- **`verdicts.jsonl`** — one machine row per result:
  `{sample_key, method_id, dataset_id, verdict:"PASS"|"NOT PASS", reason, defect_modes}`.
- **`results/full/<combination_id>/scores.<dataset_id>.jsonl`** — the raw
  per-sample scorer rows (`score`, `sub_scores`, `ok`).
- `human/review_model.json` + `detection.json` — feed the optional web app.

A per-sample scoring failure surfaces as an `ERROR` row (never a fabricated
verdict) so real problems are visible.

## What gets auto-detected

- **Garment category** (upper / lower / dress / two-piece-suit) is derived from
  the **image** inside the judge read — never from the file path — so it is
  correct on datasets whose folders are bare timestamps.
- **Judge weights** are auto-scanned (`/dev/shm/models`, then
  `/mnt/image-edit/models/Qwen/Qwen3.5-122B-A10B`); override with `--model-path`.
- **Free GPUs** are auto-picked (4 cards) after killing the GPU watchdog; the
  watchdog is restarted after the run.

## Useful flags

| flag | effect |
|------|--------|
| `--endpoint URL [--served NAME]` | attach to a running judge (no vLLM launch) |
| `--dataset {ces_old,whq_new,all}` | which configured dataset to score (default all) |
| `--limit N` | score only the first N samples **per method** (quick smoke) |
| `--detect-only` | print per-dataset sample counts and exit (no GPU) |
| `--no-webapp` | stop after REPORT.md / verdicts.jsonl (headless) |
| `--light` | skip torch/timm — pure-VLM judging (verdicts unchanged) |
| `--port N` | vLLM serve port when launching (default 8741) |
| `--review-only --run runs/<ts>` | re-open the web app for a finished run |

## Layout

```
run.py                 single entrypoint (venv bootstrap + orchestration)
config.yaml            trimmed dataset config (ces_old + whq_new, absolute paths)
setup.sh               one command to build <pkg>/.venv
requirements.txt       fastapi, uvicorn, numpy, Pillow, PyYAML
requirements-metrics.txt  torch, torchvision, timm (optional; --light skips)
scorer/                byte-frozen iter_038 closure + era_eval_common.py (patched)
                       + serve_judge.py (vLLM launcher)
evaluator/             gpu.py, verdicts.py, review_model.py
webapp/                FastAPI + built React static (optional review UI)
runs/<ts>/             REPORT.md, verdicts.jsonl, results/, human/, logs/
```

## Running on a different dataset

Point `config.yaml`'s `data.datasets[].methods[].path` / `output_file` /
`input_roles` at your data and re-run. Inputs (`input_cloth.png`,
`input_model.png`) are expected inside each method's `<sample_key>/` dir
alongside the generated `output_file` (the co-located layout the config
describes). The scorer resolves paths only from `config.yaml` — there is no repo
coupling.
