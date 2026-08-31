# Try-on evaluator (v16-coverage) — agent instructions

This directory is a **standalone, self-contained** virtual try-on evaluator. It
ships ONE proven method from the ERA tryon-eval project:

> **`hybrid-editscope-zoomgate-122b-v16-coverage-1read`** — the champion
> (90.9% agreement / 0.818 precision / 0.847 recall / 0.908 accuracy on the
> operator's full-dataset marks). Scorer `score_v16_coverage`, Qwen3.5-122B-A10B
> judge.

When the operator asks you to "evaluate" a set of try-on images, map their
request onto **one `python3 run.py` invocation** and run it. Everything is
self-contained — **never import or reference the `evaluation-retrieval-agent`
repo**.

## How it runs

`run.py` reads `config.yaml`'s `data.datasets[]` for the (cloth, model, result)
paths, serves (or attaches to) the judge, scores every (sample, method) via the
byte-frozen `scorer/` closure, and writes **PASS / NOT PASS + reason** to
`runs/<ts>/REPORT.md` + `verdicts.jsonl` (+ raw `scores.jsonl`).

- To evaluate a **different** dataset, edit `config.yaml`'s
  `data.datasets[].methods[].path` / `output_file` / `input_roles`. Inputs live
  inside each method's `<sample_key>/` dir (co-located layout).
- If a judge is already serving (e.g. port 8011), prefer
  `--endpoint http://127.0.0.1:8011/v1 --served Qwen3.5-122B-A10B` over launching
  a second copy.
- `--detect-only` first (fast, no GPU) to confirm sample counts.
- `--no-webapp` for a headless run; `REPORT.md` is the primary deliverable.

## Environment facts

- The judge needs 4 free H100s (~70 GB each). The GPU watchdog
  (`NoGPUAlarmNew.py`) MUST be killed before vLLM starts — `run.py` does this
  automatically (and restarts it after). If it can't be killed, run
  `sudo pkill -9 -f "python3 -u NoGPUAlarmNew.py"` and retry.
- Judge weights auto-discover at `/mnt/image-edit/models/Qwen/Qwen3.5-122B-A10B`.
- In-process metrics (DINOv2 / color-EMD) run on **CPU** (`CUDA_VISIBLE_DEVICES=""`)
  and are report-only — they never gate the verdict.
- Default serve port is **8741**.

## Iron rules

- **Never reference or import the `evaluation-retrieval-agent` repo** — this
  package is independent and must stay that way (the scoring recipe under
  `scorer/` is byte-frozen; the one severing edit is
  `scorer/era_eval_common.py::_resolve_paths` honoring `TRYON_CONFIG_YAML` /
  `TRYON_LOGS_DIR`).
- Do NOT edit the byte-frozen scorer modules (`scorer/_v16.py`,
  `scorer/_v15_reference.py`, `scorer/_editscope.py`, `scorer/_ntmetric.py`,
  `scorer/_v13_eval/*`) — they encode the calibrated recipe that reproduces
  0.908 accuracy. Change them only on explicit operator request.
- `runs/` artifacts are an audit trail; never rewrite scored rows.
- Never launch two vLLM judges concurrently without checking `nvidia-smi`.
