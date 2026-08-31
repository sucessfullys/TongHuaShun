# ERA Stage 0 — Task Init (behavioral prompt)

You are running `/era:init`. Your job: turn the operator's mission into a fully
scaffolded ERA project workspace. Work through the steps **in order**. Do not
skip the probe. Do not fabricate data — every value comes from a probe result
or an explicit operator answer.

**Working directory:** the ERA repo root (contains `plugin/`, `era/`, `docs/`).
All Python is `.venv/bin/python3 -m era.cli ...` run from there.

The operator's initial mission text was passed as the command argument.

---

## Step 0 — Banner

Print a short banner:

```
======================================================================
  ERA  ·  Task Init  (Stage 0)
======================================================================
```

## Step 1 — Capture the mission

Take the command argument as the mission. If it is empty, ask the operator:
"Describe the evaluation task: what was generated/edited, and where is the
data?" Do not continue without a mission.

## Step 2 — Extract environment paths from the mission

Read the mission text and extract, as best you can:

- **data roots** — directory paths holding input and/or generated images;
- **model root** — directory of local pretrained checkpoints;
- **GPU ids** — which GPUs ERA may use (expand ranges, e.g. "4-7" → 4,5,6,7);
- **env file** — path to a `.env` with API keys (default: `.env` in the ERA
  repo root if present).

For anything the mission does not state, ask the operator. Do not guess paths.

## Step 3 — Probe the environment

Use the **Write tool** to create `.era_probe_in.json` in the ERA repo root
(it is git-ignored):

```json
{
  "gpu_ids": [4, 5, 6, 7],
  "data_roots": ["/abs/path/to/data_root"],
  "model_root": "/abs/path/to/model_root",
  "env_file": "/abs/path/to/.env"
}
```

Then run, from the ERA repo root:

```bash
.venv/bin/python3 -m era.cli probe < .era_probe_in.json
```

Parse the JSON. It has four sections: `gpu`, `data`, `checkpoints`,
`credentials`. The **data probe recurses** — it discovers leaf sample
directories nested several levels deep and detects the "one root → N method
dirs" pattern on its own. Read `data.layout`, `data.methods` (each with
`method_id` / `path` / `output_file` / `output_candidates` / `sample_count`),
`data.sample_glob`, `data.sample_count`, `data.input_roles`, `data.sample_key`,
`data.confidence`, `data.scan`, and every `notes` / `error` field. Trust the
probe; inspect the filesystem by hand **only** when `data.confidence` is
`needs_confirmation` and a specific gap (e.g. multiple `output_candidates`)
needs resolving.

## Step 4 — Present findings and confirm ambiguous items

Show the operator a concise probe summary: GPUs found; data layout, sample
count, `sample_glob`, per-iter cap; each method with its detected
`output_file`; input roles; detected checkpoints; which API credentials are
present; **annotation evidence** if the dataset already carries pre-existing
operator annotations from a prior `/era:annotate` run (count + per-method
coverage + auto-validate thresholds).

Use the **AskUserQuestion** tool to confirm **only genuinely ambiguous items** —
skip anything the probe resolved cleanly (`data.confidence == "high"`, no
errors). Ask, as needed:

- **Per-method output file** — for any method whose `output_file` is empty or
  whose `output_candidates` lists more than one file, ask which file is the
  final generated output to evaluate.
- **Method ids** — the probe guesses ids from directory names; confirm a short
  id per method (e.g. `baseline`, `method_b`).
- **Input roles** — confirm the co-located input images if `input_roles` is
  empty or the names are unclear.
- **Layout** — if inputs are NOT co-located in each sample dir but live in a
  separate directory, set `layout: separate_input_dir` and an `input_root`.
- **GPU reservation** — any GPUs to reserve (exclude from VLM judges)?
  Default: none. And `max_gpus_per_run` (default: all visible GPUs).
- **Task family & adapter** — `generation` vs `editing`; adapter id
  (e.g. `virtual_tryon`, `object_replacement`, `style_transfer`, `generic`).
- **Budget** — API cost cap (USD) and wall-clock cap (hours). Defaults
  **$0 / 24 h** ($0 = no paid-API spend; local-served VLMs + metrics only).
- **Annotation evidence** — if the dataset already has operator
  annotations (a non-zero `data.annotations.central_count` from the
  probe), confirm whether Stage 2 should read them as
  *"operator-flagged failure modes"* to guide its brainstorm.
  Default **yes**. The probe summary reports how many samples are
  pre-annotated and the per-method coverage; this toggle just
  controls whether Stage 2 surfaces those notes to its sub-agents.
  Set false to skip the evidence path entirely (Stage 2 then proceeds
  from `literature.md` alone). Maps to `data.use_annotation_evidence`.
- **Auto-validation thresholds** — **always ask** the operator with
  `AskUserQuestion` unless their mission text in `$ARGUMENTS` already
  pins specific values for both pass-rate and recall-rate. The gate
  applies at Phase C (Stage 7's auto-validation step) — pinning the
  thresholds at init keeps the contract stable across iterations
  even if the dataset has no annotations yet. This question is
  **mandatory**, not "only when ambiguous": the thresholds determine
  the iteration loop's auto-revise behavior and are worth one
  explicit confirmation per project.

  Offer the operator four presets as a single-select
  `AskUserQuestion` (recommended preset first):

  - **Balanced — PASS ≥ 0.70, RECALL ≥ 0.60, MIN-PASSING = 3 (Recommended)** —
    defaults. Methods need clear agreement with the operator's
    annotations to slip through to Stage 8 human review; at least 3
    configs must clear pass/recall before the full N=50 round runs.
  - **Strict — PASS ≥ 0.85, RECALL ≥ 0.75, MIN-PASSING = 3** —
    only strongly aligned methods escape auto-validation. Fewer
    methods reach human review; more iterations needed before any
    method passes.
  - **Lenient — PASS ≥ 0.55, RECALL ≥ 0.45, MIN-PASSING = 2** —
    early-exploration phase; let more methods through and only
    require 2 to clear before the full round.
  - **Custom — operator types their own values** — fall through to the
    "Other" free-text path; parse as `pass_threshold` /
    `recall_threshold` floats in `[0.0, 1.0]` and `min_passing`
    as an int ≥ 1. If parsing fails or any value is out of range,
    re-ask.

  `auto_validate_min_passing` (Phase C-2.5) is the minimum number
  of configs that must clear BOTH pass/recall thresholds before
  Stage 6 runs the full N=50 round. Below this, Stage 6 auto-
  revises into the next iter (carrying the passing configs
  forward as `must_include_configs`). M=3 is the natural floor
  for meaningful Stage 8 comparison; M=1 restores the legacy
  "any 1 passes" semantics.

  `auto_validate_min_samples` (default **10**) is **not** asked
  separately — it's the safety floor below which the gate is
  skipped, and the default is sensible for the typical annotation
  pass. Operators with very small annotated sets (<10 samples) can
  lower it by editing `config.yaml` post-init.

  Maps to `experiment.auto_validate_pass_threshold` /
  `auto_validate_recall_threshold` / `auto_validate_min_passing` /
  `auto_validate_min_samples`.
- **Per-iter sample cap** — how many samples per generation method should
  be evaluated each iteration (and surfaced for Stage 8 human review)?
  Default **50** (recommend it as the first option). If the operator picks
  a value larger than the probed `data.sample_count`, the framework
  auto-clamps to the actual dataset size — so the operator may safely pick
  a large round number without knowing the exact total. Lower the cap to
  save GPU and API spend per iter; raise it to give the operator more
  samples to flag in human review. The same N samples are evaluated every
  iter (first N of the sorted `sample_glob`), so the cap is also the
  effective dataset size visible across the whole pipeline.
- **Project slug** — propose a kebab-case name; let the operator override.

If the GPU probe failed (`gpu.probe_ok == false`), ask the operator to supply
`gpu_model`, `visible_gpu_ids`, and `per_gpu_memory_gb` manually.

## Step 5 — Scaffold the workspace

Build the confirmed-params object and write it with the **Write tool** to
`.era_init_in.json` in the ERA repo root.

The shape is below. **The values are illustrative only — a generic editing
example. Fill every field from your probe results and the operator's answers,
never copy these example values.**

```json
{
  "project_name": "<kebab-case slug>",
  "mission": "<verbatim mission text>",
  "task_family": "<generation | editing>",
  "task_adapter": "<virtual_tryon | object_replacement | style_transfer | generic>",
  "hardware": {
    "gpu_model": "<from gpu probe>",
    "visible_gpu_ids": [0, 1],
    "reserve_gpu_ids": [],
    "max_gpus_per_run": 2,
    "per_gpu_memory_gb": 80.0,
    "driver_version": "<from gpu probe>",
    "cuda_version": "<from gpu probe>",
    "probe_ok": true
  },
  "checkpoints": {
    "local_model_root": "<from checkpoints probe>",
    "detected": ["<from checkpoints probe>"],
    "user_checkpoints": []
  },
  "data": {
    "data_root": "<from data probe>",
    "layout": "per_sample_dirs",
    "methods": [
      {"method_id": "baseline", "path": "/abs/path/to/baseline", "output_file": "edited.png"},
      {"method_id": "method_b", "path": "/abs/path/to/method_b", "output_file": "edited.png"}
    ],
    "sample_glob": "*",
    "sample_count": 200,
    "iter_sample_count": 50,
    "use_annotation_evidence": true,
    "input_roles": {"source": "source.png"},
    "input_root": "",
    "sample_key": "relpath"
  },
  "credentials": {
    "env_file": "<from credentials probe>",
    "openai": true,
    "anthropic": true,
    "google": true
  },
  "budget": {"api_cost_cap_usd": 0.0, "wallclock_cap_hours": 24.0},
  "experiment": {
    "auto_validate_pass_threshold": 0.70,
    "auto_validate_recall_threshold": 0.60,
    "auto_validate_min_passing": 3,
    "auto_validate_min_samples": 10
  },
  "probe": {
    "gpu": { ...the gpu probe section verbatim... },
    "data": { ...the data probe section verbatim... },
    "checkpoints": { ...the checkpoints probe section verbatim... },
    "credentials": { ...the credentials probe section verbatim... },
    "annotations": { ...the annotation probe section verbatim... }
  }
}
```

Notes:
- `serving` may be omitted — defaults (`ms-swift` + fallbacks) are applied.
- `headroom` / `safety_margin_gb` / `image_extensions` may be omitted.
- Put the **raw probe output** under `probe` so it is saved as an audit trail.
  Do not edit it — it records exactly what the probe found.
- Every `data.methods[].output_file` must be a real, confirmed filename.

Then run:

```bash
.venv/bin/python3 -m era.cli init-workspace < .era_init_in.json
```

Parse the result JSON.

- On `{"error": "workspace_exists", ...}` — ask the operator for a different
  project name and retry Step 5.
- On `{"error": "invalid_config", "problems": [...]}` — show the problems, fix
  the params with the operator, and retry.
- On success the result has `project_name`, `workspace_path`, `spec_path`,
  `config_path`, and `guide`.

## Step 6 — Print the guide

Output the value of the result's `guide` field **verbatim, as your own text
reply** — not inside a code block produced by Bash, and not summarized. This is
the operator's next-step instruction. After printing it, stop: no further
action until the operator starts a new session.

## Cleanup

Delete the scratch files `.era_probe_in.json` and `.era_init_in.json`.
