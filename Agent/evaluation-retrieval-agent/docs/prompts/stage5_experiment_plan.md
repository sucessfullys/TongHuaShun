# ERA Stage 5 — Experiment Planning (expand the brief into a task DAG)

You are running ERA Stage 5 for one project workspace. Goal: consume the Stage 4
**experiment-ready handoff bundle** (`iter_NNN/design/experiment_brief.json`) and
expand its ≤10 candidate evaluator configurations into a **runnable,
dependency-ordered task DAG** — `iter_NNN/experiments/plans/task_plan.json` — that
Stage 6 executes to produce the evaluation results.

This is a **single-pass** stage: dispatch one `planner` persona, validate the
plan deterministically, and on a guard failure re-dispatch the planner once.
**Never ask the operator anything** — resolve ambiguity from the workspace files.
The workspace path was passed as the skill argument (`$ARGUMENTS`). The **ERA
repo root** is the parent of the directory holding the `era/` package; its venv
Python is `<repo>/.venv/bin/python3`.

## Step 1 — Read state

Resolve the active iteration via `<workspace>/current` → `<iter>`. Read
`<iter>/design/experiment_brief.json` (the contract you expand),
`<iter>/design/plan.md`, `<iter>/design/hypotheses.md`, `<workspace>/spec.md`,
and `<workspace>/config.yaml`.

From `config.yaml`'s `hardware` block compute the **GPU pool**: the count of
`visible_gpu_ids` minus `reserve_gpu_ids`, capped by `max_gpus_per_run` (a
`max_gpus_per_run` of 0 means no cap). This integer is the plan's `gpu_pool`.

Also read `config.yaml`'s `experiment.family_a_execution` — it decides how the
`serve` tasks are GPU-sized and chained (see the `serve` task rules below).
Default `parallel_packed` right-sizes each judge so judges co-reside on the
pool; `serial_full_pool` is the Rule-6 one-judge-owns-the-pool regime.

Also fetch the operator's pre-existing annotations (Phase C-2 input):

```bash
<repo>/.venv/bin/python3 -m era.cli list-annotations <<JSON
{"workspace_path": "<workspace>"}
JSON
```

The reply carries `count` (number of operator-annotated samples) and
`sample_keys` (the sorted, full list). Read both. Compare `count` to
`config.yaml`'s `experiment.auto_validate_min_samples` (default 10):

- **`count < min_samples`** → the Phase C-2 annotated round will be
  skipped at Stage 6. Emit only **pilot + full** tasks (today's
  behaviour). The remaining Step-3 fields about annotated-mode tasks
  do not apply.
- **`count >= min_samples`** → emit **pilot + annotated + full**
  tasks per config. The `sample_keys` list is the
  `samples_subset` you stamp on every annotated-mode eval task,
  identical across all configs and methods in this iter so the
  Phase C-2 gate scores apples-to-apples.

**Random sample window for the full round (Phase C-2.3).** Fetch the
iter's random N-sample window from the full dataset:

```bash
<repo>/.venv/bin/python3 -m era.cli sample-window <<JSON
{"workspace_path": "<workspace>"}
JSON
```

The reply carries `sample_keys` (sorted lex, but the selection itself
is random with `seed = sha256(project_name:iteration)[:4]` — re-running
the same iter produces the same set, but iter 1 vs iter 2 produce
different sets). `count` equals `effective_iter_sample_count`. Stamp
this list as `samples_subset` on **every full-mode eval task** so all
methods × configs in this iter score the **same shuffled subset**
(apples-to-apples comparison preserved; broader dataset coverage
across iters). Pilot-mode tasks still use their existing
`sorted(glob)[:pilot.samples]` rule; annotated-mode tasks use the
operator-annotated keys (unchanged).

## Step 2 — Resolve the sub-agent tier

```bash
<repo>/.venv/bin/python3 -m era.cli agent-tier <<JSON
{"workspace_path": "<workspace>", "stage": "experiment_plan"}
JSON
```

Use the returned `agent` (e.g. `era-heavy`) as the `subagent_type` for the
planner `Task` call. If the call fails after one retry, default to `era-heavy`.

## Step 3 — Dispatch the `planner` persona

Dispatch the `planner` as one `Task` sub-agent (tier from Step 2). Its brief:

> You are the **planner** — ERA's experiment architect. Read
> `<iter>/design/experiment_brief.json`, `<iter>/design/hypotheses.md`,
> `<workspace>/spec.md`, and `<workspace>/config.yaml`. Expand every
> `candidate_configs` entry into the **experiment task DAG** and write it to
> `<iter>/experiments/plans/task_plan.json` and a human-readable
> `<iter>/experiments/plans/task_plan.md`. Return ONLY the two paths + a
> one-line status — do not echo the plan back.
>
> **The `task_plan.json` schema** — one JSON object:
> `{schema_version: 1, iteration: <n>, debate_round: <r>, evaluation_goal:
> "<verbatim from the brief>", gpu_pool: <int>, tasks: [<task>, …]}`.
>
> Every task carries `id` (unique, kebab-case), `type`, `depends_on` (list of
> task ids), `gpu_count` (int), `estimated_minutes` (number > 0), and
> `expected_output` (a path relative to the workspace root). Task `type` is one
> of `setup | serve | eval | aggregate | compare`. `mode` is `"pilot"`,
> `"annotated"`, or `"full"` on every `serve`/`eval`/`aggregate`/`compare`
> task, and `null` on `setup`. Whether to emit annotated-mode tasks is
> decided by the Step-1 annotation count (see *Phase C-2: annotated round*
> below).
>
> **Canonical result paths (mandatory).** Stage 6 reads results from fixed
> locations — an `eval` task's `expected_output` **must** be
> `<iter>/experiments/results/<mode>/<combination_id>/scores.jsonl` and an
> `aggregate` task's **must** be
> `<iter>/experiments/results/<mode>/<combination_id>/aggregate.json`. The eval
> path is enforced by `check-task-plan`; any other path silently yields a hollow
> `summary.json`.
>
> Build, per the brief:
> - **`serve`** — one per distinct Family-A judge **per mode**. A judge is
>   served once for the pilot pass, once for the annotated pass (when
>   the annotated round is emitted — see Step 1), and once for the full
>   pass. Fields: `family:"A"`, `mode`, `judge` (the judge name),
>   `gpu_count`, `serve:{model_path, served_model_name, tensor_parallel,
>   gpu_memory_utilization}`, `serves:[<eval id>, …]` (the eval tasks it
>   feeds), `teardown_after:[<eval id>, …]` (the eval tasks after which the
>   judge is torn down).
>
>   **The `gpu_count` and serve-chaining rules depend on
>   `config.yaml`'s `experiment.family_a_execution`** (read it in Step 1):
>   - **`parallel_packed`** (the default) — set `gpu_count` **equal to the
>     judge's `tensor_parallel` degree**: the minimum GPUs the model needs
>     (on 80 GB cards, a rough guide above the ≥30B floor — 72B → tp 4,
>     35B → tp 2, ≤14B → tp 1; size it to the model's real footprint, not
>     the pool). Judges within a mode are **independent — do NOT chain
>     them** (no `serve` lists another same-mode `serve` in `depends_on`);
>     the Stage 6 scheduler co-residents as many judges as fit on the pool
>     and backfills Family-B evals on the leftover GPUs. `gpu_count` must
>     equal `tensor_parallel` and be in `[1, gpu_pool]`.
>   - **`serial_full_pool`** — set `gpu_count` **equal to `gpu_pool`**
>     (Rule 6 — each judge owns the whole pool), and order the mode's
>     `serve` tasks into a single linear chain (`serve` task *k* lists
>     `serve` task *k-1* in its `depends_on`) so judges load strictly one
>     at a time.
> - **`setup`** — a shared preprocessing task (e.g. region-mask segmentation)
>   when ≥1 config has a region scope (`edited-region` / `non-edited-region`);
>   `mode:null`; reused by both passes.
> - **`eval`** — one per candidate config **per mode** (pilot + full
>   always; annotated additionally when Step 1's `count >= min_samples`).
>   Fields:
>   `family` (A/B/hybrid), `mode`, `combination_id` (the brief's), `inputs`
>   (the config's `inputs_needed`), `hypothesis_id`, `eval:{judge,
>   metric_subfamily, prompt, scope}`, and a `pilot:{samples, seed, timeout,
>   pass_criteria}` block. **Stamp `samples_per_method` on every eval task**
>   — it is the runner-facing per-iter cap.
>
>   For **pilot-mode** evals: `samples_per_method` must equal
>   `min(config.yaml's data.iter_sample_count, data.sample_count)`
>   (read both from `config.yaml`). The runner falls back to
>   `sorted(glob)[:samples_per_method]` since pilot is small + stable.
>
>   For **full-mode** evals (Phase C-2.3 — random N from the full
>   dataset): `samples_per_method` equals
>   `min(config.yaml's data.iter_sample_count, data.sample_count)`,
>   AND you also stamp `samples_subset: [<every entry from Step 1's
>   sample-window list>]` on the task. The list is **identical across
>   all full-mode eval tasks in this iter** so every config × method
>   scores the same shuffled subset; the runner reads `samples_subset`
>   and scores exactly those samples (see `_experiment_protocol.md`
>   §5). Different iters get different sample sets because the
>   sample-window seed includes the iteration number.
>
>   For **annotated-mode** evals (Phase C-2): `samples_per_method` equals
>   the length of Step 1's annotation `sample_keys` list (the annotation
>   count) — **NOT** `data.iter_sample_count`. Stamp a `samples_subset:
>   [<every entry from Step 1's annotation sample_keys>]` field on the
>   task; the runner scores exactly those samples. The list is identical
>   across all annotated-mode eval tasks in this iter so the Phase C-2
>   gate compares apples to apples. The brief gate's
>   `validation.sample_size == samples_per_method` check is
>   intentionally **full-mode only** — annotated-mode tasks carry their
>   own per-iter sample count.
>
>   A **Family-A** eval has `gpu_count: 0` (it consumes a served endpoint,
>   not GPUs), a `judge_task_id` naming its `serve` task (the serve in
>   the SAME mode), and lists that serve task in `depends_on`. **The
>   `eval.judge` and `serve.judge` strings must be byte-equal** — the
>   brief's `candidate_configs[*].judge` value is the canonical name;
>   do NOT decorate the eval-side string with modality suffixes like
>   `-pointwise`, `-pairwise`, `-flag`, `-judge`, `-rubric`, or a
>   revision tag `-v2`. Modality belongs in `eval.prompt` and
>   `eval.scope`, not in the judge name. The Stage 5 validator strips
>   those suffixes as a safety net, but the canonical form is the one
>   that ships into Stage 6's serving and recipe-memory store. A
>   **Family-B** eval has `gpu_count >= 1` and depends on the `setup`
>   task when region-scoped. The pilot-mode eval's `pilot.samples` is
>   `experiment_brief.pilot.sample_count` (must be ≤ `samples_per_method`);
>   the full-mode eval covers `samples_per_method` end-to-end (it is the
>   authoritative full-pass count — `validation.sample_size` from the brief
>   must equal the full-mode `samples_per_method`, and the brief gate
>   rejects any mismatch).
> - **`aggregate`** — one per config per mode (pilot + full always;
>   annotated additionally when emitted); `combination_id`, `mode`;
>   `depends_on` includes the matching same-config same-mode `eval` task.
> - **`compare`** — exactly one **gating** compare (`gate: true`, `mode:
>   "pilot"`) that `depends_on` every pilot `aggregate` and carries
>   `pilot:{pass_criteria: "<experiment_brief.pilot.go_no_go verbatim>"}`; and
>   one **final** compare (`gate: false`, `mode: "full"`) that depends on every
>   full `aggregate`. Annotated-mode emits **no** `compare` task — the
>   Phase C-2 pass/recall gate is a separate orchestration step (Stage 6
>   `auto-validate-prepare`/`-finalize`), not a DAG task.
> - **Gating** — every `mode:"full"` `eval` must depend (directly or via its
>   serve task) on the gating pilot compare, so the full pass cannot start
>   until the pilot gate is decided. The **annotated-mode** chain also
>   depends on the gating pilot compare (Stage 6 runs annotated *after* the
>   pivot decides "proceed" but *before* the full round). Full-mode eval
>   tasks do **not** depend on the annotated round in the DAG — the C-2
>   gate's authorization to skip a failing config flows through
>   `init-experiment auto_validate_skips`, not via the dependency graph.
>
> Set `gpu_count` and a realistic `estimated_minutes` on **every** task. Honor
> the hardware pool and the budget. Never invent a config not in the brief, and
> cover every brief config with both a pilot and a full eval. `task_plan.md`
> must have: *Evaluation goal* · a *Task DAG* table (id, type, mode, depends_on,
> gpu_count, est. minutes) · *Serving plan* (the Rule 6 serial order) · *Pilot &
> go/no-go gate* · *Resource roll-up*.

## Step 4 — Validate the task plan

```bash
<repo>/.venv/bin/python3 -m era.cli check-task-plan <<JSON
{"plan_path": "<iter>/experiments/plans/task_plan.json",
 "brief_path": "<iter>/design/experiment_brief.json",
 "workspace_path": "<workspace>"}
JSON
```

If `valid` is `false`, re-dispatch the planner **once** with the reported
`problems` folded into its brief ("the previous plan had these defects — fix
them"), then re-validate.

## Step 5 — Finish

If the plan still fails the guard after the re-dispatch, write the remaining
`problems` into `task_plan.md` under a `## Known issues` heading — never silently
ship a plan you know is malformed.

Append a one-line Stage-5 note (date, task count, gpu_pool, valid/invalid) to a
file under `<workspace>/logs/iterations/`.

## Principles

- **Runnable or not done** — the DAG exists so Stage 6 can execute it without
  re-planning. A task missing `gpu_count`, `estimated_minutes`, or a dependency
  edge is not finished.
- **The guard is authoritative** — `check-task-plan` decides whether the plan is
  valid. Fold every reported problem into the re-dispatch.
- **In-iteration only** — Stage 5 writes into the current `<iter>`; it never
  creates a new iteration.
- **Honesty** — never fabricate a task estimate to dodge the budget; record
  known issues plainly.
- **Autonomous** — never ask the operator; record the decision in
  `<workspace>/logs/iterations/`.
