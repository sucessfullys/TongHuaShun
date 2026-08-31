# ERA Stage 6 — Full Experiment (run the task DAG, collect the results)

You are running ERA Stage 6 for one project workspace. Goal: execute the Stage 5
task DAG (`iter_NNN/experiments/plans/task_plan.json`) — write each evaluator
runner, review it, schedule it on free GPUs, run it, recover failures, and
collect the per-config evaluation results.

**Never ask the operator anything** — resolve every ambiguity from the workspace
files and the task plan. The workspace path was passed as the skill argument
(`$ARGUMENTS`). The **ERA repo root** is the parent of the directory holding the
`era/` package; its venv Python is `<repo>/.venv/bin/python3`. Every `era.cli`
call below reads one JSON object from stdin (a heredoc) and prints one out.

**Duration is normal here.** A real Stage 6 run on the operator's production
dataset takes **several hours** (VLM judge warm-up + DAG walk + per-config
scoring) and issues **hundreds of tool calls** (background bash runner launches,
`wait-for-any-done` polls, `record-task` calls, occasional `heal-tick` cycles).
This is the autonomous-loop's design, not a cost to gate behind an operator
prompt. Drive the DAG to completion; log progress to
`<workspace>/logs/iterations/`; **never pause to ask the operator whether the
run "should continue"**. The PreToolUse hook in `.claude/settings.json` will
structurally block any `AskUserQuestion` call from this stage; you have no
escape hatch, by design.

## Step 1 — Read state

Resolve the active iteration via `<workspace>/current` → `<iter>`. Read
`<workspace>/config.yaml` (the `hardware`, `serving`, `budget`, and
`experiment` blocks), `<workspace>/spec.md`, `<iter>/design/experiment_brief.json`
(the `pilot`, `pivot_matrix`, `validation`, `resource_estimate` blocks), and
`<iter>/experiments/plans/task_plan.json`. Also read
`docs/prompts/_experiment_protocol.md` — the contract every runner you write
must obey.

If `config.yaml`'s `experiment.codex_reviewer` is `true`, runner review goes
through the separate Codex reviewer (Step 3c); if `false` (the default), you
self-review inline.

## Step 2 — Initialise the pilot pass

```bash
<repo>/.venv/bin/python3 -m era.cli init-experiment <<JSON
{"workspace_path": "<workspace>", "mode": "pilot"}
JSON
```

## Step 3 — Pilot DAG walk

Loop until the pilot pass is done. Each iteration:

**3a. Scan GPUs and claim a batch.** Run `nvidia-smi` via Bash, then:

```bash
<repo>/.venv/bin/python3 -m era.cli claim-batch <<JSON
{"workspace_path": "<workspace>", "mode": "pilot",
 "nvidia_smi_output": "<the nvidia-smi stdout>"}
JSON
```

This returns `batch` (tasks runnable now, each with its `gpu_ids`), `blocked`
(ready but gated — GPUs busy, or a Rule-6 judge resident under
`serial_full_pool`), `resident_judges` (every judge currently holding GPUs —
plural under `parallel_packed`, where several co-reside on disjoint subsets),
and the completed/remaining counts. Under the default `parallel_packed` mode a
single `batch` may contain **multiple `serve` tasks plus Family-B evals** — each
already carries its disjoint `gpu_ids`, so launch them all in the Step-3d
parallel pass exactly as written. If `batch` is empty and nothing is running and
`blocked` is empty, the pass is finished — leave the loop. If `batch` is empty
but tasks are running or blocked, skip to the wait step (3e) — there are
already in-flight runners to wait on.

**3b. Write all batch runners (cold pass — no GPUs are spinning yet).**
For *every* task in `batch`, write its runner / launch artifact to disk before
starting any of them — this keeps reviews on the cold path and lets the next
step launch the whole batch in a single shell pass.
- A **`serve`** task: write its judge launch command per `_experiment_protocol.md`
  §9 (backend `serving.backend`, `tensor_parallel` from the batch entry, a free
  port in `serving.port_range`) into a `<task_id>_launch.sh` script under
  `<iter>/experiments/configs/`.
- An **`eval`** task: write its runner to
  `<iter>/experiments/configs/<task_id>_runner.py` obeying every rule in
  `_experiment_protocol.md`. The runner **must** read `samples_per_method`
  from the task entry and score only the first `samples_per_method`
  lexicographically-sorted samples per method (`_experiment_protocol.md` §5)
  — never iterate the full `data_root`. The runner writes its own
  `.pid` / `.progress.json` / `.done.json` markers.
- A **`setup`** / **`aggregate`** / **`compare`** task: write its small
  script to `<iter>/experiments/configs/<task_id>.sh` (or `.py`).

**3c. Review each new runner — once per family.** Before the *first* runner of a
given evaluator family is launched, review it; later same-family runners
reuse the reviewed template, so review never blocks parallel dispatch.
- If `experiment.codex_reviewer` is `true`: dispatch the **`era-codex-reviewer`**
  sub-agent (a `Task` call) with the runner path. On `VERDICT: REVISE`, fix the
  runner and re-review — at most **twice** per family, then run it regardless and
  note the unresolved review.
- If `false`: re-read the runner yourself against the `_experiment_protocol.md`
  checklist, fix any defect in place, then launch.

**3d. Launch the whole batch in parallel (Phase D-1 saturation).**
Fire *every* runnable task in `batch` as a background bash job in **one**
Bash call so all runners start within the same shell pass — disjoint
`gpu_ids` are guaranteed by `claim_batch`'s fcntl lease, so the runners can
share the host without colliding. Pattern (substitute the per-task values
from the batch entries):

```bash
mkdir -p <iter>/experiments/logs
for entry in "<task_id>:<gpu_ids_comma_joined>" ...; do
  task_id="${entry%%:*}"
  gpus="${entry#*:}"
  nohup env CUDA_VISIBLE_DEVICES="$gpus" \
    <repo>/.venv/bin/python3 <iter>/experiments/configs/${task_id}_runner.py \
    > <iter>/experiments/logs/${task_id}.stdout 2>&1 &
  echo $! > <iter>/experiments/logs/${task_id}.pid
done
```

Notes:
- **`serve` tasks** launch the judge process detached the same way (their
  `_launch.sh` ends in `&` and writes `.pid`). When the judge's dry-probe
  succeeds (poll its `/health` endpoint), `record-task` it `success` with
  the `endpoint` — that does NOT block the parallel eval launches.
- **`setup` / `aggregate` / `compare`** tasks may be run inline if quick
  (`< 5 s`), or backgrounded the same way for consistency.
- Do **not** wait for any single task in this loop; the next step handles
  completion detection via marker files.

**3e. Wait for the first done — event-driven, not interval-driven.**
Instead of sleeping for `poll_interval_s`, ask the framework to return the
moment ANY running task writes its `done.json`:

```bash
<repo>/.venv/bin/python3 -m era.cli wait-for-any-done <<JSON
{"workspace_path": "<workspace>",
 "task_ids": [<every just-launched batch task_id>, <plus any prior tasks still running>],
 "timeout_s": <experiment.poll_interval_s>}
JSON
```

This returns within a few hundred milliseconds of the first task's
completion (or after `timeout_s` if nothing finishes — that's the upper
bound, not the polling cadence). It replaces the old fixed-interval sleep.
When it returns, proceed to monitoring (3f).

**3f. Monitor and recover.** Get the detection script and run it:

```bash
<repo>/.venv/bin/python3 -m era.cli experiment-status <<JSON
{"workspace_path": "<workspace>"}
JSON
```

Run the returned `detection_script` via Bash, then feed its output back:

```bash
<repo>/.venv/bin/python3 -m era.cli recover-experiment <<JSON
{"workspace_path": "<workspace>", "detection_output": "<detection stdout>"}
JSON
```

Then act on the result:
- `recovered_completed` — for a finished `eval` task, `record-task` it `success`
  (this auto-aggregates the config and refreshes `summary.json`); for a `serve`
  task whose `teardown_after` evals are now all done, call
  `era.cli shutdown-judge` (Phase D-5) — it runs the safe sequence
  (SIGTERM the pgid → poll-exit → SIGKILL on timeout → orphan sweep via
  `pkill -TERM -f <served_model_name>` → `nvidia-smi` memory verify →
  escalate to `sudo nvidia-smi --gpu-reset -i <gpu_id>` if a tp-worker
  left the GPU wedged → port-bound check → `release-gpus`). Raw
  `kill <pid>` leaks tensor-parallel workers + wedges GPUs and is
  forbidden:

  ```bash
  <repo>/.venv/bin/python3 -m era.cli shutdown-judge <<JSON
  {"workspace_path": "<workspace>", "task_id": "<serve-task-id>"}
  JSON
  ```

  Branch on `status`:
  - `"ok"` — graceful exit. The freed GPUs re-enter `claim-batch` (the next
    judge under Rule 6, or more co-resident judges / Family-B backfill under
    `parallel_packed`).
  - `"escalated_kill"` — needed SIGKILL but cleaned up. Log + proceed.
  - `"escalated_reset"` — GPU was wedged; `nvidia-smi --gpu-reset`
    cleared it. Log + proceed (the next serve will dry-probe before
    claiming).
  - `"still_stuck"` — GPU stayed allocated after the reset. This is
    a hardware-level issue; `record-task` the serve as
    `outcome: runtime_failed` with
    `failure_category: serving` (the existing taxonomy already accepts
    this as runtime-failure-eligible) so the heal loop's circuit
    breaker takes over.
- `retried` — the task is back to `pending`; it re-enters `claim-batch`. This
  also covers a **hung** runner: detection reports `HUNG` for a live process
  whose heartbeat (`.progress.json` mtime) went stale past
  `experiment.heartbeat_timeout_s`; `recover-experiment` kills it and retries it
  through the same bounded path — so a deadlocked runner cannot stall the loop.
- `recovered_failed` — heal it (3g).

`experiment-status` also reports `supervisor_heartbeat_at` /
`supervisor_idle_seconds` — this loop's own heartbeat. It is informational; keep
polling on cadence so it stays fresh.

After handling the recovered tasks, **return to 3a immediately** — do not
sleep. Freed GPUs should be re-claimed within a few hundred milliseconds,
not within `poll_interval_s`.

**3g. Heal a failed task.** Read the task's `.done.json` / captured stderr, then:

```bash
<repo>/.venv/bin/python3 -m era.cli heal-tick <<JSON
{"workspace_path": "<workspace>", "task_id": "<id>", "error_text": "<stderr>"}
JSON
```

- `retry_with_patch` — merge the returned `patch` into
  `<iter>/experiments/configs/<task_id>.override.json` (run `pip install` for a
  `pip_install` patch), then re-run the task.
- `escalate` — patch the runner script yourself in this context (one attempt),
  then re-run.
- `give_up` — the circuit breaker is open. Branch on `runtime_failure_eligible`
  in the give_up envelope:
  - **`runtime_failure_eligible: true`** AND the task is an `eval` →
    promote the task to **`outcome: "runtime_failed"`** by calling
    `record-task outcome=runtime_failed` with `failure_category` (the
    category from the envelope) and the full give_up envelope as
    `heal_history`. The Stage 6 completion gate treats a fully
    runtime_failed config as resolved (so the loop advances), up to a
    30 % cap of the chosen configs; above the cap the gate blocks the
    loop for operator triage. The framework re-validates the heal-tick
    circuit-breaker state on the server side — you cannot fabricate this
    outcome to silently drop a config. This is the **only** path that
    promotes a failure to resolved; do not invent any other.
  - Anything else (the eligible flag is false, the task is `serve` /
    `setup` / `aggregate` / `compare`, or the category is `import` /
    `missing_dir` / `config` / `runtime`) → leave the config `failed`,
    note it, and continue. The completion gate will surface it as a
    missing config and block the loop until the operator triages —
    that is the correct path for a planning miss or an agent bug.
  - One bad config must never sink the round mid-walk: keep claiming the
    other configs' tasks while a single failure is being routed.

**Budget guard.** Before each `claim-batch`, check elapsed wall-clock and any
accumulated API cost against `config.yaml`'s `budget`. On a breach, stop
claiming new tasks, let the running ones finish, and mark the rest as
**`outcome: "failure"`** with a budget-exceeded reason — **never** as
`skipped`. A budget breach is not a force-majeure escape; it is a planning
miss that the loop must surface (it will block at the Stage 6 completion
gate). The operator then either frees the budget and `/era:resume`, or
amends the Stage 4 brief to a smaller chosen set and re-plans.

## Step 4 — Pilot gate and the pivot matrix

When the pilot pass is done (`experiment-status` reports `all_done`), read
`<iter>/experiments/results/pilot/summary.json`. Evaluate it against
`experiment_brief.json`'s `pilot.go_no_go`, then match the outcome against the
brief's `pivot_matrix` and take that row's `action` (`proceed` /
`drop_configs:[…]` / `adjust` / `abort`). Stage 6 **executes** the pivot matrix —
it does not invent one. Record the decision and the matched action in
`<iter>/experiments/pilot_decision.json`. If the action is `abort`, write the
Stage-6 log note and stop here.

For a config dropped by a `drop_configs` pivot-matrix row, record each of its
already-claimed tasks with `era.cli record-task` `outcome: "skipped"` and
**`pivot_proof: "<the exact pivot_matrix[*].action string>"`** (e.g.
`pivot_proof: "drop_configs:hybrid-vlm-anchored-by-family-b"`). The CLI gate
will **reject** an `eval`-task skip without `pivot_proof`, or with a proof
that does not match any `experiment_brief.pivot_matrix[*].action`. A
pivot-matrix drop is the **only** legitimate Stage 6 skip path; anything
else (missing host deps, model too costly, runtime scope-reduction,
checkpoint resolution failure) is `outcome: "failure"`, which surfaces at
the Stage 6 completion gate and blocks the loop.

## Step 4.5 — Annotated round + pass/recall gate (Phase C-2)

When the dataset carries pre-existing operator annotations
(`config.yaml`'s `probe.data.annotations.central_count ≥
experiment.auto_validate_min_samples`), Stage 6 runs an **annotated
round** between the pilot and the full run. Each evaluation config
scores every method on the operator-annotated subset; a sub-agent then
decides per sample whether the method's output agrees with the
operator's free-text note. For **Family-B** configs the sub-agent is
scope-aware: a note about a dimension the metric does not measure is
marked out-of-scope (`applicable: false`) and excluded from the recall
denominator — the metric is not penalized for missing a defect it was
never designed to detect. Configs that meet `pass_threshold` AND
`recall_threshold` proceed to the full round, the rest are
auto-revised away.

Skip this step entirely when the probe reports fewer annotations than
`experiment.auto_validate_min_samples` — `auto-validate-prepare`
returns `skipped_for_min_samples: true` and every config flows through
to Step 5 as in pre-Phase-C-2 behaviour.

**4.5a-pre — Pre-flight: confirm annotated tasks exist in the plan
(Phase C-2.4).** Before initing the annotated pass, verify the task
plan actually contains annotated-mode eval tasks. A plan with **zero
annotated tasks** when the dataset HAS annotations means Stage 5
either skipped them (legacy v0.1.6 plan whose validator rejected
`mode: "annotated"`) or hit an LLM error. Running the gate against
such a plan produces a false-negative `any_passed: false` that would
auto-revise away the iter's real pilot+full work. The pre-flight:

```bash
<repo>/.venv/bin/python3 -c "
import json, pathlib
plan = json.loads(pathlib.Path('<iter>/experiments/plans/task_plan.json').read_text())
has = any(t.get('type') == 'eval' and t.get('mode') == 'annotated'
          for t in plan.get('tasks', []))
print(int(has))
"
```

Branch on the output:

- **`0` AND `data.annotations.central_count >= auto_validate_min_samples`**
  — Stage 5 didn't emit annotated tasks despite the dataset
  warranting them. Call `era.cli auto-revise` with the new
  Phase C-2.4 reason and **return** — do NOT proceed through Step
  4.5b/c/d/e:

  ```bash
  <repo>/.venv/bin/python3 -m era.cli auto-revise <<JSON
  {
    "workspace_path": "<workspace>",
    "reason": "stage5_missing_annotated_tasks",
    "source_stage": 5,
    "blocker_summary": "<count> annotations exist but task_plan.json has no annotated-mode tasks",
    "diagnostic": {
      "annotated_count": <central_count>,
      "expected_modes": ["pilot", "annotated", "full"],
      "observed_modes": <list of distinct modes in the plan>
    }
  }
  JSON
  ```

  Phase C-1's machinery scaffolds a fresh iter; Stage 9's advisor
  reads the trigger and steers Stage 2-4 to emit annotated tasks
  next time.

- **`1`** — annotated tasks are present in the plan. Proceed to
  4.5a (init the annotated pass).
- **`0` AND no annotations on disk** — the gate is legitimately
  not applicable. Skip Step 4.5 entirely and proceed to Step 5.

**4.5a — Init the annotated pass.**

```bash
<repo>/.venv/bin/python3 -m era.cli init-experiment <<JSON
{"workspace_path": "<workspace>", "mode": "annotated"}
JSON
```

**4.5b — Walk the annotated DAG.** Reuse the Step-3 DAG walk
(parallel runner dispatch, `wait-for-any-done`, recovery, heal) with
`mode: "annotated"`. The task plan's annotated-mode eval tasks carry
an explicit `samples_subset` field — the runner scores exactly those
sample_keys (see `_experiment_protocol.md` §5).

When all annotated-mode tasks are resolved (no `pending` or
`running`), proceed to the gate.

**4.5c — Build the sub-agent batches.**

```bash
<repo>/.venv/bin/python3 -m era.cli auto-validate-prepare <<JSON
{"workspace_path": "<workspace>", "mode": "annotated"}
JSON
```

Returns
`{batches: [{combination_id, method_id, input_path, output_path,
sample_count}, ...], thresholds, annotated_sample_count,
skipped_for_min_samples}`.

If `skipped_for_min_samples: true` (annotation count below the
floor), proceed directly to **Step 5** as if every config passed —
log the skip and move on.

**4.5d — Dispatch `era-auto-validator` sub-agents in parallel.**
For each batch, issue ONE `Task` call to the `era-auto-validator`
sub-agent. **Send all batch dispatches in a single message
(parallel tool calls)** so the agents run concurrently. Each agent's
brief is:

> You are `era:era-auto-validator`. Read the input batch at
> `{input_path}`, decide `agree: true|false` and `applicable: true|false`
> per sample using the sub-agent's decision rules (your agent file), and
> write the judgments JSON to `{output_path}`. One `Write`, then stop.

Each batch carries `scope_gating_enabled` + an `evaluation_target` block
describing what dimension that config measures. For Family-B batches
(`scope_gating_enabled: true`) the agent emits `applicable: false` on samples
whose operator note is about a different dimension — out-of-scope flags are
excluded from the recall denominator by `auto-validate-finalize`.

The sub-agents' work is bounded — light-tier Haiku, ≤3K input +
~1K output tokens per batch — so dispatching even 20 batches in
parallel is well under a minute and well under $0.10 per iter.

**4.5e — Aggregate judgments + branch.**

```bash
<repo>/.venv/bin/python3 -m era.cli auto-validate-finalize <<JSON
{"workspace_path": "<workspace>", "mode": "annotated"}
JSON
```

Returns the result.json shape: `per_config[*]` with `pass_rate`,
`recall_rate`, `passed`; `passing_configs`, `failing_configs`,
`passing_count`, `min_passing` (= `thresholds.min_passing`,
default 3 per `experiment.auto_validate_min_passing`), and
`any_passed: passing_count >= min_passing`. The result is also
written to `<iter>/auto_validate/result.json`.

Under Phase C-2.5 semantics, `any_passed` is gated on
`min_passing` — so `any_passed: false` now fires for BOTH the
all-fail case (`passing_count == 0`) AND the partial-pass case
(`0 < passing_count < min_passing`). Stage 9 differentiates
them via `diagnostic.passing_configs`.

On `error: "missing_judgments"`: **one** retry of the dispatch
(only the listed batches), then re-run `auto-validate-finalize`. If
the second pass still reports missing judgments, treat the gate as
`any_passed: false` (sub-agent dispatch is broken) and proceed to
the all-fail branch below.

On `error: "no_annotated_scores"` (Phase C-2.4 fail-loud): the
annotated round produced no scores. Branch on `missing_reason`:

- **`no_annotated_tasks_in_plan`** — same case as 4.5a-pre's
  no-tasks branch (the pre-flight should have caught this earlier
  but `aggregate_judgments` is the defensive backstop). Call
  `era.cli auto-revise reason=stage5_missing_annotated_tasks`
  with the diagnostic from the finalize error. Do NOT proceed to
  Step 5.
- **`annotated_round_didnt_run`** — annotated tasks were planned
  but every one failed at runtime (judge unavailable, OOM, etc.).
  Log this informationally in `logs/iterations/`, **skip the gate**
  (treat as if `skipped_for_min_samples: true`), and proceed to
  Step 5 with **no** `auto_validate_skips` so every config runs
  the full round. The operator still gets Stage 8 review on real
  scores; Stage 9's advisor sees the gap.

Branch on the result:

- **`any_passed: true`** (`passing_count >= min_passing`) →
  record the gate decision in the Stage-6 log note as
  `"passed C-2 gate (<passing_count>/<chosen> configs ≥ pass/recall, min=<min_passing>)"`,
  then proceed to **Step 5** passing
  `auto_validate_skips: <failing_configs>` so the failing configs'
  full-mode tasks are authorised-skipped at init.
- **`any_passed: false`** (`passing_count < min_passing`) →
  fewer than `min_passing` configs cleared the gate. Under
  Phase C-2.5 this fires for both partial-pass and all-fail.
  Call `era.cli auto-revise` with
  `reason: stage7_auto_validate_failed` and the per-config
  diagnostic INCLUDING `passing_configs: [...]` (may be empty)
  AND `failing_configs: [...]` so Stage 9's advisor can
  differentiate partial-pass (proven seeds to carry forward)
  from all-fail (every method bad). Phase C-1's machinery
  scaffolds the next iter (or forces ADVANCE at the cap). Do
  NOT proceed to Step 5.
- **`skipped_for_min_samples: true`** → annotated round wasn't
  informative; proceed to Step 5 with no `auto_validate_skips`.

The auto-revise call looks like (template handles both
partial-pass and all-fail — distinguish via `passing_configs`):

```bash
<repo>/.venv/bin/python3 -m era.cli auto-revise <<JSON
{
  "workspace_path": "<workspace>",
  "reason": "stage7_auto_validate_failed",
  "source_stage": 6,
  "blocker_summary": "<passing_count>/<chosen> configs cleared pass/recall on N annotated samples (need >= <min_passing>)",
  "diagnostic": {
    "thresholds": {...},
    "passing_configs": ["<combination_id>", ...],
    "failing_configs": ["<combination_id>", ...],
    "passing_count": <int>,
    "min_passing": <int>,
    "per_config": [
      {"combination_id": "...", "pass_rate": ..., "recall_rate": ...,
       "passed": <bool>}, ...
    ],
    "annotated_sample_count": ...
  }
}
JSON
```

## Step 5 — Full run

The full round runs only on configs that PASSED the Phase C-2 gate
(if it ran). Pass the failing configs through `auto_validate_skips`
so the orchestration layer pre-marks their full-mode tasks
`skipped` with `skip_reason: auto_validate_failed`:

```bash
<repo>/.venv/bin/python3 -m era.cli init-experiment <<JSON
{
  "workspace_path": "<workspace>",
  "mode": "full",
  "skip": [
    {
      "task_id": "eval-<combination_id>-full",
      "pivot_proof": "<exact pivot_matrix[*].action string>",
      "skip_reason": "<one-line human-readable rationale>"
    }
  ],
  "auto_validate_skips": ["<combination_id-that-failed-the-gate>", ...]
}
JSON
```

`skip` is a list of structured entries — one per task the pivot matrix dropped.
For every **eval** task the `pivot_proof` **must** match a Stage 4
`experiment_brief.pivot_matrix[*].action` exactly; the CLI gate returns
`error: "unauthorized_skip"` otherwise and leaves the state untouched. Non-eval
tasks (`serve-*`, `setup-*`) accept an empty proof. Bare task-id strings are
**no longer accepted** — they were the silent-scope-reduction back door, and the
gate now rejects them with `error: "bad_skip_entry"`.

`auto_validate_skips` is the Phase C-2 second authorised-skip path:
each combination_id must appear in
`<iter>/auto_validate/result.json.failing_configs`. The orchestration
layer re-validates against the on-disk artifact; the gate is enforced
end-to-end (CLI → init_state → completion gate).

Pilot-phase **failures** (missing host deps, model too costly, runtime crash)
**must not** populate this list — they belong in `record-task outcome=failure`
during the pilot DAG walk, which makes the Stage 6 completion gate block until
the operator fixes the blocker or amends the brief. Once the skip list is
authorized, repeat the Step-3 DAG walk with `"mode": "full"`.

## Step 6 — Collect and finish

When the full pass is done, run the **completion gate**:

```bash
<repo>/.venv/bin/python3 -m era.cli check-experiment-completion <<JSON
{"workspace_path": "<workspace>", "mode": "full"}
JSON
```

This returns the orchestration-layer "OK to advance" answer:
`complete = (every chosen_config either scored OR has a Stage 4 skip_proof
OR is fully runtime_failed within the 30 % cap) AND (no eval task is still
pending/running)`. It also returns `missing_configs`, `scored_configs`,
`skipped_with_proof`, `runtime_failed_configs`, `runtime_failure_categories`,
`failed_tasks`, and `in_progress_tasks`. If too many configs are
runtime_failed for the cap, the gate sets
`runtime_failure_cap_exceeded: {limit, observed, fraction, configs}` and
forces `complete: false` — that is the audit boundary: a systemic infra
collapse must surface, not be absorbed.

- If `complete: true`, append the Stage-6 log note (configs run, score
  means, GPU-hours and `$` spent vs. `budget`) to
  `<workspace>/logs/iterations/` and return. **Do not** advance
  `status.json` — the ralph loop owns that.
- If `complete: false`:
  - For every config in `missing_configs` that has a corresponding
    `failed_tasks` entry, drive it through one more `heal-tick` /
    `claim-batch` cycle. The healing loop's bounded circuit breaker decides
    when retries are exhausted; an exhausted task lands as `failure`, not
    `skipped`.
  - For configs whose runner produced `ok_count: 0` (real runtime crash,
    bad output path), fix the runner and re-run. **Never** "drop the
    config" with `outcome: "skipped"` — that path is gated on a Stage 4
    pivot-matrix `pivot_proof` only.
  - If after the heal-and-retry pass `check-experiment-completion` still
    reports `complete: false`, record the situation in the Stage-6 log note
    and invoke the **auto-revise** path instead of blocking. Pass the
    completion-gate output through as `diagnostic` so Stage 9's advisor
    can see exactly which configs were missing:

    ```bash
    <repo>/.venv/bin/python3 -m era.cli auto-revise <<JSON
    {
      "workspace_path": "<workspace>",
      "reason": "stage6_incomplete",
      "source_stage": 6,
      "blocker_summary": "<one-line: e.g. 4/9 configs missing after one heal pass>",
      "diagnostic": {
        "missing_configs": [...],
        "scored_configs": [...],
        "failed_tasks": [...],
        "in_progress_tasks": [...],
        "runtime_failure_categories": {...}
      }
    }
    JSON
    ```

    The reply is one of:
    - `{decision: "REVISE_SKIP_STAGE1", next_iter: N+1, ...}` — a new
      iter has been scaffolded; the ralph-loop will pick it up next pass.
    - `{decision: "ADVANCE", forced_advance: true}` — `react.max_iterations`
      reached; the ralph-loop advances `stage_index` normally and lands at
      Stage 10's terminal block.

    Either way, the pre-Stage-8 loop never sets `run_state: blocked`. The
    failure context lives in `<iter>/auto_revise/trigger.json` and is
    surfaced to Stage 9's advisor via the next iter's
    `parent_feedback.auto_revise_trigger` pointer.

The `summary.json` file (written by `write_summary`) carries the *strict*
completion view: `complete: true` iff **every** chosen config has
`ok_count > 0`, with no skip-proof exception. The orchestration layer
(`check-experiment-completion`) is the one that excuses skipped-with-proof
when deciding whether to advance; do not confuse the two answers.

## Principles

- **Results or not done** — the deliverable is real per-config evaluator scores
  in `experiments/results/`. A fabricated score is never acceptable.
- **Normalize pairwise results** — a pairwise judge config produces per-pair
  win-rates, not pointwise scalars; convert them to canonical per-method-id
  scores before any cross-config comparison or the final cost/alignment
  frontier. Never carry `<methodA>_vs_<methodB>` pair labels as method keys.
- **The scheduler and the markers are authoritative** — claim through
  `claim-batch`, learn task fate through `recover-experiment`; never guess.
- **Serve packing** — `claim-batch` enforces `experiment.family_a_execution`;
  never start a judge out of band. Under `parallel_packed` (default) several
  right-sized judges co-reside on disjoint GPU subsets and you tear each one
  down independently (via `shutdown-judge`) the moment *its own* `teardown_after`
  evals all resolve — do not wait for the other judges. Under `serial_full_pool`
  (Rule 6) exactly one judge is resident at a time, owning the whole pool.
- **Bounded healing** — `heal-tick`'s circuit breaker ends retries; a
  `give_up` config is recorded and the round continues.
- **In-iteration only** — Stage 6 runs inside `<iter>`; it never creates a new
  iteration.
- **Autonomous** — never ask the operator; record every decision in
  `<workspace>/logs/iterations/`.
