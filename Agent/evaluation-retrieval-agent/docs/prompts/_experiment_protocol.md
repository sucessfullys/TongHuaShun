# ERA Experiment Protocol — the contract every Stage 6 runner obeys

A Stage 6 **runner script** executes one task from the experiment DAG: it serves
a VLM judge, or runs an evaluator configuration over a dataset slice. Whoever
writes a runner — the experiment sub-agent — makes it obey **every** rule below.
The deterministic scheduler, recovery, and healing all depend on this contract.

The runner is written to `<iter>/experiments/configs/<task_id>_runner.py` and run
detached so it reports progress through marker files rather than stdout.

## 1. Path isolation
Every path the runner reads or writes stays under the active iteration's
`experiments/` tree (`<iter>/experiments/...`). A runner never touches another
iteration, another workspace, or the ERA repo. Resolve `<iter>` from the
argument the launcher passes; never hard-code it.

## 2. PID marker first
The runner's **first** action is to write its own PID:
`<iter>/experiments/logs/<task_id>.pid` containing `os.getpid()`. This is how
recovery tells a live task from a dead one.

## 3. Read the override file
If `<iter>/experiments/configs/<task_id>.override.json` exists, load it and honor
its keys (`batch_size`, `gpu_memory_utilization`, `port`, …). It is the healing
loop's channel — a re-run after an OOM or a port clash arrives here. Absent the
file, use the task's defaults.

## 4. Progress marker — the heartbeat, on a time cadence
Overwrite `<iter>/experiments/logs/<task_id>.progress.json` with
`{"task_id": ..., "done": <int>, "total": <int>, "updated_at": <iso>}` every N
samples **and** at least once every few minutes — the file's mtime is the
runner's **heartbeat**. The Stage 6 detection scan treats a live runner whose
`.progress.json` (or, before its first write, `.pid`) has not advanced for
`experiment.heartbeat_timeout_s` (default 1800 s) as **hung** and kills + retries
it. So refresh `.progress.json` on a timer even mid-batch — never let it go
stale while the runner is still healthy.

## 5. Sample selection — `samples_subset` is authoritative when set
Each iteration evaluates only a **fixed window** of samples per method, not
the full `data_root`. Starting at v0.1.7.1 (Phase C-2.3), the
`samples_subset` field is the **primary selection rule across all modes**;
the legacy first-N fallback only applies when `samples_subset` is absent.

**Primary rule — `samples_subset` is set on the task:**

When the task carries `samples_subset: [sample_key, ...]`, score **exactly
those samples and no others** — do NOT re-derive from `sorted(glob)`.
Sample lookup is `method_path/<sample_key>/` (POSIX-style), matching the
central annotation key format. The list is identical across all eval
tasks in the same iter so every config × method scores the **same subset**
— apples-to-apples comparison preserved by construction. The list length
is also stamped as `samples_per_method` on the task for consistency.

Stage 5 stamps `samples_subset` on:
- **`annotated` mode** — the operator-annotated keys (Phase C-2 gate).
- **`full` mode** — a deterministically-random N picked via
  `era.cli sample-window` (Phase C-2.3). Seed is
  `sha256(project_name:iteration)[:4]`, so re-running the same iter
  produces the same set; different iters pick different sets so the
  full dataset gets exercised across iters.

**Fallback — `samples_subset` is absent (pilot mode, or legacy plans):**

1. List the candidate sample directories for the method by expanding
   `data.sample_glob` under `data.methods[].path`.
2. Sort that list **lexicographically** (Python's `sorted(...)`) — the order
   must be deterministic so the same window is scored every iter.
3. Take the **first N** entries, where `N = samples_per_method` (the task
   plan stamps this; the runner does not re-derive it).

A runner that iterates the full `sample_glob` (ignoring both
`samples_subset` and `samples_per_method`) is wrong — it burns the
operator's budget and pollutes the human review with samples the
operator never wanted to see.

## 6. Per-sample scores, append-only
Append one JSON line per sample to the eval task's `scores.jsonl`. This is the
task's `expected_output`, which Stage 5 pins to the **canonical** path
`<iter>/experiments/results/<mode>/<combination_id>/scores.jsonl` (`<mode>` is
`pilot` or `full`) — `era.cli record-task` aggregates *exactly* this path, so a
runner that writes anywhere else produces a hollow `config_result.json` and a
hollow `summary.json`.

Each line is one JSON object. Use **exactly** these canonical field names and
types — the Stage 6 aggregator *and* the Stage 8 review web app read this row
directly, so a renamed or wrong-typed field silently drops the result from the
human review:

- `sample_key` — string, the sample identifier (the key, not `sample_id`).
  Just the identifier — **never the path**. Do not prefix it with the
  `method_id` or any directory component; the Stage 8 review web app appends
  `sample_key` under each method's `data.methods[].path`, so a prefixed key
  double-counts the directory and breaks image resolution + per-sample cell
  merging across methods.
- `method_id` — string, the generation method being scored (not `method`).
- `score` — a JSON **number**, never a string; the row's headline scalar.
- `sub_scores` — an object of named numeric factors (`{}` when there are none).
  Each key is a factor name, each value a number; these are shown verbatim in
  the review, so name them for a human (`garment_fidelity`, `text_alignment`, …)
  and name any pure aggregate clearly (e.g. `mean_5`).
- `scope` — string, the evaluator's scope (e.g. `pointwise-1to5`, `pairwise`,
  `region`, `whole`); it selects how the score is rendered.
- `ok` — boolean. A per-sample failure (an unreachable judge, a missing image, a
  parse error) is logged as that line with `"ok": false` and an `"error"`
  string — it **never** aborts the whole task.

## 7. DONE marker last — on success *and* failure
The runner's **last** action, whether it finished cleanly or crashed, is to
write `<iter>/experiments/logs/<task_id>.done.json`:
`{"task_id": ..., "status": "success"|"failure", "exit_code": <int>,
"summary": "...", "result_dir": "...", "endpoint": {...}?, "finished_at": <iso>}`.
A failure still gets a DONE marker — that is how recovery sees a non-zero exit.
Wrap the body in try/except so a crash still writes `status: "failure"`.

**`status` is `"success" | "failure"` only — never `"skipped"`.** A skip is an
*orchestrator* decision (a Stage 4 pivot-matrix drop, recorded by `era.cli
record-task` with a `pivot_proof`), never a runner decision. The Stage 6
recovery loop will convert any eval-task `done.json` with `status: "skipped"`
to `status: "failure"` so the silent scope-reduction path is closed; a runner
that hits "I cannot run because dep X is missing" writes `status: "failure"`
with the dep-missing message in `summary`, and the Stage 6 completion gate
surfaces the blocker to the operator.

## 8. Family A — VLM judge evaluators
A Family-A runner does **not** load a model. Its judge is already served: the
runner reads the endpoint (base URL + served model name) from the `serve` task's
recorded endpoint, passed in argv / the override file, and calls it through an
OpenAI-compatible client. When the validation protocol asks for order-swap
debiasing, the runner swaps the candidate order and averages.

## 9. Family B — metric evaluators
A Family-B runner loads its metric (CLIP / DINO / LPIPS / a segmenter / …) on
the GPU(s) the scheduler assigned — read `CUDA_VISIBLE_DEVICES`, which the
launcher sets from the task's `gpu_ids`. A region scope uses the mask or
segmenter the brief names. Probe VRAM lightly: start at the override / default
batch size and halve once on OOM rather than crashing.

## 10. A serve task's runner
A `serve` runner starts the judge process detached (backend `ms-swift`,
fallbacks `vllm` / `lmdeploy`), then **dry-probes** the endpoint — poll
`GET /v1/models` until the served model is listed, then one tiny
`POST /v1/chat/completions`. On a clean probe it writes `done.json` with
`status: "success"` and the `endpoint` block, then exits — the judge process
keeps running. On timeout it writes `status: "failure"`.

**Launch the judge in its own session** (Phase D-5). Pass
`start_new_session=True` to `subprocess.Popen` (or wrap the CLI in
`setsid`) so the parent + every tensor-parallel worker share one
**process group**. Stage 6's teardown (`era.cli shutdown-judge`)
SIGTERMs the **pgid**, not just the parent PID — pgid signalling is
the only way to reap orphaned tp-workers reliably (raw `kill <pid>`
leaks them; they keep GPU contexts open and wedge the next judge
in the Rule-6 chain). Pattern:

```python
import subprocess
proc = subprocess.Popen(
    ["python3", "-m", "vllm.entrypoints.openai.api_server",
     "--model", model_path, "--served-model-name", served_model_name,
     "--tensor-parallel-size", str(tensor_parallel),
     "--port", str(port)],
    start_new_session=True,            # new session ⇒ pgid = pid
    stdout=open(LOG, "w"), stderr=subprocess.STDOUT,
)
# Stamp the parent PID into the .pid marker the runner writes.
Path(PID_MARKER).write_text(str(proc.pid))
```

Also stamp the launched `served_model_name` and the endpoint host/port
in `done.json` (or via `record-task endpoint=...`) so
`shutdown-judge` can do the orphan-sweep `pkill -TERM -f
<served_model_name>` and the port-bound advisory check.

## 11. Never fabricate
Every score is a real metric value or a real judge response. If a judge is
unreachable or an input is missing, that sample is a logged failure — never an
invented number. Never fabricate a human-correlation result; Stage 6 produces
raw evaluator scores only.
