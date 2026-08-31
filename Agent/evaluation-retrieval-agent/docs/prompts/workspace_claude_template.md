# ERA Workspace — {{PROJECT_NAME}}

This directory is an **ERA project workspace** (ERA v{{ERA_VERSION}}), created
by `/era:init`. It evaluates a **{{TASK_FAMILY}}** task with the
**{{TASK_ADAPTER}}** adapter.

## Layout

- `config.yaml` — machine-readable project config (hardware, data, serving,
  budget, experiment policy including Phase C auto-validate thresholds).
  Source of truth for downstream stages.
- `spec.md` — human-readable, operator-editable evaluation spec.
- `status.json` — pipeline lifecycle: `stage` / `stage_index` / `iteration` /
  `run_state` (`idle` · `running` · `stopped` · `awaiting_human` · `blocked`
  · `done`). Single source of truth.
- `probe/` — raw environment-probe artifacts captured at init.
- `research/` — Stage 1 literature (workspace-global; may be refreshed by a
  Stage 9 `REVISE_RERUN_STAGE1` verdict).
- `shared/` — cross-iteration shared artifacts.
- `current` → `iter_NNN` — symlink to the active iteration.
- `iter_NNN/` — per-iteration content for Stages 2–10 (`design/`,
  `experiments/`, `serving/`, `comparison/`, `human/`, `react/`,
  `auto_revise/`, `auto_validate/`, `deliverable/`).

## Iteration loop

Stages 2–10 run inside `iter_NNN/`. After Stage 8 (human feedback), Stage 9
(ReAct) decides `ADVANCE | REVISE_SKIP_STAGE1 | REVISE_RERUN_STAGE1` and may
scaffold `iter_{N+1}/` to refine the evaluation method. `react.max_iterations`
(default 5) caps the loop — at the cap, Stage 9 forces ADVANCE and Stage 10
terminates the run.

## Rules

- Treat `config.yaml` as authoritative; if `spec.md` and `config.yaml`
  disagree, surface it to the operator rather than guessing.
- Do not edit `probe/` artifacts — they are an audit trail.
- ERA v{{ERA_VERSION}} implements Stages 0-9 end-to-end (task init, literature
  research, the Stage 2-4 design debate, the Stage 5-6 experiment, Stage 7
  pre-human comparison, Stage 8 human feedback, Stage 9 ReAct iteration
  gate); only Stage 10 (`final_report`) is a stub. `/era:start` runs the
  pipeline as a ralph-loop; the **only** operator hand-off is the Stage 8
  Continue prompt (the review web app finalize). `/era:resume` is the
  closed-terminal fallback.
- **Pre-Stage-8 never blocks** (Phase C-1): when Stages 4/5/6/7 produce
  an invalid output (e.g. Stage 6 `check-experiment-completion` returns
  `complete: false`), the stage calls `era.cli auto-revise` which writes
  `<iter>/auto_revise/trigger.json` and fires a Stage 9
  `REVISE_SKIP_STAGE1` — the loop advances to `iter_{N+1}/` rather than
  setting `run_state: blocked`. Stage 9's advisor reads the trigger via
  the next iter's `parent_feedback.auto_revise_trigger` pointer and drops
  the infeasible configs from the brief. Only Stage 8 (`awaiting_human` /
  `stopped` / `blocked`) and Stage 10 (`done`) still own blocking states.
- **Stage 6 saturates available GPUs** (Phase D-1): Stage 6 writes every
  batch runner up front, launches the whole batch as background bash jobs
  in **one** shell pass (`nohup ... &` with disjoint
  `CUDA_VISIBLE_DEVICES`), and waits via the event-driven
  `era.cli wait-for-any-done` — which returns within ~250 ms of the first
  task's `done.json` instead of sleeping for
  `experiment.poll_interval_s`. Default `experiment.family_b_schedule` is
  now `parallel_on_unallowed_gpus` so Family-B evals run on GPUs outside
  a resident judge's pool. The optional
  `experiment.max_parallel_runners` (default 0 = uncapped) caps
  concurrent runners when host CPU / disk pressure matters.
- **Parallel-packed judges** (Phase D-6): default
  `experiment.family_a_execution` is now `parallel_packed` — each VLM
  judge claims only its `tensor_parallel` GPUs, multiple judges co-reside
  on **disjoint** GPU subsets, and Family-B evals backfill the rest, so
  the pool stays saturated instead of one judge owning all cards. Stage 5
  right-sizes each `serve` task's `gpu_count` to its TP degree and keeps
  judges independent (no serve chain). `serial_full_pool` (Rule 6 — one
  judge owns the pool) stays operator-pinnable for strict isolation; the
  optional `experiment.max_concurrent_judges` (default 0 = uncapped) caps
  co-resident judges when judge-startup / host-RAM pressure matters.
- **Pass/recall auto-validation gate** (Phase C-2): between Stage 6's
  pilot and full rounds, Stage 6 runs an **annotated round** scoring
  every config on the operator's `/era:annotate` notes
  (`<dataset>/annotations/*.json`). The light-tier `era-auto-validator`
  sub-agent (Haiku, one Task per `(combination_id, method_id)` batch,
  parallel-dispatched) judges agree/disagree per sample by semantic
  comparison of the method's `score` + `sub_scores` against the
  operator's free-text annotation. `era.cli auto-validate-finalize`
  aggregates the judgments into per-config `pass_rate` / `recall_rate`
  and writes `<iter>/auto_validate/result.json`. Configs meeting both
  thresholds (defaults 0.70 / 0.60, operator-pinned at `/era:init`)
  proceed to the full N=50 round; failing configs are pre-skipped via
  `init-experiment auto_validate_skips: [cid, ...]` (a second authorized
  scope-reduction path validated against
  `result.json.failing_configs`). When ALL configs fail, Stage 6 calls
  `auto-revise reason=stage7_auto_validate_failed` and the Phase C-1
  machinery scaffolds the next iter so Stage 2 can swap evaluation
  approaches. The annotated round is skipped on datasets with fewer
  than `auto_validate_min_samples` annotations (default 10).
- **Annotation-aware Stages 2-4** (Phase C-2.2): when
  `data.use_annotation_evidence` is `true` (default) and the dataset
  carries operator notes, Stage 2's brainstorm personas read
  `era.cli list-annotations` + the operator's per-method notes and
  design evaluators that target the flagged failure modes (each
  candidate's `hypothesis_id` text carries a tag naming the
  targeted mode). Stage 3's rigor-critic scores **annotation
  coverage** 0-3 per candidate. Stage 4's synthesizer biases
  `chosen_configs` so the union covers every flagged failure mode
  at least once — materially raising the Phase C-2 gate's
  first-iter pass probability. The task-plan validator's
  pilot-block requirement is pilot-mode-only (annotated/full eval
  tasks may omit `pilot:`); `_normalize_judge` strips modality
  suffixes (`-pointwise`, `-pairwise`, `-flag`, `-judge`,
  `-rubric`, `-v\d+`) before comparing `serve.judge` vs
  `eval.judge`, so cosmetic planner variance no longer blocks
  Stage 5.
- **Gate fail-loud** (Phase C-2.4): Stage 6 pre-flights the
  Phase C-2 gate by checking the task plan contains
  annotated-mode eval tasks before calling auto-validate. A plan
  with no annotated tasks despite annotations existing (legacy
  v0.1.6 plan or Stage 5 LLM drift) triggers
  `auto-revise reason=stage5_missing_annotated_tasks` instead of
  the silent false-negative that previously discarded the iter's
  pilot+full work. If you see `stage5_missing_annotated_tasks`
  in `logs/iterations/`, the next iter's Stage 2-4 will be told
  to emit annotated tasks — no operator action needed.
- **Autonomy hook — no operator prompts during `/era:start`**
  (Phase D-3): `.claude/settings.json` ships a PreToolUse hook
  that intercepts every `AskUserQuestion` call and runs
  `era.cli check-autonomy`. The hook reads `status.json.run_state`
  and **structurally blocks** the call when the loop is `running`
  or `blocked` (Stages 1-7 + 9-10), exit code 2 with a stderr
  message explaining the iron rule. The only allowed site is
  `run_state: awaiting_human` (the `era-human-feedback` skill
  sets this BEFORE its Continue / Still working / Cancel prompt
  — the hook recognises the marker and lets it through). To
  pin custom hooks (e.g. an audit log on every Bash call) edit
  `.claude/settings.local.json` (gitignored, per-machine);
  operator-added matchers survive the ERA merge.
- **M-passing threshold + EA-style iter memory** (Phase C-2.5):
  `experiment.auto_validate_min_passing` (default **3**, set at
  /era:init) gates the full N=50 round on the count of configs
  that cleared the C-2 pass/recall thresholds. `any_passed` is
  redefined as `passing_count >= min_passing`; below M, Stage 6
  auto-revises into the next iter. Stage 9's advisor writes
  three EA primitives into `evolution_state.json`:
  **must_include_configs** (elitism — the prior iter's PROVEN
  configs are hard requirements for the next brief, enforced by
  `check-experiment-brief`'s `evolution_state_path` argument),
  **hall_of_fame** (top-K across all iters by `fitness_composite
  = (pass × recall × human_endorsement)^(1/3)`, capped at
  `max(min_passing, 5)`), and **lessons_learned** (structured
  success_patterns + failure_patterns + open_questions with
  `confidence: low|medium|high` graduating by evidence count
  across iters). Stage 9 also writes a natural-language
  `react/lessons.md` digest; Stage 2's brainstorm in the next
  iter reads all four — success patterns as priors, failure
  patterns as anti-patterns, hall_of_fame as pre-allocated
  must-include slots. Optional `candidate_configs[*].parent_hypothesis_ids`
  tracks lineage so Stage 9 can spot drift ("we keep mutating
  the same losing branch"). The **≥30B VLM judge floor** is
  encoded in Stage 1/2/4 prompts and Stage 9's `model_upsize`
  ladder (named references: Qwen3.6-35B-A3B, Gemma-4-31B-it);
  sub-30B candidates are forbidden unless the operator
  explicitly asks for them in human feedback.
- **Random N samples per iter** (Phase C-2.3): the full round's
  N samples are picked **randomly** from the full dataset via
  `era.cli sample-window` (seed = `sha256(project_name:iteration)`
  truncated to 32 bits — deterministic per (workspace, iter) so
  re-runs reproduce, but different iters pick different sets so
  the whole dataset is exercised across iters). Stage 5 stamps
  the picked list as `samples_subset` on every full-mode eval
  task, so all methods × configs in one iter score the **same
  shuffled subset** (apples-to-apples comparison preserved by
  construction). The validator's `samples_subset` field is no
  longer annotated-only; it is the primary selection rule across
  all modes when present, with `sorted(glob)[:N]` as the legacy
  fallback (pilot mode + pre-v0.1.7.1 plans).
- **Unattended permissions** (Phase D-2): `.claude/settings.json`
  pre-approves the Bash / Write / Edit / Task patterns ERA's
  ralph-loop uses (every `era.cli` subcommand, `nvidia-smi`,
  `nohup` background runners, `kill`, read-only Bash utilities,
  localhost curl, workspace-internal writes, every
  `Task(era:*)`), and denies a catastrophic set
  (`git push --force`, `git config --global`, `rm -rf /`). The
  merge semantics ensure this block lands on **existing**
  workspaces too on the next `/era:resume`. **Repo-root
  inheritance** (Phase D-2 extension): patterns added to the
  repo-root `<era_repo>/.claude/settings.json`'s
  `permissions.allow` / `permissions.deny` are inherited by this
  workspace at `/era:init` and re-applied on every `/era:resume`
  (`ensure_claude_settings` unions ERA defaults + repo-root pins
  on top of whatever this workspace's settings.json already
  holds). Operators pin repo-wide custom patterns once at the
  repo root instead of editing source. To pin extra patterns
  per-machine without committing them, edit
  `.claude/settings.local.json` (already gitignored) and add to
  its `permissions.allow` / `permissions.deny` arrays. **If your
  `/era:start` run still prompts on an unexpected pattern**, add
  it to `.claude/settings.local.json`'s `permissions.allow` and
  resume.

## GPU environment

This machine runs a GPU watchdog, `NoGPUAlarmNew.py` (in
`/mnt/image-edit/datasets/xywang/code/GPU_OCU/`), that holds otherwise-idle
cards. A GPU whose `nvidia-smi` signature looks like:

```
NVIDIA H100 80GB HBM3 | **°C, 100 % | 35061 / 81559 MB
```

— roughly 35 GB used at 100 % utilization — is running **only the watchdog, not
a real job**. The watchdog releases the card automatically as soon as a genuine
GPU task starts, so **treat such a GPU as free** when scheduling experiments.

If the watchdog does not release a card on its own, stop it manually:

```bash
sudo pkill -9 -f "python3 -u NoGPUAlarmNew.py"
```

Both forms (with and without `sudo`) are pre-approved in
`.claude/settings.json`'s `permissions.allow` (Phase D-4 extension), so
the autonomous loop can stop a misbehaving watchdog without an operator
prompt. The patterns are scoped to the watchdog name — blanket
`Bash(pkill *)` / `Bash(sudo pkill *)` are intentionally NOT auto-allowed.

After all experiments in an iteration finish, if `NoGPUAlarmNew.py` was killed
and has not restarted itself, restart it so idle cards stay protected:

```bash
cd /mnt/image-edit/datasets/xywang/code/GPU_OCU/ && bash start.sh
```

**Stage 8 auto-ensures the watchdog.** Before the human-feedback web app
launches (a potentially long wait with idle experiment GPUs), Stage 8 calls
`era.cli ensure-watchdog` — it checks process liveness with `pgrep` and runs
the `start.sh` above **only** when no watchdog is up, so it re-arms a watchdog
that died between Stage 6 and the hand-off without ever spawning a duplicate.
This step is non-blocking — a watchdog miss is logged, never gating the
operator hand-off.

**Phantom GPU reset.** If a card's `nvidia-smi` signature shows stuck memory
or a phantom process that won't release after killing its PID (a crashed VLM
judge can leave the GPU compute context wedged), force-reset the specific
card:

```bash
sudo nvidia-smi --gpu-reset -i <gpu_id>
```

This is pre-approved in `.claude/settings.json`'s `permissions.allow`
(Phase D-4), so the autonomous loop can auto-clear zombie GPUs without an
operator prompt. Use the per-card `-i <id>` form rather than resetting all
cards at once — a Stage 6 judge running on a different GPU should not be
collateral.

**Safe vLLM teardown (Phase D-5).** When a Stage 6 judge's
`teardown_after` evals all complete, Stage 6 calls `era.cli
shutdown-judge` rather than raw `kill <pid>` — it runs the safe
sequence: SIGTERM the **process group** (the runner launches vLLM
with `start_new_session=True` so parent + tensor-parallel workers
share one pgid) → poll-exit up to `graceful_timeout_s` (default
30 s) → SIGKILL the pgid on timeout → orphan sweep via `pkill
-TERM -f <served_model_name>` → `nvidia-smi` memory verify →
escalate to `sudo nvidia-smi --gpu-reset` if wedged → release the
GPU lease. Returns one of four statuses: `ok`, `escalated_kill`,
`escalated_reset`, or `still_stuck` (hardware-level — record-task
as `runtime_failed` with `failure_category: serving`). Raw `kill
<pid>` from a runner is forbidden — it leaks tensor-parallel
workers and wedges GPUs.
