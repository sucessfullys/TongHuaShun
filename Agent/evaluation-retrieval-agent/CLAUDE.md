# ERA — General AIGC Evaluation Retrieval Agent

## What ERA is

ERA is a config-driven, multi-stage agent (hosted by the Claude Code CLI) that
**researches, builds, and human-validates evaluation protocols for AIGC tasks**
— and *retrieves* the evaluation method best aligned with human judgement at the
lowest cost. The task (image generation, image editing, or another AIGC task) is
never hardwired: it is supplied at init via the operator's mission and captured
as `config.yaml` + a task adapter. ERA is modeled on `AutoResearch-SibylSystem`
— consult that repo's patterns before changing architecture.

The full pipeline has 12 stages (0–11); see `overall_plan.md` for the design.

## Iron rule — autonomy

Once `/era:start` runs, every stage runs unattended **except for the Stage 8
operator confirmation prompt**:

- **Never stop to ask the operator.** Inside the runtime loop and its skills,
  do not use `AskUserQuestion` — **except for the Stage 8 wait prompt**,
  which is the pipeline's single in-loop operator hand-off (the operator
  selects Continue once they have clicked Finalize in the review web app).
  Resolve every other ambiguity from `config.yaml` / `spec.md`, decide, and
  record the decision in `<workspace>/logs/iterations/`.
- **Never halt on transient errors.** Diagnose, fix, retry, or route around.
- ERA has two interactive steps: `/era:init` (Stage 0, which confirms probe
  results with the operator) and the Stage 8 wait prompt (which gates Stage 9
  on the operator finalizing feedback). Every other stage runs unattended.

## Status (v0.1.7)

Implemented: **Stage 0 — Task Init** (`/era:init`), **Stage 1 — Research
(literature)**, **Stages 2–4 — the idea-generation + debate loop**
(`plan_brainstorm` → `multi_review` → `plan_decision`), **Stages 5–6 — the
experiment** (`experiment_plan` → `full_experiment`), **Stages 7–8 — the
human-feedback handoff** (`pre_human_comparison` → `human_feedback`), and
**Stage 9 — ReAct** (`react`), plus the `/era:start` Ralph-loop runtime and
the `/era:status` / `/era:stop` / `/era:resume` lifecycle commands.
**`/era:annotate <dataset_root>`** is a standalone web app (not part of the
pipeline) that lets the operator browse a multi-method try-on dataset,
type free-text notes per sample, and save them as
`<dataset_root>/<sample_key>__annotation.json` for later pickup by Stage 7
(problem display) and Stage 9 (evolution input). Stages
2–4 fan out eval-domain persona sub-agents over 3 model tiers and emit an
experiment-ready handoff bundle in `iter_NNN/design/`. Stage 5 expands that
bundle into a dependency-ordered task DAG
(`iter_NNN/experiments/plans/task_plan.json`, guarded by
`era.cli check-task-plan`); Stage 6 walks the DAG — writing, reviewing,
GPU-scheduling, running, recovering, and healing the evaluator tasks — and
collects per-config results into `iter_NNN/experiments/results/`. Stage 7
(`pre_human_comparison`) distils those results into
`iter_NNN/comparison/comparison.json`; Stage 8 (`human_feedback`) launches a
**review web app** (FastAPI + React, `era/webapp/`) as a detached background
server, sets `run_state: awaiting_human` for `/era:status` visibility, prints
the SSH-tunnel block, and **blocks in the same ralph-loop iteration on an
`AskUserQuestion` prompt**; the operator finalizes feedback in the web app
and selects **Continue**, the skill verifies via `era.cli feedback-status`,
writes `iter_NNN/human/{feedback.json,human_labels.json}`, stops the web
server, and returns `run_state: running` — so the loop flows directly into
Stage 9 with no manual `/era:resume`. `/era:resume` is the closed-terminal
fallback (it re-enters Stage 8, which detects already-finalized feedback and
proceeds). Stage 9
(`react`) dispatches the `era-react-advisor` sub-agent to synthesize
`iter_NNN/react/evolution_state.json` (cumulative_feedback across all
iters, operator-veto `exclude_list`, typed `evolution_proposals`,
`general_failure_modes`) and picks one of three verdicts: **ADVANCE** →
Stage 10, **REVISE_SKIP_STAGE1** → new `iter_{N+1}/` starting at Stage 2,
or **REVISE_RERUN_STAGE1** → new `iter_{N+1}/` starting at Stage 1 with a
targeted `literature_update_brief.md` Stage 9 wrote. A bounded
`react.max_iterations` (default 5) forces ADVANCE at the cap. There is no
`dataset_ship` stage (the slot was retired in v0.1.6) and no standalone
deployment stage: ERA retrieves an evaluation protocol, nothing is
"deployed". Only Stage 10 (`final_report`) is still a stub.

**v0.1.7 hardenings on top of the base pipeline** — already shipped:
**Phase C-1** (auto-revise — pre-Stage-8 stages never block; `era.cli
auto-revise` writes `<iter>/auto_revise/trigger.json` and fires Stage 9
`REVISE_SKIP_STAGE1`), **Phase D-1** (Stage 6 GPU saturation — parallel
background-bash runner dispatch + the event-driven `era.cli
wait-for-any-done`; `experiment.family_b_schedule` default flipped to
`parallel_on_unallowed_gpus`; new `experiment.max_parallel_runners`
config), **Phase D-6** (parallel-packed judges — `experiment.family_a_execution`
default flipped from `serial_full_pool` to **`parallel_packed`**: each VLM
judge claims only its `tensor_parallel` GPUs, multiple judges co-reside on
**disjoint** GPU subsets, and Family-B evals backfill the rest, so the pool
stays saturated instead of one judge owning all cards. Stage 5 right-sizes
each `serve` task's `gpu_count` to its TP degree and emits independent judges
— no Rule-6 serial chain — guarded by `check-task-plan`'s `family_a_execution`
branch; `claim_batch` co-residents them and prunes per-judge leases on each
judge's own teardown. New `experiment.max_concurrent_judges` (default 0 =
uncapped) caps co-resident judges; `serial_full_pool` stays operator-pinnable),
and **Phase C-2** (pass/recall auto-validation gate — new
`annotated` task-plan mode; the light-tier `era-auto-validator`
sub-agent judges agree/disagree per sample; `era.cli
auto-validate-prepare` / `auto-validate-finalize` aggregate per-config
pass_rate / recall_rate; `era.cli list-annotations` feeds Stage 5;
new authorized-skip path `init-experiment auto_validate_skips`),
and **Phase C-2.5** (M-passing threshold + EA-style iter-to-iter
memory — `experiment.auto_validate_min_passing` (default **3**)
gates the full N=50 round on the count of configs that cleared
the C-2 pass/recall thresholds, with `any_passed` redefined as
`passing_count >= min_passing`; below M, Stage 6 auto-revises
into the next iter via Phase C-1's machinery. The Stage 9
advisor now writes three EA primitives into
`evolution_state.json`: `must_include_configs` (elitism — the
prior iter's PROVEN configs carry forward as hard requirements,
enforced by `check-experiment-brief`'s `evolution_state_path`
parameter), `hall_of_fame` (population memory — top-K across
all iters by `fitness_composite = (pass × recall ×
human_endorsement)^(1/3)`, capped at `max(min_passing, 5)`),
and `lessons_learned` (structured success/failure patterns
with `confidence: low|medium|high` graduating across iters,
plus `open_questions`). Stage 9 also writes
`react/lessons.md` (natural-language digest) and Stage 2's
brainstorm reads all four — success_patterns as priors,
failure_patterns as anti-patterns, hall_of_fame as
pre-allocated must-include slots. Stage 4's brief gate enforces
`candidate_configs ⊇ must_include_configs`. Lineage tracking
via optional `candidate_configs[*].parent_hypothesis_ids`. The
**≥30B VLM judge floor** (Qwen3.6-35B-A3B, Gemma-4-31B-it as
named references) is encoded in Stage 1/2/4 prompts and
Stage 9's `model_upsize` ladder — sub-30B candidates are
forbidden unless the operator explicitly asks for them in
human feedback. See `feedback-vlm-min-size`,
`feedback-ea-memory-stage9`, and
`feedback-auto-validate-min-passing` in operator memory).
See the per-phase paragraphs below for the wire diagrams.

## Python environment (mandatory)

This project uses a **venv** at `.venv/`. **All Python must use
`.venv/bin/python3`** — never bare `python3` (the system interpreter lacks the
dependencies). Rebuild with `python3 -m venv .venv && .venv/bin/pip install -e .`.

The deterministic logic is the `era` package, driven through
`.venv/bin/python3 -m era.cli <subcommand>` — each subcommand reads one JSON
object from stdin and prints one JSON object to stdout.

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
pkill -9 -f "python3 -u NoGPUAlarmNew.py"
```

or

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
launches (a potentially long wait with idle experiment GPUs), the
`era-human-feedback` skill calls `era.cli ensure-watchdog` —
`gpu_scheduler.ensure_watchdog_alive` checks process liveness with `pgrep`
and runs the `start.sh` above **only** when no watchdog is up (so it never
spawns a duplicate), re-arming a watchdog that died between Stage 6 and the
hand-off. Unlike `resume_watchdog` (sentinel-gated), this is a true liveness
ensure; it is non-blocking — a watchdog miss is logged, never gating the
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
escalate to `sudo nvidia-smi --gpu-reset` if a tp-worker left
the GPU wedged → release the GPU lease. Returns one of four
statuses: `ok` (graceful), `escalated_kill` (needed SIGKILL but
cleaned up), `escalated_reset` (needed GPU reset), `still_stuck`
(hardware-level — record-task as `runtime_failed` with
`failure_category: serving`).

## Repository layout

```
era/                Python package — config, probes, workspace, orchestration, CLI
  probe/            environment probes (gpu, data, checkpoints, credentials)
  orchestration/    workspace scaffolding, guide, ralph-prompt compiler,
                    Claude Code wiring (.claude/settings.json, .mcp.json),
                    lifecycle, debate loop, the Stage 5-6 task plan, GPU
                    scheduler, experiment state / results / error-heal, the
                    Stage 8 human-feedback backbone, the Stage 9 ReAct
                    iteration-gate backbone (react.py), the Phase C-1
                    auto-revise machinery (auto_revise.py), and the
                    Phase C-2 pass/recall gate backbone (auto_validate.py)
  webapp/           the Stage 8 review web app — FastAPI backend (app, data,
                    store, images, server) + the built React frontend in static/
  annotate/         the /era:annotate standalone image-annotation web app
                    (FastAPI backend + vanilla-JS frontend in static/index.html)
plugin/             Claude Code plugin
  commands/         /era:init, /era:start, /era:status, /era:stop, /era:resume,
                    /era:annotate
  scripts/          environment preflight (preflight.sh)
  skills/           model-invoked skills (era-literature; era-plan-brainstorm,
                    era-multi-review, era-plan-decision — the Stage 2-4 debate;
                    era-experiment-plan, era-experiment — Stages 5-6;
                    era-pre-human-comparison, era-human-feedback — Stages 7-8;
                    era-react — Stage 9)
  agents/           sub-agents (literature-scout; era-heavy/standard/light
                    tiers; era-codex-reviewer — opt-in Stage 6 code review;
                    era-react-advisor — Stage 9 synthesizer;
                    era-auto-validator — Phase C-2 pass/recall semantic judge)
docs/prompts/       ERA runtime behavioral prompts/templates — flat, unversioned
docs/mcp-servers.md MCP setup (arXiv / Google Scholar) for literature research
knowledge/          coding-task plans/prompts for BUILDING ERA — never runtime
workspaces/         per-project workspaces (git-ignored)
tests/              pytest unit + integration tests
```

## Runtime model

- `/era:init <mission>` scaffolds `workspaces/{project}/` (`config.yaml`,
  `spec.md`, `status.json`, `CLAUDE.md`, `.claude/settings.json`, `.mcp.json`,
  `iter_001/` + `current`). The init also writes
  `data.iter_sample_count` (default **50**) — the per-iter samples-per-method
  cap that Stage 6 runners actually score and Stage 8 surfaces for human
  feedback. The effective cap is `min(iter_sample_count, sample_count)`, so
  setting the cap above the dataset size silently uses the dataset size.
  The Stage 4 brief gate (`check-experiment-brief`) enforces that
  `validation.sample_size` equals this effective cap and
  `pilot.sample_count` ≤ cap. **Phase C-2.3** (v0.1.7.1): the full
  round's N samples are picked **randomly** from the full dataset via
  `era.cli sample-window` (seed = `sha256(project_name:iteration)[:4]`
  — deterministic per (workspace, iter), so re-runs reproduce; different
  iters pick different sets so the whole dataset gets exercised across
  iters). Stage 5 stamps the picked list as `samples_subset` on every
  full-mode eval task, so all methods × configs in one iter score the
  **same shuffled subset** (apples-to-apples preserved by construction).
  Pilot-mode keeps `sorted(glob)[:pilot.samples]` (small + stable);
  annotated-mode uses the operator-annotated keys (Phase C-2). The init also runs
  `era.probe.annotations.probe_annotations` against the dataset to
  surface any pre-existing operator annotations from a prior
  `/era:annotate` run — count, per-method coverage, sync state — and
  the post-init guide shows them alongside the auto-validation gate
  thresholds (`experiment.auto_validate_pass_threshold` default **0.70**,
  `auto_validate_recall_threshold` default **0.60**,
  `auto_validate_min_samples` default **10**, all operator-pinnable at
  /era:init). `data.use_annotation_evidence` (default **true**) controls
  whether Stage 2's brainstorm reads the annotations as
  operator-flagged failure modes.
- **Shared serving memory.** `~/.era/memory/serving_recipes/` holds one
  JSON per `(judge_model, backend)` pair, cross-project + per-user.
  Captures install commands, working launch flags, runner-template
  snippets, and known quirks so the next project's Stage 5-6
  doesn't re-derive them. API: `era/orchestration/serving_memory.py`
  (`read_recipe` / `write_recipe` / `list_recipes` / `forget_recipe`);
  CLI: `era.cli serving-memory <<< '{"verb": "list|read|write|forget", …}'`.
  Phase A ships the read/write API + manual capture path; active
  capture from Stage 6 lands in Phase C.
- `/era:start` runs the pipeline as a **Ralph loop** via the official
  `ralph-loop` plugin: it compiles `<workspace>/.claude/ralph-prompt.txt` from
  `docs/prompts/ralph_loop.md` and advances `status.json` one stage per
  iteration. `/era:status|stop|resume` manage a run. When `jq` (the plugin's
  Stop hook needs it) or the plugin itself is unavailable, `/era:start` runs the
  loop in a **manual fallback** mode (no Stop hook) — `preflight.sh` reports
  which mode applies.
- Claude Code is wired **per launch directory** — Claude Code reads
  `.claude/settings.json` and `.mcp.json` only from the directory it starts in.
  `era/orchestration/ralph.py:ensure_claude_settings` writes `.claude/settings.json`
  (enables the `ralph-loop` plugin, auto-trusts MCP, **pre-approves the
  ralph-loop runtime permissions**) and
  `era/orchestration/mcp.py:ensure_mcp_config` writes `.mcp.json` (registers the
  arXiv + GitHub + Codex MCP servers, from the `MCP_SERVERS` registry) into the
  repo root and every workspace — so `/ralph-loop` and the `mcp__arxiv-*` /
  `mcp__github__*` / `mcp__codex__*` tools load wherever Claude Code starts.
  `era.cli write-mcp-config` (re)registers a directory on demand.
- **Permissions** (v0.1.7.1 Phase D-2): `ensure_claude_settings` also
  populates `permissions.allow` with the specific Bash / Edit / Write /
  Task patterns ERA's ralph-loop uses (every `era.cli` subcommand,
  `nvidia-smi`, `nohup` background runners, `kill` for hung judges,
  read-only Bash utilities, localhost curl health probes, writes under
  `iter_*/`, `logs/`, `.claude/`, `shared/`, every `Task(era:*)`
  sub-agent), and `permissions.deny` with a small catastrophic set
  (`git push --force`, `git config --global`, `rm -rf /`). The merge
  semantics ensure existing workspaces upgrade on next
  `/era:resume`: ERA-managed keys are added when absent, operator-
  added keys (including extra `permissions.allow` patterns) are
  preserved. Operators who want broader or narrower permissions pin
  them in `.claude/settings.local.json` (gitignored, per-machine).
  **Repo-root inheritance** (Phase D-2 extension): patterns added to
  the repo-root `.claude/settings.json` `permissions.allow` /
  `permissions.deny` are inherited by every new workspace at
  `/era:init` and re-applied on every `/era:resume` — the same
  `_merge_into_settings` union runs once with the ERA defaults and
  once with the repo-root file. This lets operators pin repo-wide
  custom patterns (e.g. an extra `Bash(...)` allow for a
  project-specific tool) without editing `ERA_ALLOW_PATTERNS` in
  source. This is what closes the prompt-loop the autonomy rule
  already closed at the AskUserQuestion layer — without it, every
  Bash / Edit / Task during Stage 6 still asked the operator.
- **Autonomy hook** (v0.1.7.1 Phase D-3): `ensure_claude_settings`
  also writes a `hooks.PreToolUse` block with matcher
  `AskUserQuestion` that runs `era.cli check-autonomy` before any
  `AskUserQuestion` call. The hook reads `status.json.run_state`
  and exits 2 (block) when the loop is `running` or `blocked` —
  structurally preventing the outer ralph-loop agent from
  prompting the operator during Stages 1-7 + 9-10 even when its
  harness-level priors push it to confirm a long-running stage.
  The only allowed sites are `run_state: awaiting_human` (the
  legitimate Stage 8 hand-off, set by the `era-human-feedback`
  skill before its Continue prompt) and `idle` / `stopped` /
  `done` (operator outside `/era:start`). The merge semantics
  union the ERA matcher with any operator-added
  `hooks.PreToolUse` entries for other matchers.
- `status.json` is the **single source of truth** for pipeline lifecycle —
  `stage` / `stage_index` / `iteration` / `run_state` (`idle` · `running` ·
  `stopped` · `awaiting_human` · `blocked` · `done`). `config.yaml` holds only
  static project facts and `ralph-state.json` only prompt-compile metadata —
  neither carries lifecycle state. There is no daemon.
- Workspace scope: Stage 0 + Stage 1 are workspace-**global** by default
  (Stage 1 may be re-run by Stage 9's `REVISE_RERUN_STAGE1` to refresh the
  literature); Stages 2–10 run per-iteration inside `iter_NNN/`; Stage 9
  (ReAct) restarts Stages 2→end (or Stages 1→end on literature refresh) as a
  new iteration carrying the prior human feedback + the prior
  `evolution_state.json`.
- Stages 2–4 (the debate loop) fan out **persona sub-agents** to 3 model tiers —
  `era-heavy` / `era-standard` / `era-light` (`plugin/agents/`); `config.yaml`'s
  `agent_modes` block maps each debate stage to a tier. Stage 4 owns a bounded
  ADVANCE/REVISE refinement loop (`debate.max_rounds`, enforced by
  `era.cli debate-tick`) and emits the experiment-ready bundle; `era.cli
  check-experiment-brief` validates it against Rules 4 & 5.
- Stages 5–6 (the experiment) consume that bundle. Stage 5 expands it into a
  task DAG (`era.cli check-task-plan` is the guard); Stage 6 walks the DAG with
  a deterministic backbone — `gpu_scheduler.py` (cross-workspace `fcntl` GPU
  leases under `.era/scheduler/`, `parallel_packed` co-resident judges by
  default — `serial_full_pool` Rule-6 judges when pinned), the
  `experiment_state.py` task/marker/recovery model, and `error_heal.py`'s
  bounded auto-fix circuit breaker — driven via the `era.cli` Stage-6
  subcommands (`init-experiment`, `gpu-scan`, `claim-batch`, `release-gpus`,
  `experiment-status`, `record-task`, `recover-experiment`, `heal-tick`). The
  `experiment` block in `config.yaml` carries the policy (Rule 6 mode, poll
  cadence, retry cap); `experiment.codex_reviewer` (default false) opts into the
  separate `era-codex-reviewer` sub-agent for runner code review.
  **Stage 6 cannot silently scope-reduce** (v0.1.6 hardening): the
  orchestration-layer guard `era.cli check-experiment-completion` is the
  authoritative advance gate — `complete: true` iff every chosen_config from
  the Stage 4 brief either produced real per-sample scores OR was skipped
  with a Stage 4 pivot-matrix `skip_proof`. `record-task outcome=skipped` on
  an eval task is rejected (`unauthorized_skip`) unless the request carries
  a `pivot_proof` matching `experiment_brief.pivot_matrix[*].action`; a
  runner's `done.json` with `status: skipped` for an eval task is converted
  to `status: failure` by `apply_detection`; runtime "missing deps" /
  "budget tight" must be recorded as `outcome: failure`. **Pre-Stage-8 the
  loop never blocks** (v0.1.6 Phase C-1): when
  `check-experiment-completion` returns `complete: false`, Stage 6 calls
  `era.cli auto-revise` — which writes `<iter>/auto_revise/trigger.json`,
  fires `react_tick("REVISE_SKIP_STAGE1", ...)`, and either scaffolds a
  next iter (under-cap) or returns `forced_advance: True` (cap reached,
  loop advances to Stage 10's terminal block). The same auto-revise hook
  replaces every other pre-Stage-8 block site (Stages 4/5/7 brief or
  task-plan or comparison failures). Stage 9's advisor reads the prior
  iter's trigger via the new `parent_feedback.auto_revise_trigger`
  pointer and uses `diagnostic.missing_configs` to drop infeasible
  configs from the next iter. Only Stage 8 and Stage 10 still own
  blocking states (`awaiting_human` / `stopped` / `blocked` for Stage 8,
  `done` for Stage 10). **Stage 6 saturates available GPUs**
  (v0.1.7 Phase D-1): the Stage 6 skill writes every batch runner up
  front, launches the whole batch as background bash jobs in one
  shell pass (`nohup ... &` with disjoint `CUDA_VISIBLE_DEVICES`),
  and waits via the new event-driven `era.cli wait-for-any-done` —
  which returns within ~250 ms of the first task's `done.json`
  instead of sleeping for `experiment.poll_interval_s`. The
  Family-B default flipped to `parallel_on_unallowed_gpus` so
  Family-B evals run on GPUs outside a resident judge's pool, and
  the optional `experiment.max_parallel_runners` (default 0 =
  uncapped) caps concurrent runners when host CPU / disk pressure
  matters. **Pass/recall auto-validation gate** (v0.1.7 Phase C-2):
  between Stage 6's pilot and full rounds, the pipeline now runs an
  **annotated round** that scores every config on the operator's
  pre-existing `/era:annotate` notes (`<dataset>/annotations/*.json`).
  Stage 6 calls `era.cli auto-validate-prepare` to build one input
  batch per `(combination_id, method_id)` under
  `<iter>/auto_validate/inputs/`, then dispatches the **light-tier
  `era-auto-validator` sub-agent** (one Task per batch, parallel) to
  decide agree/disagree per sample by semantic comparison of the
  method's structured output against the operator's free-text note.
  The batch carries `scope_gating_enabled` + an `evaluation_target`
  block (the config's measured dimension, incl. `hypothesis_text`); for
  **Family-B** configs the sub-agent is scope-aware — a note about a
  dimension the metric does not measure is marked `applicable: false`
  and excluded from the recall denominator (it is not penalized for
  missing an out-of-scope defect), while Family-A judges stay fully
  accountable. Then `auto-validate-finalize` aggregates judgments into per-config
  `pass_rate` / `recall_rate` and writes
  `<iter>/auto_validate/result.json`. Configs that meet
  `experiment.auto_validate_pass_threshold` AND
  `auto_validate_recall_threshold` proceed to the full round; the
  rest are pre-skipped via the new authorized-skip path
  `init-experiment auto_validate_skips: [cid, ...]` (validated against
  `result.json.failing_configs`, with the same anti-fabrication
  guarantees as Stage 4 pivot proofs). When `any_passed: false`,
  Stage 6 calls `auto-revise reason=stage7_auto_validate_failed` and
  Phase C-1's machinery scaffolds the next iter so Stage 2 can
  search for better evaluation methods. The annotated round is
  skipped when annotation count < `auto_validate_min_samples`
  (default 10), preserving today's behaviour on un-annotated
  datasets. **Phase C-2.2 hardening:** Stages 2-4 now consume
  annotation evidence as a first-class input. Stage 2's brainstorm
  fetches `era.cli list-annotations` (gated on
  `data.use_annotation_evidence`) and tells each persona to design
  evaluators that catch the operator's flagged failure modes; each
  candidate's `hypothesis_id` text carries a tag naming the
  targeted mode. Stage 3's rigor-critic scores **annotation
  coverage** 0-3 per candidate. Stage 4's synthesizer biases
  `chosen_configs` so the union covers every flagged failure mode
  at least once. The validator's pilot-block requirement is now
  pilot-mode-only (annotated and full eval tasks may omit `pilot:`),
  and a `_normalize_judge` helper strips modality suffixes
  (`-pointwise`, `-pairwise`, `-flag`, `-judge`, `-rubric`, `-v\d+`)
  before comparing `serve.judge` against `eval.judge`, so
  cosmetic LLM variance no longer fails Stage 5. **Phase C-2.4
  fail-loud:** Stage 6 pre-flights the gate by checking that the
  task plan actually contains annotated-mode eval tasks before
  calling `auto-validate-prepare`. When the plan has no annotated
  tasks despite the dataset carrying annotations (the legacy
  v0.1.6 case), Stage 6 calls
  `auto-revise reason=stage5_missing_annotated_tasks`; Phase C-1's
  machinery scaffolds a fresh iter where Stage 2-4 emit them.
  `auto_validate.build_batches` and `aggregate_judgments` also
  return `error: "no_annotated_scores"` (with a `missing_reason`
  distinguishing `no_annotated_tasks_in_plan` from
  `annotated_round_didnt_run`) instead of silently emitting an
  all-fail result. The two cases route differently: planner drift
  → auto-revise; runtime failure → skip the gate and proceed to
  the full round with all configs.
- Stages 7–8 (the human-feedback handoff) follow. Stage 7
  (`pre_human_comparison`) assembles `iter_NNN/comparison/comparison.json` from
  the Stage 6 results. Stage 8 (`human_feedback`) launches the `era/webapp/`
  review web app as a detached background server, writes the SSH-tunnel
  instructions to `logs/iterations/`, sets `run_state: awaiting_human`, and
  **blocks in the same ralph-loop iteration on an `AskUserQuestion`
  confirmation prompt** (Continue / Still working / Cancel; **Other** accepts
  free text). On Continue the skill verifies `feedback-status`, stops the
  server, sets `run_state: running`, and returns — the loop advances directly
  into Stage 9. The deterministic backbone is
  `era/orchestration/human_feedback.py`, driven via the `era.cli` subcommands
  `serve-feedback` / `feedback-status` / `finalize-feedback` / `stop-feedback`.
  `/era:resume` remains as the fallback for the closed-terminal case
  (re-enters Stage 8; the new Step 7 pre-check skips the wait prompt when
  feedback is already finalized).
- Stage 9 (`react`) is the iteration gate. The `era-react` skill dispatches
  the `era-react-advisor` sub-agent to read every iter's
  `human/{feedback,human_labels}.json` plus the prior iter's
  `react/evolution_state.json`, the prior iter's optional
  `auto_revise/trigger.json` (when this iter is the recovery from a
  pre-Stage-8 auto-revise), and `research/literature.md` §F (EA /
  iterative-refinement literature, added to Stage 1's first-iter
  decomposition), and synthesize `iter_NNN/react/evolution_state.json`
  (cumulative_feedback trajectory, operator-veto `exclude_list`, typed
  `evolution_proposals`, `general_failure_modes`). It picks one of
  `ADVANCE | REVISE_SKIP_STAGE1 | REVISE_RERUN_STAGE1`. The deterministic
  backbone is `era/orchestration/react.py`, driven via the `era.cli`
  subcommands `react-aggregate` / `react-tick` / `create-next-iteration` /
  `check-evolution-state`. `react.max_iterations` (default 5 in
  `config.yaml`) caps the loop — `react-tick` forces ADVANCE at the cap.
  On `REVISE_*`, `create-next-iteration` atomically scaffolds `iter_{N+1}/`,
  swaps `current`, populates `iter_{N+1}/iteration.json.parent_feedback`,
  and resets `stage_index` to the *last completed* stage so the next ralph
  pass dispatches the right stage — `0` for `REVISE_RERUN_STAGE1` (next
  pass runs Stage 1: research) or `1` for `REVISE_SKIP_STAGE1` (next pass
  runs Stage 2: plan_brainstorm).

## Conventions

- **Plugin layout.** ERA ships as a Claude Code **plugin** (loaded with
  `claude --plugin-dir ./plugin`). Per plugin convention every component lives
  directly under `plugin/` — skills in `plugin/skills/<name>/SKILL.md`, subagents
  in `plugin/agents/<name>.md`, slash-commands in `plugin/commands/<name>.md`,
  helper scripts in `plugin/scripts/`; **only** `plugin.json` sits in
  `plugin/.claude-plugin/`. Do **not** use `.claude/skills` or `.claude/agents` —
  those are for project-scoped *non-plugin* assets, which ERA does not use.
  Plugin skills/agents are namespaced `era:<name>` (e.g. `era:era-literature`).
- **Prompts** live flat in `docs/prompts/` (unversioned). `knowledge/` holds
  plans for *building* ERA — it is never loaded at runtime.
- **Skills and agents declare their tool allowlist** (`allowed-tools` /
  `tools`) and omit `AskUserQuestion` — this structurally enforces the autonomy
  rule. Slash-commands do not declare `allowed-tools`.
- Behavioral prompts are thin-command-driven: a `/era:*` command or a skill
  reads a prompt under `docs/prompts/` and follows it; the `era` package stays
  deterministic.
- Tests: `.venv/bin/python3 -m pytest -q`. Add tests with new behavior.
- Git: develop on the version branch (e.g. `v0.1.3`); `master` is
  release-only. Do not `git add .` — stage files explicitly.
