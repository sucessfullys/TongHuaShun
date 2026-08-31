# ERA — Evaluation Retrieval Agent

**General AIGC Evaluation Retrieval Agent.** A config-driven, multi-stage agent
(hosted by the Claude Code CLI) that researches, builds, and human-validates
evaluation protocols for AIGC tasks — image generation, image editing, and
beyond — and *retrieves* the evaluation method best aligned with human
judgement at the lowest cost. The task is never hardwired; it is supplied at
init via the operator's mission.

This release (**v0.1.7**) implements **Stages 0–9 end-to-end**:
**Stage 0 — Task Init** (`/era:init`), **Stage 1 — Research (literature)**,
**Stages 2–4 — the idea-generation + debate loop** (`plan_brainstorm` →
`multi_review` → `plan_decision`), **Stages 5–6 — the experiment**
(`experiment_plan` → `full_experiment`), **Stages 7–8 — pre-human
comparison + human feedback** (`pre_human_comparison` → `human_feedback`,
a review web app over an SSH tunnel), and **Stage 9 — ReAct** (the
iteration gate that decides ADVANCE / REVISE between iters), plus the
autonomous `/era:start` runtime loop and the `/era:status` / `/era:stop`
/ `/era:resume` lifecycle commands. Only **Stage 10 (`final_report`)**
is still a stub; the `dataset_ship` slot was retired in v0.1.6 because
ERA retrieves an evaluation protocol, it does not ship a dataset.

v0.1.7 ships three hardenings on top of the base pipeline:

- **Phase C-1 — auto-revise.** Pre-Stage-8 stages never block the
  loop. When Stage 4/5/6/7 produces an invalid output, `era.cli
  auto-revise` fires a Stage 9 `REVISE_SKIP_STAGE1` and scaffolds the
  next iter (capped by `react.max_iterations`).
- **Phase D-1 — Stage 6 GPU saturation.** Stage 6 launches every
  batch task as a background bash job in one shell pass and waits
  via the event-driven `era.cli wait-for-any-done` (returns within
  ~250 ms of any task's `done.json`). `family_b_schedule` default
  flipped to `parallel_on_unallowed_gpus`.
- **Phase C-2 — pass/recall auto-validation gate.** Between Stage 6's
  pilot and full rounds, the pipeline runs an **annotated round** on
  the operator's pre-existing `/era:annotate` notes, dispatches the
  light-tier `era-auto-validator` sub-agent to judge agree/disagree
  per sample, then either runs the full N=50 round on the passing
  configs only or auto-revises to the next iter when all configs
  fail. **`/era:annotate <dataset_root>`** is the standalone web app
  the operator uses to author those annotations. The full round's
  N samples are picked **randomly per iter** from the full dataset
  via `era.cli sample-window` (deterministic seed = workspace +
  iteration; all methods × configs in one iter score the **same**
  shuffled subset, but iter 2 picks a different set so the whole
  dataset is exercised across iters).

## Quickstart

```bash
# 1. From the ERA repo root, create the virtualenv and install the package.
python3 -m venv .venv
.venv/bin/pip install -e .
```

Then, in Claude Code:

```bash
# 2. Launch Claude Code with the ERA plugin (from the repo root).
claude --plugin-dir ./plugin --dangerously-skip-permissions

# 3. Initialize a project. Describe YOUR task — any AIGC generation/editing
#    evaluation. Illustrative example:
/era:init Evaluate the edited images under /path/to/edits, comparing two
  editing methods. Each sample directory holds source.png and edited.png.
  Use GPUs 0-1. Pretrained models are in /path/to/models.

# 4. cd into the new workspace (the guide printed by /era:init names the
#    path), relaunch Claude Code there, and run the autonomous pipeline.
/era:start
```

`/era:init` probes the environment (GPUs, data roots, checkpoints, `.env`
credentials), confirms ambiguous items with you, and scaffolds
`workspaces/{project}/`. `/era:start` then runs the pipeline **autonomously** —
it does not stop to ask you anything until the human-feedback stage.

## Commands

| Command | Purpose |
|---------|---------|
| `/era:init <mission>` | Stage 0 — probe the environment and scaffold a workspace (interactive). |
| `/era:start [project]` | Run the pipeline autonomously as a Ralph loop. |
| `/era:status [project]` | Show stage / iteration / run-state for ERA project(s). |
| `/era:stop [project]` | Halt a running pipeline (resumable). |
| `/era:resume [project]` | Continue a stopped or interrupted pipeline. |

## Workspace layout

```
workspaces/{project}/
  config.yaml  spec.md  status.json  CLAUDE.md        # global Stage 0 output
  .claude/settings.json  .mcp.json                    # ralph-loop plugin + arXiv MCP
  .claude/ralph-prompt.txt                            # compiled runtime prompt
  probe/                                              # raw probe artifacts
  research/literature.md                              # Stage 1 survey (global)
  shared/   logs/iterations/                          # global
  current -> iter_001                                 # active iteration
  iter_001/  design/ experiments/ serving/ comparison/ human/ deliverable/
```

Stage 0 and Stage 1 are workspace-global; Stages 2–11 run per-iteration inside
`iter_NNN/`. After Stage 8 (human feedback), Stage 10 (ReAct) restarts Stages 2
→ end as new iterations.

## Ralph loop (runtime engine)

`/era:start` runs ERA's pipeline as a **Ralph loop** — one pipeline stage per
iteration, recovering state from the workspace each turn. ERA does not ship its
own loop script; it uses the official **`ralph-loop`** plugin (the Ralph Wiggum
technique, in-session via a Stop hook).

On a fresh machine, install the plugin once (CLI):

```bash
claude plugin marketplace add anthropics/claude-plugins-official   # if not already added
claude plugin install ralph-loop@claude-plugins-official
```

ERA then keeps the plugin enabled wherever you launch Claude Code: the repo
ships a checked-in `.claude/settings.json`, and `/era:init` scaffolds one into
every workspace. Both enable `ralph-loop@claude-plugins-official` (and declare
its marketplace via `extraKnownMarketplaces`), so `/ralph-loop` loads
automatically in the repo root and in any `workspaces/<project>/`.

`/era:start` compiles `workspaces/{project}/.claude/ralph-prompt.txt` from
`docs/prompts/ralph_loop.md` and hands it to `ralph-loop:ralph-loop` with
`--max-iterations 12 --completion-promise 'ERA_PIPELINE_COMPLETE'`. The loop runs
fully autonomously from Stage 1 until the human-feedback stage; stop it early
with `/era:stop`, continue with `/era:resume`.

The plugin's Stop hook needs the **`jq`** CLI. Install it for the best loop
experience (one fresh context window per iteration); if `jq` (or the plugin)
is absent, `/era:start` automatically runs a **manual fallback** loop instead —
`preflight.sh` reports which mode applies and the pipeline still completes.

## Idea generation & debate (Stages 2–4)

After the literature survey, ERA brainstorms and debates **candidate evaluation
protocols**, then commits to one:

- **Stage 2 — `plan_brainstorm`** fans out 4 eval-domain generator personas
  (judge-advocate, metrics-advocate, cost-pragmatist, hybrid-innovator) over the
  Stage 1 survey, producing a scored candidate pool (`design/candidates.json`).
- **Stage 3 — `multi_review`** runs 3 critic personas (alignment, feasibility,
  rigor) that score every candidate (`design/reviews.md`).
- **Stage 4 — `plan_decision`** synthesizes the chosen evaluation experiment
  design and runs a bounded ADVANCE/REVISE refinement loop (capped by
  `debate.max_rounds`). It emits the **experiment-ready handoff bundle** the
  later Experiment stages execute: `design/plan.md`, `design/experiment_brief.json`,
  `design/hypotheses.md`, and `design/decision.json`. A deterministic guard
  (`era.cli check-experiment-brief`) validates the brief against the full
  handoff contract — Rules 4 & 5, every executable per-config field, and
  resolution of every `hypothesis_id` against `hypotheses.md`; a failing brief
  is itself a REVISE trigger.

Each debate persona is dispatched to one of 3 model tiers — `era-heavy` /
`era-standard` / `era-light`; `config.yaml`'s `agent_modes` block maps each
debate stage to a tier, so model spend is tunable per stage without editing the
skills.

## The experiment (Stages 5–6)

With the chosen design fixed, ERA plans and runs the experiment that produces
the evaluation results:

- **Stage 5 — `experiment_plan`** expands the Stage 4 `experiment_brief.json`
  into a runnable, dependency-ordered **task DAG** (`experiments/plans/task_plan.json`)
  — serve / eval / aggregate / compare tasks, a pilot pass gated before the full
  pass. A deterministic guard (`era.cli check-task-plan`) checks the DAG is
  acyclic, covers every candidate config, and honors Rule 6.
- **Stage 6 — `full_experiment`** walks that DAG: it writes each evaluator
  runner, reviews it, schedules it on free GPUs with a Sibyl-parity scheduler
  (cross-workspace `fcntl` GPU leases; Rule 6 — one VLM judge resident at a
  time, owning the whole pool), runs it, recovers failures from marker files,
  heals errors with a bounded circuit breaker, gates the pilot pass on the
  Stage-4 pivot matrix, and collects per-config scores into
  `experiments/results/`.

Stage 6's evaluator runners are reviewed before they run. By default ERA
self-reviews each runner inline; set `experiment.codex_reviewer: true` in
`config.yaml` (and install the `codex` CLI) to instead get an independent review
from a separate Codex model via the `era-codex-reviewer` sub-agent.

## Human feedback (Stages 7–8)

Once the experiment has results, ERA brings a human into the loop:

- **Stage 7 — `pre_human_comparison`** distils the Stage 6 results into a single
  `iter_NNN/comparison/comparison.json` — the comparison view the review opens
  with (and, from iteration 2 on, folds in the prior iteration's human labels).
- **Stage 8 — `human_feedback`** launches a **review web app** (FastAPI +
  React) as a detached background server and pauses the pipeline at
  `run_state: awaiting_human`. The operator reaches it from a laptop over an
  SSH tunnel:

  ```bash
  ssh -N -L 8731:127.0.0.1:8731 <user>@<gpu-box>   # then open http://localhost:8731/
  ```

  The app shows each result with its input(s) and what every judge scored it.
  The human flags any Family-A / hybrid judgement that is wrong and any
  Family-B relative ranking that is wrong; everything left unflagged is recorded
  as endorsed/correct. Finalizing writes `iter_NNN/human/feedback.json` and the
  derived `iter_NNN/human/human_labels.json` — the ground truth a later
  iteration's Stage 7 compares new candidate protocols against. The operator
  then runs `/era:resume` to continue.

`era.cli serve-feedback` launches the same web app standalone against any
iteration's results — to review a finished run without driving the loop.

## Literature research (Stage 1) — MCP setup

Stage 1 surveys evaluation methods, metrics, and benchmarks via parallel
`literature-scout` sub-agents over arXiv + GitHub (MCP) + web search. ERA
registers these **automatically**: `arxiv-mcp-server` is a pip dependency
(installed by `pip install -e .`), the `github` MCP is GitHub's hosted endpoint
for repository/code search, and `/era:init` scaffolds a `.mcp.json` into every
workspace so they resolve wherever `/era:start` runs. Register them for the repo
root once with `.venv/bin/python3 -m era.cli write-mcp-config`. GitHub search
needs a `GITHUB_PERSONAL_ACCESS_TOKEN` exported in your environment. See
**`docs/mcp-servers.md`** for details, Google Scholar, and verification. If no
MCP server resolves, the survey degrades gracefully to built-in web search.

## Repository

```
plugin/            Claude Code plugin
  commands/        /era:init, /era:start, /era:status, /era:stop, /era:resume
  scripts/         environment preflight (preflight.sh)
  skills/          model-invoked skills (era-literature, the Stage 2-4 debate,
                   era-experiment-plan + era-experiment for Stages 5-6,
                   era-pre-human-comparison + era-human-feedback for Stages 7-8)
  agents/          sub-agents (literature-scout; era-heavy/standard/light tiers;
                   era-codex-reviewer for Stage 6 code review)
era/               Python package — config, workspace, probes, orchestration, CLI
  probe/           environment probes (gpu, data, checkpoints, credentials)
  orchestration/   workspace scaffolding, ralph compiler, lifecycle, debate loop,
                   the Stage 5-6 task plan, GPU scheduler, experiment state /
                   results / error-heal, the Stage 8 human-feedback backbone
  webapp/          the Stage 8 review web app — FastAPI backend + React frontend
docs/prompts/      ERA runtime prompts/templates (flat, unversioned)
docs/mcp-servers.md  MCP setup for Stage 1 literature research
knowledge/         coding-task plans/prompts for building ERA (not runtime)
tests/             pytest unit + integration tests
config.example.yaml  annotated config schema reference
CLAUDE.md          project instructions for working in this repo
```

## Development

```bash
.venv/bin/python3 -m pytest          # run tests
.venv/bin/python3 -m era.cli status  # summarize workspaces (reads JSON stdin)
```
