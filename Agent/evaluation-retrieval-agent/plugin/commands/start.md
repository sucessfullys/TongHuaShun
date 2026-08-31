---
description: "Run the ERA evaluation pipeline autonomously (Stage 1+)"
argument-hint: "[project name]"
---

# /era:start

Run the ERA evaluation-retrieval pipeline for an initialized workspace as an
**autonomous Ralph loop**. From Stage 1 until the human-feedback stage the loop
**never stops to ask the operator** — it decides from `spec.md` / `config.yaml`
and records every decision in the iteration log.

## Prerequisite

The `ralph-loop` plugin (Anthropic's `claude-plugins-official` marketplace) runs
the loop. ERA enables it automatically: the workspace's `.claude/settings.json`
(scaffolded by `/era:init`) and the repo's checked-in `.claude/settings.json`
both list it under `enabledPlugins`, so `/ralph-loop` loads on session start.
If it is unavailable — never installed
(`claude plugin install ralph-loop@claude-plugins-official`), or `jq` is missing
(the plugin's Stop hook needs `jq`) — ERA runs the loop in **manual-fallback
mode** instead (step 3b); the pipeline still completes.

## Setup

You are normally launched from inside the workspace directory. Resolve:

- **`REPO`** — the ERA repo root, `${CLAUDE_PLUGIN_ROOT}/..` (contains `era/`,
  `plugin/`, `workspaces/`).
- **`PY`** — `${CLAUDE_PLUGIN_ROOT}/../.venv/bin/python3`.
- **`WS`** — the workspace **absolute path**: if `$ARGUMENTS` names a project,
  `REPO/workspaces/$ARGUMENTS`; otherwise the current working directory. It must
  contain `status.json`, `spec.md`, `config.yaml`, and `current` — if not, tell
  the operator to run `/era:init` first, then stop.

## Preflight — before anything else

Run the environment check:

```bash
bash "${CLAUDE_PLUGIN_ROOT}/scripts/preflight.sh"
```

If it exits **non-zero**, print its output verbatim to the operator and
**stop** — do not run any `era.cli` command and do not enter the loop. A
non-zero result means the ERA repo itself is broken (mid-merge, conflict
markers, or a missing virtualenv) and must be fixed before the pipeline can run.

On success it also prints a **`loop engine = …`** line — note it; step 3 uses
it to pick the loop mode (`ralph-loop plugin` when `jq` is present, `manual
fallback` when not — the plugin's Stop hook needs `jq`).

## Steps

1. **Mark the run started** — set `run_state: running`:

   ```bash
   "$PY" -m era.cli update-status <<JSON
   {"workspace_path": "<WS>", "run_state": "running"}
   JSON
   ```

2. **Compile the Ralph-loop prompt:**

   ```bash
   "$PY" -m era.cli write-ralph-prompt <<JSON
   {"workspace_path": "<WS>"}
   JSON
   ```

   It writes `<WS>/.claude/ralph-prompt.txt` and prints a JSON result with
   `prompt_path`. On `error`, report it and stop.

3. **Run the loop.** Pick the mode from preflight's `loop engine` line:

   **a. `ralph-loop plugin`** (jq present) — use the **Skill** tool to call
   `ralph-loop:ralph-loop` with:
   - prompt (a single line): `Follow the instructions in <prompt_path> to run
     the ERA evaluation pipeline for project <project name>.`
   - arguments: `--max-iterations 12 --completion-promise 'ERA_PIPELINE_COMPLETE'`

   If the `ralph-loop:ralph-loop` skill turns out to be unavailable, switch to
   mode **b**.

   **b. `manual fallback`** (jq missing, or `/ralph-loop` unavailable) — run the
   loop yourself, with **no** Stop hook: read `<WS>/.claude/ralph-prompt.txt`
   and follow it directly — read `<WS>/status.json`, run the next stage, advance
   it with `era.cli update-status`, then **re-read `status.json` and continue**.
   Loop until `run_state` is terminal (`blocked`, `awaiting_human`, or `done`) —
   do not stop after a single stage.

   Either way the loop advances one pipeline stage per iteration — Stage 1
   (literature research) first. Do not intervene and do not ask the operator
   anything while it runs.

## Lifecycle

`/era:status` — progress · `/era:stop` — halt · `/era:resume` — continue.

## Status: v0.1.4

ERA implements **Stage 0 (init)**, **Stage 1 (literature research)**,
**Stages 2–4 (the idea-generation + debate loop)**, and **Stages 5–6 (the
experiment — plan the task DAG, then run it on free GPUs)**. The loop runs
through Stage 6 and collects per-config evaluation results in
`iter_NNN/experiments/results/`. Stage runners for Stages 7–11 are stubs — the
loop then records the first unimplemented stage as pending and completes. Each
new stage runner makes the same loop do more.
