---
description: "Resume a stopped or interrupted ERA pipeline (the closed-terminal fallback)"
argument-hint: "[project name]"
---

# /era:resume

Resume an ERA project that was stopped (`/era:stop`) or interrupted. The loop
recovers its position from `status.json` and continues from the current stage —
it does not restart the pipeline.

`/era:resume` is the **fallback** path for the Stage 8 hand-off: the primary
hand-off is the in-loop confirmation prompt (`AskUserQuestion`) the
`era-human-feedback` skill blocks on after launching the review web app.
Use `/era:resume` only when the operator closed the terminal during that
prompt (or otherwise interrupted the run); re-entering Stage 8 detects the
already-finalized feedback in its Step 7 pre-check and advances without
re-prompting.

## Setup

Resolve these once:

- **`REPO`** — the ERA repo root, `${CLAUDE_PLUGIN_ROOT}/..` (contains `era/`,
  `plugin/`, `workspaces/`).
- **`PY`** — `${CLAUDE_PLUGIN_ROOT}/../.venv/bin/python3`.
- **`WS`** — the workspace **absolute path**: if `$ARGUMENTS` names a project,
  `REPO/workspaces/$ARGUMENTS`; otherwise the current working directory. It must
  contain `status.json` — if not, tell the operator to run `/era:init` first,
  then stop.

## Preflight — before anything else

Run `bash "${CLAUDE_PLUGIN_ROOT}/scripts/preflight.sh"`. If it exits
**non-zero**, print its output verbatim and **stop** — the ERA repo is broken
(mid-merge, conflict markers, or a missing virtualenv) and must be fixed before
the loop can resume. On success, note its `loop engine = …` line — it selects
the loop mode in step 3.

## Human-feedback gate — before resuming

Read `<WS>/status.json`. If `run_state` is **`awaiting_human`**, the run is
paused at Stage 8 for human review — check whether the human has finished:

```bash
"$PY" -m era.cli feedback-status <<JSON
{"workspace_path": "<WS>"}
JSON
```

- If `feedback.finalized` is **`false`**, the human has not finalized their
  review yet. Print the feedback web-app URL (`server.url`) and the SSH-tunnel
  command from the latest `<WS>/logs/iterations/` note, tell the operator to
  finish the review and click **Finalize** in the web app, and **stop without
  changing `run_state`**.
- If `feedback.finalized` is **`true`**, the human is done. Stop the web server
  (`era.cli stop-feedback`), then continue to the Steps below — the loop will
  advance past Stage 8.

If `run_state` is anything else (`stopped`, `blocked`, `idle`), continue
straight to the Steps.

## Steps

1. **Clear the stop flag** — set `run_state: running`:

   ```bash
   "$PY" -m era.cli update-status <<JSON
   {"workspace_path": "<WS>", "run_state": "running"}
   JSON
   ```

2. **Recompile the Ralph-loop prompt:**

   ```bash
   "$PY" -m era.cli write-ralph-prompt <<JSON
   {"workspace_path": "<WS>"}
   JSON
   ```

3. **Relaunch the loop** exactly as `/era:start` step 3 — pick the mode from
   preflight's `loop engine` line: the `ralph-loop:ralph-loop` skill
   (`--max-iterations 12 --completion-promise 'ERA_PIPELINE_COMPLETE'`) when
   `jq` is present, or the **manual fallback** (drive the loop yourself with no
   Stop hook, looping until `run_state` is terminal) when `jq` is missing or
   `/ralph-loop` is unavailable. The loop reads `status.json` and continues from
   where it left off, autonomously, without asking the operator.

> Resuming a `blocked` project only makes progress once the blocking stage has
> a runner; until then the loop will re-block at the same stage. Resuming an
> `awaiting_human` project only proceeds once the human has finalized their
> feedback in the Stage 8 web app (the human-feedback gate above).
