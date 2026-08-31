---
description: "Stop a running ERA pipeline"
argument-hint: "[project name]"
---

# /era:stop

Halt a running ERA pipeline. The run is paused, not discarded — resume it later
with `/era:resume`.

## Setup

Resolve these once:

- **`REPO`** — the ERA repo root, `${CLAUDE_PLUGIN_ROOT}/..` (contains `era/`,
  `plugin/`, `workspaces/`).
- **`PY`** — `${CLAUDE_PLUGIN_ROOT}/../.venv/bin/python3`.
- **`WS`** — the workspace **absolute path**: if `$ARGUMENTS` names a project,
  `REPO/workspaces/$ARGUMENTS`; otherwise the current working directory. It must
  contain `status.json`.

## Steps

1. **Mark the run stopped** — set `run_state: stopped`. The Ralph loop checks
   `run_state` at the start of every iteration and exits cleanly when it is not
   `running`:

   ```bash
   "$PY" -m era.cli update-status <<JSON
   {"workspace_path": "<WS>", "run_state": "stopped"}
   JSON
   ```

2. **Cancel the Ralph loop** — use the **Skill** tool to call
   `ralph-loop:cancel-ralph`. If no loop is active this is a harmless no-op.

3. Tell the operator the project is stopped and can be continued with
   `/era:resume`.
