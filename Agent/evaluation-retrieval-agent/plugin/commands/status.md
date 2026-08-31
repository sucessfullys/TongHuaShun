---
description: "Show ERA project progress"
argument-hint: "[project name]"
---

# /era:status

Report the progress of ERA project workspace(s).

## Setup

Resolve these once:

- **`REPO`** — the ERA repo root, `${CLAUDE_PLUGIN_ROOT}/..` (contains `era/`,
  `plugin/`, `workspaces/`).
- **`PY`** — `${CLAUDE_PLUGIN_ROOT}/../.venv/bin/python3`.
- **`WS`** — *optional for this command.* If `$ARGUMENTS` names a project, set
  it to `REPO/workspaces/$ARGUMENTS`; with no argument, leave `WS` unset to
  summarize **every** workspace.

## Steps

- Run `era.cli status`. With no `WS`, send an empty object to list every
  workspace:

  ```bash
  "$PY" -m era.cli status <<JSON
  {}
  JSON
  ```

  To scope to one project, send `{"workspace_path": "<WS>"}` instead.

- Render the JSON `projects` array as a readable table — one row per project
  with: project name, `stage`, `stage_index`, `iteration`, `run_state`,
  `updated_at`, and whether the literature survey (`has_literature`) exists.
- `run_state` values: `idle` (initialized, not started) · `running` ·
  `awaiting_human` (paused for operator feedback) · `blocked` (stopped at a
  not-yet-implemented stage) · `done` · `stopped`.
- If `count` is 0, tell the operator no initialized workspaces were found and
  to run `/era:init` first.
