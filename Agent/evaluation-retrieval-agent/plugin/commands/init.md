---
description: "Initialize an ERA evaluation project (Stage 0 — Task Init)"
argument-hint: "[mission text]"
---

# /era:init

**Stage 0 — Task Init.** Interactively initialize a General Image Generation /
Editing Evaluation Retrieval Agent (ERA) project: parse the mission, probe the
environment, confirm ambiguous items with the operator, and scaffold a workspace.

The operator's mission is `$ARGUMENTS`.

## How to run this command

The **ERA repo root** is the directory containing `plugin/`, `era/`, and
`knowledge/` — it is your current working directory (the parent of
`${CLAUDE_PLUGIN_ROOT}`).

1. Read the behavioral prompt at
   `docs/prompts/stage0_init.md` (relative to the ERA repo root).
2. Follow its steps **exactly**, in order.
3. Treat `$ARGUMENTS` as the operator's initial mission text. If it is empty,
   the prompt tells you to ask for it.

All Python is invoked with `.venv/bin/python3` from the ERA repo root. The
behavioral prompt specifies every command, every probe, and when to ask the
operator questions versus when to scaffold the workspace.
