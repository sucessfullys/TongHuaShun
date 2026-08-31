---
name: era-pre-human-comparison
description: ERA Stage 7 — assemble the pre-human comparison view. Confirm the Stage 6 results are complete, distil them into iter_NNN/comparison/comparison.json (per-config scores + cross-family findings), and, from iteration 2 on, fold in the prior iteration's human labels so the human review opens with full context.
allowed-tools: Read, Write, Glob, Grep, Bash
---

# ERA Pre-human Comparison (Stage 7)

Assemble the comparison view the Stage 8 human-feedback web app opens with —
the bridge between the finished experiment (Stage 6) and the human review
(Stage 8).

The workspace path is passed as `$ARGUMENTS`.

## How to run

The **ERA repo root** is the parent of `${CLAUDE_PLUGIN_ROOT}` — it contains
`era/`, `plugin/`, `docs/`, and `workspaces/`.

1. Read the behavioral prompt `docs/prompts/stage7_pre_human_comparison.md`
   (relative to the ERA repo root).
2. Follow its steps **exactly**, for the workspace given as `$ARGUMENTS`.

**Never ask the operator anything** — this skill runs inside ERA's autonomous
loop. Resolve every ambiguity from `config.yaml` / `spec.md` / the result files,
decide, and record the decision in `<workspace>/logs/iterations/`. Do not
advance `status.json` — the ralph loop owns the stage transition.
