---
name: era-plan-brainstorm
description: ERA Stage 2 — brainstorm candidate evaluation protocols for a project's AIGC task by fanning out 4 generator persona sub-agents (judge-advocate, metrics-advocate, cost-pragmatist, hybrid-innovator) over the Stage 1 literature survey, then merging their proposals into design/candidates.md and a structured design/candidates.json.
allowed-tools: Read, Write, Glob, Grep, Bash, Task
---

# ERA Plan Brainstorm (Stage 2)

Brainstorm **candidate evaluation protocols** — concrete evaluator
configurations — for this ERA project's AIGC task, and write a structured
candidate pool into the active iteration for Stage 3 to debate.

The workspace path is passed as `$ARGUMENTS`.

## How to run

The **ERA repo root** is the parent of `${CLAUDE_PLUGIN_ROOT}` — it contains
`era/`, `plugin/`, `docs/`, and `workspaces/`.

1. Read the behavioral prompt `docs/prompts/stage2_brainstorm.md` (relative to
   the ERA repo root).
2. Follow its steps **exactly**, for the workspace given as `$ARGUMENTS`.

You are the **orchestrator**: the behavioral prompt has you read the Stage 1
survey, resolve the configured sub-agent tier, fan the brainstorm out to 4
parallel generator persona sub-agents (via the `Task` tool), and merge their
proposals into `<workspace>/<current iteration>/design/candidates.md` plus the
structured `design/candidates.json`.

**Resilience:** if a generator sub-agent fails or returns nothing, continue with
the others and note the gap — never block on one persona. Retry a transient tool
failure once before treating it as real.

**Sub-agent contract:** the generator personas do the heavy reasoning in their
own context windows and write their proposals to files; they return only a path
+ a one-line status. Merge by reading those files from disk — this keeps the
skill fast and low-context.

**Never ask the operator anything** — this skill runs inside ERA's autonomous
loop. Resolve every ambiguity from `spec.md` / `config.yaml` /
`research/literature.md`, decide, and record the decision in
`<workspace>/logs/iterations/`.
