---
name: era-experiment-plan
description: ERA Stage 5 — expand the Stage 4 experiment_brief.json into a runnable, dependency-ordered experiment task DAG (experiments/plans/task_plan.json + task_plan.md) via a planner persona sub-agent, and validate it deterministically.
allowed-tools: Read, Write, Glob, Grep, Bash, Task
---

# ERA Experiment Planning (Stage 5)

Expand the Stage 4 experiment-ready handoff bundle into the **experiment task
DAG** — the dependency-ordered plan of serve / eval / aggregate / compare tasks
that Stage 6 executes to produce the evaluation results.

The workspace path is passed as `$ARGUMENTS`.

## How to run

The **ERA repo root** is the parent of `${CLAUDE_PLUGIN_ROOT}` — it contains
`era/`, `plugin/`, `docs/`, and `workspaces/`.

1. Read the behavioral prompt `docs/prompts/stage5_experiment_plan.md` (relative
   to the ERA repo root).
2. Follow its steps **exactly**, for the workspace given as `$ARGUMENTS`.

This is a **single-pass** stage (not the Stage 2-4 debate loop): the behavioral
prompt has you dispatch one `planner` persona sub-agent, validate the task plan
deterministically (`era.cli check-task-plan`), and — if the guard reports
problems — re-dispatch the planner once with those problems folded in.

**Resilience:** retry a transient tool failure once before treating it as real;
if the planner sub-agent fails twice, fall back to planning in this context —
never leave a half-written `task_plan.json`.

**Sub-agent contract:** the planner does the heavy reasoning in its own context
window and writes `experiments/plans/task_plan.json` + `task_plan.md`, returning
only the paths + a one-line status. Re-read the plan from disk to validate it.

**Never ask the operator anything** — this skill runs inside ERA's autonomous
loop. Resolve every ambiguity from the workspace files, decide, and record the
decision in `<workspace>/logs/iterations/`.
