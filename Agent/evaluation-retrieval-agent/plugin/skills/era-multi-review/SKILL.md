---
name: era-multi-review
description: ERA Stage 3 — critique the brainstormed evaluation-protocol candidates with 3 debate critic persona sub-agents (alignment-critic, feasibility-critic, rigor-critic), then merge their reviews into design/reviews.md with a per-candidate consensus verdict.
allowed-tools: Read, Write, Glob, Grep, Bash, Task
---

# ERA Multi-Review (Stage 3)

Run a multi-persona **debate** over the Stage 2 candidate evaluation protocols —
critique and score every candidate — and write a consolidated review into the
active iteration for Stage 4 to decide on.

The workspace path is passed as `$ARGUMENTS`.

## How to run

The **ERA repo root** is the parent of `${CLAUDE_PLUGIN_ROOT}` — it contains
`era/`, `plugin/`, `docs/`, and `workspaces/`.

1. Read the behavioral prompt `docs/prompts/stage3_review.md` (relative to the
   ERA repo root).
2. Follow its steps **exactly**, for the workspace given as `$ARGUMENTS`.

You are the **orchestrator**: the behavioral prompt has you read
`design/candidates.json`, resolve the configured sub-agent tier, fan the
critique out to 3 parallel critic persona sub-agents (via the `Task` tool), and
merge their reviews into `<workspace>/<current iteration>/design/reviews.md`.

**Resilience:** if a critic sub-agent fails or returns nothing, continue with the
others and note the gap — never block on one persona. Retry a transient tool
failure once before treating it as real.

**Sub-agent contract:** the critic personas reason in their own context windows
and write their reviews to files; they return only a path + a one-line status.
Merge by reading those files from disk — this keeps the skill fast and
low-context.

**Never ask the operator anything** — this skill runs inside ERA's autonomous
loop. Resolve every ambiguity from `spec.md` / `config.yaml` and the candidate
files, decide, and record the decision in `<workspace>/logs/iterations/`.
