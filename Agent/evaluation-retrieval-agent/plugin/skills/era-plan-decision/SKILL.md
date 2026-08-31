---
name: era-plan-decision
description: ERA Stage 4 — synthesize the chosen evaluation experiment design from the candidates and reviews via a synthesizer persona sub-agent, run the bounded ADVANCE/REVISE debate loop, and emit the experiment-ready bundle (design/plan.md, design/experiment_brief.json, design/hypotheses.md, design/decision.json).
allowed-tools: Read, Write, Glob, Grep, Bash, Task
---

# ERA Plan Decision (Stage 4)

Synthesize the Stage 2 candidates and the Stage 3 reviews into the **chosen
evaluation experiment design**, run a bounded refinement loop, and write the
experiment-ready handoff bundle the later Experiment stages execute.

The workspace path is passed as `$ARGUMENTS`.

## How to run

The **ERA repo root** is the parent of `${CLAUDE_PLUGIN_ROOT}` — it contains
`era/`, `plugin/`, `docs/`, and `workspaces/`.

1. Read the behavioral prompt `docs/prompts/stage4_decision.md` (relative to the
   ERA repo root).
2. Follow its steps **exactly**, for the workspace given as `$ARGUMENTS`.

You **own the debate loop**: the behavioral prompt has you dispatch the
`synthesizer` persona sub-agent, validate the experiment brief deterministically
(`era.cli check-experiment-brief`), apply the round cap (`era.cli debate-tick`),
and — on a REVISE verdict — re-run the Stage 2 + Stage 3 persona fan-out
internally before re-synthesizing. The loop is **in-iteration refinement**: it
never creates a new `iter_NNN/`. On ADVANCE you write the full bundle:
`design/plan.md`, `design/experiment_brief.json`, `design/hypotheses.md`,
`design/decision.json`.

**Resilience:** retry a transient tool failure once before treating it as real;
if the synthesizer sub-agent fails twice, fall back to synthesizing in this
context — never leave a half-written bundle.

**Sub-agent contract:** the synthesizer — and, on a REVISE round, the
re-dispatched Stage 2/3 personas — do the heavy reasoning in their own context
windows and write to files, returning only a path + a one-line status. Re-read
artifacts from disk between debate rounds so the skill's context stays flat
regardless of `debate.max_rounds`.

**Never ask the operator anything** — this skill runs inside ERA's autonomous
loop. Resolve every ambiguity from the workspace files, decide, and record the
decision in `<workspace>/logs/iterations/`. The round cap, not an operator,
ends the loop.
