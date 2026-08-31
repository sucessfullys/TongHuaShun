---
name: era-react
description: ERA Stage 9 — the iteration gate. Read every iteration's finalized human feedback, dispatch the era-react-advisor sub-agent to synthesize iter_NNN/react/evolution_state.json (cumulative_feedback, exclude_list, evolution_proposals, general_failure_modes) and decide ADVANCE | REVISE_SKIP_STAGE1 | REVISE_RERUN_STAGE1, validate the file deterministically, record the verdict (capped by react.max_iterations), and on REVISE_* atomically advance to iter_{N+1} so the loop re-runs Stage 2 (or Stage 1 first when the literature needs refreshing).
allowed-tools: Read, Write, Glob, Grep, Bash, Task
---

# ERA ReAct (Stage 9)

Decide whether the current evaluation protocol is good enough to ship — or
whether ERA should iterate again, optionally refreshing the literature first.

The workspace path is passed as `$ARGUMENTS`.

## How to run

The **ERA repo root** is the parent of `${CLAUDE_PLUGIN_ROOT}` — it contains
`era/`, `plugin/`, `docs/`, and `workspaces/`. Its venv Python is
`<repo>/.venv/bin/python3`.

1. Read the behavioral prompt `docs/prompts/stage9_react.md` (relative to the
   ERA repo root).
2. Follow its steps **exactly**, for the workspace given as `$ARGUMENTS`. The
   prompt's six-step flow is:
   - read `status.json` + `config.yaml`'s `react` block;
   - run `era.cli react-aggregate` to build the deterministic cumulative
     summary;
   - dispatch the `era-react-advisor` sub-agent (via `Task`) — it writes
     `iter_N/react/evolution_state.json` (and on `REVISE_RERUN_STAGE1` also
     `iter_N/react/literature_update_brief.md`);
   - validate the file with `era.cli check-evolution-state` — a malformed
     payload is a blocker the skill surfaces and re-dispatches the sub-agent;
   - record the verdict with `era.cli react-tick` — this is where
     `react.max_iterations` is enforced (any `REVISE_*` is forced to
     `ADVANCE` once `status.iteration` reaches the cap);
   - on `REVISE_*`, atomically advance with `era.cli create-next-iteration`
     (scaffolds `iter_{N+1}/`, swaps `current`, populates
     `iter_{N+1}/iteration.json.parent_feedback`, and resets `stage_index`
     to the *last completed* stage — `0` for rerun so the next ralph pass
     dispatches Stage 1 / `research`, or `1` for skip so it dispatches
     Stage 2 / `plan_brainstorm`).

You **own the iteration gate**: this skill is the only place that creates a
new `iter_NNN/` after Stage 0. ADVANCE leaves `status` at `stage_index = 9`;
the ralph loop then advances to index 10 (final_report). REVISE_* leaves
`create-next-iteration` to update `status`; the loop's next iteration re-reads
`status.json` and resumes at the new iter's `stage_index`.

**Resilience:** retry a transient tool failure once before treating it as
real; if the advisor sub-agent fails twice, synthesize a minimal
`evolution_state.json` in this context from the `react-aggregate` output (no
proposals, no excludes, `decision: ADVANCE`, `rationale: "advisor unreachable;
shipping the best config from cumulative_feedback"`) and continue — never
leave a Stage 9 iteration half-decided.

**Sub-agent contract:** the advisor reads every prior feedback artifact in its
own context window and writes one file. Re-read it from disk before
validating so this skill's context stays flat.

**Never ask the operator anything** — this skill runs inside ERA's autonomous
loop. Resolve every ambiguity from the workspace files, decide, and record
the decision in `<workspace>/logs/iterations/`. The iteration cap, not an
operator, ends the loop.
