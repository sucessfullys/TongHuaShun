---
name: era-experiment
description: ERA Stage 6 — execute the Stage 5 task DAG. Write each evaluator runner, review it, schedule it on free GPUs (parallel_packed co-resident VLM judges by default; serial_full_pool when pinned), run it, recover failures, heal errors, and collect per-config evaluation results into experiments/results/.
allowed-tools: Read, Write, Glob, Grep, Bash, Task
---

# ERA Full Experiment (Stage 6)

Execute the experiment task DAG that Stage 5 produced — bring up VLM judges,
run the evaluator configurations over the dataset, recover failures, and collect
the per-config evaluation results that the later comparison stages consume.

The workspace path is passed as `$ARGUMENTS`.

## How to run

The **ERA repo root** is the parent of `${CLAUDE_PLUGIN_ROOT}` — it contains
`era/`, `plugin/`, `docs/`, and `workspaces/`.

1. Read the behavioral prompt `docs/prompts/stage6_experiment.md` (relative to
   the ERA repo root), and the runner contract `docs/prompts/_experiment_protocol.md`.
2. Follow the Stage 6 prompt's steps **exactly**, for the workspace given as
   `$ARGUMENTS`.

You own the **DAG walk**: the prompt has you initialise the experiment state,
claim runnable batches with the deterministic GPU scheduler (`era.cli
claim-batch` — it enforces `experiment.family_a_execution`: `parallel_packed`
(default) co-residents right-sized judges on disjoint GPU subsets and backfills
Family-B evals, while `serial_full_pool` runs Rule 6 — one judge owning the
whole pool), write + review + run each task's runner, monitor marker files,
recover and heal failures with bounded retries, gate the pilot pass on the
Stage-4 pivot matrix, then run the full pass and collect `experiments/results/`.

**Code review:** when `config.yaml`'s `experiment.codex_reviewer` is `true`,
review each new runner with the `era-codex-reviewer` sub-agent (an independent
Codex perspective); when `false` (the default), self-review the runner inline
against the protocol checklist.

**Resilience:** retry a transient tool failure once before treating it as real.
A failed evaluator task is healed through `era.cli heal-tick`'s bounded circuit
breaker; a `give_up` config is recorded and the round continues — one bad config
never sinks the experiment.

**Never ask the operator anything** — this skill runs inside ERA's autonomous
loop. Resolve every ambiguity from `config.yaml` / `spec.md` / the task plan,
decide, and record the decision in `<workspace>/logs/iterations/`. Do not
advance `status.json` — the ralph loop owns the stage transition.
