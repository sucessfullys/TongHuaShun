---
name: era-human-feedback
description: ERA Stage 8 — pause the autonomous loop for human review. Normalize the iteration's stage 2-6 results into a review model, ensure the GPU watchdog is protecting idle cards, launch the feedback web app as a detached background server, print the operator's SSH-tunnel command and URL, set run_state to awaiting_human, then block on an AskUserQuestion confirmation prompt in the same loop iteration. On Continue, verify the operator finalized feedback in the web app (via era.cli feedback-status), stop the server, set run_state to running, and return so the loop flows directly into Stage 9.
allowed-tools: Read, Write, Glob, Grep, Bash, Task, AskUserQuestion
---

# ERA Human Feedback (Stage 8)

Hand the finished evaluation to a human. This skill starts the review web app
in the background, tells the operator exactly how to reach it over an SSH
tunnel, and then **blocks in the same ralph-loop iteration on a confirmation
prompt** until the operator selects **Continue** — at which point the skill
verifies feedback is finalized, stops the web server, and returns so the
loop advances into Stage 9.

The workspace path is passed as `$ARGUMENTS`.

## How to run

The **ERA repo root** is the parent of `${CLAUDE_PLUGIN_ROOT}` — it contains
`era/`, `plugin/`, `docs/`, and `workspaces/`.

1. Read the behavioral prompt `docs/prompts/stage8_human_feedback.md` (relative
   to the ERA repo root).
2. Follow its steps **exactly**, for the workspace given as `$ARGUMENTS`. Step 2
   normalizes the review model — it runs `era.cli build-review-model` and then
   dispatches the **`era-review-adapter`** sub-agent (via `Task`) so the web app
   renders correctly for any task and any iteration, not just the demo. The
   sub-agent is for *semantic* enrichment only (score-display factor labels,
   description prettification, judge `display` re-rendering); the deterministic
   build owns structure (`sample_key` normalization, cell merging, Family-B
   per-sample ranking aggregation). Dispatch it even when warnings are empty.

**`AskUserQuestion` is allowed here, by design.** Stage 8 is the **one
explicit exception** to the autonomy iron rule (every other stage and skill
must run unattended): the whole point of Stage 8 is the operator hand-off.
Use `AskUserQuestion` only for the wait-prompt described in
`stage8_human_feedback.md` Step 8 — never to disambiguate workspace state or
to escape from a transient error. Do not advance `status.json`'s `stage_index`
— the ralph loop owns the stage transition; this skill manages `run_state`
(`awaiting_human` while waiting, then `running` on Continue or `stopped` on
Cancel).
