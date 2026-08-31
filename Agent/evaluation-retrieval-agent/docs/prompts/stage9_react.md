# ERA Stage 9 — ReAct (decide ADVANCE / REVISE_*, advance the iteration)

You are running ERA Stage 9 for one project workspace. Goal: read every
iteration's finalized human feedback, synthesize the cumulative learning
state, decide one of three verdicts — `ADVANCE | REVISE_SKIP_STAGE1 |
REVISE_RERUN_STAGE1` — and, on REVISE_*, atomically advance the workspace
to the next iteration so the loop continues.

You **own the iteration gate**. **Never ask the operator anything** — the
iteration cap ends the loop, not an operator. The advisor sub-agent dispatch
typically takes 5-10 minutes and is unattended; the Phase D-3 PreToolUse
hook will structurally block any `AskUserQuestion` call from this stage.
The workspace path was passed as the skill argument (`$ARGUMENTS`). The
**ERA repo root** is the parent of the directory holding the `era/`
package; its venv Python is `<repo>/.venv/bin/python3`.

## Step 1 — Read state

Resolve the active iteration via `<workspace>/current` → `<iter>`. Read
`<workspace>/status.json` (note its `iteration` field, call it `N`),
`<workspace>/config.yaml`'s `react` block (`max_iterations`,
`endorsement_threshold`, `min_alignment_samples`), and confirm
`<iter>/human/human_labels.json` exists and is finalized — if it does not,
Stage 8 did not finish; record a `pending` note in
`<workspace>/logs/iterations/` and exit (the ralph loop will re-enter).

## Step 2 — Aggregate cumulative feedback

Run the deterministic aggregator — it walks every iter's finalized feedback
and returns the structured `cumulative_feedback` block the advisor will
embed in `evolution_state.json`:

```bash
<repo>/.venv/bin/python3 -m era.cli react-aggregate <<JSON
{"workspace_path": "<workspace>"}
JSON
```

The result carries `cumulative_feedback` with `iters_analyzed`,
`per_config_trajectory` (per-config endorsement rate by iter + wrong themes
from `item_marks` comments), and `general_feedback_themes` (verbatim
operator notes, one per iter). Keep this output — pass it to the advisor.

## Step 3 — Dispatch the advisor sub-agent

Dispatch **one** `Task` sub-agent of type `era-react-advisor` (this is a
heavy-tier opus agent — its identity is `era:era-react-advisor`). Pass its
brief:

> You are the **Stage 9 ReAct advisor**. The workspace is `<workspace>`. The
> active iteration is `<N>` (`<iter>`). The prior iteration is
> `<N-1>` (or `null` for `N == 1`). The deterministic aggregator returned the
> `cumulative_feedback` shown below — embed it (possibly enriched) in your
> `evolution_state.json`. Read every prior iter's
> `iter_*/human/{feedback,human_labels}.json`, every prior
> `iter_*/react/evolution_state.json`, this iter's
> `<iter>/design/experiment_brief.json` + `<iter>/design/hypotheses.md`, and
> `<workspace>/research/literature.md` §F (evolutionary / iterative-
> refinement strategies for auto-eval).
>
> **Auto-revise context** — when this iter's
> `iteration.json.parent_feedback.auto_revise_trigger` is set, the prior
> iter blocked at a pre-Stage-8 stage and was auto-revised. Read that
> trigger file (`iter_{N-1}/auto_revise/trigger.json`) for the
> `reason`, `source_stage`, `blocker_summary`, and `diagnostic` fields.
> Use them to inform the decision:
>
> - `reason: stage6_incomplete` with `diagnostic.missing_configs` —
>   add those configs to `exclude_list` (scope: `combination_id`,
>   reason: "stage6_incomplete: <one-line cause>") so Stage 2's
>   brainstorm drops them. Verdict typically `REVISE_SKIP_STAGE1`.
> - `reason: stage7_auto_validate_failed` with
>   `diagnostic.passing_configs`, `diagnostic.failing_configs`,
>   `diagnostic.passing_count`, `diagnostic.min_passing`, and
>   `diagnostic.per_config[*].pass_rate/recall_rate` — fewer than
>   `min_passing` configs (default 3) cleared the operator's
>   pass/recall thresholds. Phase C-2.5 splits this into two
>   sub-cases:
>
>     - **Partial-pass** (`diagnostic.passing_configs` non-empty
>       — 1 or 2 configs cleared the gate but fewer than
>       `min_passing`): the passing configs are PROVEN on the
>       operator's annotations and must be re-run in the next
>       iter. DO NOT add them to `exclude_list`. Set
>       `evolution_state.must_include_configs =
>       diagnostic.passing_configs` so Stage 4's brief gate
>       forces them into the next iter's `candidate_configs`.
>       Add ONLY the `failing_configs` to `exclude_list` (scope:
>       `combination_id`, reason "stage7_partial_pass: pass=X,
>       recall=Y"). Also surface each passing config as an
>       `evolution_proposals` entry of type `hybrid_compose` /
>       `prompt_rewrite` so Stage 2 brainstorms complementary
>       variants — different scope, prompt structure, sub-score
>       weighting — that target the same operator-flagged
>       failure modes from a different angle. The total
>       `chosen_configs` in the next iter should be ≥
>       `min_passing` so the gate has a chance to clear.
>       Verdict `REVISE_SKIP_STAGE1`.
>     - **All-fail** (`diagnostic.passing_configs` empty — no
>       config cleared the gate): every method failed. Add all
>       failing configs to `exclude_list` (reason
>       "stage7_auto_validate_failed: pass=X, recall=Y"), add a
>       `general_failure_modes` entry naming the *kind* of
>       evaluator that failed (judge size, prompt template,
>       hybrid composition), and leave `must_include_configs`
>       empty. Use `evolution_proposals` of type `prompt_rewrite`
>       / `model_upsize` / `hybrid_compose` to propose
>       replacements that swap to a different family / size /
>       prompt structure. Verdict `REVISE_SKIP_STAGE1`.
> - `reason: stage5_missing_annotated_tasks` with
>   `diagnostic.annotated_count` and `diagnostic.observed_modes` —
>   Stage 5's task plan omitted `mode: "annotated"` eval tasks
>   despite the dataset carrying `annotated_count` operator
>   annotations (≥ `auto_validate_min_samples`). This is a Stage 5
>   planner-prompt drift, not a candidate-evaluator failure: the
>   chosen_configs themselves may be fine, the planner just forgot
>   to emit the annotated round. Add a
>   `general_failure_modes` entry with key
>   `"stage5_planner_omitted_annotated_tasks"` naming the omission
>   as a hard requirement for the next iter so Stage 2-4 surface it
>   in the brief. Do NOT add the candidate configs to
>   `exclude_list` — they didn't fail. Verdict
>   `REVISE_SKIP_STAGE1`.
> - `reason: stage4_brief_invalid` / `stage5_task_plan_invalid` —
>   note the failure mode in `general_failure_modes` so Stage 2-4
>   tighten their Rule-5 slot allocation. Verdict
>   `REVISE_SKIP_STAGE1`.
> - `reason: stage1_literature_missing` — rare; verdict
>   `REVISE_RERUN_STAGE1` (write a `literature_update_brief.md` that
>   tells Stage 1 which sections it failed to produce).
> - Any other auto-revise reason — verdict `REVISE_SKIP_STAGE1` with
>   the blocker summary captured in `general_failure_modes`.
>
> Synthesize the exclude_list (operator vetos), evolution_proposals (typed:
> prompt_rewrite / model_upsize / hybrid_compose / prompt_restructure),
> general_failure_modes, AND the Phase C-2.5 EA primitives:
> `must_include_configs` (elitism — proven configs the next iter MUST
> re-evaluate), `lessons_learned` (structured success/failure pattern
> records with `confidence: low|medium|high` graduated by evidence count
> across iters, plus `open_questions`), and `hall_of_fame` (top-K configs
> across ALL iters by `fitness_composite =
> (pass_rate * recall_rate * human_endorsement_rate)^(1/3)`, capped at
> `max(min_passing, 5)` — `must_include_configs` MUST be a subset of
> hall_of_fame.combination_ids). Pick one verdict: `ADVANCE |
> REVISE_SKIP_STAGE1 | REVISE_RERUN_STAGE1`. On REVISE_RERUN_STAGE1,
> also write `<iter>/react/literature_update_brief.md` with three
> sections (Drop / Deepen / Add) naming exactly what Stage 1 should
> refresh. ALWAYS write `<iter>/react/lessons.md` — natural-language
> companion to lessons_learned (success patterns + failure patterns +
> hall-of-fame one-liners + open questions); Stage 2's next-iter
> brainstorm reads it as a context document. Write
> `<iter>/react/evolution_state.json` in a single `Write`, then end
> with `VERDICT: <DECISION>`.
>
> ```json
> <react-aggregate output here>
> ```

Re-read `<iter>/react/evolution_state.json` from disk after the sub-agent
returns — never trust its echo. Its last line is the `VERDICT:` claim; the
authoritative decision is the `decision` field of the file on disk.

## Step 4 — Validate the file

Run the deterministic schema guard:

```bash
<repo>/.venv/bin/python3 -m era.cli check-evolution-state <<JSON
{"state_path": "<iter>/react/evolution_state.json"}
JSON
```

If `valid` is `false`, fold the reported `problems` into a short revision
brief and re-dispatch the advisor once (carrying the same
`cumulative_feedback` plus the validator's `problems` list). If the second
attempt is still invalid, fall back: rewrite the file in this context to a
minimal valid payload with `decision: "ADVANCE"`, an empty `exclude_list` /
`evolution_proposals` / `general_failure_modes`, `literature_update_requested:
false`, and a `rationale` naming the validation failure — never leave a
malformed `evolution_state.json` on disk.

## Step 5 — Record the verdict (the bounded loop)

The verdict you just wrote is the **advisor's preference**. The bounded
counter is the deterministic `react.max_iterations` cap from `config.yaml` —
record the verdict through `react-tick`, which forces `ADVANCE` when
`status.iteration >= react.max_iterations`:

```bash
<repo>/.venv/bin/python3 -m era.cli react-tick <<JSON
{"workspace_path": "<workspace>",
 "verdict": "<decision from evolution_state.json>",
 "rationale": "<rationale from evolution_state.json (truncate to one line)>"}
JSON
```

This returns `decision` (the *effective* decision after the cap is applied),
`forced` (`true` when a `REVISE_*` was overridden), and writes
`<iter>/react/decision.json` + appends to `<iter>/react/history.jsonl`.
If `forced` is `true`, patch `evolution_state.json` to set
`"forced": true` and append a one-line note to its `rationale` —
`"iteration cap reached; advisor preferred <original verdict>"` — so the
audit trail is consistent.

## Step 6 — Branch on the effective decision

- **`ADVANCE`** — leave `status` at `stage_index = 9`. The ralph loop's next
  pass will advance to `stage_index = 10` (`final_report`). Append a
  one-line Stage 9 note (date, iteration, decision, forced) to
  `<workspace>/logs/iterations/`. Done.

- **`REVISE_SKIP_STAGE1`** or **`REVISE_RERUN_STAGE1`** — atomically advance
  to the next iteration:

  ```bash
  <repo>/.venv/bin/python3 -m era.cli create-next-iteration <<JSON
  {"workspace_path": "<workspace>",
   "rerun_stage1": <true if REVISE_RERUN_STAGE1 else false>}
  JSON
  ```

  This scaffolds `iter_{N+1}/`, swaps `current`, populates
  `iter_{N+1}/iteration.json.parent_feedback` with workspace-relative
  pointers to the prior iter's `human_labels.json`, `evolution_state.json`,
  and (on rerun) `literature_update_brief.md`, and updates `status.json`:
  `iteration += 1`, `run_state: "running"`, and `stage_index` set to the
  *last completed* stage so the next ralph pass dispatches the right
  stage — `0` for rerun (so Stage 1 / `research` runs next) or `1` for
  skip (so Stage 2 / `plan_brainstorm` runs next). Append a one-line
  Stage 9 note to `<workspace>/logs/iterations/` naming the new
  iteration, the rerun flag, and the count of `exclude_list` /
  `evolution_proposals` items carried forward. Done — the next ralph
  pass will pick up the new iter's `stage_index`.

## Principles

- **The cap is authoritative** — `react-tick` decides termination. Never
  override its `forced ADVANCE`; never end early by skipping the advisor
  when the cap is not yet hit.
- **One file is the source of truth** — `iter_NNN/react/evolution_state.json`
  is what Stage 2 / Stage 7 will read in iter `N+1`. Validate it
  deterministically before you trust it; never leave a malformed file on
  disk.
- **Atomic advance** — only `create-next-iteration` may scaffold a new
  `iter_NNN/`. Never mix `iter_N/` writes after you have called it; never
  call it twice for the same iteration.
- **Honesty** — a forced advance is stamped `forced: true` in both
  `evolution_state.json` and `decision.json`. Never silently downgrade the
  advisor's verdict.
- **Autonomous** — never ask the operator; record every decision in
  `<workspace>/logs/iterations/`.
