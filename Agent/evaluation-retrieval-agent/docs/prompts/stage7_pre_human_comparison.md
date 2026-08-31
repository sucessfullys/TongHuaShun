# ERA Stage 7 — Pre-human Comparison (assemble the comparison view)

You are running ERA Stage 7 for one project workspace. Goal: confirm the
Stage 6 experiment finished cleanly and distil its results into a single
`iter_NNN/comparison/comparison.json` — the comparison view the Stage 8
human-feedback web app opens with.

**Never ask the operator anything** — resolve every ambiguity from the workspace
files. The workspace path was passed as the skill argument (`$ARGUMENTS`). The
**ERA repo root** is the parent of the directory holding the `era/` package.

This is a deliberately **thin** stage: ERA retrieves an evaluation protocol —
there is nothing to "deploy" — so Stage 7 only assembles the comparison, it does
not run models.

## Step 1 — Read state

Resolve the active iteration via `<workspace>/current` → `<iter>`. Read:

- `<iter>/experiments/results/summary.json` — the Stage 6 run summary.
- `<iter>/experiments/results/full/final_comparison.json` — the cross-family
  comparison, if it exists.
- `<workspace>/status.json` — to read the current `iteration` number.

## Step 2 — Readiness check

Confirm `summary.json` exists (file-existence is still a precondition).
Then run the orchestration-layer completion gate — this is the same answer
Stage 6 and the ralph loop use, so Stage 4's pre-validated pivot-matrix
drops (each stamped with a `skip_proof` that matches one of the brief's
`pivot_matrix[*].action` strings) are correctly excused even when
`summary.json`'s own strict `complete` flag is `false`:

```bash
<repo>/.venv/bin/python3 -m era.cli check-experiment-completion <<JSON
{"workspace_path": "<workspace>", "mode": "full"}
JSON
```

If `complete: true`, proceed to Step 3. Otherwise — `missing_configs[]` is
non-empty, or `in_progress_tasks[]` is non-empty — the experiment did not
finish: append the returned report (the JSON object) to
`<workspace>/logs/iterations/`, **do not** write `comparison.json`, and
stop. The ralph loop's verification will then block the run. Do not
fabricate results.

## Step 3 — Prior human labels (iteration 2 and later)

If the iteration number is `> 1`, look for the most recent prior iteration's
human labels — walk `<workspace>/iter_{N-1}/human/human_labels.json` down to
`iter_001`. If one is found, read its `config_summary` for a one-line
`prior_labels_summary`; if none exists, `prior_labels_summary` is `null`.

Also read the most recent prior iteration's
`<workspace>/iter_{N-1}/react/evolution_state.json` — walk down the same way.
If one is found, derive a one-line `prior_react_decision`:

```json
{"iteration_dir": "iter_NNN",
 "decision": "<ADVANCE|REVISE_SKIP_STAGE1|REVISE_RERUN_STAGE1>",
 "forced": <bool>,
 "exclude_list_count": <int>,
 "evolution_proposals_count": <int>}
```

If none exists, `prior_react_decision` is `null`. This is what the operator
sees in the Stage 8 web app to know what Stage 9 just decided.

## Step 4 — Write `comparison.json`

Write `<iter>/comparison/comparison.json` with exactly this shape:

```json
{
  "schema_version": "1.0",
  "iteration": <iteration number>,
  "iteration_dir": "<iter dir name>",
  "mode": "<summary.json mode>",
  "config_count": <summary.json config_count>,
  "configs": <summary.json configs array, copied verbatim>,
  "cross_family_findings": <final_comparison.json cross_family_findings, or []>,
  "prior_labels_summary": <null, or {"iteration_dir": "...", "config_summary": [...]}>,
  "prior_react_decision": <null, or the one-line object above>,
  "produced_at": "<current UTC ISO-8601 timestamp>"
}
```

Copy the `configs` array straight from `summary.json` — do not recompute scores.
If `final_comparison.json` is absent, use `[]` for `cross_family_findings`.

## Step 5 — Log and finish

Append a one-line note to `<workspace>/logs/iterations/` (date, `stage 7
pre_human_comparison`, "comparison view assembled — N configs"). Do **not**
advance `status.json` — the ralph loop owns the stage transition.
