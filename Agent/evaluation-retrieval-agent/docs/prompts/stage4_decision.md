# ERA Stage 4 — Plan Decision (synthesize the experiment-ready plan)

You are running ERA Stage 4 for one project workspace. Goal: synthesize the
Stage 2 candidates and the Stage 3 reviews into the **chosen evaluation
experiment design**, run a bounded ADVANCE/REVISE refinement loop, and — on
ADVANCE — emit the **experiment-ready handoff bundle** the later Experiment
stages execute.

You **own the debate loop**. **Never ask the operator anything** — the round cap
ends the loop, not an operator. The workspace path was passed as the skill
argument (`$ARGUMENTS`). The **ERA repo root** is the parent of the directory
holding the `era/` package; its venv Python is `<repo>/.venv/bin/python3`.

## Step 1 — Read state

Resolve the active iteration via `<workspace>/current` → `<iter>`. Read
`<iter>/design/candidates.json`, `<iter>/design/candidates.md`,
`<iter>/design/reviews.md`, plus `<workspace>/spec.md`,
`<workspace>/config.yaml`, and `<workspace>/research/literature.md`.

**Annotation evidence (Phase C-2 — coverage bias for chosen_configs).**
When `config.yaml`'s `data.use_annotation_evidence` is `true` (default)
and the dataset has annotations, fetch them:

```bash
<repo>/.venv/bin/python3 -m era.cli list-annotations <<JSON
{"workspace_path": "<workspace>"}
JSON
```

If `count >= experiment.auto_validate_min_samples`, surface the
`sample_keys` + `method_coverage` to the synthesizer below; the
synthesizer must bias `chosen_configs` so the union of their declared
failure-mode tags (carried in each candidate's `hypothesis_id` text by
Stage 2) covers every operator-flagged failure mode at least once.
Read up to 5 per-method central annotation files
(`<data_root>/annotations/<sample_key>.json`) to give the synthesizer
the operator's verbatim notes — those are the ground truth the Phase
C-2 gate will compare against. A chosen set that leaves a flagged
mode uncovered should be rejected; if no candidate addresses a mode,
note the gap in the brief's `general_failure_modes` so Stage 9's
advisor knows to evolve toward it in the next iter.

Read the debate-loop state:

```bash
<repo>/.venv/bin/python3 -m era.cli debate-state <<JSON
{"workspace_path": "<workspace>"}
JSON
```

This returns `round` and `max_rounds` (initializing the state on first use).

## Step 2 — Resolve the sub-agent tier

```bash
<repo>/.venv/bin/python3 -m era.cli agent-tier <<JSON
{"workspace_path": "<workspace>", "stage": "plan_decision"}
JSON
```

Use the returned `agent` (e.g. `era-heavy`) as the `subagent_type` for the
synthesizer `Task` call. If the call fails after one retry, default to
`era-heavy`.

## Step 3 — Synthesize (the debate loop)

Run this loop until it converges:

**3a. Dispatch the `synthesizer` persona** as one `Task` sub-agent (tier from
Step 2). Pass it the current debate **round** so it can stamp the brief — on the
first loop pass that is the `round` from Step 1's `debate-state`; on every later
pass it is the `round` returned by `debate-tick` in Step 3c (the post-increment
value). Its brief:

> You are the **synthesizer** — ERA's senior evaluation architect. Read the
> candidate pool (`candidates.json` / `candidates.md`) and the multi-review
> (`reviews.md`). Assemble the **chosen evaluation experiment design**: select
> and, where the reviews demand, refine the candidate evaluator configurations
> into a coherent comparison of **≤10 configurations** allocated by Rule 5's
> slots — `metric_baseline` (1-2), `vlm_scale` (≤3, the VLM scale ablation),
> `hybrid` (≤3), `task_specific` (≤3), `wildcard` (≤1). The set must satisfy the
> Rule 4 scale-comparison invariant (≥2 **distinct** VLM judge scales, or ≥1 VLM
> scale + ≥1 metric-only baseline) — two configs with the same `judge` do not
> count as a scale comparison. **VLM-scale floor (hard):** for image-edit /
> try-on workspaces, every VLM scale slot's `judge` must be **≥30B class**
> (named reference models: Qwen3.6-35B-A3B, Gemma-4-31B-it; upper rung 72B).
> **Sub-30B picks are forbidden** unless the operator explicitly asked for
> them in human feedback (a quoted line in
> `parent_feedback.cumulative_feedback` like "try the 7B" or "include a
> sub-30B baseline"). The synthesizer rejects any sub-30B candidate that
> lacks this operator-request evidence — see `feedback-vlm-min-size`. **Every field in the Step 4 schemas is
> mandatory** and the deterministic guard rejects a brief that omits any of
> them: each `candidate_configs` entry needs its slot-/family-correct
> combination tuple (a Family-A config needs `judge` + `prompt`; a Family-B
> config needs `metric_subfamily`; the `slot` must match the family —
> `metric_baseline`⇒B, `vlm_scale`⇒A, `hybrid`⇒hybrid), a non-empty
> `inputs_needed`, `scope`, `gpu_estimate`, a numeric `cost_estimate_usd`,
> `feasible: true`, and a `hypothesis_id` that names a heading in
> `hypotheses.md`; the `validation` / `pilot` / `resource_estimate` /
> `pivot_matrix` blocks must be fully populated; stamp `iteration` and
> `debate_round` (= the round you were given). Then **write the three bundle
> files** (Step 4 schemas). **Return ONLY** a verdict — `ADVANCE` if the design
> is sound enough to plan an experiment around, or `REVISE` with a concise
> revision brief if a Stage-3 critic raised a blocking, fixable defect — plus
> the paths you wrote; do not echo the bundle contents back. Honor the budget
> and the hardware pool. Never fabricate a metric or a human-correlation number.
>
> **Annotation-coverage bias (Phase C-2).** When Step 1 surfaced
> operator annotations (`count >= experiment.auto_validate_min_samples`),
> bias the chosen_configs selection so the **union of the chosen
> configs' targeted failure modes covers every operator-flagged failure
> mode at least once**. Each candidate's `hypothesis_id` text carries a
> Stage-2-author tag naming the failure mode(s) it primarily targets;
> Stage 3's rigor-critic scored annotation coverage 0-3 in
> `reviews.md`. Prefer candidates scored ≥2 on annotation coverage.
> When a flagged failure mode has no candidate addressing it, do NOT
> spend a slot on it — instead, note the gap in
> `experiment_brief.json`'s `general_failure_modes` block so Stage 9's
> advisor knows to evolve toward it in the next iter. Configs that
> fail the Phase C-2 gate auto-revise the loop; an annotation-aware
> chosen set materially raises the gate's first-iter pass probability
> and saves iter-cap budget. Carry through each chosen_config's
> failure-mode tag into the brief's `hypothesis_id` text so the
> Phase C-2 gate's diagnostic + Stage 9's advisor can trace causation.

**3b. Validate the experiment brief** deterministically — this guard enforces
the full v0.1.3 handoff contract, including that every `hypothesis_id` resolves
to a heading in `hypotheses.md`:

```bash
<repo>/.venv/bin/python3 -m era.cli check-experiment-brief <<JSON
{"brief_path": "<iter>/design/experiment_brief.json",
 "hypotheses_path": "<iter>/design/hypotheses.md",
 "config_path": "<workspace>/config.yaml",
 "evolution_state_path": "<resolved iteration.json.parent_feedback.evolution_state, or omit if absent>"}
JSON
```

`config_path` enables the per-iter-cap cross-check: the gate enforces
`validation.sample_size == effective_iter_sample_count` and
`pilot.sample_count <= effective_iter_sample_count`, where the effective
count is `min(data.iter_sample_count, data.sample_count)`.

`evolution_state_path` enables Phase C-2.5's elitism cross-check: when
set, the gate enforces `candidate_configs ⊇
evolution_state.must_include_configs` so the prior iter's PROVEN
configs (those that cleared the C-2 gate but weren't enough to meet
`min_passing`) are forced into this iter's brief by their
`combination_id`. Read the path from this iter's `iteration.json`:
`parent_feedback.evolution_state` (already workspace-relative). Omit
this field for iter_001 (no prior iter) or when the prior iter has no
must-include carry-forward.

If `valid` is `false`, the verdict is **REVISE** regardless of what the
synthesizer returned — fold the reported `problems` into the revision brief.

**3c. Apply the round cap:**

```bash
<repo>/.venv/bin/python3 -m era.cli debate-tick <<JSON
{"workspace_path": "<workspace>", "verdict": "<ADVANCE or REVISE>",
 "reason": "<on a REVISE, the synthesizer's one-line revision brief>"}
JSON
```

This returns `action` — `advance` or `revise` — `forced` (`true` when a REVISE
was overridden to `advance` because the cap was hit), the new `round`, and
`cleared` (on a `revise`, the transient debate dirs it emptied — see 3d).

**3d. Branch on `action`:**
- **`revise`** — refine in place by re-running the Stage 2 + Stage 3 fan-out.
  `debate-tick` has already emptied `design/candidates/` and `design/reviews/`
  (its `cleared` field lists them), so this round's fan-out starts from clean
  transient dirs and the merge globs only this round's files.
  **Delegate the work to sub-agents** so your own context stays flat across
  rounds: re-dispatch the 4 Stage-2 generator persona `Task` sub-agents (the
  briefs in `docs/prompts/stage2_brainstorm.md` §3), each carrying the revision
  brief **and the new `round`** (the `round` returned by `debate-tick`) and
  writing `design/candidates/<persona>.md`; merge them by **reading the files
  from disk** into refreshed `candidates.md` + `candidates.json` stamped with
  that new round. Then re-dispatch the 3 Stage-3 critic persona `Task`
  sub-agents (`docs/prompts/stage3_review.md` §3), each writing
  `design/reviews/<critic>.md`; merge from disk into a refreshed `reviews.md`.
  Every sub-agent returns only a path + one-line status — never echo file
  contents. Then go back to **3a** (the synthesizer there gets that same new
  `round`). This is in-iteration refinement — **never create a new `iter_NNN/`**.
- **`advance`** — leave the loop; go to Step 5.

If `forced` is `true`, the design ships as-is despite an open critique — note
that honestly in `plan.md` and `decision.json`.

## Step 4 — The bundle the synthesizer writes

Each loop pass, the synthesizer writes all three, each in a single complete
`Write`:

**`<iter>/design/experiment_brief.json`** — the machine-readable handoff Stage 5
executes:

```json
{
  "iteration": <n>, "debate_round": <r>,
  "evaluation_goal": "what human-aligned 'good' means for this task",
  "candidate_configs": [
    {"combination_id": "vlm-72b-pairwise-whole",
     "slot": "metric_baseline|vlm_scale|hybrid|task_specific|wildcard",
     "family": "A|B|hybrid",
     "judge": "...", "metric_subfamily": "...", "prompt": "...", "scope": "...",
     "inputs_needed": ["..."],
     "gpu_estimate": "...", "cost_estimate_usd": 0.0,
     "hypothesis_id": "H1", "feasible": true}
  ],
  "validation": {"human_rating_set": "...", "alignment_metric": "...",
                 "sample_size": 0, "debiasing": "..."},
                 # ^ sample_size MUST equal config.yaml's effective per-iter
                 # cap: min(data.iter_sample_count, data.sample_count). The
                 # cap is operator-set at Stage 0; the brief gate rejects any
                 # mismatch. Stage 6 runners score exactly this many samples
                 # per method (the same window across iters).
  "pilot": {"sample_count": 0, "go_no_go": "criteria text"},
            # ^ pilot.sample_count MUST be <= validation.sample_size — the
            # pilot is a subset of the iter window, not a different set.
  "scale_comparison": "how the Rule 4 invariant is met",
  "resource_estimate": {"gpu_hours": 0.0, "wallclock_hours": 0.0,
                        "api_cost_usd": 0.0},
  "pivot_matrix": [{"pilot_outcome": "...", "action": "..."}]
}
```

**`<iter>/design/hypotheses.md`** — the falsifiable **alignment hypotheses**,
one per candidate configuration (or per family), each structured:

```markdown
## H<k>: <title>
**Statement** — a precise, falsifiable claim about human alignment.
**Expected outcome** — the quantitative prediction (the threshold + metric).
**Falsification criterion** — the exact condition (+ statistical test) that
disproves it.
**Confound controls** — the design choices that isolate the effect (order-swap,
region masking, sample size, ...).
```

Each `hypothesis_id` in `experiment_brief.json` must name a hypothesis here.

**`<iter>/design/plan.md`** — the human-readable evaluation experiment plan,
with **every** section: *Evaluation goal* · *Candidate evaluator configurations*
(the ≤10, grouped by Rule 5 slot) · *Validation protocol* (human-rating set,
alignment metric, sample size, order-swap debiasing) · *Pilot & go/no-go gate* ·
*Ablations & the scale-comparison invariant* · *Resource estimate* (GPU-hours,
serving plan, wall-clock, `$` vs. budget) · *Risk assessment* (a table) ·
*Pivot matrix*.

## Step 5 — Finish (on `advance`)

Write `<iter>/design/decision.json` in a single `Write`:

```json
{"decision": "ADVANCE", "round": <final round>, "forced": <bool>,
 "chosen_configs": ["combination_id", "..."],
 "rejected": ["candidate_id", "..."],
 "rationale": "why this design — 2-4 sentences"}
```

`chosen_configs` lists the `combination_id`s of the configs you kept — it must
equal the set of `experiment_brief.json`'s `candidate_configs[].combination_id`.
`rejected` lists the `candidate_id`s from `candidates.json` that were dropped
(the synthesizer refines candidates into configs, so a kept candidate's
`candidate_id` and its `combination_id` need not be identical).

Re-run `check-experiment-brief` once more as a final guard (with the same
`brief_path` + `hypotheses_path` as Step 3b); if it still reports problems after
a forced advance, record them in `decision.json` under a `"known_issues"` key —
never silently ship a brief you know is malformed.

Append a one-line Stage-4 note (date, decision, round, chosen count) to
`<workspace>/logs/iterations/`.

## Principles

- **Experiment-ready or not done** — the bundle exists so Stage 5 can build a
  runnable experiment without re-debating. A plan with no falsifiable
  hypotheses, no validation set, or no pilot gate is not finished.
- **The cap is authoritative** — `debate-tick` decides when the loop ends. Never
  loop past `max_rounds`; never end early by ignoring a blocking defect.
- **In-iteration only** — the REVISE loop refines within `<iter>`; it never
  creates a new iteration. New iterations belong to Stage 9 (ReAct).
- **Honesty** — a forced advance with an open critique is stated plainly, not
  hidden. Never fabricate evaluation numbers or citations.
- **Autonomous** — never ask the operator; record every decision in
  `<workspace>/logs/iterations/`.
