# ERA Stage 2 — Plan Brainstorm (candidate evaluation protocols)

You are running ERA Stage 2 for one project workspace. Goal: brainstorm
**candidate evaluation protocols** — concrete evaluator configurations — for this
project's AIGC task, and write a structured candidate pool
(`<iter>/design/candidates.md` + `<iter>/design/candidates.json`) for Stage 3 to
debate and Stage 4 to synthesize into an experiment-ready plan.

You are the **orchestrator**: you fan the brainstorm out to 4 parallel generator
persona sub-agents and merge their proposals. **Never ask the operator
anything** — decide from the workspace and record the decision. The workspace
path was passed as the skill argument (`$ARGUMENTS`).

The **ERA repo root** is the parent of the directory holding the `era/` package;
its venv Python is `<repo>/.venv/bin/python3`.

## Step 1 — Read the project

Resolve the active iteration: follow `<workspace>/current` (a symlink, or
`current.txt`) to the live `iter_NNN/` directory — call it `<iter>`.

Read `<workspace>/spec.md`, `<workspace>/config.yaml`, and the Stage 1 survey
`<workspace>/research/literature.md`. Extract:
- the **task** — `task_family` / `task_adapter` — and the mission's notion of a
  *human-aligned good result* (what an evaluator must reward / penalize);
- the **data shape** — input roles, methods, sample count, whether paired
  ground-truth exists — which decides what metrics even apply;
- the **budget** — `budget.api_cost_cap_usd` (0 ⇒ local-only VLM judges) and
  `wallclock_cap_hours`; the **hardware** — the allowed GPU pool;
- from `literature.md`: the **§3 Candidate evaluation methods** table and the
  **§7 Recommendations for Stage 2** — this is your seed material.

**If `<iter>/design/reviews.md` already exists**, this is a REVISE round: also
read it and the revision brief carried in your invocation, and improve the
candidates to address every blocking critique — do not just regenerate.

**If `<iter>/iteration.json.parent_feedback.evolution_state` is set**, this
iteration was spawned by Stage 9 with cumulative learning. Read the prior
iteration's `evolution_state.json` (the path is workspace-relative — resolve
against `<workspace>/`). It carries three carry-forward inputs the generators
must honor:

- **`exclude_list`** — operator vetos. Discard any candidate whose attribute
  matches an active exclusion before scoring. The exclusion's `scope` names
  the field to match: `judge_size` → match against the judge's parameter
  size; `judge_family` → match against the judge name; `metric_subfamily` →
  match against the metric family; `combination_id` → exact-match against
  the proposed `combination_id`; `prompt_variant` → match against a prompt
  identifier. Matching is lowercase substring. Each active exclusion's
  `reason` should appear once in `candidates.md`'s frontmatter so a reader
  sees why an obvious candidate was skipped.
- **`evolution_proposals`** — typed change suggestions
  (`prompt_rewrite | model_upsize | hybrid_compose | prompt_restructure`)
  the advisor synthesized from prior feedback. **Each proposal is a
  first-class candidate seed for the generators** — pass them alongside
  `literature.md` §3 to the personas, with a note that the generator may
  refine the proposal (e.g. pick a different upsize target than the
  proposed one, justify why) but should *evaluate* whether the proposed
  change has merit.
- **`general_failure_modes`** — recurring error patterns across configs.
  Surface them in the generator briefs so each persona knows what *not* to
  repeat (e.g. "prior judges over-credit similar-print garments — design a
  rubric or region scope that addresses this").
- **`must_include_configs`** (Phase C-2.5 elitism) — list of
  `combination_id` strings the prior iter's Stage 9 advisor marked as
  PROVEN: they cleared the C-2 pass/recall gate but the iter had fewer
  than `min_passing` (default 3) survivors so the loop auto-revised.
  These configs MUST appear verbatim in this iter's `candidate_configs`
  — Stage 4's brief gate (`check-experiment-brief`) rejects any brief
  that drops a must-include. Quote each must-include's
  `combination_id`, `judge`, `prompt`, `scope`, `metric_subfamily`,
  etc. from the prior iter's `iter_{N-1}/design/candidates.json` (or
  the equivalent from the prior brief) into the new brief — Stage 2's
  personas brainstorm the COMPLEMENTARY candidates only. The new iter's
  total `candidate_configs` should reach ≥ `min_passing` so the gate
  has a chance to clear.
- **`lessons_learned`** (Phase C-2.5 EA reflection) — structured
  record of `success_patterns`, `failure_patterns`, and
  `open_questions` synthesized by the prior iter's Stage 9 advisor.
  Each pattern carries a `confidence: low|medium|high` graduated by
  evidence count across iters. Surface to every persona's brief:
  treat `success_patterns` with `confidence: high` (3+ iters
  confirming) as **design priors** — new candidates should
  preserve those design dimensions unless a persona has a specific
  refutation. Treat `failure_patterns` as **anti-patterns** — any
  candidate matching a `failure_pattern` is rejected at Stage 3 unless
  it carries a concrete fix referencing the `remedy_proposed`.
  Open questions are concrete uninvestigated design choices — at
  least one new candidate should investigate one of them.
- **`hall_of_fame`** (Phase C-2.5 EA population memory) — top-K
  configs across ALL prior iters, ranked by `fitness_composite`.
  Every `must_include_configs` entry is also in `hall_of_fame`.
  Personas should treat hall-of-fame entries as the seeds to
  hybridise / mutate / extend — not as configs to ignore. Lineage
  from a hall-of-fame entry goes in the new candidate's
  optional `parent_hypothesis_ids` list.
- **`lessons.md`** (Phase C-2.5) — natural-language companion to
  `lessons_learned`. The path is workspace-relative at
  `iter_{N-1}/react/lessons.md`. Each persona sub-agent should be
  given this file as a context document (alongside
  `research/literature.md`) so the EA priors and anti-patterns are
  visible without re-deriving them from the JSON.

**Annotation evidence (Phase C-2 — operator's known failure modes).** When
`config.yaml`'s `data.use_annotation_evidence` is `true` (default) and the
operator has authored notes under `<data_root>/annotations/`, fetch them:

```bash
<repo>/.venv/bin/python3 -m era.cli list-annotations <<JSON
{"workspace_path": "<workspace>"}
JSON
```

Read the returned `count`, `sample_keys`, and `method_coverage`. If
`count >= experiment.auto_validate_min_samples` (default 10), the Phase C-2
pass/recall gate WILL run against this annotated subset between Stage 6's
pilot and full rounds. For up to **5 annotated samples per generation
method**, read the central annotation file at
`<data_root>/annotations/<sample_key>.json` (use the `Read` tool — small
JSON files) to extract the operator's per-method free-text notes. These
notes are the operator's **known failure modes**: the defects each
candidate evaluator must be able to catch. Pass the surfaced notes to
every persona below; configs that fail the Phase C-2 gate auto-revise the
loop, so spend slots on candidates whose design directly addresses these
specific defects.

If `count < auto_validate_min_samples` or the dataset has no annotations,
the gate is skipped at Stage 6 — proceed without this evidence path.

## Step 2 — Resolve the sub-agent tier

Run, from the ERA repo root:

```bash
<repo>/.venv/bin/python3 -m era.cli agent-tier <<JSON
{"workspace_path": "<workspace>", "stage": "plan_brainstorm"}
JSON
```

Use the returned `agent` value (e.g. `era-standard`) as the `subagent_type` for
every generator `Task` call below. If the call fails after one retry, default to
`era-standard`.

## Step 3 — Fan out the 4 generator personas

Spawn **4 generator persona sub-agents, all in parallel** — issue every `Task`
call in a single turn, each with `subagent_type` = the tier from Step 2. Give
each persona: the task family/adapter, the human-aligned notion of "good", the
data shape and budget, the path to `research/literature.md`, and — on a REVISE
round — the revision brief. Instruct each to **write its proposal to
`<iter>/design/candidates/<persona>.md`** and to propose **1–3 candidate
evaluator configurations**, each carrying the experiment-ready fields in Step 4.

The 4 personas and their briefs:

1. **judge-advocate** — Champion a **Family A (VLM/LMM judge)** protocol.
   Design the rubric, the scoring mode (pointwise vs. pairwise — prefer
   pairwise + order-swap debiasing where the literature shows it aligns better),
   the judge model and scale, and the prompt. Honor the budget: if
   `api_cost_cap_usd == 0`, every judge must be locally served on the allowed
   GPU pool. Include ≥2 VLM scales so the scale-comparison invariant (Rule 4)
   can hold. **Capability-tier floor (hard default):** for image-edit /
   try-on workspaces, every candidate VLM judge must be **≥30B-class**
   (named reference models: Qwen3.6-35B-A3B, Gemma-4-31B-it). Both
   Rule-4 scales should be ≥30B (e.g. ≥30B vs. 72B); **sub-30B candidates
   are forbidden** unless the operator explicitly asked for them in human
   feedback — i.e. a quoted line in `parent_feedback.general_feedback` or
   `cumulative_feedback` like "try the 7B" or "include a sub-30B baseline".
   Without that operator request, sub-30B candidates are rejected by
   Stage 3's reviewers and dropped at Stage 4 synthesis. See
   `feedback-vlm-min-size`. Optimize human alignment.

2. **metrics-advocate** — Champion a **Family B (metric / feature / region)**
   protocol. Select metrics (CLIP, DINO, LPIPS, SSIM, PSNR, IQA, segmentation-
   IoU, OCR, face-ID, pose — whichever apply) and the **scope** each runs at
   (whole-image / edited-region / non-edited-region). If the data has no paired
   ground truth, build a reference-free decomposed scorecard. Optimize cost,
   determinism, and reproducibility.

3. **cost-pragmatist** — Champion the **cheapest** protocol that still clears a
   credible minimum alignment bar — a learned reference-free scorer, or a small
   metric ensemble. Treat GPU-hours, wall-clock, and `$` as first-class; honor
   `budget`. ERA's deliverable is the *smallest* evaluator that agrees with
   humans — make that case concretely.

4. **hybrid-innovator** — Champion a **calibrated hybrid**: a VLM judge gated by
   or fused with region metrics, or a learned scorer calibrated against a small
   human-rated set. Find the non-obvious combination that sits on the
   alignment/cost Pareto frontier. Justify why the combination beats either
   family alone.

**Annotated failure modes (when surfaced in Step 1).** Every persona
above receives the operator's per-method annotation notes when the gate
will run. Each candidate config must be designed to catch the kinds of
defects those notes describe — for example, *"color is too white"* →
include a color-fidelity sub-score (CLIP-vit or a hue-histogram delta);
*"logo blurred"* → include a structural-detail / sharpness sub-score
(LPIPS, region-DINO, or a judge prompt that explicitly asks about logo
clarity); *"sleeve warped"* → include a region-aware shape sub-score
(garment mask + DINO). **Tag each candidate's `hypothesis_id` text with
one line naming the failure mode(s) it primarily targets** (e.g.
`"H2: tighter color-fidelity rubric — targets the operator's 'color
shift' annotations on samples s1/s4/s11"`). Stage 3's critics score
that coverage; Stage 4's synthesizer biases the final selection so the
union of chosen configs covers every flagged failure mode at least once.
Configs that fail the Phase C-2 gate auto-revise the loop; spend slots
on candidates the annotations suggest will actually agree.

**Sub-agent return contract — keep the orchestrator context small.** Each
generator persona reasons in its own context window and **writes** its proposal
to `<iter>/design/candidates/<persona>.md`. Instruct every persona to **return
ONLY the path it wrote plus a one-line status** (e.g.
`design/candidates/judge-advocate.md — 2 candidates`) — never to echo the file
contents back. You reconstruct the full picture in Step 4 by reading those
files from disk, not from the personas' replies. This keeps Stage 2 fast and
bounds your context.

**Resilience:** if a persona sub-agent fails or returns nothing, retry once;
then continue with the survivors and note the gap. Three of four is still a
valid brainstorm.

## Step 4 — Merge & save

Each candidate evaluator configuration must carry these **experiment-ready
fields** (so Stage 4 can assemble a plan and Stage 5 can run it):
- a short kebab-case `combination_id`;
- the **combination tuple** — `judge` · `metric_subfamily` · `prompt` · `scope`
  (any field that does not apply to the family is `null`/"n/a");
- `family` — `A` (VLM judge), `B` (metric/feature/region), or `hybrid`;
- `inputs_needed` — which input roles + the output it consumes;
- a **draft alignment hypothesis** — a falsifiable claim with a *quantitative
  threshold* (e.g. "reaches Kendall τ ≥ 0.6 vs. the human-rating set, above
  every Family-B metric");
- the **validation set** — a human-rated dataset from `literature.md` §5 — and
  the **alignment metric** (Spearman / Kendall τ / PLCC / pairwise-accuracy);
- a **GPU/cost estimate** and an **API-cost estimate** (USD);
- 2–4 **key risks**.

**Read each `<iter>/design/candidates/<persona>.md` from disk**, then merge the
proposals — deduplicate near-identical configurations, keep every distinct one —
and write **two** files, each in a single complete `Write`:

`<iter>/design/candidates.md` — human-readable, with this structure:

```markdown
# ERA Stage 2 — Candidate Evaluation Protocols — <project name>

**Task:** <task_family> / <task_adapter>   **Iteration:** <n>   **Debate round:** <r>

## Candidate comparison
| candidate_id | family | combination (judge·metric·prompt·scope) | est. GPU/cost | est. alignment | proposing persona |
|--------------|--------|------------------------------------------|---------------|----------------|-------------------|

## <candidate_id> — <title>
<proposing persona · summary · combination tuple · inputs needed · draft
alignment hypothesis (with threshold) · validation set + metric · GPU/cost +
API-cost estimate · key risks>

<...one section per candidate...>
```

`<iter>/design/candidates.json` — the structured pool Stage 3 and Stage 4 read:

```json
{
  "iteration": <n>,
  "debate_round": <r>,
  "candidates": [
    {"candidate_id": "...", "title": "...", "proposing_persona": "...",
     "family": "A|B|hybrid", "summary": "...",
     "combination": {"judge": "...", "metric_subfamily": "...",
                     "prompt": "...", "scope": "..."},
     "inputs_needed": ["..."],
     "draft_hypothesis": "claim + quantitative threshold",
     "validation_set": "...", "alignment_metric": "...",
     "gpu_cost_estimate": "...", "api_cost_estimate_usd": 0.0,
     "key_risks": ["..."],
     "novelty_score": 0, "feasibility_score": 0, "alignment_score": 0,
     "status": "candidate"}
  ]
}
```

Scores are 1–10 (your own honest assessment, merging the personas' views).

Append a one-line Stage-2 note (date, candidate count, any persona gap) to
`<workspace>/logs/iterations/`.

## Principles

- **Evaluation-method focused** — every candidate is a way to *evaluate*
  generated outputs, never a way to generate them.
- **Experiment-ready by construction** — a candidate without a falsifiable
  hypothesis, a threshold, and a named validation set is not done.
- **Honor the rules** — respect the budget ($0 ⇒ local VLMs only), the hardware
  pool, and the ≤10-configuration cap that Stage 4 must hit (Rule 5).
- **Honesty** — never fabricate a metric, a paper, or a human-correlation
  number; if `literature.md` is thin on a point, say so.
- **Autonomous** — never ask the operator; record decisions in
  `<workspace>/logs/iterations/`.
