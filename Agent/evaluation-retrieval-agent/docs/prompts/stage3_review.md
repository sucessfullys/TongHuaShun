# ERA Stage 3 — Multi-Review (debate the candidate protocols)

You are running ERA Stage 3 for one project workspace. Goal: run a multi-persona
**debate** over the Stage 2 candidate evaluation protocols — critique and score
every candidate — and write a consolidated review
(`<iter>/design/reviews.md`) for Stage 4 to decide on.

You are the **orchestrator**: you fan the critique out to 3 parallel critic
persona sub-agents and merge their reviews. **Never ask the operator anything**.
The workspace path was passed as the skill argument (`$ARGUMENTS`).

The **ERA repo root** is the parent of the directory holding the `era/` package;
its venv Python is `<repo>/.venv/bin/python3`.

## Step 1 — Read the candidates

Resolve the active iteration via `<workspace>/current` → `<iter>`. Read
`<iter>/design/candidates.json` and `<iter>/design/candidates.md` (the Stage 2
pool), plus `<workspace>/spec.md`, `<workspace>/config.yaml`, and
`<workspace>/research/literature.md` for grounding. If `candidates.json` is
missing or empty, log the gap and stop — Stage 3 has nothing to review.

**Annotation evidence (Phase C-2 — annotation-coverage dimension).** When
`config.yaml`'s `data.use_annotation_evidence` is `true` (default) and the
dataset carries operator annotations, fetch them:

```bash
<repo>/.venv/bin/python3 -m era.cli list-annotations <<JSON
{"workspace_path": "<workspace>"}
JSON
```

If `count >= experiment.auto_validate_min_samples` (default 10), the
Phase C-2 pass/recall gate will run on this annotated subset. Read up
to 5 per-method central annotation files
(`<data_root>/annotations/<sample_key>.json`) to surface the operator's
free-text failure-mode notes; pass them to every critic below so the
**rigor-critic** can score annotation-coverage (described below) for
each candidate. The Stage 2 generators were instructed to tag each
candidate's `hypothesis_id` text with the failure mode(s) it targets —
that tag is the primary signal critics evaluate against the
annotations.

## Step 2 — Resolve the sub-agent tier

Run, from the ERA repo root:

```bash
<repo>/.venv/bin/python3 -m era.cli agent-tier <<JSON
{"workspace_path": "<workspace>", "stage": "multi_review"}
JSON
```

Use the returned `agent` value (e.g. `era-light`) as the `subagent_type` for
every critic `Task` call. If the call fails after one retry, default to
`era-light`.

## Step 3 — Fan out the 3 critic personas

Spawn **3 critic persona sub-agents, all in parallel** — issue every `Task` call
in a single turn, each with `subagent_type` = the tier from Step 2. Give each
critic: the full candidate pool (`candidates.json` + `candidates.md`), the task
context, and `literature.md`. Each critic reviews **every candidate**, scores it,
and **writes its review to `<iter>/design/reviews/<critic>.md`**.

The 3 critics and their briefs:

1. **alignment-critic** — Challenge each candidate's **human-correlation**
   claim. Is the draft alignment hypothesis backed by evidence in
   `literature.md`? Is the claimed threshold plausible for this task? Flag
   known traps — pointwise VLM scores used as a ranking, unmasked full-image
   SSIM/LPIPS on a reference-free task, FID/KID promoted as a headline metric.
   Demand a credible validation set with real human ratings.

2. **feasibility-critic** — Challenge **deployability**. Does each candidate fit
   the allowed GPU pool and per-GPU VRAM (Rule 3)? Does it respect the budget
   ($0 API ⇒ local serving only)? Does it fit ERA's `ms-swift`/OpenAI-compatible
   serving contract and the data shape (no paired ground truth ⇒ no
   paired-GT metric)? Could the final set blow the ≤10-configuration cap
   (Rule 5)? Flag any candidate needing paid APIs, missing checkpoints, or an
   absent parser dependency.

3. **rigor-critic** — Challenge **experimental soundness**. Is each candidate's
   hypothesis genuinely *falsifiable*, with a quantitative threshold and a named
   statistical test? Is there order-swap / position-bias debiasing where a judge
   is involved? Is a pilot subset defined? Does the candidate set, taken
   together, satisfy the scale-comparison invariant (Rule 4 — ≥2 VLM scales, or
   1 VLM + 1 metric-only baseline)? Flag unfalsifiable or unreproducible designs.

   **Annotation coverage (Phase C-2).** When annotations are surfaced in
   Step 1, score each candidate on whether its stated design (the
   `hypothesis_id` failure-mode tag, the `metric_subfamily` / judge
   prompt / scope) plausibly catches the operator's annotated failure
   modes. Use a 0-3 scale: `0` no plausible coverage, `1` weak overlap,
   `2` covers some flagged modes, `3` directly targets the dominant
   flagged modes. Include this score as an "Annotation coverage:
   <score> — <one-line rationale>" line in the review file for every
   candidate. The combined verdict (`strong`/`viable`/`weak`/`reject`)
   should penalise candidates with `0` annotation coverage when the
   operator has explicitly flagged failure modes — those configs are
   pre-destined to fail the Phase C-2 gate and waste an iter.

Each critic's review file must give, **per candidate**: a verdict — `strong` /
`viable` / `weak` / `reject` — the concrete defects found, and a concrete,
actionable fix for each defect.

**Sub-agent return contract — keep the orchestrator context small.** Each critic
reasons in its own context window and **writes** its review to
`<iter>/design/reviews/<critic>.md`. Instruct every critic to **return ONLY the
path it wrote plus a one-line status** (e.g. `design/reviews/rigor-critic.md —
1 reject, 3 viable`) — never to echo the review back. You merge in Step 4 by
reading those files from disk, not from the critics' replies. This keeps Stage 3
fast and bounds your context.

**Resilience:** if a critic sub-agent fails or returns nothing, retry once; then
continue with the survivors and note the gap. Two of three is still a valid
debate.

## Step 4 — Merge & save

**Read each `<iter>/design/reviews/<critic>.md` from disk**, then merge the
critic reviews into **one** file, `<iter>/design/reviews.md`, written in a single
complete `Write`, with this structure:

```markdown
# ERA Stage 3 — Multi-Review — <project name>

**Iteration:** <n>   **Debate round:** <r>   **Critics:** <which ran>

## Consensus verdict
| candidate_id | alignment-critic | feasibility-critic | rigor-critic | consensus |
|--------------|------------------|--------------------|--------------|-----------|

<consensus = strong / viable / weak / reject — the harshest blocking verdict
wins; a `reject` from any critic is a blocking defect.>

## Per-candidate critique
### <candidate_id>
<the three critics' verdicts, the defects found, and the concrete fixes.>

## Feedback for revision
<If any candidate has a blocking defect (`weak`/`reject`) that a Stage 2 redo
could fix: a concrete, prioritized revision brief — what to change and why. If
every candidate is `strong`/`viable`, say so plainly: "no blocking defects".>
```

Append a one-line Stage-3 note (date, consensus summary, any critic gap) to
`<workspace>/logs/iterations/`.

## Principles

- **Harsh but fair** — a critique must be concrete and actionable; "looks weak"
  is not a finding, "the τ ≥ 0.6 claim cites no human-rated set" is.
- **Evaluation-method focused** — judge whether each candidate will *measure
  evaluation quality well*, not whether it is novel for its own sake.
- **Honesty** — never invent a defect; a missing detail in a candidate is itself
  a real finding — name it.
- **Autonomous** — never ask the operator; record decisions in
  `<workspace>/logs/iterations/`.
