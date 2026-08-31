# ERA Stage 1 — Research (literature)

You are running ERA Stage 1 for one project workspace. Goal: survey the
**evaluation methods, metrics, benchmarks, and human-correlation studies**
relevant to this project's AIGC task, and write a single structured survey
`<workspace>/research/literature.md` for the later pipeline stages to consume.

You are the **orchestrator**: you fan the search out to parallel
`literature-scout` sub-agents and merge their digests. **Never ask the operator
anything** — decide from the workspace and record the decision. The workspace
path was passed as the skill argument (`$ARGUMENTS`).

**Duration is normal here.** A real Stage 1 run takes **20-60 minutes** as the
MCP literature scouts walk arXiv / GitHub / web search across the candidate
topic decomposition. This is the autonomous-loop's design, not a cost to gate
behind an operator prompt — never pause to confirm. The PreToolUse hook in
`.claude/settings.json` (Phase D-3) will structurally block any
`AskUserQuestion` call from this stage; you have no escape hatch, by design.

## Step 1 — Read the project

Read `<workspace>/spec.md` and `<workspace>/config.yaml`. Extract:
- the **task** — `task_family` (generation / editing / …) and `task_adapter`;
- the **mission** and what a *human-aligned good result* means for this task —
  the properties an evaluator must reward and the defects it must penalize;
- the **data shape** (input roles, methods, sample count) — context for which
  metrics are even applicable.

## Step 2 — Decompose into search directions

### 2a — Default first-iteration decomposition

If this is the first invocation (`iter_001`, no carry-forward brief — see §2b),
split the survey into **5–7 search directions**. A good default set, adapted to
the task:

1. **VLM / LMM judge protocols** — multimodal-model evaluators, scoring rubrics,
   prompt designs for this task family. **Capability-tier floor (hard):** for
   image-edit / try-on tasks, treat **≥30B-class VLMs** as the floor for
   serious judges (named reference models: Qwen3.6-35B-A3B, Gemma-4-31B-it)
   — see `feedback-vlm-min-size`. Sub-30B (incl. 7B-class) entries may
   appear in the literature survey **only** as cost-baseline context, never
   as serious candidates. They are only allowed to enter Stage 2's brainstorm
   when the operator explicitly requested them in human feedback ("try the
   7B"); without that request, Stage 4 rejects them at brief synthesis.
2. **Metric / feature-based metrics** — FID, CLIP-score, LPIPS, SSIM, PSNR,
   DINO, segmentation-IoU, OCR, face-ID, pose, IQA — whichever apply.
3. **Task-specific benchmarks** — benchmarks built for this task / adapter
   (e.g. image-editing benchmarks, generation leaderboards).
4. **Human-correlation / meta-evaluation studies** — papers measuring how well
   automatic metrics agree with human judgement.
5. **Datasets with human ratings** — datasets usable to validate an evaluator.
6. **Open-source implementations** — repos/libraries for the above metrics and
   judges; scouts search these on GitHub directly via the GitHub MCP
   (`mcp__github__search_repositories` / `search_code`).
7. **Evolutionary / iterative-refinement strategies for auto-eval protocols** —
   how prior work refines judges between rounds (rubric tightening, prompt
   evolution, judge ensembling, scale escalation, EA-style selection/mutation
   of candidate evaluators). The Stage 9 ReAct advisor reads this section
   (§F in the merged survey) to ground its `evolution_proposals` carry-forward.

Merge or drop directions that do not fit the task; keep it to 5–7.

### 2b — Re-invocation with a literature_update_brief

If this iteration was spawned by Stage 9 with `REVISE_RERUN_STAGE1`, the
deterministic backbone placed a brief into the new iter's
`iteration.json.parent_feedback.literature_update_brief` (a workspace-
relative path, typically `iter_{N-1}/react/literature_update_brief.md`).
**Skip the full fan-out** and instead:

1. Read the brief — it has three sections (Drop / Deepen / Add) naming
   exactly which §s of `research/literature.md` to refresh.
2. Read the existing `research/literature.md` so you know what is already
   surveyed.
3. Spawn **2–3 targeted** `literature-scout` sub-agents (one per Add /
   Deepen bullet that needs new evidence). Each scout's brief is the
   single section to rewrite plus the operator context the brief names.
4. **Rewrite only the affected sections** of `research/literature.md` (keep
   every other section byte-stable); on a Drop bullet, mark the entry as
   `[deprioritized iter_N: <reason>]` rather than deleting it.
5. Prepend a short note to the file's frontmatter — under "Survey date",
   add `**Iteration N updates:** <one-line summary of what changed>` — so a
   reader can see at a glance which sections were touched.

The full Step 3 fan-out and Step 4 rewrite are not needed when only sections
were refreshed; the targeted-rewrite path skips both. The Principles below
still apply (resilience, honesty, autonomy).

## Step 3 — Fan out to parallel scouts

Spawn **one `literature-scout` sub-agent per direction, all in parallel** —
issue every Task call in a single turn. Give each scout: its one direction, the
task family/adapter, and the human-aligned notion of "good". Each scout searches
arXiv + GitHub (MCP) + web and returns a structured digest. Wait for all of them.

If sub-agents are unavailable, do the same searches yourself, one direction at a
time, using the arXiv + GitHub MCP servers + `WebSearch` / `WebFetch`. If a
search MCP is not registered, degrade to web search only — still produce the
survey.

If a scout fails or returns nothing, continue with the others and note the gap
— never block on one scout. Retry a transient sub-agent or tool failure once
before treating it as real.

## Step 4 — Summarize & save

Merge the scout digests — deduplicate, keep only directly relevant work — then
write **one** file, `<workspace>/research/literature.md`, in a **single
complete `Write`** (compose the whole document first, then write it once — never
append incrementally, so a half-written survey can never exist), with exactly
this structure:

```markdown
# ERA Literature Survey — <project name>

**Task:** <task_family> / <task_adapter>
**Survey date:** <YYYY-MM-DD>
**arXiv queries:** <all arXiv queries used>
**Web queries:** <all web queries used>

## 1. Task & evaluation framing
<2–3 sentences: what is being generated/edited, and what a human-aligned "good"
result means here — the properties an evaluator must reward / penalize.>

## 2. Field overview
<2–3 paragraphs: how this task is evaluated today; judge-based vs metric-based
paradigms; what counts as SOTA evaluation practice.>

## 3. Candidate evaluation methods
| id | Family | What it measures | Inputs needed | GPU / cost | Reported human correlation | Source |
|----|--------|------------------|---------------|-----------|----------------------------|--------|

<One row per concrete evaluator. Family = A (VLM/LMM judge) or B (metric /
feature / region). `id` is a short kebab-case handle. This table is the
structured handoff Stage 2 consumes — make it complete and concrete.>

## 4. Benchmarks & human-correlation studies
<Benchmarks for this task, and meta-evaluation studies: which metrics correlate
with humans, by how much, on what data.>

## 5. Datasets with human ratings
<Datasets carrying human preference / quality ratings that ERA could use to
validate a candidate evaluator.>

## 6. Available implementations
| Repo / paper | Match | License | Strategy | Rationale |
|--------------|-------|---------|----------|-----------|

<Match = high / medium / low. Strategy ∈ Adopt / Extend / Compose / Build.>

## 7. Recommendations for Stage 2
<Which evaluation methods ERA should prototype first and why; which to skip;
the open gaps. Concrete enough to seed Stage 2's protocol brainstorming.>
```

Strategy definitions: **Adopt** — use the implementation directly; **Extend** —
fork/wrap and modify; **Compose** — combine 2–3 tools; **Build** — implement
from scratch, referencing surveyed designs.

## Principles

- **Evaluation-method focused** — ERA retrieves *how to evaluate*, not how to
  generate. Every entry must inform evaluator choice.
- **Speed over exhaustiveness** — 5–10 strong references per direction.
- **Resilient** — retry a transient MCP / web / sub-agent failure up to 3 times
  with a short backoff before recording it as a failed search.
- **Honesty** — never fabricate papers, numbers, or correlations; if a search
  yielded little, say so plainly.
- **Autonomous** — never ask the operator; record decisions and a one-line
  Stage-1 note in `<workspace>/logs/iterations/`.
