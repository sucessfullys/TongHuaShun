# Skill: Aggregating Scores

## Purpose

Compute per-pair overall scores and per-category scores from raw per-sample
sub-scores, then rank prompt pairs and write scoring artifacts for downstream
memory injection and reporting.

---

## Owner Module

`src/system_prompt_retrieval_agent/scoring/aggregate.py`

Supporting modules:
- `scoring/category.py` — per-category aggregation
- `scoring/ranking.py` — ranking, CSV/JSON artifact writing, next-round context

Public surface (re-exported from `scoring/__init__.py`):
- `compute_overall_score`
- `compute_category_balance_bonus`
- `compute_category_scores`
- `rank_pairs`
- `write_ranking_artifacts`
- `build_next_round_context`

---

## Inputs

| Name | Type | Description |
|---|---|---|
| `sub_scores` | `dict` or `PromptPairSubScores` | Per-pair sub-scores: `qwen_pass_rate`, `edit_correctness`, `garment_transfer_correctness`, `preservation`, `artifact_penalty` |
| `weights` | `dict` or `EvaluationWeights` | Weight for each sub-score key; accepts Pydantic model via `.model_dump()` |
| `category_balance_bonus` | `float or None` | Pre-computed balance bonus to inject into overall score; `None` means 0.0 contribution |
| `per_sample_scores` | `list[dict]` | Per-sample records with `sample_id`, `category`, `qwen_status`, and numeric sub-score fields |
| `categories` | `list[str]` | Categories to compute; default `["dress", "lower", "upper"]` |
| `pairs` | `list[PromptPair]` | Fully-scored PromptPair objects for ranking |
| `round_dir` | `Path` | Round output directory for writing artifacts |
| `round_id` | `int` | Round number stamped into CSV rows |

---

## Outputs

| Name | Location | Description |
|---|---|---|
| `overall_score` | `float` in `[0.0, 1.0]` | Weighted sum of sub-scores with hard gate and clamp |
| `category_scores` | `dict[str, CategoryScoreContext]` | Per-category weighted and total scores |
| `ranked_pairs` | `list[PromptPair]` | Pairs sorted descending by `overall_score`; `None` scores last |
| `weighted_scores.csv` | `{round_dir}/scoring/` | One row per pair with all score columns |
| `category_scores.csv` | `{round_dir}/scoring/` | One row per (pair, category) |
| `ranking.json` | `{round_dir}/scoring/` | `[{prompt_pair_id, overall_score, rank}]` |

---

## Scoring Formula

```
overall = w_qwen  * qwen_pass_rate
        + w_edit  * edit_correctness
        + w_gt    * garment_transfer_correctness
        + w_pres  * preservation
        + w_art   * artifact_penalty      # weight is typically negative
        + w_cat   * category_balance_bonus_or_zero
```

Hard gate: if `qwen_pass_rate < hard_gate_threshold` (default 0.5), multiply
result by `hard_gate_multiplier` (default 0.3).

Clamp: output is clamped to `[0.0, 1.0]`.

Missing or `None` sub-scores are treated as `0.0` and logged at DEBUG level.

---

## Category Balance Bonus

```python
ratio = min(category_scores) / max(category_scores)
bonus = 1.0 if ratio >= min_ratio else ratio
# Returns 0.0 when < 2 categories or any score is 0
```

Default `min_ratio = 0.70`.

---

## Key Behaviours

- Pure functions for all computation; IO confined to `write_ranking_artifacts`.
- `rank_pairs` is a stable sort: ties broken ascending by `tie_break_field`
  (default `prompt_pair_id`).
- `build_next_round_context` returns deep copies of top-N pairs with scores
  attached, ready for memory injection.
- `compute_category_scores` excludes `parse_fail_*` qwen statuses from the
  pass-rate denominator.
- Missing categories return `None` in the output dict; callers may supply
  `missing_score_reason` when constructing a `CategoryScoreContext`.
- Weights default from `AppConfig.evaluation.weights` semantics; pass a
  `EvaluationWeights` Pydantic instance or a plain dict.

---

## Constraints

- No network calls, no OpenAI API usage.
- Import schemas only from `schemas.py`; never redefine `PromptPair`,
  `PromptPairScoreContext`, `PromptPairSubScores`, `CategoryScoreContext`.
- Do not write to `config.py`, `schemas.py`, or any path outside `scoring/`.
