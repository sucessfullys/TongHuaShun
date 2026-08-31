# Skill: summarizing-failures

## Purpose

Identify the lowest-scoring try-on samples from an evaluation round, then use
an LLM to generate structured failure summaries with diagnostic tags. These
summaries feed back into prompt generation to guide iterative improvement.

## Inputs

| Parameter | Type | Description |
|-----------|------|-------------|
| `pair_samples` | `list[dict]` | Per-sample eval results with `overall_score`, `sample_id`, `category`, sub-scores, and `notes` |
| `prompt_pair_id` | `str` | ID of the prompt pair being analyzed |
| `k` | `int` | Number of lowest-scoring examples to summarize (default: 10) |
| `cfg` | `AppConfig` | App configuration |
| `openai_client` | optional | Injectable client for testing |
| `budget_guard` | optional | `BudgetGuard` for cost tracking |

## Outputs

Returns `list[FailureSummary]` (from `schemas.py`), each containing:
- `sample_id`: sample identifier
- `category`: garment category
- `summary`: 1-3 sentence failure description
- `failure_tags`: list of diagnostic tag strings
- `score`: overall_score of this sample

## Implementation

Module: `system_prompt_retrieval_agent.evaluation.failure_summary`

Key functions:
- `select_low_score_examples(samples, k)` — selects k lowest by `overall_score`; stable tie-break by `sample_id`
- `summarize_one(sample, cfg, openai_client, budget_guard)` — single LLM call
- `summarize_failures(...)` — orchestrates selection + concurrent summarization
- `append_summaries_jsonl(jsonl_path, summaries, prompt_pair_id, round_id)` — persist to JSONL

### Quick usage

```python
from system_prompt_retrieval_agent.evaluation.failure_summary import (
    summarize_failures, append_summaries_jsonl
)

summaries = await summarize_failures(
    pair_samples,
    prompt_pair_id="pair_001",
    cfg=cfg,
    k=10,
)
append_summaries_jsonl(output_path / "failures.jsonl", summaries, "pair_001", round_id=3)
```

## JSONL Schema

Each line in the output JSONL file:
```json
{
  "schema_version": "1.0",
  "round": 3,
  "prompt_pair_id": "pair_001",
  "sample_id": "dress/sample_042",
  "category": "dress",
  "summary": "The garment color was not correctly transferred...",
  "failure_tags": ["wrong_garment_color", "texture_loss"],
  "score": 0.31
}
```

## Rate-Limit and Budget Contract

- Each call routed through `get_rate_limiter().acquire()`.
- Fixed stub cost: `$0.0005` per call.
- `BudgetGuard.charge(0.0005)` called before each API request.
- `CostExhausted` propagates upward without retry.
- Concurrency limited by `cfg.evaluation.max_concurrent` via `asyncio.Semaphore`.
