# Skill: summarizing-failures

## Purpose

Given the K lowest-scoring samples for a `(prompt_pair_id, user_prompt_id)`
bucket, generate a concise failure-mode summary and a list of failure tags.
Results are appended to `failures.jsonl` and
`memory/rounds/{round_id}/failures/failure_summaries.jsonl`.

## Inputs

| Field | Type | Description |
|---|---|---|
| `sample_id` | str | `category/filename` identifier for the sample. |
| `category` | str or null | Garment category (`dress`, `lower`, `upper`). |
| `overall_score` | float or null | Aggregated score for the sample. |
| `user_prompt_id` | str | User phrasing ID this failure is attributed to. |
| `notes` | str | Evaluation notes from the local VLM evaluator. |
| `edit_correctness` | float | Sub-score (0–1). |
| `garment_transfer_correctness` | float | Sub-score (0–1). |
| `preservation` | float | Sub-score (0–1). |
| `artifact_penalty` | float | Sub-score (0–1, higher = worse). |

## Selection

Top-K (default 10) samples by ascending `overall_score` (lowest-scoring first).
Stable tie-break by `sample_id` (ascending lexicographic).

## Outputs

JSON object matching `schema.json`:

```json
{
  "summary": "1-3 sentence description of the failure mode.",
  "failure_tags": ["misaligned_garment", "color_bleed"]
}
```

## Persisted JSONL Record

Each summary is appended with additional context fields:

```json
{
  "schema_version": "1.0",
  "round": <int>,
  "prompt_pair_id": "<str>",
  "user_prompt_id": "<str>",
  "sample_id": "<str>",
  "category": "<str or null>",
  "summary": "<str>",
  "failure_tags": ["<str>"],
  "score": <float or null>
}
```

The `user_prompt_id` field enables per-phrasing brittleness analysis: failures
grouped by `user_prompt_id` reveal which phrasings most expose system-prompt
weaknesses.

## Rate Limiting

All OpenAI calls are RateLimiter-gated at ≤ 3 req/s. Concurrency bounded by
`evaluation.max_concurrent`.

## Budget

Each call charges `$0.0005` against BudgetGuard. Raises `CostExhausted` if
either cap is reached.

## Model

`gpt-5.4-mini` (configured via `cfg.api.failure_summary_model`).
