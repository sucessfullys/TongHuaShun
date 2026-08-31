# Skill: evaluating-locally

## Purpose

Evaluate generated virtual try-on images using a local OpenAI Vision API call
(GPT-5.4). Each evaluation cell is keyed by `(prompt_pair_id, user_prompt_id,
sample_id)`. Returns sub-scores consumed by the scoring aggregator.

## Inputs

| Field | Type | Description |
|---|---|---|
| `prompt_pair_id` | str | Unique ID for the system/negative prompt pair. |
| `user_prompt_id` | str | ID of the user phrasing used during FLUX generation. |
| `sample_id` | str | `category/filename` or dataset sample identifier. |
| `model_image_path` | str | Path to the original model image. |
| `cloth_image_path` | str | Path to the target cloth image. |
| `generated_image_path` | str | Path to the FLUX-generated result. |
| `intermediate_prompt` | str | Gemma-produced intermediate caption for FLUX. |

## Outputs

JSON object matching `schema.json`:

```json
{
  "edit_correctness": 0.85,
  "garment_transfer_correctness": 0.80,
  "preservation": 0.90,
  "artifact_penalty": 0.05,
  "notes": "Brief evaluation notes."
}
```

## Sub-score Definitions

- `edit_correctness` (0–1): how well the edit follows the intermediate prompt
- `garment_transfer_correctness` (0–1): accuracy of garment placement and transfer
- `preservation` (0–1): how well the model identity and background are preserved
- `artifact_penalty` (0–1): visible artifacts (higher = more artifacts, worse result)
- `notes` (string): brief qualitative evaluation notes

## Rate Limiting

All OpenAI calls are RateLimiter-gated at ≤ 3 req/s. Concurrency bounded by
`evaluation.max_concurrent` (default 4).

## Budget

Each call charges `$0.002` against BudgetGuard. Raises `CostExhausted` if
either the daily or per-round cap is reached before the call.

## Persistence

Per-phrasing rows appended to `api_eval/{prompt_pair_id}/{user_prompt_id}.jsonl`
alongside standard fields: `prompt_pair_id`, `user_prompt_id`, `sample_id`,
all sub-scores, `usd_spent`.

## Retry Policy

- API retries: `max_api_retries=6`, exponential backoff + jitter.
- All calls pass through `rate_limiter.py`; never bypass the limiter.
