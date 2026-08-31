# Skill: evaluating-locally

## Purpose

Run GEdit-Bench-style local evaluation on generated try-on samples using an
OpenAI vision model. Produces per-sample sub-scores: `edit_correctness`,
`garment_transfer_correctness`, `preservation`, `artifact_penalty`, and `notes`.

## Inputs

| Key | Type | Description |
|-----|------|-------------|
| `sample_id` | `str` | Unique identifier for the sample |
| `model_image_path` | `str\|Path` | Path to the model person image |
| `cloth_image_path` | `str\|Path` | Path to the target garment image |
| `generated_image_path` | `str\|Path` | Path to the generated try-on image |
| `intermediate_prompt` | `str` | Text prompt used during generation |

## Outputs

Returns a dict matching `schema.json`:

```json
{
  "sample_id": "dress/sample_001",
  "edit_correctness": 0.85,
  "garment_transfer_correctness": 0.80,
  "preservation": 0.90,
  "artifact_penalty": 0.05,
  "notes": "Brief evaluation notes.",
  "usd_spent": 0.002
}
```

## Implementation

Module: `system_prompt_retrieval_agent.evaluation.local_api_eval.LocalApiEvaluator`

The evaluator:
1. Encodes all three images as base64 data URIs (via Pillow).
2. Constructs a vision message with `image_url` content blocks.
3. Calls `cfg.api.api_eval_model` with `response_format: json_object`.
4. Enforces `cfg.evaluation.max_concurrent` via `asyncio.Semaphore`.
5. Rate-limits via `get_rate_limiter().acquire()` (max 3 req/s).
6. Charges `$0.002` per call to both `RateLimiter.add_cost` and `BudgetGuard.charge`.
7. Raises `CostExhausted` if budget caps are exceeded.

### Quick usage

```python
from system_prompt_retrieval_agent.evaluation import LocalApiEvaluator, BudgetGuard

guard = BudgetGuard(daily_usd_cap=50.0, per_round_usd_cap=10.0)
evaluator = LocalApiEvaluator(cfg, budget_guard=guard)
results = await evaluator.evaluate_many(samples)
```

## Rate-Limit Contract

All calls go through `rate_limiter.get_rate_limiter()`. Maximum 3 req/s, burst 5.
Never bypass for any reason.

## Config toggle

Set `evaluation.run_local_api_eval: false` in `config.yaml` to skip evaluation
entirely. `evaluate_many` returns an empty list when this flag is false.
