# Skill: generating-prompt-pairs

## Purpose

Generate N `(system_prompt, negative_prompt)` pairs for the virtual try-on
pipeline. Each pair is scored by the remote 3×H100 workflow (Gemma → FLUX →
Qwen) and the local OpenAI evaluator, then ranked to drive the next round.

## Inputs

| Field | Type | Description |
|---|---|---|
| `round_id` | int | Current round number. |
| `N` | int | Number of pairs to generate. |
| `history_context` | PromptPairHistoryContext | Five populated context blocks (see below). |
| `tryon_prompts` | list[dict] | Enabled TRYON_PROMPTS entries (text + language tags). |

### Five context blocks (plan §8.2)

1. `long_memory_prompt_pairs` — top pairs from long-term memory CSV.
2. `project_memory_prompt_pairs` — pairs from project shared memory.
3. `previous_top_prompt_pairs` — top-ranked pairs from all prior rounds.
4. `previous_round_prompt_pairs` — pairs from the immediately prior round.
5. `random_history_prompt_pairs` — random historical pairs for diversity.

Each pair in every block carries **inline** scores (`PromptPairScoreContext`),
a `per_user_prompt_breakdown` block (zh/en grouped, plus `worst_user_prompt_id`
and `cross_lingual_gap`), and `failure_summaries`. No detached score tables.

### Fixed user-prompt library (V0.2.1 §8.2)

A static `user_prompt_library` block embeds all enabled TRYON_PROMPTS entries
(text + language tags) so GPT can see the full set of phrasings the system
prompt must withstand during evaluation.

## Outputs (plan §8.3)

JSON object matching `schema.json`:

```json
{
  "prompt_pairs": [
    {
      "system_prompt": "<string, 20–4000 chars>",
      "negative_prompt": "<string or null>",
      "rationale": "<non-empty string>",
      "expected_improvement_target": "<string>",
      "risk": "<string>",
      "cross_lingual_robustness_strategy": "<non-empty string>"
    }
  ]
}
```

`user_prompts` is **not** a valid output field. The schema uses
`additionalProperties: false` to reject any attempt to propose new user prompts.

## Retry policy (plan §8.5)

- Parse retries: `max_parse_retries=3`, wait 60 s between attempts.
- API retries: `max_api_retries=6`, exponential + jitter.
- Fallback order on exhaustion: best_existing → best_in_context → seed pair.

## Deduplication (plan §8.7)

`pair_hash(sys, neg) = (sha256(sys)[:12], sha256(neg or "")[:12])`.
One regeneration slot per duplicate; accept with `duplicate=True` if still
colliding after regeneration.

## Model

`gpt-5.4` with Structured Outputs when supported; fallback to `json_object` +
jsonschema validation. All calls are RateLimiter-gated (≤ 3 req/s).
