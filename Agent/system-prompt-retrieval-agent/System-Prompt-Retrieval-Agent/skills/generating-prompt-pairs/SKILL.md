# Skill: generating-prompt-pairs

## Name
`generating-prompt-pairs`

## Description
Generates a batch of N `PromptPair` candidates for the next optimization round using the OpenAI API. The generator receives a rich history context (long-memory references, previous-round pairs with scores, failure summaries) and produces structured prompt pairs with rationale and expected improvement targets.

## Inputs
| Field | Type | Description |
|-------|------|-------------|
| `round_id` | `int` | Current round number (1-based). Used to build pair IDs and load memory context. |
| `N` | `int` | Number of prompt pairs to generate (typically 3). |
| `ctx` | `PromptPairHistoryContext` | Populated history context with 5 optional blocks: long_memory, project_memory, previous_top, previous_round, random_history. Each block contains `PromptPair` objects with attached `scores` and `failure_summaries`. |

## Outputs
`tuple[list[PromptPair], bool]`
- `list[PromptPair]`: N validated, deduplicated prompt pairs with IDs `sys_r{round:02d}_{slot:03d}`, `neg_r...`, `pair_r...`.
- `bool`: `fallback_used` — True if any pair was sourced from fallback chain (best existing, previous round, or seed).

## Constraints
- Rate-limited: ≤ 3 requests/second, enforced via `get_rate_limiter().acquire()` before each API call.
- Never logs API keys or environment variable values.
- Schema validation: responses must conform to `skills/generating-prompt-pairs/schema.json` with `minItems=maxItems=N`.
- Retry policy: 3 parse retries (60s wait each), 6 API retries (exponential+jitter backoff, max 60s).
- Fallback chain on full failure: best existing pair by `overall_score` → previous round's best → seed from `temp_wxy/PROMPTS.py`.
- `negative_prompt` values of `None`, `"None"`, `"none"`, `"NULL"`, `""`, or whitespace → Python `None`.

## Owner module
`src/system_prompt_retrieval_agent/prompt_generation/generator.py`

## Entry point
```python
from system_prompt_retrieval_agent.prompt_generation import generate_prompt_pairs

pairs, fallback_used = await generate_prompt_pairs(cfg, memory_manager, round_id=1, N=3, existing_pairs=[])
```
