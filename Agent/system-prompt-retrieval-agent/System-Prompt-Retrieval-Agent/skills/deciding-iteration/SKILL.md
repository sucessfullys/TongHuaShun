# Skill: deciding-iteration

Decide whether to start the next round based on score, budget, fallback streak, and signal state (plan §15).

## Stops when any hold
1. `max(overall_score) >= workflow.score_threshold` (default 0.80)
2. `round_id >= workflow.max_rounds` (default 6)
3. `usd_spent >= budget.daily_usd_cap` or `>= budget.per_round_usd_cap`
4. 3 consecutive rounds `fallback_prompt_generation=true`
5. Unrecoverable SSH tunnel failure (>10 min) or fatal remote error
6. SIGTERM/SIGINT: flush memory, write `best_pair.yaml`, exit 130

## Owner module
`system_prompt_retrieval_agent.agent_loop.AgentLoop`
