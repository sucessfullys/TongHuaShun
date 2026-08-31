from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from system_prompt_retrieval_agent.config import AppConfig
from system_prompt_retrieval_agent.rate_limiter import get_rate_limiter
from system_prompt_retrieval_agent.schemas import FailureSummary
from system_prompt_retrieval_agent.evaluation.budget_guard import BudgetGuard, CostExhausted

# Cost per failure summary call (fixed stub)
_USD_PER_SUMMARY = 0.0005

SCHEMA_VERSION = "1.0"


def select_worst_per_category(
    cell_results: list[dict],
    *,
    k_per_category: int = 7,
    score_weights: dict | None = None,  # accepted for API symmetry; unused (_cell_score is unweighted)
    categories: tuple[str, ...] = ("dress", "lower", "upper"),
) -> list[dict]:
    """Pick the worst ``k_per_category`` cells per category (dress/lower/upper).

    Each ``cell_results`` row needs a ``sample_id`` — category is derived from
    its ``__``-delimited prefix. Overall score uses :func:`scoring_v022._cell_score`,
    the same per-cell aggregator the ranking pipeline uses (mean over available
    sub-scores; axis weighting is applied later at the per-pair total).

    Returns a flat list of at most ``len(categories) * k_per_category`` rows,
    each annotated with ``category`` and ``overall_score`` for downstream
    :func:`summarize_one` calls.
    """
    from system_prompt_retrieval_agent.scoring_v022 import _cell_score

    bucketed: dict[str, list[tuple[float, dict]]] = {c: [] for c in categories}
    for r in cell_results:
        sid = r.get("sample_id", "")
        if not sid:
            continue
        cat = sid.split("__", 1)[0]
        if cat not in bucketed:
            continue
        score = _cell_score(r)
        if score is None:
            continue
        annotated = dict(r)
        annotated["category"] = cat
        annotated["overall_score"] = score
        bucketed[cat].append((score, annotated))

    out: list[dict] = []
    for cat in categories:
        bucketed[cat].sort(key=lambda x: x[0])  # ascending: worst first
        out.extend(row for _, row in bucketed[cat][:k_per_category])
    return out


async def summarize_pair_for_improvement(
    *,
    prompt_pair_id: str,
    system_prompt: str,
    ranking_row: Any,
    failure_summaries: list[FailureSummary],
    cfg: AppConfig,
    openai_client: Any = None,
    budget_guard: BudgetGuard | None = None,
) -> str:
    """One gpt-5.4-mini call per pair per round → 2-4 sentence improvement directive.

    Inputs the model receives: current system prompt, pair_overall, axis means,
    per-category weighted scores, the failure_tag frequency table, and a few
    representative failure summaries. Output is a single concise directive
    that ChatGPT will read in the next round's ``shared_rules`` block.
    """
    if openai_client is None:
        import openai

        api_key = os.environ.get(cfg.api.openai_api_key_env)
        openai_client = openai.AsyncOpenAI(
            api_key=api_key,
            max_retries=int(cfg.rate_limits.max_api_retries),
        )

    rl = get_rate_limiter()
    await rl.acquire()
    if budget_guard is not None:
        budget_guard.charge(_USD_PER_SUMMARY)
    rl.add_cost(_USD_PER_SUMMARY)

    tag_counts: dict[str, int] = {}
    for fs in failure_summaries:
        for t in fs.failure_tags or []:
            tag_counts[t] = tag_counts.get(t, 0) + 1
    top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:10]

    axis_means = {
        "qwen_pass_rate": getattr(ranking_row, "mean_qwen_pass_rate", None),
        "edit_correctness": getattr(ranking_row, "mean_edit_correctness", None),
        "garment_transfer": getattr(ranking_row, "mean_garment_transfer_correctness", None),
        "preservation": getattr(ranking_row, "mean_preservation", None),
        "artifact_penalty": getattr(ranking_row, "mean_artifact_penalty", None),
    }
    per_cat = getattr(ranking_row, "per_category", {}) or {}
    per_cat_brief = {
        cat: {
            "n_cells": v.get("n_cells"),
            "weighted": v.get("weighted_score"),
        }
        for cat, v in per_cat.items()
    }

    examples_block = "\n".join(
        f"- [{fs.category or 'unknown'}] {fs.sample_id} (score={fs.score}): "
        f"{fs.summary} | tags={fs.failure_tags}"
        for fs in failure_summaries[:8]
    ) or "(no per-sample summaries this round)"

    user_msg = (
        f"You are improving an image-edit system prompt for try-on.\n\n"
        f"CURRENT SYSTEM PROMPT (id={prompt_pair_id}):\n"
        f"------\n{system_prompt}\n------\n\n"
        f"ROUND METRICS:\n"
        f"  pair_overall = {getattr(ranking_row, 'pair_overall', None)}\n"
        f"  axis_means   = {json.dumps(axis_means)}\n"
        f"  per_category = {json.dumps(per_cat_brief)}\n"
        f"  top_failure_tags = {top_tags}\n\n"
        f"WORST SAMPLES:\n{examples_block}\n\n"
        f"TASK: Output a 2-4 sentence directive for the NEXT round's prompt "
        f"author. Name the weakest axis and the one or two concrete "
        f"changes to the system prompt that would most likely fix the "
        f"observed failures. Do NOT rewrite the prompt; only say what to "
        f"change. Return JSON: {{\"improvement_summary\": string}}."
    )

    response = await openai_client.chat.completions.create(
        model=cfg.api.failure_summary_model,
        messages=[{"role": "user", "content": user_msg}],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    try:
        return str(json.loads(content).get("improvement_summary", "")).strip()
    except (ValueError, TypeError):
        return ""


def select_low_score_examples(samples: list[dict], k: int = 10) -> list[dict]:
    """
    Pick k samples with lowest 'overall_score'.
    Stable tie-break by sample_id (ascending lexicographic).
    """
    sorted_samples = sorted(
        samples,
        key=lambda s: (s.get("overall_score") if s.get("overall_score") is not None else float("inf"), s.get("sample_id", "")),
    )
    return sorted_samples[:k]


async def summarize_one(
    sample: dict,
    *,
    cfg: AppConfig,
    openai_client: Any = None,
    budget_guard: BudgetGuard | None = None,
) -> FailureSummary:
    """
    GPT call via cfg.api.failure_summary_model. Rate-limited. Budget-charged ($0.0005 each).
    Returns FailureSummary with prompt_pair_id attached via sample['prompt_pair_id'].
    """
    if openai_client is None:
        import openai

        api_key = os.environ.get(cfg.api.openai_api_key_env)
        openai_client = openai.AsyncOpenAI(
            api_key=api_key,
            max_retries=int(cfg.rate_limits.max_api_retries),
        )

    rl = get_rate_limiter()
    await rl.acquire()

    if budget_guard is not None:
        budget_guard.charge(_USD_PER_SUMMARY)
    rl.add_cost(_USD_PER_SUMMARY)

    sample_id = sample.get("sample_id", "")
    category = sample.get("category")
    overall_score = sample.get("overall_score")
    notes = sample.get("notes", "")
    sub_scores = {
        k: sample.get(k)
        for k in ("edit_correctness", "garment_transfer_correctness", "preservation", "artifact_penalty")
        if sample.get(k) is not None
    }

    prompt_text = (
        f"Sample ID: {sample_id}\n"
        f"Category: {category or 'unknown'}\n"
        f"Overall score: {overall_score}\n"
        f"Sub-scores: {json.dumps(sub_scores)}\n"
        f"Notes: {notes}\n\n"
        "Analyze this low-scoring sample and provide:\n"
        "1. A concise summary (1-3 sentences) of the failure mode\n"
        "2. A list of failure tags (e.g., 'misaligned_garment', 'color_bleed', 'pose_artifact')\n\n"
        "Return JSON: {\"summary\": string, \"failure_tags\": [string]}"
    )

    response = await openai_client.chat.completions.create(
        model=cfg.api.failure_summary_model,
        messages=[{"role": "user", "content": prompt_text}],
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    result = json.loads(content)

    return FailureSummary(
        sample_id=sample_id,
        category=category,
        summary=str(result.get("summary", "")),
        failure_tags=list(result.get("failure_tags", [])),
        score=float(overall_score) if overall_score is not None else None,
    )


async def summarize_failures(
    pair_samples: list[dict],
    *,
    prompt_pair_id: str,
    cfg: AppConfig,
    k: int = 10,
    openai_client: Any = None,
    budget_guard: BudgetGuard | None = None,
    user_prompt_id: str | None = None,
) -> list[FailureSummary]:
    """
    Select lowest k samples, call summarize_one for each (gathered with semaphore),
    return list[FailureSummary] with sample_id + category + summary + failure_tags + score.

    V0.2.1: ``user_prompt_id`` is attached to each sample so that the failure
    bucket key includes ``(prompt_pair_id, user_prompt_id)``.  This makes
    per-phrasing brittleness visible in the failure summaries.
    """
    if openai_client is None:
        import openai

        api_key = os.environ.get(cfg.api.openai_api_key_env)
        openai_client = openai.AsyncOpenAI(
            api_key=api_key,
            max_retries=int(cfg.rate_limits.max_api_retries),
        )

    # Attach prompt_pair_id (and optionally user_prompt_id) to each sample
    for s in pair_samples:
        s.setdefault("prompt_pair_id", prompt_pair_id)
        if user_prompt_id is not None:
            s.setdefault("user_prompt_id", user_prompt_id)

    selected = select_low_score_examples(pair_samples, k=k)
    semaphore = asyncio.Semaphore(cfg.evaluation.max_concurrent)

    async def _bounded(sample: dict) -> FailureSummary:
        async with semaphore:
            return await summarize_one(
                sample,
                cfg=cfg,
                openai_client=openai_client,
                budget_guard=budget_guard,
            )

    tasks = [_bounded(s) for s in selected]
    return list(await asyncio.gather(*tasks))


def append_summaries_jsonl(
    jsonl_path: Path,
    summaries: list[FailureSummary],
    prompt_pair_id: str,
    round_id: int,
    *,
    user_prompt_id: str | None = None,
) -> None:
    """
    Append each summary as JSONL with schema_version, round, prompt_pair_id,
    sample_id, summary, failure_tags, score.

    V0.2.1: when ``user_prompt_id`` is provided it is included in each record,
    making the bucket key ``(prompt_pair_id, user_prompt_id)`` so per-phrasing
    brittleness is queryable from the JSONL.
    """
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("a", encoding="utf-8") as fh:
        for fs in summaries:
            record: dict = {
                "schema_version": SCHEMA_VERSION,
                "round": round_id,
                "prompt_pair_id": prompt_pair_id,
            }
            if user_prompt_id is not None:
                record["user_prompt_id"] = user_prompt_id
            record.update({
                "sample_id": fs.sample_id,
                "category": fs.category,
                "summary": fs.summary,
                "failure_tags": fs.failure_tags,
                "score": fs.score,
            })
            fh.write(json.dumps(record) + "\n")
