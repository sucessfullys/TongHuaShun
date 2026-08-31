"""context_builder.py — Build and validate PromptPairHistoryContext for generation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from system_prompt_retrieval_agent.memory.shared import resolve_prompt_path
from system_prompt_retrieval_agent.schemas import (
    PromptPair,
    PromptPairHistoryContext,
    PromptPairScoreContext,
)

logger = logging.getLogger(__name__)

_CONTEXT_BLOCKS = [
    "long_memory_prompt_pairs",
    "project_memory_prompt_pairs",
    "previous_top_prompt_pairs",
    "previous_round_prompt_pairs",
    "random_history_prompt_pairs",
]


def build_history_context(memory_manager, round_id: int, N: int) -> PromptPairHistoryContext:
    """Delegate to memory_manager.load_for_generation and return history context.

    Args:
        memory_manager: Object with ``load_for_generation(round_id, N)`` method.
        round_id: Current round number.
        N: Number of pairs requested per generation call.

    Returns:
        PromptPairHistoryContext populated by the memory manager.
    """
    return memory_manager.load_for_generation(round_id, N)


def assert_rich_context(ctx: PromptPairHistoryContext) -> None:
    """Assert that all pairs in the context have scores properly attached.

    For each non-None PromptPair across the five context blocks, the pair's
    ``scores`` field must be either:
      - a ``PromptPairScoreContext`` instance (which may contain ``missing_score_reason``
        if scores are legitimately absent for this pair), or
      - Python ``None`` is NOT acceptable — it indicates a detached-table shape
        where scores were stored outside the pair object.

    Raises:
        AssertionError: If any pair has ``scores=None`` (Python None), indicating
            scores are detached from pair objects.
    """
    for block_name in _CONTEXT_BLOCKS:
        pairs: Optional[list[PromptPair]] = getattr(ctx, block_name, None)
        if pairs is None:
            continue
        for i, pair in enumerate(pairs):
            assert pair.scores is not None, (
                f"Pair {pair.prompt_pair_id!r} in block {block_name!r}[{i}] has "
                f"scores=None — detached-table shape detected. "
                f"Use PromptPairScoreContext(missing_score_reason=...) instead of None "
                f"when scores are not yet available."
            )
            assert isinstance(pair.scores, PromptPairScoreContext), (
                f"Pair {pair.prompt_pair_id!r} in block {block_name!r}[{i}] has "
                f"scores of unexpected type {type(pair.scores)!r}."
            )


# ---------------------------------------------------------------------------
# V0.2.1 per-pair breakdown helpers (plan §8.1, §8.2, §8.5)
# ---------------------------------------------------------------------------

def _build_per_user_prompt_breakdown(scores: PromptPairScoreContext) -> dict[str, Any]:
    """Build the per_user_prompt_breakdown block for a single pair's scores.

    Groups scores by language (zh/en) based on user_prompt_id prefix.
    Includes worst_user_prompt_id and cross_lingual_gap inline.

    Args:
        scores: The PromptPairScoreContext attached to the pair.

    Returns:
        Dict with keys:
          - ``by_user_prompt``: per-user-prompt sub-scores dict
          - ``zh_mean_score``: mean score over zh prompts (or None)
          - ``en_mean_score``: mean score over en prompts (or None)
          - ``worst_user_prompt_id``: id of the worst user-prompt (or None)
          - ``cross_lingual_gap``: abs(zh_mean - en_mean) (or None)
    """
    by_user_prompt: dict[str, Any] = {}

    # Collect per-user-prompt sub-scores
    if scores.per_user_prompt:
        for uid, sub in scores.per_user_prompt.items():
            by_user_prompt[uid] = {
                "language": "zh" if uid.startswith("zh") else "en",
                "qwen_pass_rate": sub.qwen_pass_rate,
                "edit_correctness": sub.edit_correctness,
                "garment_transfer_correctness": sub.garment_transfer_correctness,
                "preservation": sub.preservation,
                "artifact_penalty": sub.artifact_penalty,
            }

    # Group mean scores by language from mean_score_by_user_prompt
    zh_scores: list[float] = []
    en_scores: list[float] = []
    if scores.mean_score_by_user_prompt:
        for uid, score in scores.mean_score_by_user_prompt.items():
            if uid.startswith("zh"):
                zh_scores.append(score)
            elif uid.startswith("en"):
                en_scores.append(score)

    zh_mean = (sum(zh_scores) / len(zh_scores)) if zh_scores else scores.zh_mean_score
    en_mean = (sum(en_scores) / len(en_scores)) if en_scores else scores.en_mean_score

    # cross_lingual_gap
    cross_lingual_gap = scores.cross_lingual_gap
    if cross_lingual_gap is None and zh_mean is not None and en_mean is not None:
        cross_lingual_gap = abs(zh_mean - en_mean)

    return {
        "by_user_prompt": by_user_prompt,
        "zh_mean_score": zh_mean,
        "en_mean_score": en_mean,
        "worst_user_prompt_id": scores.worst_user_prompt_id,
        "cross_lingual_gap": cross_lingual_gap,
    }


def _pair_to_context_dict(pair: PromptPair) -> dict[str, Any]:
    """Serialize a PromptPair to a context dict with inline scores and breakdown.

    The scores are embedded directly on the pair dict (no detached table).
    The per_user_prompt_breakdown is also embedded inline.
    Failure summaries are attached per pair.

    Args:
        pair: A PromptPair with scores attached.

    Returns:
        Dict with pair fields + ``scores`` inline + ``per_user_prompt_breakdown``
        inline + ``failure_summaries`` list.
    """
    base = {
        "prompt_pair_id": pair.prompt_pair_id,
        "system_prompt_id": pair.system_prompt_id,
        "negative_prompt_id": pair.negative_prompt_id,
        "round_id": pair.round_id,
        "system_prompt": pair.system_prompt,
        "negative_prompt": pair.negative_prompt,
        "rationale": pair.rationale,
        "expected_improvement_target": pair.expected_improvement_target,
        "risk": pair.risk,
        "selection_role": pair.selection_role,
    }

    # Inline scores (never detached)
    if pair.scores is not None:
        base["scores"] = {
            "overall_score": pair.scores.overall_score,
            "total_score": pair.scores.total_score,
            "missing_score_reason": pair.scores.missing_score_reason,
            "zh_mean_score": pair.scores.zh_mean_score,
            "en_mean_score": pair.scores.en_mean_score,
            "prompt_sensitivity": pair.scores.prompt_sensitivity,
            "cross_lingual_gap": pair.scores.cross_lingual_gap,
            "worst_user_prompt_id": pair.scores.worst_user_prompt_id,
            "worst_user_prompt_score": pair.scores.worst_user_prompt_score,
        }
        # Inline per_user_prompt_breakdown (V0.2.1 §8.1, §8.2)
        base["per_user_prompt_breakdown"] = _build_per_user_prompt_breakdown(pair.scores)
    else:
        base["scores"] = None
        base["per_user_prompt_breakdown"] = None

    # Failure summaries attached per pair (never in a separate table)
    base["failure_summaries"] = [
        {
            "sample_id": fs.sample_id,
            "category": fs.category,
            "summary": fs.summary,
            "failure_tags": fs.failure_tags,
            "score": fs.score,
        }
        for fs in pair.failure_summaries
    ]

    return base


def build_generation_input_blocks(
    ctx: PromptPairHistoryContext,
    tryon_prompts: list[dict[str, Any]],
    enabled_user_prompt_ids: list[str] | None = None,
    *,
    memory_root: Path | None = None,
) -> dict[str, Any]:
    """Assemble the full generation input dict from context + TRYON_PROMPTS library.

    Builds five context blocks (each pair has inline scores + per_user_prompt_breakdown
    + failure_summaries — no detached score tables) plus a static ``user_prompt_library``
    block listing the enabled TRYON_PROMPTS entries (text + language tags).

    Args:
        ctx: History context with the five blocks of PromptPairs.
        tryon_prompts: Full TRYON_PROMPTS list from seed_loader.load_tryon_prompts().
        enabled_user_prompt_ids: If provided, filter the library to only these IDs.
            If None, uses entries with ``enabled=True``.

    Returns:
        Dict with keys matching the five context block names + ``user_prompt_library``.
        No detached score tables anywhere in the output.
    """
    result: dict[str, Any] = {}

    for block_name in _CONTEXT_BLOCKS:
        pairs: Optional[list[PromptPair]] = getattr(ctx, block_name, None)
        if pairs is None:
            result[block_name] = []
        else:
            result[block_name] = [_pair_to_context_dict(p) for p in pairs]

    # Build fixed user-prompt library — enabled entries only
    library_entries: list[dict[str, Any]] = []
    for entry in tryon_prompts:
        if not isinstance(entry, dict):
            continue
        uid = entry.get("user_prompt_id", "")
        if enabled_user_prompt_ids is not None:
            if uid not in enabled_user_prompt_ids:
                continue
        else:
            if not entry.get("enabled", False):
                continue
        library_entries.append({
            "user_prompt_id": uid,
            "language": entry.get("language", ""),
            "text": entry.get("text", ""),
        })

    result["user_prompt_library"] = library_entries

    # Cross-round shared rules surface (project's shared memory).
    # Each rule carries rule_id / text / metadata, plus the real system_prompt
    # text resolved from prompt_path so the LLM can read the prompt inline
    # without cross-referencing the long_memory block by id.
    rules = getattr(ctx, "shared_rules", None)
    result["shared_rules"] = []
    if rules:
        import yaml as _yaml  # local import; yaml is already a project dep
        for r in rules:
            if not (r.get("rule_id") and r.get("text")):
                continue
            entry: dict[str, Any] = {
                "rule_id": r.get("rule_id"),
                "text": r.get("text"),
                "confidence": r.get("confidence"),
                "source_round": r.get("source_round"),
                "last_seen": r.get("last_seen"),
                "tags": r.get("tags"),
                "system_prompt_id": r.get("system_prompt_id"),
                "prompt_pair_id": r.get("prompt_pair_id"),
                "source_run_id": r.get("source_run_id"),
                "n_cells": r.get("n_cells"),
                "pair_overall": r.get("pair_overall"),
                "improvement_summary": r.get("improvement_summary"),
                "axis_means_json": r.get("axis_means_json"),
                "worst_failure_tags": r.get("worst_failure_tags"),
                "worst_user_prompt_id": r.get("worst_user_prompt_id"),
                "zh_mean_score": r.get("zh_mean_score"),
                "en_mean_score": r.get("en_mean_score"),
                "delta_vs_prev": r.get("delta_vs_prev"),
            }
            prompt_path = r.get("prompt_path")
            # CSV stores prompt_path as relative to memory_root; resolve at use-time.
            resolved = resolve_prompt_path(prompt_path, memory_root) if memory_root else (
                Path(prompt_path) if prompt_path else None
            )
            if resolved is not None and resolved.is_file():
                try:
                    data = _yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
                    entry["system_prompt"] = data.get("system_prompt")
                    entry["negative_prompt"] = data.get("negative_prompt")
                except (OSError, _yaml.YAMLError) as exc:
                    logger.warning("shared_rules: failed to load %s: %s", resolved, exc)
            result["shared_rules"].append(entry)

    return result
