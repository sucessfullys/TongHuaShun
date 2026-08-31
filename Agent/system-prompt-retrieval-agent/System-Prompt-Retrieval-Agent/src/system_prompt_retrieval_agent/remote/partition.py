"""Partition a StageManifest into surviving and failed prompt pairs.

Plan reference: §6.4, §11.5, S04.06.
"""
from __future__ import annotations

from typing import Optional

from system_prompt_retrieval_agent.schemas import PerPairManifest, StageManifest


def partition_stage_pairs(
    stage_manifest: StageManifest,
    allow_partial: bool,
    enabled_user_prompt_ids: list[str],
    expected_samples_for_pair: dict[str, list[str]],
    manifest_purpose: Optional[str] = None,
) -> tuple[list[str], list[dict]]:
    """Partition manifest pairs into surviving and failed.

    Parameters
    ----------
    stage_manifest:
        The V0.2.1 stage manifest returned by the remote controller.
    allow_partial:
        If ``False`` (strict mode, default), a pair survives only when
        **every** enabled ``(pair, user_prompt, sample)`` cell has
        ``errors == 0``.  The ``min_surviving_user_prompts_per_pair``
        knob is overridden by strict mode (plan §6.4).
        If ``True`` (partial mode), a pair survives when at least one
        enabled ``(pair, user_prompt)`` cell succeeded.
    enabled_user_prompt_ids:
        The full enabled user-prompt ID list for the run.
    expected_samples_for_pair:
        Mapping of ``prompt_pair_id`` → list of expected ``sample_id``s.
        Used to compute the expected total cell count per pair.
    manifest_purpose:
        When ``"resume_missing_cells"``, the expected total check is
        skipped — omitted cells are assumed already valid on disk.
        ``None`` or ``"prior_stage_survivor_cells"`` use the full count
        check.

    Returns
    -------
    surviving_pairs:
        List of ``prompt_pair_id`` strings that survived.
    failed_pairs:
        List of dicts ``{prompt_pair_id, failure_reason}`` for pairs that
        did not survive.
    """
    surviving_pairs: list[str] = []
    failed_pairs: list[dict] = []

    for pair_id, per_pair in stage_manifest.pairs.items():
        failure_reason = _check_pair(
            pair_id=pair_id,
            per_pair=per_pair,
            allow_partial=allow_partial,
            enabled_user_prompt_ids=enabled_user_prompt_ids,
            expected_sample_ids=expected_samples_for_pair.get(pair_id, []),
            manifest_purpose=manifest_purpose,
        )
        if failure_reason is None:
            surviving_pairs.append(pair_id)
        else:
            failed_pairs.append({"prompt_pair_id": pair_id, "failure_reason": failure_reason})

    return surviving_pairs, failed_pairs


def _check_pair(
    pair_id: str,
    per_pair: PerPairManifest,
    allow_partial: bool,
    enabled_user_prompt_ids: list[str],
    expected_sample_ids: list[str],
    manifest_purpose: Optional[str] = None,
) -> Optional[str]:
    """Return a failure reason string, or ``None`` when the pair survives."""
    # Rule 1 — count integrity.
    if per_pair.ok + per_pair.errors != per_pair.total:
        return "incomplete_count"

    # Rule 3 — no duplicate cell keys within per_user_prompt totals.
    # (Cell-level data is not present in PerPairManifest; check per-user-prompt
    # totals sum correctly against pair totals.)
    up_ok_sum = sum(u.ok for u in per_pair.per_user_prompt.values())
    up_err_sum = sum(u.errors for u in per_pair.per_user_prompt.values())
    up_total_sum = sum(u.total for u in per_pair.per_user_prompt.values())
    if (up_ok_sum != per_pair.ok
            or up_err_sum != per_pair.errors
            or up_total_sum != per_pair.total):
        return "duplicate_sample_id"

    # Rule 2 — strict vs. partial.
    if not allow_partial:
        if manifest_purpose == "resume_missing_cells":
            # Omitted cells are assumed already valid on disk; skip count check.
            # But dispatched cells must have errors == 0.
            if per_pair.errors > 0:
                return "incomplete_count"
        else:
            # Strict mode: every enabled (pair, user_prompt) cell must have errors == 0.
            expected_total = len(enabled_user_prompt_ids) * len(expected_sample_ids)
            if per_pair.errors > 0:
                return "incomplete_count"
            if expected_total > 0 and per_pair.total < expected_total:
                return "incomplete_count"
    else:
        # Partial mode: pair survives if at least one (pair, user_prompt) cell succeeded.
        surviving_up = sum(
            1 for u in per_pair.per_user_prompt.values() if u.ok > 0 and u.errors == 0
        )
        if surviving_up == 0:
            return "incomplete_count"

    return None
