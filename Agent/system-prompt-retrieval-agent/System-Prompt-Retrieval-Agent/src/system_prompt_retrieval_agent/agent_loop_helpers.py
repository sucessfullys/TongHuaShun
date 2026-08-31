"""S09 V0.2.1 agent-loop helpers (parent-only).

Adds the cross-round barrier exception, cell-id-index writer + collision
detector, and the strict-mode user-prompt coupling assertion. These
helpers are designed for direct unit testing without rewriting
``agent_loop.run()`` end-to-end. They are pulled into ``agent_loop`` at
the appropriate dispatch points.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Tuple

CellKey = Tuple[str, str, str]  # (prompt_pair_id, user_prompt_id, sample_id)


class CrossRoundBarrierViolation(RuntimeError):
    """Raised when round K+1 attempts to start before round K's
    finalize step has completed (workflow.cross_round_pipelining=False).
    Plan §11.5, §S09.09.
    """

    def __init__(self, current_round: int, prior_round_finalized: bool, *, detail: str = ""):
        self.current_round = current_round
        self.prior_round_finalized = prior_round_finalized
        msg = (
            f"CrossRoundBarrierViolation: round {current_round} cannot begin "
            f"before round {current_round - 1} finalize is complete "
            f"(prior_round_finalized={prior_round_finalized})."
        )
        if detail:
            msg += f" {detail}"
        super().__init__(msg)


class CrossRoundCellIdCollision(RuntimeError):
    """Raised when the cell_id_index detects the same
    `(round_id, prompt_pair_id, user_prompt_id, sample_id)` written
    twice across rounds. Plan §S09.10.
    """


def assert_cross_round_barrier(
    current_round: int,
    prior_round_finalized: bool,
    cross_round_pipelining_enabled: bool,
) -> None:
    """Strict cross-round barrier check (workflow §11.5).

    When `cross_round_pipelining` is False (V0.2.1 default), round K+1
    must NOT begin before round K is fully finalized.
    """
    if cross_round_pipelining_enabled:
        return
    if current_round <= 1:
        return
    if not prior_round_finalized:
        raise CrossRoundBarrierViolation(
            current_round=current_round,
            prior_round_finalized=prior_round_finalized,
        )


def write_cell_id_index_row(index_path: Path, run_id: str, round_id: int, cell: CellKey) -> None:
    """Append one cell-id-index row. Used for cross-round collision detection."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pair_id, up_id, sample_id = cell
    row = {
        "run_id": run_id,
        "round_id": round_id,
        "prompt_pair_id": pair_id,
        "user_prompt_id": up_id,
        "sample_id": sample_id,
    }
    with index_path.open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def detect_cross_round_collisions(index_path: Path) -> list[dict[str, Any]]:
    """Read cell_id_index.jsonl and return any (pair, up, sample) tuples
    that appear in more than one round_id. Empty list = clean.
    Plan §S09.10."""
    if not index_path.exists():
        return []
    seen: dict[CellKey, set[int]] = {}
    with index_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key: CellKey = (
                row["prompt_pair_id"],
                row["user_prompt_id"],
                row["sample_id"],
            )
            seen.setdefault(key, set()).add(int(row["round_id"]))

    collisions: list[dict[str, Any]] = []
    for key, rounds in seen.items():
        if len(rounds) > 1:
            collisions.append({
                "prompt_pair_id": key[0],
                "user_prompt_id": key[1],
                "sample_id": key[2],
                "round_ids": sorted(rounds),
            })
    return collisions


def assert_no_cross_round_cell_id_collision(index_path: Path) -> None:
    """Raise CrossRoundCellIdCollision if any cell repeats across rounds."""
    collisions = detect_cross_round_collisions(index_path)
    if collisions:
        sample = collisions[0]
        raise CrossRoundCellIdCollision(
            f"Cross-round cell_id collision: pair={sample['prompt_pair_id']!r} "
            f"user_prompt={sample['user_prompt_id']!r} sample={sample['sample_id']!r} "
            f"appears in rounds {sample['round_ids']}. Total collisions: {len(collisions)}."
        )


def assert_config_strict_user_prompt_coupling(cfg: Any) -> None:
    """Plan §6.4 strict-mode coupling assertion + §S09.06a.

    Under workflow.allow_partial_samples=False, the
    gemma_user_prompts.min_surviving_user_prompts_per_pair knob is
    silently overridden to enabled-count. This helper warns the user
    if the config silently weakens strict mode.
    """
    if cfg.workflow.allow_partial_samples:
        return  # knob honored; no coupling violation
    enabled = sum(1 for e in cfg.gemma_user_prompts.library if e.enabled)
    knob = cfg.gemma_user_prompts.min_surviving_user_prompts_per_pair
    if enabled and knob < enabled:
        # Strict mode wins; surface a structured warning so callers can log it.
        raise StrictModeUserPromptCouplingWarning(
            f"strict mode (allow_partial_samples=False) overrides "
            f"min_surviving_user_prompts_per_pair={knob} "
            f"(enabled-count={enabled}). All enabled cells must succeed."
        )


class StrictModeUserPromptCouplingWarning(Exception):
    """Surfaced when config implicitly weakens strict mode. Plan §S09.06a."""
