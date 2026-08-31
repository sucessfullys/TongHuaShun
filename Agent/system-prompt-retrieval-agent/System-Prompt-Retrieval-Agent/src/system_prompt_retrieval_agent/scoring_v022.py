"""V0.2.2 scoring wiring (S06.01–S06.11).

Bridges the V0.2.2 merged-view manifests + on-disk artifacts produced
by the orchestrator (``agent_loop_v022.V022Orchestrator``) into the
existing scoring/eval primitives in ``scoring/`` and ``evaluation/``.

Responsibilities:

* S06.01 — build evaluation cells from canonical FLUX ``result.png``,
  Qwen ``eval.json``, Gemma ``intermediate_prompt.txt``, plus
  per-sample model/cloth dataset paths.
* S06.02 — reject empty evaluation-cell lists for any pair that
  survived Qwen.
* S06.03 — call ``LocalApiEvaluator.evaluate_many_cells`` instead of
  ``evaluate_many``.
* S06.04 — gather with ``return_exceptions=True`` so per-cell failures
  are captured (delegated to existing implementation; this module
  surfaces results and per-cell error rows).
* S06.05 — apply skip / penalty rules to failed cells.
* S06.06 — call ``build_score_context_for_pair`` for V0.2.2.
* S06.07 — call ``rank_pairs_with_gate`` for production ranking.
* S06.08 — call ``write_user_prompt_scores_csv``.
* S06.09 — move pairs missing required scoring input into
  ``failed_pairs[]`` with reason
  :data:`canonical_paths.SCORING_FAILED_REASON_MISSING_INPUT`. If no
  rankable pairs remain, abort before ranking/memory.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .remote._vendored import canonical_paths as cp


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalCell:
    prompt_pair_id: str
    user_prompt_id: str
    sample_id: str
    intermediate_prompt: str
    generated_image_path: str
    qwen_eval_json_path: str
    model_image_path: str
    cloth_image_path: str

    def as_evaluator_input(self) -> dict[str, Any]:
        return {
            "prompt_pair_id": self.prompt_pair_id,
            "user_prompt_id": self.user_prompt_id,
            "sample_id": self.sample_id,
            "intermediate_prompt": self.intermediate_prompt,
            "generated_image_path": self.generated_image_path,
            "model_image_path": self.model_image_path,
            "cloth_image_path": self.cloth_image_path,
            "qwen_eval_json_path": self.qwen_eval_json_path,
        }


class ScoringInputMissing(ValueError):
    """A pair lacks required scoring input — moved to failed_pairs[]."""


class NoRankablePairsError(RuntimeError):
    """Aborted before ranking/memory because every pair failed."""


# ---------------------------------------------------------------------------
# S06.01 — build evaluation cells
# ---------------------------------------------------------------------------


def build_evaluation_cells(
    *,
    artifact_root: Path,
    run_id: str,
    round_id: int,
    qwen_merged: Mapping[str, Any],
    flux_merged: Mapping[str, Any],
    gemma_merged: Mapping[str, Any],
    sample_lookup: Mapping[str, Mapping[str, str]],
) -> list[EvalCell]:
    """Build evaluation cells from V0.2.2 merged-view manifests.

    A cell makes the eval input list iff it appears as successful
    (status ∈ ok/carried_over) in **all three** stages' merged views
    AND the on-disk artifacts for FLUX (``result.png``), Qwen
    (``eval.json``), and Gemma (``intermediate_prompt.txt``) are
    present.

    ``sample_lookup`` maps ``sample_id`` to a dict with
    ``model_image_path`` and ``cloth_image_path``.
    """
    qwen_keys = _ok_cell_keys(qwen_merged)
    flux_keys = _ok_cell_keys(flux_merged)
    gemma_keys = _ok_cell_keys(gemma_merged)
    common = qwen_keys & flux_keys & gemma_keys

    out: list[EvalCell] = []
    for pid, upid, sid in sorted(common):
        flux_path = artifact_root / cp.cell_artifact_path(
            run_id, "flux", round_id, pid, upid, sid
        )
        qwen_path = artifact_root / cp.cell_artifact_path(
            run_id, "qwen", round_id, pid, upid, sid
        )
        gemma_path = artifact_root / cp.cell_artifact_path(
            run_id, "gemma", round_id, pid, upid, sid
        )
        if not (flux_path.is_file() and qwen_path.is_file() and gemma_path.is_file()):
            continue
        sample = sample_lookup.get(sid, {})
        out.append(
            EvalCell(
                prompt_pair_id=pid,
                user_prompt_id=upid,
                sample_id=sid,
                intermediate_prompt=gemma_path.read_text(encoding="utf-8"),
                generated_image_path=str(flux_path),
                qwen_eval_json_path=str(qwen_path),
                model_image_path=sample.get("model_image_path", ""),
                cloth_image_path=sample.get("cloth_image_path", ""),
            )
        )
    return out


def _ok_cell_keys(merged: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (c["prompt_pair_id"], c["user_prompt_id"], c["sample_id"])
        for c in merged.get("cells", [])
        if c.get("status") in cp.SUCCESSFUL_CELL_STATUSES
    }


# ---------------------------------------------------------------------------
# S06.02 — reject empty per-pair evaluation
# ---------------------------------------------------------------------------


def reject_empty_evaluation_per_pair(
    eval_cells: Sequence[EvalCell],
    *,
    qwen_surviving_pairs: Iterable[str],
) -> dict[str, list[EvalCell]]:
    """Group eval cells by pair_id; raise if a Qwen-surviving pair has no cells."""
    by_pair: dict[str, list[EvalCell]] = {}
    for cell in eval_cells:
        by_pair.setdefault(cell.prompt_pair_id, []).append(cell)
    for pid in qwen_surviving_pairs:
        if not by_pair.get(pid):
            raise ScoringInputMissing(
                f"pair {pid!r} survived Qwen but has zero evaluation cells"
            )
    return by_pair


# ---------------------------------------------------------------------------
# S06.05 — failed-cell skip / penalty
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailedCellPolicy:
    skip: bool = True  # default: skip failed cells from aggregation
    penalty_score: float = 0.0  # used when skip=False


def apply_failed_cell_policy(
    eval_results: Sequence[Mapping[str, Any]],
    policy: FailedCellPolicy,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in eval_results:
        if r.get("error") or r.get("status") == "error":
            if policy.skip:
                continue
            r2 = dict(r)
            r2["score"] = policy.penalty_score
            out.append(r2)
        else:
            out.append(dict(r))
    return out


# ---------------------------------------------------------------------------
# S06.09 — failed_pairs[] gating
# ---------------------------------------------------------------------------


@dataclass
class ScoringPartition:
    rankable_pairs: list[str] = field(default_factory=list)
    failed_pairs: list[dict[str, str]] = field(default_factory=list)


def partition_pairs_by_scoring_input(
    *,
    qwen_surviving_pairs: Iterable[str],
    eval_cells_by_pair: Mapping[str, Sequence[EvalCell]],
) -> ScoringPartition:
    """Move pairs with no eval cells into failed_pairs[] (S06.09)."""
    out = ScoringPartition()
    for pid in qwen_surviving_pairs:
        cells = eval_cells_by_pair.get(pid, ())
        if cells:
            out.rankable_pairs.append(pid)
        else:
            out.failed_pairs.append(
                {
                    "prompt_pair_id": pid,
                    "failure_reason": cp.SCORING_FAILED_REASON_MISSING_INPUT,
                }
            )
    return out


def assert_some_pairs_rankable(partition: ScoringPartition) -> None:
    """Abort before ranking/memory if no pair is rankable (S06.09)."""
    if not partition.rankable_pairs:
        raise NoRankablePairsError(
            f"no rankable pairs remain; failed_pairs={partition.failed_pairs}"
        )


# ---------------------------------------------------------------------------
# S05B.28 — V0.2.2-native ranking rows (run_id-keyed, replaces legacy
# scoring.ranking.pair_row_for_csv which has no run_id field)
# ---------------------------------------------------------------------------


_SUB_SCORE_KEYS = (
    "qwen_pass_rate",
    "edit_correctness",
    "garment_transfer_correctness",
    "preservation",
    "artifact_penalty",
)


_DEFAULT_SCORE_WEIGHTS: dict[str, float] = {
    "qwen_pass_rate": 0.40,
    "edit_correctness": 0.20,
    "garment_transfer_correctness": 0.15,
    "preservation": 0.15,
    "artifact_penalty": -0.10,
}


_CATEGORIES = ("dress", "lower", "upper")
_WORST_CELLS_PER_PAIR = 10


def _category_of(sample_id: str) -> str | None:
    if "__" not in sample_id:
        return None
    head = sample_id.split("__", 1)[0]
    return head if head in _CATEGORIES else None


def _weighted_total(means: Mapping[str, float | None], weights: Mapping[str, float]) -> float | None:
    pieces: list[float] = []
    for k, w in weights.items():
        v = means.get(k)
        if v is None:
            continue
        pieces.append(float(v) * float(w))
    return sum(pieces) if pieces else None


@dataclass(frozen=True)
class PairRankingRow:
    """V0.2.2 ranking row (run_id-keyed; suitable for memory_v022).

    The legacy ``scoring.ranking.pair_row_for_csv`` is forbidden in
    the V0.2.2 production path because it has no ``run_id`` field and
    emits the legacy long-memory schema (see plan §8.6.1, S05B.28).
    """

    run_id: str
    round_id: int
    prompt_pair_id: str
    pair_overall: float
    n_cells: int
    n_user_prompts: int
    per_user_prompt_overall: dict[str, float] = field(default_factory=dict)
    mean_qwen_pass_rate: float | None = None
    mean_edit_correctness: float | None = None
    mean_garment_transfer_correctness: float | None = None
    mean_preservation: float | None = None
    mean_artifact_penalty: float | None = None
    total_score: float | None = None
    per_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    worst_cells: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "round_id": self.round_id,
            "prompt_pair_id": self.prompt_pair_id,
            "pair_overall": self.pair_overall,
            "n_cells": self.n_cells,
            "n_user_prompts": self.n_user_prompts,
            "per_user_prompt_overall": dict(self.per_user_prompt_overall),
            "mean_qwen_pass_rate": self.mean_qwen_pass_rate,
            "mean_edit_correctness": self.mean_edit_correctness,
            "mean_garment_transfer_correctness": self.mean_garment_transfer_correctness,
            "mean_preservation": self.mean_preservation,
            "mean_artifact_penalty": self.mean_artifact_penalty,
            "total_score": self.total_score,
            "per_category": {
                cat: dict(block) for cat, block in self.per_category.items()
            },
            "worst_cells": [dict(c) for c in self.worst_cells],
        }


def _cell_score(cell_result: Mapping[str, Any]) -> float | None:
    """Extract a single overall score from one cell-eval result.

    Tolerates V0.2.1 sub-score dicts (``edit_correctness`` etc.) and
    V0.2.2 pre-aggregated ``score`` keys. Returns ``None`` when the
    cell has no usable score (e.g. error rows the policy chose to keep
    with ``score=None``).
    """
    if "score" in cell_result and cell_result["score"] is not None:
        try:
            return float(cell_result["score"])
        except (TypeError, ValueError):
            return None
    sub_keys = (
        "qwen_pass_rate", "edit_correctness",
        "garment_transfer_correctness", "preservation",
    )
    vals = [cell_result.get(k) for k in sub_keys if cell_result.get(k) is not None]
    if not vals:
        return None
    try:
        return float(sum(float(v) for v in vals) / len(vals))
    except (TypeError, ValueError):
        return None


_REQUIRED_EVAL_AXES = (
    "edit_correctness",
    "garment_transfer_correctness",
    "preservation",
    "artifact_penalty",
)


def _dedupe_complete(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Defensive dedupe + completeness filter for per-cell eval rows.

    Group by ``(prompt_pair_id, user_prompt_id, sample_id)`` and keep
    the latest row that has all four required axes parseable as floats.
    Drops malformed/incomplete rows so phantom-zero ranking can never
    sneak in via this path.

    Latest-wins is by list order (caller is responsible for passing
    rows in append order; this matches how ``_load_prior_records``
    already orders rows by mtime + line index).
    """
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        if any(a not in r for a in _REQUIRED_EVAL_AXES):
            continue
        try:
            for a in _REQUIRED_EVAL_AXES:
                float(r[a])
        except (TypeError, ValueError):
            continue
        k = (
            str(r.get("prompt_pair_id") or ""),
            str(r.get("user_prompt_id") or ""),
            str(r.get("sample_id") or ""),
        )
        by_key[k] = r
    return list(by_key.values())


def build_pair_ranking_rows(
    *,
    run_id: str,
    round_id: int,
    eval_cells_by_pair: Mapping[str, Sequence[EvalCell]],
    cell_results_by_pair: Mapping[str, Sequence[Mapping[str, Any]]],
    score_weights: Mapping[str, float] | None = None,
) -> list[PairRankingRow]:
    """Aggregate per-cell eval results into V0.2.2 ranking rows.

    For each pair we emit:
      * ``pair_overall`` — language-agnostic mean over all usable cell
        scores (a cell whose ``_cell_score`` is ``None`` is dropped).
      * ``per_user_prompt_overall`` — per-user-prompt mean over the
        same usable cells.
      * ``mean_*`` — direct arithmetic mean of each of the five Qwen
        sub-scores across cells where the field is non-null.
      * ``total_score`` — weighted combination of the five means using
        ``score_weights`` (defaults to ``EvaluationWeights``).
      * ``per_category`` — same five sub-score means bucketed by
        ``sample_id.split("__", 1)[0]`` (``dress``/``lower``/``upper``)
        plus a ``weighted_score`` and ``n_cells`` per category. Cells
        whose sample_id does not match the convention are excluded
        from the per-category aggregates (but not from ``pair_overall``).
        Categories with zero matching cells are emitted with
        ``missing_score_reason="no_cells_in_category"``.
      * ``worst_cells`` — bottom ``_WORST_CELLS_PER_PAIR`` cells by
        ``_cell_score`` ascending, each carrying its sub-score
        breakdown and the resolved category. Emitted unconditionally
        (no hard threshold) so downstream consumers always have a
        diagnostic surface even when overall pass-rate is healthy.

    Pairs with **zero** usable cells are emitted with
    ``pair_overall=0.0`` and ``n_cells=0``; the runner is expected to
    have already moved them into ``failed_pairs[]`` via
    :func:`partition_pairs_by_scoring_input`. We do not silently
    drop them here so the ranking surface stays observable.
    """
    weights = dict(score_weights) if score_weights else dict(_DEFAULT_SCORE_WEIGHTS)
    rows: list[PairRankingRow] = []
    for pid, cells in eval_cells_by_pair.items():
        # Defensive dedupe: drop malformed/incomplete rows and collapse
        # any duplicate (pair, up, sample) keys to the latest complete
        # record. evaluate_many_cells normally guarantees this already,
        # but a future caller may bypass it.
        results = _dedupe_complete(list(cell_results_by_pair.get(pid, ())))
        per_up_buckets: dict[str, list[float]] = {}
        all_scores: list[float] = []
        sub_score_lists: dict[str, list[float]] = {k: [] for k in _SUB_SCORE_KEYS}
        per_cat_sub: dict[str, dict[str, list[float]]] = {
            cat: {k: [] for k in _SUB_SCORE_KEYS} for cat in _CATEGORIES
        }
        per_cat_n: dict[str, int] = {cat: 0 for cat in _CATEGORIES}
        cell_overalls: list[tuple[float, dict[str, Any]]] = []

        result_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
        for r in results:
            upid = r.get("user_prompt_id", "")
            sid = r.get("sample_id", "")
            if upid and sid:
                result_by_key[(upid, sid)] = r
        for cell in cells:
            r = result_by_key.get((cell.user_prompt_id, cell.sample_id))
            if r is None:
                continue
            score = _cell_score(r)
            if score is None:
                continue
            all_scores.append(score)
            per_up_buckets.setdefault(cell.user_prompt_id, []).append(score)

            # Direct sub-score collection (skip None per field).
            cat = _category_of(cell.sample_id)
            cell_subs: dict[str, float | None] = {}
            for k in _SUB_SCORE_KEYS:
                v = r.get(k)
                if v is None:
                    cell_subs[k] = None
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    cell_subs[k] = None
                    continue
                cell_subs[k] = fv
                sub_score_lists[k].append(fv)
                if cat is not None:
                    per_cat_sub[cat][k].append(fv)
            if cat is not None:
                per_cat_n[cat] += 1

            cell_overalls.append((score, {
                "user_prompt_id": cell.user_prompt_id,
                "sample_id": cell.sample_id,
                "category": cat,
                "overall": score,
                **{k: cell_subs[k] for k in _SUB_SCORE_KEYS},
            }))

        pair_overall = (sum(all_scores) / len(all_scores)) if all_scores else 0.0
        per_up_means = {
            upid: (sum(vs) / len(vs)) for upid, vs in per_up_buckets.items()
        }
        means: dict[str, float | None] = {
            k: (sum(vs) / len(vs)) if vs else None
            for k, vs in sub_score_lists.items()
        }
        total_score = _weighted_total(means, weights) if all_scores else None

        per_category: dict[str, dict[str, Any]] = {}
        for cat in _CATEGORIES:
            n = per_cat_n[cat]
            if n == 0:
                per_category[cat] = {
                    "n_cells": 0,
                    "missing_score_reason": "no_cells_in_category",
                }
                continue
            cat_means: dict[str, float | None] = {
                k: (sum(vs) / len(vs)) if vs else None
                for k, vs in per_cat_sub[cat].items()
            }
            cat_weighted = _weighted_total(cat_means, weights)
            block: dict[str, Any] = {
                "n_cells": n,
                "weighted_score": cat_weighted,
                "total_score": cat_weighted,
                "missing_score_reason": None,
            }
            for k in _SUB_SCORE_KEYS:
                block[k] = cat_means[k]
            per_category[cat] = block

        cell_overalls.sort(key=lambda x: x[0])
        worst_cells = [c for _, c in cell_overalls[:_WORST_CELLS_PER_PAIR]]

        rows.append(
            PairRankingRow(
                run_id=run_id, round_id=round_id, prompt_pair_id=pid,
                pair_overall=pair_overall,
                n_cells=len(all_scores),
                n_user_prompts=len(per_up_means),
                per_user_prompt_overall=per_up_means,
                mean_qwen_pass_rate=means["qwen_pass_rate"],
                mean_edit_correctness=means["edit_correctness"],
                mean_garment_transfer_correctness=means["garment_transfer_correctness"],
                mean_preservation=means["preservation"],
                mean_artifact_penalty=means["artifact_penalty"],
                total_score=total_score,
                per_category=per_category,
                worst_cells=worst_cells,
            )
        )
    rows.sort(key=lambda r: (-r.pair_overall, r.prompt_pair_id))
    return rows


def build_next_round_context_v022(
    ranking_rows: Sequence[PairRankingRow], *, top_n: int = 3
) -> list[dict[str, Any]]:
    """Return V0.2.2-native next-round context entries.

    Pure dicts (no PromptPair coupling) so the runner can hand them
    straight to the schedule without touching legacy scoring helpers.
    """
    return [r.as_dict() for r in ranking_rows[:top_n]]
