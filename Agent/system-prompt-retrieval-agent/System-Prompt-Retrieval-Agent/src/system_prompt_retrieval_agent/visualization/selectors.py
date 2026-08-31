"""Score selectors for comparison grid composition.

Provides utilities to join per-sample scores across two rounds and identify
top-K improvements and top-K decreases by score delta.

V0.2.1 additions:
- ``CellScore`` type keyed by ``(pair_id, user_prompt_id, sample_id)``.
- ``group_by_pair_and_sample``: groups cell scores so each grid row covers one
  ``(pair_id, sample_id)`` with columns per ``user_prompt_id``.
"""

from __future__ import annotations

from typing import TypedDict


class SampleScore(TypedDict):
    """Per-sample score record for a single round (V0.2 / backward-compat)."""

    sample_id: str
    overall_score: float
    category: str | None


class CellScore(TypedDict):
    """Per-cell score record keyed by (pair_id, user_prompt_id, sample_id).

    Used in V0.2.1 multi-phrasing runs where each (pair, sample) is exercised
    under multiple user_prompt_ids (e.g. zh_001, en_001).
    """

    pair_id: str
    user_prompt_id: str
    sample_id: str
    overall_score: float
    category: str | None
    failure_reason: str | None  # non-None when the cell failed (no image produced)


def pair_scores_by_sample(
    prev: list[SampleScore],
    curr: list[SampleScore],
) -> list[dict]:
    """Inner-join prev and curr score lists by sample_id.

    Returns a list of dicts with keys:
        sample_id, prev, curr, delta (curr - prev), category.

    Samples missing in either side are excluded (missing-sample handling).
    When both sides have a category, the curr category wins; if only prev has
    one it is used.
    """
    prev_map: dict[str, SampleScore] = {s["sample_id"]: s for s in prev}
    curr_map: dict[str, SampleScore] = {s["sample_id"]: s for s in curr}

    result: list[dict] = []
    for sample_id, curr_score in curr_map.items():
        if sample_id not in prev_map:
            continue
        prev_score = prev_map[sample_id]
        category = curr_score.get("category") or prev_score.get("category")
        delta = curr_score["overall_score"] - prev_score["overall_score"]
        result.append(
            {
                "sample_id": sample_id,
                "prev": prev_score["overall_score"],
                "curr": curr_score["overall_score"],
                "delta": delta,
                "category": category,
            }
        )
    return result


def select_max_improvements(deltas: list[dict], k: int = 6) -> list[dict]:
    """Return top-k samples by delta descending.

    Stable tie-break: sample_id ascending.
    """
    sorted_deltas = sorted(deltas, key=lambda d: (-d["delta"], d["sample_id"]))
    return sorted_deltas[:k]


def select_max_decreases(deltas: list[dict], k: int = 6) -> list[dict]:
    """Return top-k samples by delta ascending (largest decreases first).

    Stable tie-break: sample_id ascending.
    """
    sorted_deltas = sorted(deltas, key=lambda d: (d["delta"], d["sample_id"]))
    return sorted_deltas[:k]


def group_by_pair_and_sample(
    cells: list[CellScore],
) -> dict[tuple[str, str], dict[str, CellScore]]:
    """Group cell scores by (pair_id, sample_id), sub-keyed by user_prompt_id.

    Returns::

        {
            (pair_id, sample_id): {
                user_prompt_id: CellScore,
                ...
            },
            ...
        }

    This layout drives the V0.2.1 grid where each row is one ``(pair, sample)``
    and each column corresponds to one ``user_prompt_id`` (e.g. zh_001, en_001).
    Cells that failed (``failure_reason`` is not None) are included so the grid
    can render a placeholder tile instead of being skipped.
    """
    grouped: dict[tuple[str, str], dict[str, CellScore]] = {}
    for cell in cells:
        key = (cell["pair_id"], cell["sample_id"])
        if key not in grouped:
            grouped[key] = {}
        grouped[key][cell["user_prompt_id"]] = cell
    return grouped
