"""Top-level helpers for composing per-round comparison grids.

V0.2 entry point
-----------------
``compose_round_grids`` handles the round-1 no-op (no previous scores).

V0.2.1 entry point
-------------------
``compose_round_grids_v021`` discovers FLUX artifacts at::

    flux/round_{round_id}/{pair_id}/{user_prompt_id}/{sample_id}/result.png

and emits grids at::

    outputs/v02/runs/{run_id}/grids/round_{round_id}/{pair_id}.png

with a ``grid_meta.yaml`` sidecar containing
``(pair_id, round_id, user_prompt_ids, sample_ids, corpus_hash)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .selectors import (
    SampleScore,
    CellScore,
    pair_scores_by_sample,
    select_max_improvements,
    select_max_decreases,
    group_by_pair_and_sample,
)
from .comparison_grid import (
    compose_comparison_grid,
    write_grid_metadata,
    compose_user_prompt_grid,
    write_grid_meta_yaml,
)


# ---------------------------------------------------------------------------
# V0.2 (backward-compatible)
# ---------------------------------------------------------------------------

def compose_round_grids(
    prev_scores: Optional[list[SampleScore]],
    curr_scores: list[SampleScore],
    image_map: dict,
    visualizations_dir: Path,
    k: int = 6,
) -> Optional[dict]:
    """Compose improvement and decrease grids for a round.

    Parameters
    ----------
    prev_scores:
        Per-sample scores from round K-1. Pass ``None`` or an empty list for
        the first round (no-op: returns ``None`` and writes no files).
    curr_scores:
        Per-sample scores from round K.
    image_map:
        ``{sample_id: {"model": Path, "cloth": Path, "prev_result": Path,
        "curr_result": Path}}``.
    visualizations_dir:
        Directory under which grids and metadata are written.
    k:
        Number of top/bottom samples to include in each grid.

    Returns
    -------
    dict or None
        ``{"improvement": Path, "decrease": Path, "metadata": Path}`` on
        success, or ``None`` when there are no previous scores to compare.
    """
    if not prev_scores:
        return None

    deltas = pair_scores_by_sample(prev_scores, curr_scores)

    # Sub-select samples
    improvements = select_max_improvements(deltas, k=k)
    decreases = select_max_decreases(deltas, k=k)

    visualizations_dir = Path(visualizations_dir)
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    improvement_path = visualizations_dir / "max_improvement_grid.png"
    decrease_path = visualizations_dir / "max_decrease_grid.png"
    metadata_path = visualizations_dir / "grid_metadata.json"

    compose_comparison_grid(improvements, image_map, improvement_path)
    compose_comparison_grid(decreases, image_map, decrease_path)

    # Combine both sets for the metadata
    all_deltas = {d["sample_id"]: d for d in improvements + decreases}
    write_grid_metadata(metadata_path, list(all_deltas.values()), improvement_path)

    return {
        "improvement": improvement_path,
        "decrease": decrease_path,
        "metadata": metadata_path,
    }


# ---------------------------------------------------------------------------
# V0.2.1
# ---------------------------------------------------------------------------

def discover_flux_artifacts(
    flux_root: Path,
    round_id: int,
    pair_id: str,
    user_prompt_ids: list[str],
    sample_ids: list[str],
) -> dict[tuple[str, str, str], dict]:
    """Discover FLUX artifact paths for a grid.

    Looks for::

        {flux_root}/round_{round_id}/{pair_id}/{user_prompt_id}/{sample_id}/result.png

    Returns a ``cell_map`` suitable for ``compose_user_prompt_grid``::

        {
            (pair_id, user_prompt_id, sample_id): {
                "image_path": Path | None,
                "failure_reason": str | None,
                "language": None,
            }
        }

    A cell whose ``result.png`` is absent is recorded with
    ``failure_reason="artifact_missing"``.
    """
    cell_map: dict[tuple[str, str, str], dict] = {}
    round_dir = flux_root / f"round_{round_id}" / pair_id

    for up_id in user_prompt_ids:
        for sid in sample_ids:
            img_path = round_dir / up_id / sid / "result.png"
            if img_path.exists():
                cell_map[(pair_id, up_id, sid)] = {
                    "image_path": img_path,
                    "failure_reason": None,
                    "language": None,
                }
            else:
                cell_map[(pair_id, up_id, sid)] = {
                    "image_path": None,
                    "failure_reason": "artifact_missing",
                    "language": None,
                }
    return cell_map


def compose_round_grids_v021(
    pair_id: str,
    round_id: int,
    run_id: str,
    user_prompt_ids: list[str],
    sample_ids: list[str],
    corpus_hash: str,
    flux_root: Path,
    output_root: Path,
    *,
    cell_map_override: Optional[dict[tuple[str, str, str], dict]] = None,
    portrait_h: int = 1376,
    landscape_w: int = 768,
) -> dict:
    """Compose a V0.2.1 per-pair comparison grid for a round.

    Artifact discovery
    ------------------
    FLUX result images are discovered at::

        {flux_root}/round_{round_id}/{pair_id}/{user_prompt_id}/{sample_id}/result.png

    Pass ``cell_map_override`` to inject a pre-built cell map (useful in tests).

    Output paths
    ------------
    Grid PNG::

        {output_root}/outputs/v02/runs/{run_id}/grids/round_{round_id}/{pair_id}.png

    Grid meta YAML::

        {output_root}/outputs/v02/runs/{run_id}/grids/round_{round_id}/{pair_id}_meta.yaml

    Parameters
    ----------
    pair_id:
        Prompt-pair ID.
    round_id:
        Round number (1-based).
    run_id:
        Run identifier string.
    user_prompt_ids:
        Ordered list of user_prompt_ids for the columns
        (e.g. ``["zh_001", "zh_003", "en_001", "en_003"]``).
    sample_ids:
        Ordered list of sample_ids for the rows.
    corpus_hash:
        SHA-256 hex digest of the enabled user-prompt corpus.
    flux_root:
        Root directory containing FLUX outputs
        (i.e. ``{flux_root}/round_{round_id}/...`` exists).
    output_root:
        Root under which the grid output tree is built.
    cell_map_override:
        If provided, used instead of auto-discovery from ``flux_root``.
    portrait_h:
        Target height for portrait images.
    landscape_w:
        Target width for landscape/square images.

    Returns
    -------
    dict
        ``{"grid": Path, "meta": Path}``
    """
    if cell_map_override is not None:
        cell_map = cell_map_override
    else:
        cell_map = discover_flux_artifacts(
            flux_root, round_id, pair_id, user_prompt_ids, sample_ids
        )

    rows = [(pair_id, sid) for sid in sample_ids]

    grids_dir = (
        Path(output_root)
        / "outputs"
        / "v02"
        / "runs"
        / run_id
        / "grids"
        / f"round_{round_id}"
    )
    grid_path = grids_dir / f"{pair_id}.png"
    meta_path = grids_dir / f"{pair_id}_meta.yaml"

    compose_user_prompt_grid(
        rows=rows,
        cell_map=cell_map,
        user_prompt_ids=user_prompt_ids,
        output_path=grid_path,
        portrait_h=portrait_h,
        landscape_w=landscape_w,
    )

    write_grid_meta_yaml(
        meta_path,
        pair_id=pair_id,
        round_id=round_id,
        user_prompt_ids=user_prompt_ids,
        sample_ids=sample_ids,
        corpus_hash=corpus_hash,
    )

    return {"grid": grid_path, "meta": meta_path}
