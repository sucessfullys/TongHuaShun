"""Visualization package for composing per-round comparison grids."""

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
from .round_grids import (
    compose_round_grids,
    compose_round_grids_v021,
    discover_flux_artifacts,
)

__all__ = [
    # V0.2 / backward-compat
    "SampleScore",
    "pair_scores_by_sample",
    "select_max_improvements",
    "select_max_decreases",
    "compose_comparison_grid",
    "write_grid_metadata",
    "compose_round_grids",
    # V0.2.1 additions
    "CellScore",
    "group_by_pair_and_sample",
    "compose_user_prompt_grid",
    "write_grid_meta_yaml",
    "compose_round_grids_v021",
    "discover_flux_artifacts",
]
