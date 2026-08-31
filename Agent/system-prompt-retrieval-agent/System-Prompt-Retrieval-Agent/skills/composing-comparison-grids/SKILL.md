# Skill: composing-comparison-grids

## Purpose

Compose per-round visual comparison grids that highlight which samples improved
or degraded the most between consecutive rounds of the retrieval agent loop.

V0.2.1 adds multi-phrasing grids: each column corresponds to one
``user_prompt_id`` (e.g. ``zh_001``, ``en_001``), and each row covers one
``(pair_id, sample_id)``.  Failed cells render a placeholder tile so the grid
is always dense.

---

## V0.2 API (backward-compatible)

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `prev_scores` | `list[SampleScore] \| None` | Per-sample scores from round K-1. `None` or empty triggers the round-1 no-op. |
| `curr_scores` | `list[SampleScore]` | Per-sample scores from round K. |
| `image_map` | `dict[str, dict[str, Path]]` | Maps `sample_id` to a dict with keys `model`, `cloth`, `prev_result`, `curr_result` (each a `Path` to a PNG). |
| `visualizations_dir` | `Path` | Directory where output files are written. |
| `k` | `int` (default 6) | Number of top/bottom samples per grid. |

`SampleScore` fields: `sample_id: str`, `overall_score: float`, `category: str | None`.

### Outputs

When `prev_scores` is `None` or empty (round 1), returns `None` and writes **no files**.

Otherwise, writes three files under `visualizations_dir`:

| File | Description |
|------|-------------|
| `max_improvement_grid.png` | Grid of top-K samples with highest score delta (`curr - prev`). |
| `max_decrease_grid.png` | Grid of top-K samples with lowest (most negative) score delta. |
| `grid_metadata.json` | JSON summary of all samples included in either grid. |

Returns `{"improvement": Path, "decrease": Path, "metadata": Path}`.

### Grid layout (V0.2)

Each row = one sample. Columns (left to right):

| Column | Content |
|--------|---------|
| `model` | Model reference image |
| `cloth` | Clothing reference image |
| `prev_result` | Generation result from round K-1 |
| `curr_result` | Generation result from round K |
| `delta_label` | Colour-coded text: green = improvement, red = decrease |

Image sizing:
- Portrait (h > w): resize to `portrait_h=1376` px height, aspect preserved.
- Landscape/square (w >= h): resize to `landscape_w=768` px width, aspect preserved.

---

## V0.2.1 API (multi-phrasing)

### Entry point

```python
from system_prompt_retrieval_agent.visualization import compose_round_grids_v021

result = compose_round_grids_v021(
    pair_id="pair_r03_001",
    round_id=3,
    run_id="run_20241201_abc",
    user_prompt_ids=["zh_001", "zh_003", "en_001", "en_003"],
    sample_ids=["s01", "s02", "s03"],
    corpus_hash="<sha256-from-S00.10>",
    flux_root=Path("outputs/v02/runs/run_xyz/flux"),
    output_root=Path("outputs/v02/runs/run_xyz"),
)
# result["grid"] → outputs/v02/runs/run_xyz/grids/round_3/pair_r03_001.png
# result["meta"] → outputs/v02/runs/run_xyz/grids/round_3/pair_r03_001_meta.yaml
```

### Artifact discovery

FLUX results are discovered at::

    {flux_root}/round_{round_id}/{pair_id}/{user_prompt_id}/{sample_id}/result.png

Cells whose ``result.png`` is absent are recorded with
``failure_reason="artifact_missing"`` and rendered as placeholder tiles.

### Output path naming

| File | Path |
|------|------|
| Grid PNG | `{output_root}/outputs/v02/runs/{run_id}/grids/round_{round_id}/{pair_id}.png` |
| Meta YAML | `{output_root}/outputs/v02/runs/{run_id}/grids/round_{round_id}/{pair_id}_meta.yaml` |

### Grid layout (V0.2.1)

- Header row: one tile per ``user_prompt_id``, labelled with the id.
- Data rows: one row per ``(pair_id, sample_id)``; one column per ``user_prompt_id``.
- Failed cells render a placeholder tile (muted red background, failure reason text).
- Absent cells (key missing from ``cell_map``) render a ``"missing"`` placeholder.

### ``grid_meta.yaml`` schema

```yaml
pair_id: pair_r03_001
round_id: 3
user_prompt_ids:
  - zh_001
  - zh_003
  - en_001
  - en_003
sample_ids:
  - s01
  - s02
corpus_hash: <sha256>
```

---

## Owner module

`src/system_prompt_retrieval_agent/visualization/`

| Module | Responsibility |
|--------|---------------|
| `selectors.py` | `pair_scores_by_sample`, `select_max_improvements`, `select_max_decreases` (V0.2); `CellScore`, `group_by_pair_and_sample` (V0.2.1) |
| `comparison_grid.py` | `compose_comparison_grid`, `write_grid_metadata` (V0.2); `compose_user_prompt_grid`, `write_grid_meta_yaml` (V0.2.1) |
| `round_grids.py` | `compose_round_grids` (V0.2); `compose_round_grids_v021`, `discover_flux_artifacts` (V0.2.1) |
| `__init__.py` | Re-exports all public symbols |

---

## Constraints

- Pure PIL + stdlib + PyYAML. No matplotlib, no network calls.
- `ImageDraw.text` with the default PIL font for delta labels and placeholder tiles.
- `tmp_path` for test fixtures; do not commit binary PNG files.
- Import `schemas.py` if needed; never redefine its types.
- Failed cells must be rendered as placeholder tiles — never silently skipped.
- ``grid_meta.yaml`` must be emitted adjacent to every V0.2.1 grid PNG.
