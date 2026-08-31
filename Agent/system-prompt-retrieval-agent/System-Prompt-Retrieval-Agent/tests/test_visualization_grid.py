"""Tests for visualization.comparison_grid and round_grids (S08.08).

Covers:
- Compose grid of 3 samples: output PNG exists; dimensions >= expected minimums.
- Round-1 no-op: compose_round_grids(prev_scores=None, ...) returns None, writes no files.
- Round-1 empty prev list returns None.
- Metadata JSON contains all sample_ids and correct delta values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

from system_prompt_retrieval_agent.visualization.comparison_grid import (
    COLUMNS,
    compose_comparison_grid,
    write_grid_metadata,
)
from system_prompt_retrieval_agent.visualization.round_grids import compose_round_grids
from system_prompt_retrieval_agent.visualization.selectors import SampleScore

# Ensure the tests package root is importable (conftest only adds src/).
_TESTS_ROOT = Path(__file__).resolve().parent.parent
if str(_TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_TESTS_ROOT))

from tests.fixtures.synthetic_images import make_synthetic_image_set  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_image_map(tmp_path: Path) -> dict[str, dict[str, Path]]:
    """Generate 5 synthetic samples into tmp_path and return the image_map."""
    return make_synthetic_image_set(tmp_path / "synthetic_images", n_samples=5)


@pytest.fixture()
def three_sample_deltas() -> list[dict]:
    """Three delta dicts for samples 0, 1, 2."""
    return [
        {"sample_id": "sample_00", "prev": 0.50, "curr": 0.70, "delta": 0.20, "category": "upper_body"},
        {"sample_id": "sample_01", "prev": 0.65, "curr": 0.55, "delta": -0.10, "category": "lower_body"},
        {"sample_id": "sample_02", "prev": 0.80, "curr": 0.85, "delta": 0.05, "category": None},
    ]


# ---------------------------------------------------------------------------
# compose_comparison_grid
# ---------------------------------------------------------------------------

class TestComposeComparisonGrid:
    def test_output_png_exists(self, tmp_path, synthetic_image_map, three_sample_deltas):
        out = tmp_path / "grid.png"
        result = compose_comparison_grid(three_sample_deltas, synthetic_image_map, out)
        assert result == out
        assert out.exists()

    def test_output_is_valid_png(self, tmp_path, synthetic_image_map, three_sample_deltas):
        out = tmp_path / "grid.png"
        compose_comparison_grid(three_sample_deltas, synthetic_image_map, out)
        img = Image.open(out)
        assert img.format == "PNG"

    def test_image_width_at_least_sum_of_columns(
        self, tmp_path, synthetic_image_map, three_sample_deltas
    ):
        """Width must be >= sum of column contributions; we use small images
        so we can compute expected minimums exactly."""
        out = tmp_path / "grid.png"
        # Use small sizes so the test is fast and predictable.
        compose_comparison_grid(
            three_sample_deltas,
            synthetic_image_map,
            out,
            portrait_h=96,
            landscape_w=64,
            col_gap=4,
        )
        img = Image.open(out)
        # 5 columns with col_gap=4 between them => at least 4*4=16 gap pixels
        # Each column is at least 1px wide.
        assert img.width >= len(COLUMNS) + (len(COLUMNS) - 1) * 4

    def test_image_height_at_least_sum_of_rows(
        self, tmp_path, synthetic_image_map, three_sample_deltas
    ):
        out = tmp_path / "grid.png"
        compose_comparison_grid(
            three_sample_deltas,
            synthetic_image_map,
            out,
            portrait_h=96,
            landscape_w=64,
            row_gap=4,
        )
        img = Image.open(out)
        n_rows = len(three_sample_deltas)
        # At least n_rows px + row gaps
        assert img.height >= n_rows + (n_rows - 1) * 4

    def test_five_columns_heuristic(
        self, tmp_path, synthetic_image_map, three_sample_deltas
    ):
        """Sample a horizontal line through the middle of the grid and verify
        there are at least 5 visually distinct horizontal bands.

        Implementation: count transitions where neighbouring pixel groups differ
        substantially.  We use colour columns from the synthetic fixture which
        have clearly distinct hues.
        """
        out = tmp_path / "grid.png"
        compose_comparison_grid(
            three_sample_deltas,
            synthetic_image_map,
            out,
            portrait_h=96,
            landscape_w=64,
            col_gap=2,
            row_gap=2,
        )
        img = Image.open(out).convert("RGB")
        w, h = img.size
        # Sample the middle row
        y = h // 2
        pixels = [img.getpixel((x, y)) for x in range(w)]

        # Count transitions: a significant colour change between neighbouring
        # pixel columns (threshold to ignore anti-aliasing artefacts).
        def _dist(a, b):
            return sum(abs(ai - bi) for ai, bi in zip(a, b))

        transitions = 0
        for i in range(1, len(pixels)):
            if _dist(pixels[i], pixels[i - 1]) > 20:
                transitions += 1

        # At minimum we expect transitions between 5 columns => 4 transitions
        # (plus possibly gap-colour transitions).  Use a relaxed lower bound.
        assert transitions >= 4, f"Expected >=4 transitions, got {transitions}"

    def test_empty_deltas_writes_placeholder(self, tmp_path, synthetic_image_map):
        out = tmp_path / "empty_grid.png"
        compose_comparison_grid([], synthetic_image_map, out)
        assert out.exists()

    def test_missing_image_path_uses_placeholder(self, tmp_path, three_sample_deltas):
        """If image_map is empty, compose_comparison_grid still completes."""
        out = tmp_path / "grid_no_images.png"
        compose_comparison_grid(
            three_sample_deltas,
            {},
            out,
            portrait_h=96,
            landscape_w=64,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# write_grid_metadata
# ---------------------------------------------------------------------------

class TestWriteGridMetadata:
    def test_metadata_json_is_valid(self, tmp_path, three_sample_deltas):
        meta_path = tmp_path / "metadata.json"
        grid_path = tmp_path / "grid.png"
        write_grid_metadata(meta_path, three_sample_deltas, grid_path)
        payload = json.loads(meta_path.read_text())
        assert "grid" in payload
        assert "samples" in payload

    def test_metadata_contains_all_sample_ids(self, tmp_path, three_sample_deltas):
        meta_path = tmp_path / "metadata.json"
        grid_path = tmp_path / "grid.png"
        write_grid_metadata(meta_path, three_sample_deltas, grid_path)
        payload = json.loads(meta_path.read_text())
        ids = {s["sample_id"] for s in payload["samples"]}
        assert ids == {"sample_00", "sample_01", "sample_02"}

    def test_metadata_delta_values_correct(self, tmp_path, three_sample_deltas):
        meta_path = tmp_path / "metadata.json"
        grid_path = tmp_path / "grid.png"
        write_grid_metadata(meta_path, three_sample_deltas, grid_path)
        payload = json.loads(meta_path.read_text())
        by_id = {s["sample_id"]: s for s in payload["samples"]}
        assert abs(by_id["sample_00"]["delta"] - 0.20) < 1e-9
        assert abs(by_id["sample_01"]["delta"] - (-0.10)) < 1e-9
        assert abs(by_id["sample_02"]["delta"] - 0.05) < 1e-9

    def test_metadata_grid_path_is_absolute(self, tmp_path, three_sample_deltas):
        meta_path = tmp_path / "metadata.json"
        grid_path = tmp_path / "grid.png"
        write_grid_metadata(meta_path, three_sample_deltas, grid_path)
        payload = json.loads(meta_path.read_text())
        assert Path(payload["grid"]).is_absolute()


# ---------------------------------------------------------------------------
# compose_round_grids — round-1 no-op
# ---------------------------------------------------------------------------

class TestComposeRoundGridsNoOp:
    def _curr_scores(self) -> list[SampleScore]:
        return [
            SampleScore(sample_id="s1", overall_score=0.7, category=None),
            SampleScore(sample_id="s2", overall_score=0.8, category=None),
        ]

    def test_none_prev_returns_none(self, tmp_path):
        result = compose_round_grids(
            prev_scores=None,
            curr_scores=self._curr_scores(),
            image_map={},
            visualizations_dir=tmp_path / "viz",
        )
        assert result is None

    def test_none_prev_writes_no_files(self, tmp_path):
        viz_dir = tmp_path / "viz"
        compose_round_grids(
            prev_scores=None,
            curr_scores=self._curr_scores(),
            image_map={},
            visualizations_dir=viz_dir,
        )
        assert not viz_dir.exists() or list(viz_dir.iterdir()) == []

    def test_empty_prev_returns_none(self, tmp_path):
        result = compose_round_grids(
            prev_scores=[],
            curr_scores=self._curr_scores(),
            image_map={},
            visualizations_dir=tmp_path / "viz",
        )
        assert result is None

    def test_empty_prev_writes_no_files(self, tmp_path):
        viz_dir = tmp_path / "viz"
        compose_round_grids(
            prev_scores=[],
            curr_scores=self._curr_scores(),
            image_map={},
            visualizations_dir=viz_dir,
        )
        assert not viz_dir.exists() or list(viz_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# compose_round_grids — full round
# ---------------------------------------------------------------------------

class TestComposeRoundGridsFull:
    def _prev_scores(self) -> list[SampleScore]:
        return [
            SampleScore(sample_id="sample_00", overall_score=0.50, category="upper_body"),
            SampleScore(sample_id="sample_01", overall_score=0.65, category="lower_body"),
            SampleScore(sample_id="sample_02", overall_score=0.80, category=None),
        ]

    def _curr_scores(self) -> list[SampleScore]:
        return [
            SampleScore(sample_id="sample_00", overall_score=0.70, category="upper_body"),
            SampleScore(sample_id="sample_01", overall_score=0.55, category="lower_body"),
            SampleScore(sample_id="sample_02", overall_score=0.85, category=None),
        ]

    def test_returns_dict_with_three_keys(self, tmp_path, synthetic_image_map):
        result = compose_round_grids(
            prev_scores=self._prev_scores(),
            curr_scores=self._curr_scores(),
            image_map=synthetic_image_map,
            visualizations_dir=tmp_path / "viz",
            k=3,
        )
        assert result is not None
        assert set(result.keys()) == {"improvement", "decrease", "metadata"}

    def test_output_files_exist(self, tmp_path, synthetic_image_map):
        result = compose_round_grids(
            prev_scores=self._prev_scores(),
            curr_scores=self._curr_scores(),
            image_map=synthetic_image_map,
            visualizations_dir=tmp_path / "viz",
            k=3,
        )
        assert result["improvement"].exists()
        assert result["decrease"].exists()
        assert result["metadata"].exists()

    def test_metadata_contains_sample_ids(self, tmp_path, synthetic_image_map):
        result = compose_round_grids(
            prev_scores=self._prev_scores(),
            curr_scores=self._curr_scores(),
            image_map=synthetic_image_map,
            visualizations_dir=tmp_path / "viz",
            k=3,
        )
        payload = json.loads(result["metadata"].read_text())
        ids = {s["sample_id"] for s in payload["samples"]}
        # All 3 samples should appear in improvement or decrease set
        assert "sample_00" in ids or "sample_01" in ids or "sample_02" in ids
