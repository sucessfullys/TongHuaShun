"""Tests for V0.2.1 visualization: multi-user-prompt column grids (W3).

Covers:
- ``group_by_pair_and_sample``: correct grouping, including failed cells.
- ``compose_user_prompt_grid``: output PNG exists and has expected structure.
- Failed cell renders a placeholder tile (no image produced, grid is dense).
- ``write_grid_meta_yaml``: YAML sidecar schema validation.
- ``compose_round_grids_v021``: artifact discovery + output path naming.
- ``discover_flux_artifacts``: absent artifact recorded as ``artifact_missing``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

# Ensure the project root (parent of tests/) is importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from system_prompt_retrieval_agent.visualization.selectors import (
    CellScore,
    group_by_pair_and_sample,
)
from system_prompt_retrieval_agent.visualization.comparison_grid import (
    compose_user_prompt_grid,
    write_grid_meta_yaml,
)
from system_prompt_retrieval_agent.visualization.round_grids import (
    compose_round_grids_v021,
    discover_flux_artifacts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_cell(
    pair_id: str,
    user_prompt_id: str,
    sample_id: str,
    score: float = 0.7,
    failure_reason: str | None = None,
    category: str | None = None,
) -> CellScore:
    return CellScore(
        pair_id=pair_id,
        user_prompt_id=user_prompt_id,
        sample_id=sample_id,
        overall_score=score,
        category=category,
        failure_reason=failure_reason,
    )


def _make_result_png(path: Path, size: tuple[int, int] = (64, 96)) -> Path:
    """Write a small solid-colour PNG at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(100, 150, 200)).save(path)
    return path


# ---------------------------------------------------------------------------
# group_by_pair_and_sample
# ---------------------------------------------------------------------------

class TestGroupByPairAndSample:
    def _cells(self) -> list[CellScore]:
        return [
            _make_cell("pair_01", "zh_001", "s01", score=0.8),
            _make_cell("pair_01", "en_001", "s01", score=0.75),
            _make_cell("pair_01", "zh_001", "s02", score=0.6),
            _make_cell("pair_01", "en_001", "s02", score=0.65),
            _make_cell("pair_02", "zh_001", "s01", score=0.9),
        ]

    def test_groups_by_pair_and_sample(self):
        grouped = group_by_pair_and_sample(self._cells())
        assert ("pair_01", "s01") in grouped
        assert ("pair_01", "s02") in grouped
        assert ("pair_02", "s01") in grouped

    def test_sub_keyed_by_user_prompt_id(self):
        grouped = group_by_pair_and_sample(self._cells())
        sub = grouped[("pair_01", "s01")]
        assert "zh_001" in sub
        assert "en_001" in sub

    def test_cell_values_preserved(self):
        grouped = group_by_pair_and_sample(self._cells())
        cell = grouped[("pair_01", "s01")]["zh_001"]
        assert abs(cell["overall_score"] - 0.8) < 1e-9

    def test_failed_cell_included(self):
        """Failed cells must appear in the grouping (not be filtered out)."""
        cells = [
            _make_cell("pair_01", "zh_001", "s01", failure_reason="worker_crash"),
            _make_cell("pair_01", "en_001", "s01", score=0.7),
        ]
        grouped = group_by_pair_and_sample(cells)
        sub = grouped[("pair_01", "s01")]
        assert "zh_001" in sub
        assert sub["zh_001"]["failure_reason"] == "worker_crash"

    def test_empty_returns_empty(self):
        assert group_by_pair_and_sample([]) == {}

    def test_single_cell(self):
        grouped = group_by_pair_and_sample(
            [_make_cell("pair_01", "zh_001", "s01")]
        )
        assert len(grouped) == 1
        assert ("pair_01", "s01") in grouped


# ---------------------------------------------------------------------------
# compose_user_prompt_grid — happy path
# ---------------------------------------------------------------------------

class TestComposeUserPromptGrid:
    def _cell_map(self, tmp_path: Path) -> dict[tuple[str, str, str], dict]:
        user_prompts = ["zh_001", "en_001"]
        samples = ["s01", "s02"]
        cell_map: dict[tuple[str, str, str], dict] = {}
        for up in user_prompts:
            for sid in samples:
                img = tmp_path / f"{up}_{sid}.png"
                _make_result_png(img)
                cell_map[("pair_01", up, sid)] = {
                    "image_path": img,
                    "failure_reason": None,
                    "language": up.split("_")[0],
                }
        return cell_map

    def test_output_png_exists(self, tmp_path):
        cell_map = self._cell_map(tmp_path)
        out = tmp_path / "grid.png"
        result = compose_user_prompt_grid(
            rows=[("pair_01", "s01"), ("pair_01", "s02")],
            cell_map=cell_map,
            user_prompt_ids=["zh_001", "en_001"],
            output_path=out,
            portrait_h=96,
            landscape_w=64,
        )
        assert result == out
        assert out.exists()

    def test_output_is_valid_png(self, tmp_path):
        cell_map = self._cell_map(tmp_path)
        out = tmp_path / "grid.png"
        compose_user_prompt_grid(
            rows=[("pair_01", "s01"), ("pair_01", "s02")],
            cell_map=cell_map,
            user_prompt_ids=["zh_001", "en_001"],
            output_path=out,
            portrait_h=96,
            landscape_w=64,
        )
        img = Image.open(out)
        assert img.format == "PNG"

    def test_image_has_header_plus_rows(self, tmp_path):
        """Grid height should exceed the header height (>= header + 1 data row)."""
        from system_prompt_retrieval_agent.visualization.comparison_grid import _HEADER_HEIGHT
        cell_map = self._cell_map(tmp_path)
        out = tmp_path / "grid.png"
        compose_user_prompt_grid(
            rows=[("pair_01", "s01"), ("pair_01", "s02")],
            cell_map=cell_map,
            user_prompt_ids=["zh_001", "en_001"],
            output_path=out,
            portrait_h=96,
            landscape_w=64,
        )
        img = Image.open(out)
        assert img.height > _HEADER_HEIGHT

    def test_empty_rows_writes_placeholder(self, tmp_path):
        out = tmp_path / "empty.png"
        compose_user_prompt_grid(
            rows=[],
            cell_map={},
            user_prompt_ids=["zh_001"],
            output_path=out,
        )
        assert out.exists()

    def test_empty_user_prompt_ids_writes_placeholder(self, tmp_path):
        out = tmp_path / "empty2.png"
        compose_user_prompt_grid(
            rows=[("pair_01", "s01")],
            cell_map={},
            user_prompt_ids=[],
            output_path=out,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# compose_user_prompt_grid — failed cell placeholder
# ---------------------------------------------------------------------------

class TestUserPromptGridFailedCell:
    def test_failed_cell_does_not_skip_row(self, tmp_path):
        """Grid must be produced even when all cells fail."""
        cell_map = {
            ("pair_01", "zh_001", "s01"): {
                "image_path": None,
                "failure_reason": "worker_crash",
                "language": "zh",
            },
            ("pair_01", "en_001", "s01"): {
                "image_path": None,
                "failure_reason": "artifact_missing",
                "language": "en",
            },
        }
        out = tmp_path / "grid.png"
        compose_user_prompt_grid(
            rows=[("pair_01", "s01")],
            cell_map=cell_map,
            user_prompt_ids=["zh_001", "en_001"],
            output_path=out,
            portrait_h=96,
            landscape_w=64,
        )
        assert out.exists()
        img = Image.open(out)
        assert img.width >= 1 and img.height >= 1

    def test_mixed_cells_grid_produced(self, tmp_path):
        """A grid with one good cell and one failed cell is still produced."""
        good_img = tmp_path / "good.png"
        _make_result_png(good_img, size=(64, 96))
        cell_map = {
            ("pair_01", "zh_001", "s01"): {
                "image_path": good_img,
                "failure_reason": None,
                "language": "zh",
            },
            ("pair_01", "en_001", "s01"): {
                "image_path": None,
                "failure_reason": "incomplete_count",
                "language": "en",
            },
        }
        out = tmp_path / "mixed_grid.png"
        compose_user_prompt_grid(
            rows=[("pair_01", "s01")],
            cell_map=cell_map,
            user_prompt_ids=["zh_001", "en_001"],
            output_path=out,
            portrait_h=96,
            landscape_w=64,
        )
        assert out.exists()

    def test_absent_cell_treated_as_missing(self, tmp_path):
        """A (pair_id, user_prompt_id, sample_id) key absent from cell_map
        produces a 'missing' placeholder tile — grid is still dense."""
        cell_map: dict = {}  # intentionally empty
        out = tmp_path / "absent.png"
        compose_user_prompt_grid(
            rows=[("pair_01", "s01")],
            cell_map=cell_map,
            user_prompt_ids=["zh_001"],
            output_path=out,
            portrait_h=96,
            landscape_w=64,
        )
        assert out.exists()


# ---------------------------------------------------------------------------
# write_grid_meta_yaml
# ---------------------------------------------------------------------------

class TestWriteGridMetaYaml:
    def test_yaml_schema(self, tmp_path):
        meta = tmp_path / "grid_meta.yaml"
        write_grid_meta_yaml(
            meta,
            pair_id="pair_01",
            round_id=3,
            user_prompt_ids=["zh_001", "en_001"],
            sample_ids=["s01", "s02"],
            corpus_hash="abc123def456",
        )
        payload = yaml.safe_load(meta.read_text())
        assert payload["pair_id"] == "pair_01"
        assert payload["round_id"] == 3
        assert payload["user_prompt_ids"] == ["zh_001", "en_001"]
        assert payload["sample_ids"] == ["s01", "s02"]
        assert payload["corpus_hash"] == "abc123def456"

    def test_parent_dir_created(self, tmp_path):
        meta = tmp_path / "nested" / "deep" / "grid_meta.yaml"
        write_grid_meta_yaml(
            meta,
            pair_id="pair_x",
            round_id=1,
            user_prompt_ids=["zh_001"],
            sample_ids=["s01"],
            corpus_hash="deadbeef",
        )
        assert meta.exists()

    def test_round_id_is_int(self, tmp_path):
        meta = tmp_path / "grid_meta.yaml"
        write_grid_meta_yaml(
            meta,
            pair_id="pair_01",
            round_id=7,
            user_prompt_ids=["zh_001"],
            sample_ids=["s01"],
            corpus_hash="hash_7",
        )
        payload = yaml.safe_load(meta.read_text())
        assert isinstance(payload["round_id"], int)


# ---------------------------------------------------------------------------
# discover_flux_artifacts
# ---------------------------------------------------------------------------

class TestDiscoverFluxArtifacts:
    def test_present_artifact_discovered(self, tmp_path):
        flux_root = tmp_path / "flux"
        result_path = flux_root / "round_2" / "pair_01" / "zh_001" / "s01" / "result.png"
        _make_result_png(result_path)
        cell_map = discover_flux_artifacts(
            flux_root=flux_root,
            round_id=2,
            pair_id="pair_01",
            user_prompt_ids=["zh_001"],
            sample_ids=["s01"],
        )
        key = ("pair_01", "zh_001", "s01")
        assert key in cell_map
        assert cell_map[key]["failure_reason"] is None
        assert cell_map[key]["image_path"] == result_path

    def test_absent_artifact_marked_missing(self, tmp_path):
        flux_root = tmp_path / "flux"
        cell_map = discover_flux_artifacts(
            flux_root=flux_root,
            round_id=2,
            pair_id="pair_01",
            user_prompt_ids=["en_001"],
            sample_ids=["s01"],
        )
        key = ("pair_01", "en_001", "s01")
        assert key in cell_map
        assert cell_map[key]["failure_reason"] == "artifact_missing"
        assert cell_map[key]["image_path"] is None

    def test_partial_discovery(self, tmp_path):
        """One present, one missing."""
        flux_root = tmp_path / "flux"
        present = flux_root / "round_1" / "pair_01" / "zh_001" / "s01" / "result.png"
        _make_result_png(present)
        cell_map = discover_flux_artifacts(
            flux_root=flux_root,
            round_id=1,
            pair_id="pair_01",
            user_prompt_ids=["zh_001", "en_001"],
            sample_ids=["s01"],
        )
        assert cell_map[("pair_01", "zh_001", "s01")]["failure_reason"] is None
        assert cell_map[("pair_01", "en_001", "s01")]["failure_reason"] == "artifact_missing"

    def test_all_cells_in_result(self, tmp_path):
        """Number of discovered cells equals len(user_prompts) * len(samples)."""
        flux_root = tmp_path / "flux"
        cell_map = discover_flux_artifacts(
            flux_root=flux_root,
            round_id=1,
            pair_id="pair_01",
            user_prompt_ids=["zh_001", "zh_003", "en_001"],
            sample_ids=["s01", "s02"],
        )
        assert len(cell_map) == 6  # 3 user_prompts × 2 samples


# ---------------------------------------------------------------------------
# compose_round_grids_v021
# ---------------------------------------------------------------------------

class TestComposeRoundGridsV021:
    def _minimal_cell_map(
        self, pair_id: str, user_prompt_ids: list[str], sample_ids: list[str]
    ) -> dict[tuple[str, str, str], dict]:
        """All cells are failed (no real images needed)."""
        return {
            (pair_id, up, sid): {
                "image_path": None,
                "failure_reason": "artifact_missing",
                "language": None,
            }
            for up in user_prompt_ids
            for sid in sample_ids
        }

    def test_output_grid_exists(self, tmp_path):
        result = compose_round_grids_v021(
            pair_id="pair_01",
            round_id=2,
            run_id="run_abc",
            user_prompt_ids=["zh_001", "en_001"],
            sample_ids=["s01", "s02"],
            corpus_hash="hash123",
            flux_root=tmp_path / "flux",
            output_root=tmp_path,
            cell_map_override=self._minimal_cell_map(
                "pair_01", ["zh_001", "en_001"], ["s01", "s02"]
            ),
        )
        assert result["grid"].exists()

    def test_output_meta_yaml_exists(self, tmp_path):
        result = compose_round_grids_v021(
            pair_id="pair_01",
            round_id=2,
            run_id="run_abc",
            user_prompt_ids=["zh_001", "en_001"],
            sample_ids=["s01", "s02"],
            corpus_hash="hash123",
            flux_root=tmp_path / "flux",
            output_root=tmp_path,
            cell_map_override=self._minimal_cell_map(
                "pair_01", ["zh_001", "en_001"], ["s01", "s02"]
            ),
        )
        assert result["meta"].exists()

    def test_output_path_naming(self, tmp_path):
        """Grid must be at outputs/v02/runs/{run_id}/grids/round_{round_id}/{pair_id}.png."""
        result = compose_round_grids_v021(
            pair_id="pair_01",
            round_id=3,
            run_id="run_xyz",
            user_prompt_ids=["zh_001"],
            sample_ids=["s01"],
            corpus_hash="abc",
            flux_root=tmp_path / "flux",
            output_root=tmp_path,
            cell_map_override=self._minimal_cell_map("pair_01", ["zh_001"], ["s01"]),
        )
        expected = (
            tmp_path / "outputs" / "v02" / "runs" / "run_xyz"
            / "grids" / "round_3" / "pair_01.png"
        )
        assert result["grid"] == expected

    def test_meta_yaml_content(self, tmp_path):
        result = compose_round_grids_v021(
            pair_id="pair_01",
            round_id=2,
            run_id="run_abc",
            user_prompt_ids=["zh_001", "en_001"],
            sample_ids=["s01", "s02"],
            corpus_hash="hash_content_test",
            flux_root=tmp_path / "flux",
            output_root=tmp_path,
            cell_map_override=self._minimal_cell_map(
                "pair_01", ["zh_001", "en_001"], ["s01", "s02"]
            ),
        )
        payload = yaml.safe_load(result["meta"].read_text())
        assert payload["pair_id"] == "pair_01"
        assert payload["round_id"] == 2
        assert payload["corpus_hash"] == "hash_content_test"
        assert set(payload["user_prompt_ids"]) == {"zh_001", "en_001"}
        assert set(payload["sample_ids"]) == {"s01", "s02"}

    def test_real_artifact_discovery(self, tmp_path):
        """compose_round_grids_v021 auto-discovers artifacts when no override."""
        flux_root = tmp_path / "flux"
        result_path = flux_root / "round_1" / "pair_01" / "zh_001" / "s01" / "result.png"
        _make_result_png(result_path)

        result = compose_round_grids_v021(
            pair_id="pair_01",
            round_id=1,
            run_id="run_discover",
            user_prompt_ids=["zh_001"],
            sample_ids=["s01"],
            corpus_hash="disc_hash",
            flux_root=flux_root,
            output_root=tmp_path,
        )
        assert result["grid"].exists()
        assert result["meta"].exists()
