"""Tests for the recursive data-root probe."""

from __future__ import annotations

from pathlib import Path

from era.probe import probe_data_roots


def test_per_sample_dirs_colocated(per_sample_root: Path):
    """One root, sample dirs are direct children, inputs co-located."""
    result = probe_data_roots([str(per_sample_root)])
    assert result["layout"] == "per_sample_dirs"
    assert result["sample_count"] == 3
    assert result["sample_glob"] == "*"
    assert result["sample_key"] == "relpath"
    assert len(result["methods"]) == 1
    method = result["methods"][0]
    assert method["output_file"] == "tryon_result_flux2klein.png"
    assert set(result["input_roles"].values()) == {
        "input_cloth.png", "input_model.png",
    }
    assert result["confidence"] == "high"


def test_nested_multi_method(nested_methods_root: Path):
    """One root with 3 method dirs, sample dirs nested 3 levels deep."""
    result = probe_data_roots([str(nested_methods_root)])
    assert result["layout"] == "per_sample_dirs"
    assert result["sample_glob"] == "*/*/*"          # category/scene/sample
    assert result["sample_count"] == 4               # per method
    assert len(result["methods"]) == 3
    assert set(result["input_roles"].values()) == {
        "input_cloth.png", "input_model.png",
    }

    by_id = {m["method_id"]: m for m in result["methods"]}
    # methods 1 & 2 have a single, unambiguous output
    assert by_id["tryon_results"]["output_file"] == "tryon_result_flux2klein.png"
    # method 3 differs; intermediates dropped, final guessed, flagged
    v5 = by_id["tryon_results_pe_v5"]
    assert v5["output_file"] == "tryon_result_flux2klein_agent.png"
    assert len(v5["output_candidates"]) == 4         # final + 3 intermediates
    assert result["confidence"] == "needs_confirmation"
    assert result["scan"]["leaves_found"] == 12      # 3 methods x 4 samples


def test_flat_directory(flat_root: Path):
    result = probe_data_roots([str(flat_root)])
    assert result["layout"] == "flat_files"
    assert result["sample_count"] == 5
    assert result["confidence"] == "needs_confirmation"


def test_multi_method_roots(multi_method_roots: list[str]):
    """Two method dirs passed as separate roots."""
    result = probe_data_roots(multi_method_roots)
    assert result["layout"] == "per_sample_dirs"
    assert len(result["methods"]) == 2
    assert result["sample_count"] == 4
    assert set(result["input_roles"].values()) == {"source.png"}
    for method in result["methods"]:
        assert method["output_file"] == "result.png"


def test_missing_root(tmp_path: Path):
    result = probe_data_roots([str(tmp_path / "does_not_exist")])
    assert result["layout"] == "unknown"
    assert result["methods"] == []
    assert any("not found" in n for n in result["notes"])


def test_deeply_nested_single_method(tmp_path: Path):
    """A single root whose samples nest under one branch."""
    from conftest import touch
    root = tmp_path / "ds"
    for i in range(3):
        leaf = root / "outputs" / f"s{i}"
        touch(leaf / "src.png")
        touch(leaf / "edited.png")
    result = probe_data_roots([str(root)])
    assert result["layout"] == "per_sample_dirs"
    assert result["sample_count"] == 3
    assert len(result["methods"]) == 1
