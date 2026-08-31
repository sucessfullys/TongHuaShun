"""Tests for the FastAPI annotation app (era/annotate/app.py)."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from era.annotate.app import build_app
from era.annotate.store import annotation_path


def _make_real_shape_dataset(root: Path) -> None:
    """2 methods × 2 categories × 3 samples each, per-method output filename."""
    methods_outputs = {
        "method_a": "tryon_result_method_a.png",
        "method_b": "tryon_result_method_b.png",
    }
    for method, output_file in methods_outputs.items():
        for category in ("dress", "upper"):
            for n in (1, 2, 3):
                sample_dir = root / method / category / f"{category}{n:02d}"
                sample_dir.mkdir(parents=True)
                (sample_dir / "input_cloth.png").write_bytes(
                    f"PNG-cloth-{category}{n}".encode())
                (sample_dir / "input_model.png").write_bytes(
                    f"PNG-model-{category}{n}".encode())
                (sample_dir / output_file).write_bytes(
                    f"PNG-{method}-{category}{n}".encode())


def test_health_reports_dataset(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.get("/api/health")
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "ok"
    assert payload["method_count"] == 2
    assert payload["sample_count"] == 6
    assert "input_cloth" in payload["input_roles"]
    assert "input_model" in payload["input_roles"]
    assert payload["output_role"] == "output"


def test_samples_lists_every_sample(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.get("/api/samples")
    assert r.status_code == 200
    payload = r.json()
    assert payload["total_count"] == 6
    assert payload["annotated_count"] == 0
    # Sample keys carry slashes
    sample_keys = [row["sample_key"] for row in payload["samples"]]
    for sk in sample_keys:
        assert "/" in sk
    assert payload["methods"] == ["method_a", "method_b"]


def test_sample_endpoint_returns_per_method_roles_with_path_arg(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    sample_key = "dress/dress01"
    r = client.get(f"/api/sample/{sample_key}")
    assert r.status_code == 200
    payload = r.json()
    assert payload["sample_key"] == sample_key
    method_ids = [m["method_id"] for m in payload["methods"]]
    assert method_ids == ["method_a", "method_b"]
    for m in payload["methods"]:
        assert m["present"] is True
        # Every role resolves to a relpath under the dataset root
        for role in ("input_cloth", "input_model", "output"):
            rel = m["roles"][role]
            assert rel is not None
            assert sample_key in rel
    # Per-method annotation is empty initially
    assert payload["per_method"] == {}


def test_sample_endpoint_404s_unknown(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.get("/api/sample/totally/not/a/sample")
    assert r.status_code == 404


def test_image_endpoint_serves_per_method_output(tmp_path: Path):
    """Each method's output URL must serve THAT method's distinct bytes."""
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.get("/api/image", params={
        "method": "method_a", "sample": "dress/dress01", "role": "output",
    })
    assert r.status_code == 200
    assert r.content == b"PNG-method_a-dress1"

    r = client.get("/api/image", params={
        "method": "method_b", "sample": "dress/dress01", "role": "output",
    })
    assert r.status_code == 200
    assert r.content == b"PNG-method_b-dress1"


def test_image_endpoint_404s_unknown(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.get("/api/image", params={
        "method": "method_a", "sample": "ghost/sample", "role": "output",
    })
    assert r.status_code == 404


def test_image_endpoint_refuses_traversal(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.get("/api/image", params={
        "method": "method_a",
        "sample": "../../../etc/passwd",
        "role": "input_cloth",
    })
    assert r.status_code == 404


def test_put_annotation_persists_one_method(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    sample_key = "dress/dress01"
    r = client.put(f"/api/sample/{sample_key}/annotation",
                   json={"method_id": "method_a",
                         "annotation": "logo blurred"})
    assert r.status_code == 200
    payload = r.json()
    assert payload["per_method"] == {"method_a": "logo blurred"}
    # File lands at <dataset>/annotations/dress/dress01.json
    on_disk = annotation_path(tmp_path, sample_key)
    assert on_disk.is_file()


def test_put_annotation_two_methods_independent(tmp_path: Path):
    """Editing method_b must not clobber method_a's slot for the same sample.

    This is the crux of the per-method design — two operators (or one
    operator on two tabs) can edit different methods of the same sample
    without losing each other's work.
    """
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    sample_key = "upper/upper02"
    client.put(f"/api/sample/{sample_key}/annotation",
               json={"method_id": "method_a", "annotation": "A note"})
    r = client.put(f"/api/sample/{sample_key}/annotation",
                   json={"method_id": "method_b", "annotation": "B note"})
    assert r.status_code == 200
    assert r.json()["per_method"] == {"method_a": "A note",
                                       "method_b": "B note"}
    # And the GET reflects the merge
    g = client.get(f"/api/sample/{sample_key}")
    assert g.json()["per_method"] == {"method_a": "A note",
                                       "method_b": "B note"}


def test_put_annotation_empty_clears_one_slot(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    sample_key = "dress/dress01"
    client.put(f"/api/sample/{sample_key}/annotation",
               json={"method_id": "method_a", "annotation": "A"})
    client.put(f"/api/sample/{sample_key}/annotation",
               json={"method_id": "method_b", "annotation": "B"})
    # Clear method_a only — method_b should survive
    r = client.put(f"/api/sample/{sample_key}/annotation",
                   json={"method_id": "method_a", "annotation": ""})
    assert r.status_code == 200
    assert r.json()["per_method"] == {"method_b": "B"}


def test_put_annotation_404s_unknown_sample(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.put("/api/sample/totally/not/a/sample/annotation",
                   json={"method_id": "method_a", "annotation": "x"})
    assert r.status_code == 404


def test_put_annotation_422_unknown_method(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.put("/api/sample/dress/dress01/annotation",
                   json={"method_id": "method_z", "annotation": "x"})
    assert r.status_code == 422


def test_put_annotation_422_missing_method_id(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.put("/api/sample/dress/dress01/annotation",
                   json={"annotation": "x"})
    assert r.status_code == 422


def test_put_annotation_422_missing_annotation_field(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.put("/api/sample/dress/dress01/annotation",
                   json={"method_id": "method_a"})
    assert r.status_code == 422


def test_samples_annotated_counter_updates_after_save(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    client.put("/api/sample/dress/dress01/annotation",
               json={"method_id": "method_a", "annotation": "x"})
    payload = client.get("/api/samples").json()
    assert payload["annotated_count"] == 1
    by_key = {row["sample_key"]: row for row in payload["samples"]}
    assert by_key["dress/dress01"]["annotated"] is True
    assert by_key["dress/dress01"]["annotated_methods"] == 1


def test_index_serves_html(tmp_path: Path):
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.get("/")
    assert r.status_code == 200
    assert "ERA" in r.text and "Annotation" in r.text


def test_built_react_bundle_is_served(tmp_path: Path):
    """The served /api root must be the built React bundle (not the
    placeholder HTML). Trips a regression when someone forgets to run
    `npm run build` after editing the frontend source."""
    _make_real_shape_dataset(tmp_path)
    client = TestClient(build_app(tmp_path))
    r = client.get("/")
    assert r.status_code == 200
    # React mount point present
    assert '<div id="root">' in r.text
    # At least one hashed asset (Vite emits ./assets/index-<hash>.js / .css)
    assert "./assets/" in r.text or "/assets/" in r.text
    # Verify both asset types are referenced
    assert ".js" in r.text and ".css" in r.text
