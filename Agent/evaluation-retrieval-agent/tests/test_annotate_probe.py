"""Tests for era.cli annotate-probe (the read-only probe subcommand)."""

from __future__ import annotations

import io
import json
from pathlib import Path

from era.cli import main


def _real_shape_dataset(root: Path) -> None:
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
                    f"cloth-{category}{n}".encode())
                (sample_dir / "input_model.png").write_bytes(
                    f"model-{category}{n}".encode())
                (sample_dir / output_file).write_bytes(
                    f"{method}-{category}{n}".encode())


def _ambiguous_output_dataset(root: Path) -> None:
    """Method has TWO non-input image files per sample → output_candidates
    has 2 entries → probe confidence is needs_confirmation."""
    for sample in ("s001", "s002"):
        d = root / "only_method" / sample
        d.mkdir(parents=True)
        (d / "input_cloth.png").write_bytes(b"cloth")
        (d / "input_model.png").write_bytes(b"model")
        # Two output-looking files, no intermediate suffix → probe can't pick
        (d / "result_a.png").write_bytes(b"a")
        (d / "result_b.png").write_bytes(b"b")


def _run_probe(monkeypatch, capsys, payload: dict) -> tuple[int, dict]:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    code = main(["annotate-probe"])
    out = json.loads(capsys.readouterr().out)
    return code, out


# ---- happy path -----------------------------------------------------------

def test_probe_reports_methods_and_samples(
    tmp_path: Path, monkeypatch, capsys,
):
    _real_shape_dataset(tmp_path)
    code, out = _run_probe(monkeypatch, capsys, {"dataset_root": str(tmp_path)})
    assert code == 0
    assert out["status"] == "ok"
    assert out["methods"] == ["method_a", "method_b"]
    assert out["sample_count"] == 6        # 2 categories × 3 samples
    assert "input_cloth" in out["input_roles"]
    assert "input_model" in out["input_roles"]
    assert out["method_outputs"]["method_a"] == "tryon_result_method_a.png"
    assert out["method_outputs"]["method_b"] == "tryon_result_method_b.png"


def test_probe_first_sample_resolves_all_true(
    tmp_path: Path, monkeypatch, capsys,
):
    """For a clean dataset, every (method, role) resolves to a real file."""
    _real_shape_dataset(tmp_path)
    _, out = _run_probe(monkeypatch, capsys, {"dataset_root": str(tmp_path)})
    assert out["first_sample_key"] is not None
    for method_id, roles_map in out["first_sample_resolves"].items():
        assert all(roles_map.values()), (
            f"method {method_id} has missing roles: {roles_map}"
        )


def test_probe_confidence_high_when_unambiguous(
    tmp_path: Path, monkeypatch, capsys,
):
    _real_shape_dataset(tmp_path)
    _, out = _run_probe(monkeypatch, capsys, {"dataset_root": str(tmp_path)})
    assert out["confidence"] == "high"


# ---- ambiguous probe ------------------------------------------------------

def test_probe_surfaces_output_candidates_for_ambiguous_method(
    tmp_path: Path, monkeypatch, capsys,
):
    """When the probe can't pick a single output, slash command needs the
    candidate list to drive AskUserQuestion."""
    _ambiguous_output_dataset(tmp_path)
    _, out = _run_probe(monkeypatch, capsys, {"dataset_root": str(tmp_path)})
    assert out["confidence"] == "needs_confirmation"
    # The output_candidates entry for the (one) method must list both files
    candidates = out["output_candidates"]
    assert len(candidates) == 1
    method_id = next(iter(candidates))
    assert sorted(candidates[method_id]) == ["result_a.png", "result_b.png"]


# ---- refusal paths --------------------------------------------------------

def test_probe_missing_dataset_returns_error(
    tmp_path: Path, monkeypatch, capsys,
):
    code, out = _run_probe(
        monkeypatch, capsys, {"dataset_root": str(tmp_path / "ghost")})
    assert code == 1
    assert out["error"] == "no_dataset"


def test_probe_empty_dataset_returns_zero_samples(
    tmp_path: Path, monkeypatch, capsys,
):
    """An empty dir (no method subdirs) is a *valid* probe response with
    zero samples — the slash command refuses to launch on this."""
    code, out = _run_probe(monkeypatch, capsys, {"dataset_root": str(tmp_path)})
    assert code == 0
    assert out["sample_count"] == 0
    assert out["methods"] == []
    assert out["first_sample_key"] is None


def test_probe_missing_dataset_root_param(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    code = main(["annotate-probe"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["error"] == "missing_dataset_root"


def test_probe_bad_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json {{{"))
    code = main(["annotate-probe"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["error"] == "bad_stdin_json"


# ---- end-to-end: probe → server (overrides applied) ----------------------

def test_overrides_change_image_resolution(tmp_path: Path):
    """build_app with output_overrides resolves to the override filename,
    not the probe's auto-picked one — proving the overrides flow lands."""
    _ambiguous_output_dataset(tmp_path)
    # The probe's method_id depends on its single-nested-branch detection,
    # which uses tmp_path's basename — discover it at runtime, don't hardcode.
    from era.annotate.app import build_app
    from era.annotate.data import walk_dataset
    from fastapi.testclient import TestClient

    ds = walk_dataset(tmp_path)
    method_id = ds.methods[0]
    sample_key = ds.sample_keys[0]
    app = build_app(
        tmp_path, output_overrides={method_id: "result_b.png"},
    )
    client = TestClient(app)
    r = client.get("/api/image", params={
        "method": method_id, "sample": sample_key, "role": "output",
    })
    assert r.status_code == 200
    assert r.content == b"b"


def test_input_role_overrides_register_a_role(tmp_path: Path):
    """input_role_overrides can override or add roles."""
    _real_shape_dataset(tmp_path)
    from era.annotate.app import build_app
    from fastapi.testclient import TestClient

    # Add a new fabricated role mapping that doesn't exist on disk
    # (the resolver should return None / 404).
    app = build_app(
        tmp_path, input_role_overrides={"input_cloth": "ghost.png"},
    )
    client = TestClient(app)
    r = client.get("/api/image", params={
        "method": "method_a", "sample": "dress/dress01",
        "role": "input_cloth",
    })
    # The override points at a non-existent filename → 404.
    assert r.status_code == 404


# ---- serve_annotate plumbing ---------------------------------------------

def test_cli_annotate_mirror_backfills(tmp_path: Path, monkeypatch, capsys):
    """The CLI subcommand writes per-method copies for every central file."""
    import json as _json
    from era.annotate.store import (
        ANNOTATIONS_DIR, PER_METHOD_FILENAME,
    )
    _real_shape_dataset(tmp_path)

    # Hand-write one central annotation file (simulating the operator
    # having saved a note before the mirroring feature shipped).
    central = tmp_path / ANNOTATIONS_DIR / "dress" / "dress01.json"
    central.parent.mkdir(parents=True, exist_ok=True)
    central.write_text(_json.dumps({
        "schema_version": "2.0",
        "sample_key": "dress/dress01",
        "per_method": {"method_a": "A note", "method_b": "B note"},
        "created_at": "2026-05-26T12:00:00+00:00",
        "updated_at": "2026-05-26T12:00:00+00:00",
    }), encoding="utf-8")

    monkeypatch.setattr("sys.stdin", io.StringIO(
        _json.dumps({"dataset_root": str(tmp_path)})))
    code = main(["annotate-mirror"])
    out = _json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["status"] == "ok"
    assert out["scanned"] == 1
    assert out["written"] == 2     # 2 methods × 1 sample
    # Both per-method copies landed
    for method in ("method_a", "method_b"):
        copy = (tmp_path / method / "dress" / "dress01" / PER_METHOD_FILENAME)
        assert copy.is_file(), f"missing per-method copy at {copy}"
        body = _json.loads(copy.read_text(encoding="utf-8"))
        assert body["method_id"] == method
        assert body["annotation"] == (
            "A note" if method == "method_a" else "B note")


def test_cli_annotate_mirror_missing_dataset(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))
    code = main(["annotate-mirror"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["error"] == "missing_dataset_root"


def test_serve_annotate_passes_overrides_via_env(tmp_path: Path, monkeypatch):
    """End-to-end: serve_annotate sets ERA_ANNOTATE_OUTPUT_OVERRIDES in the
    spawned process's env. Subprocess is monkey-patched to capture it."""
    from era.orchestration import annotate as orchestrate

    _real_shape_dataset(tmp_path)

    captured_env: dict = {}

    class _FakePopen:
        def __init__(self, *args, **kwargs):
            self.pid = 99001
            self.kwargs = kwargs
            captured_env.update(kwargs.get("env", {}))

        def poll(self):
            return None

    monkeypatch.setattr(orchestrate.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(orchestrate, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(orchestrate, "_wait_responsive",
                        lambda h, p, timeout: True)
    monkeypatch.setattr(orchestrate, "_port_responds", lambda h, p: True)
    monkeypatch.setattr(orchestrate, "_pick_port", lambda h, p: 8801)

    result = orchestrate.serve_annotate(
        tmp_path,
        output_overrides={"method_a": "result_X.png"},
        input_role_overrides={"input_cloth": "alt_cloth.png"},
    )
    assert result["status"] == "ok"

    # The env passed into Popen must carry both JSON-encoded overrides
    out_env = captured_env.get("ERA_ANNOTATE_OUTPUT_OVERRIDES")
    in_env = captured_env.get("ERA_ANNOTATE_INPUT_ROLE_OVERRIDES")
    assert out_env is not None and json.loads(out_env) == {
        "method_a": "result_X.png"}
    assert in_env is not None and json.loads(in_env) == {
        "input_cloth": "alt_cloth.png"}

    # The pidfile also records them for audit
    pidfile = (tmp_path / orchestrate.PIDFILE_NAME)
    assert pidfile.is_file()
    rec = json.loads(pidfile.read_text(encoding="utf-8"))
    assert rec["output_overrides"] == {"method_a": "result_X.png"}
    assert rec["input_role_overrides"] == {"input_cloth": "alt_cloth.png"}

    # Cleanup
    pidfile.unlink()
