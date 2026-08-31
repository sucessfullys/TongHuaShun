"""Tests for the shared cross-project serving-recipe memory."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from era.cli import main
from era.orchestration import serving_memory as sm


@pytest.fixture(autouse=True)
def _redirect_memory_dir(tmp_path: Path, monkeypatch):
    """Point ERA_MEMORY_DIR at the test tmp so we never touch the user's
    real ~/.era/ tree."""
    monkeypatch.setenv("ERA_MEMORY_DIR", str(tmp_path / "memory"))


def _sample_recipe(model_id="Qwen/Qwen2.5-VL-72B-Instruct",
                   backend="ms_swift") -> dict:
    return {
        "model_id": model_id,
        "backend": backend,
        "install": {
            "preflight": ["pip install ms-swift==1.0.0"],
            "model_root": "/dev/shm/models/Qwen2.5-VL-72B-Instruct",
        },
        "launch": {
            "command": "swift deploy --model_id_or_path … --tp 4",
            "timeout_s": 600,
        },
        "known_quirks": ["needs FLASHINFER=1"],
    }


# ---- write + read ------------------------------------------------------

def test_write_then_read_roundtrip():
    record = sm.write_recipe(_sample_recipe())
    assert record["model_id"] == "Qwen/Qwen2.5-VL-72B-Instruct"
    assert record["backend"] == "ms_swift"
    assert record["schema_version"] == "1.0"
    # _ident keeps both org+model parts (so org_a/foo != org_b/foo)
    assert record["model_slug"] == "qwen_qwen2_5_vl_72b_instruct"
    assert record["captured_at"] and record["last_validated"]

    reread = sm.read_recipe("Qwen/Qwen2.5-VL-72B-Instruct", "ms_swift")
    assert reread is not None
    assert reread["model_id"] == record["model_id"]
    assert reread["known_quirks"] == ["needs FLASHINFER=1"]


def test_read_missing_returns_none():
    assert sm.read_recipe("not/a/model", "ms_swift") is None


def test_read_with_empty_args_returns_none():
    assert sm.read_recipe("", "") is None


def test_write_rejects_missing_required_keys():
    with pytest.raises(ValueError):
        sm.write_recipe({"backend": "ms_swift"})  # no model_id
    with pytest.raises(ValueError):
        sm.write_recipe({"model_id": "x"})        # no backend


def test_write_preserves_captured_at_on_update():
    """Re-writing the same recipe preserves captured_at but bumps last_validated."""
    first = sm.write_recipe(_sample_recipe())
    second = sm.write_recipe(_sample_recipe())
    assert first["captured_at"] == second["captured_at"]
    assert second["last_validated"] >= first["last_validated"]


def test_write_preserves_captured_by_workspace_across_kwargless_rewrite():
    """M2 — audit fields survive a rewrite that doesn't re-supply the
    kwargs. Otherwise a second project re-validating a recipe would
    silently erase the original capturer's provenance."""
    sm.write_recipe(
        _sample_recipe(),
        captured_by_workspace="orig-workspace",
        captured_by_iteration=7,
    )
    # Second write — no kwargs (e.g. Phase C's active capture from a
    # different project doesn't know who originally captured).
    after = sm.write_recipe(_sample_recipe())
    assert after["captured_by_workspace"] == "orig-workspace"
    assert after["captured_by_iteration"] == 7


def test_write_overrides_captured_by_workspace_when_supplied():
    """The original capturer is preserved by default, but an explicit
    kwarg on the rewrite wins (so a second project CAN claim ownership
    if they want)."""
    sm.write_recipe(
        _sample_recipe(),
        captured_by_workspace="orig", captured_by_iteration=1,
    )
    after = sm.write_recipe(
        _sample_recipe(),
        captured_by_workspace="new", captured_by_iteration=2,
    )
    assert after["captured_by_workspace"] == "new"
    assert after["captured_by_iteration"] == 2


def test_write_refuses_overwrite_when_disabled():
    sm.write_recipe(_sample_recipe(), captured_by_workspace="orig")
    again = sm.write_recipe(
        {"model_id": "Qwen/Qwen2.5-VL-72B-Instruct", "backend": "ms_swift",
         "install": {"preflight": ["NEW"]}},
        overwrite=False,
    )
    # The returned record is the EXISTING one, not the new payload
    assert again["captured_by_workspace"] == "orig"
    assert again["install"]["preflight"] != ["NEW"]


def test_write_records_capture_metadata():
    record = sm.write_recipe(
        _sample_recipe(),
        captured_by_workspace="tryon-eval",
        captured_by_iteration=3,
    )
    assert record["captured_by_workspace"] == "tryon-eval"
    assert record["captured_by_iteration"] == 3


# ---- slug normalization ------------------------------------------------

def test_slug_normalizes_model_id():
    """Model_id with slashes / case / dashes becomes a safe filename slug.
    Both org + model parts are preserved (org_a/foo != org_b/foo)."""
    sm.write_recipe({
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "backend": "ms_swift",
    })
    files = list(sm.serving_recipes_dir().glob("*.json"))
    assert len(files) == 1
    assert files[0].name == "qwen_qwen2_5_vl_72b_instruct__ms_swift.json"


def test_slug_distinguishes_backends():
    """Same model, different backends → two separate files."""
    sm.write_recipe({"model_id": "X/Foo", "backend": "ms_swift"})
    sm.write_recipe({"model_id": "X/Foo", "backend": "vllm"})
    files = sorted(p.name for p in sm.serving_recipes_dir().glob("*.json"))
    assert files == ["x_foo__ms_swift.json", "x_foo__vllm.json"]


def test_slug_distinguishes_orgs():
    """Different orgs with the same model name don't collide."""
    sm.write_recipe({"model_id": "org_a/Foo", "backend": "ms_swift"})
    sm.write_recipe({"model_id": "org_b/Foo", "backend": "ms_swift"})
    files = sorted(p.name for p in sm.serving_recipes_dir().glob("*.json"))
    assert files == ["org_a_foo__ms_swift.json", "org_b_foo__ms_swift.json"]


# ---- list / forget ------------------------------------------------------

def test_list_empty_when_no_recipes():
    assert sm.list_recipes() == []


def test_list_returns_summary_rows_sorted_by_recency():
    # L3: _now_iso uses millisecond precision now, so back-to-back writes
    # on a fast machine still produce strictly increasing timestamps. No
    # time.sleep needed.
    sm.write_recipe({"model_id": "alpha", "backend": "ms_swift"})
    sm.write_recipe({"model_id": "beta", "backend": "vllm"})
    rows = sm.list_recipes()
    assert len(rows) == 2
    # Most recent first
    assert rows[0]["model_id"] == "beta"
    assert rows[1]["model_id"] == "alpha"
    # Summary shape
    for row in rows:
        assert set(row) >= {"slug", "model_id", "backend",
                            "captured_at", "last_validated"}


def test_forget_removes_file():
    sm.write_recipe({"model_id": "Qwen", "backend": "ms_swift"})
    r = sm.forget_recipe("Qwen", "ms_swift")
    assert r["deleted"] is True
    assert sm.read_recipe("Qwen", "ms_swift") is None


def test_forget_missing_is_idempotent():
    r = sm.forget_recipe("does/not/exist", "ms_swift")
    assert r["status"] == "ok"
    assert r["deleted"] is False


# ---- atomic write ------------------------------------------------------

def test_write_leaves_no_partial_file():
    sm.write_recipe(_sample_recipe())
    leftovers = list(sm.serving_recipes_dir().glob("*.tmp"))
    assert leftovers == []


def test_memory_dir_respects_env_var(tmp_path: Path, monkeypatch):
    """ERA_MEMORY_DIR overrides ~/.era/memory/ for tests + sandboxing."""
    monkeypatch.setenv("ERA_MEMORY_DIR", str(tmp_path / "alt"))
    assert sm.memory_root() == (tmp_path / "alt").resolve()


# ---- CLI roundtrip -----------------------------------------------------

def test_cli_list_empty(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"verb": "list"})))
    code = main(["serving-memory"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out == {"status": "ok", "recipes": []}


def test_cli_write_then_read(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "verb": "write",
        "recipe": _sample_recipe(),
        "captured_by_workspace": "tryon-eval",
    })))
    code = main(["serving-memory"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["status"] == "ok"
    assert out["model_id"] == "Qwen/Qwen2.5-VL-72B-Instruct"

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "verb": "read",
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "backend": "ms_swift",
    })))
    code = main(["serving-memory"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["captured_by_workspace"] == "tryon-eval"


def test_cli_read_missing(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "verb": "read", "model_id": "ghost", "backend": "vllm",
    })))
    code = main(["serving-memory"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["error"] == "not_found"


def test_cli_bad_verb_rejected(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "verb": "delete-everything",
    })))
    code = main(["serving-memory"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["error"] == "bad_verb"


def test_cli_forget(monkeypatch, capsys):
    sm.write_recipe(_sample_recipe())
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({
        "verb": "forget",
        "model_id": "Qwen/Qwen2.5-VL-72B-Instruct",
        "backend": "ms_swift",
    })))
    code = main(["serving-memory"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["deleted"] is True


# ---- M1: unknown_backend flag ----------------------------------------

def test_unknown_backend_carries_flag():
    """M1 — a backend outside KNOWN_BACKENDS gets a structured marker
    so Stage 6 reader (and the CLI list view) can warn."""
    record = sm.write_recipe({
        "model_id": "Custom/Llama-3-70B",
        "backend": "internal_fork",
    })
    assert record["unknown_backend"] is True
    # Re-reading the file confirms the flag persisted to disk
    reread = sm.read_recipe("Custom/Llama-3-70B", "internal_fork")
    assert reread["unknown_backend"] is True


def test_known_backend_does_not_carry_flag():
    """A whitelisted backend does NOT get the flag (avoids spurious
    'unknown_backend: false' noise)."""
    record = sm.write_recipe(_sample_recipe())   # backend=ms_swift
    assert "unknown_backend" not in record


def test_known_backends_constant_lists_supported_set():
    """Lock the supported-backends list so we know exactly what
    counts as 'known' for the M1 flag check."""
    assert sm.KNOWN_BACKENDS == ("ms_swift", "vllm", "lmdeploy")


# ---- T2: malformed-type stdin guard ---------------------------------

def test_cli_rejects_stdin_array(monkeypatch, capsys):
    """T2 — piping a JSON array (not an object) into serving-memory
    must produce a structured bad_stdin_json error, not a TypeError
    traceback from .get() on a list."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(["list"])))
    code = main(["serving-memory"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["error"] == "bad_stdin_json"
    assert "object" in out["message"]


def test_cli_rejects_stdin_bare_string(monkeypatch, capsys):
    """Same guard for a bare JSON string."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps("list")))
    code = main(["serving-memory"])
    out = json.loads(capsys.readouterr().out)
    assert code == 1
    assert out["error"] == "bad_stdin_json"
