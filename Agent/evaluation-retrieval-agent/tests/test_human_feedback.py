"""Tests for ERA Stage 8 — the human-feedback orchestration + CLI subcommands."""

from __future__ import annotations

import io
import json
from pathlib import Path

import yaml

from era.cli import main
from era.orchestration import human_feedback as hf
from era.workspace import Workspace

# A flagging config (Family A) and a comparison config (Family B).
_BRIEF = {
    "candidate_configs": [
        {
            "combination_id": "vlm-pointwise-72b", "family": "A",
            "judge": "Qwen2.5-VL-72B", "slot": "vlm_scale",
            "metric_subfamily": "n/a", "prompt": "pointwise-rubric",
            "scope": "pointwise-1to5", "hypothesis_id": "H1",
        },
        {
            "combination_id": "region-metric-baseline", "family": "B",
            "judge": "n/a", "slot": "metric_baseline",
            "metric_subfamily": "DINOv2 + ArcFace + MANIQA", "prompt": "n/a",
            "scope": "region", "hypothesis_id": "H2",
        },
    ],
}


def make_feedback_workspace(
    tmp_path: Path,
    name: str = "fb-demo",
    *,
    task_family: str = "editing",
    input_roles: dict | None = None,
    sample_metadata: dict | None = None,
):
    """Build a workspace with one iteration of Stage-6 results. Returns (ws, iter_dir).

    ``task_family`` / ``input_roles`` let a test build a generation workspace
    (``task_family="generation"``, ``input_roles={}``); ``sample_metadata``, if
    given, is written as each sample's ``metadata.json`` (so the review payload
    surfaces an input prompt / instruction).
    """
    if input_roles is None:
        input_roles = {"input_cloth": "input_cloth.png",
                        "input_model": "input_model.png"}
    data_root = tmp_path / "dataset"
    methods = []
    for mid in ("method_a", "method_b"):
        mpath = data_root / mid
        for sk in ("sample_001", "sample_002"):
            sd = mpath / sk
            sd.mkdir(parents=True)
            for fn in ("input_cloth.png", "input_model.png", "out.png"):
                (sd / fn).write_bytes(b"\x89PNG\r\n\x1a\n")
            if sample_metadata is not None:
                (sd / "metadata.json").write_text(
                    json.dumps(sample_metadata), encoding="utf-8")
        methods.append(
            {"method_id": mid, "path": str(mpath), "output_file": "out.png"})

    ws = Workspace(tmp_path / "workspaces", name)
    ws.scaffold()
    ws.create_iteration(1)
    ws.set_current(1)
    ws.write_status({
        "project_name": name, "stage": "full_experiment", "stage_index": 6,
        "iteration": 1, "run_state": "blocked",
    })
    (ws.root / "config.yaml").write_text(yaml.safe_dump({
        "project_name": name, "task_family": task_family, "task_adapter": "x",
        "hardware": {"visible_gpu_ids": [0], "max_gpus_per_run": 1,
                     "per_gpu_memory_gb": 10.0},
        "data": {
            "data_root": str(data_root), "layout": "per_sample_dirs",
            "methods": methods, "sample_glob": "*", "sample_count": 2,
            "input_roles": input_roles,
            "sample_key": "relpath",
        },
    }), encoding="utf-8")

    iter_dir = ws.root / "iter_001"
    (iter_dir / "design" / "experiment_brief.json").write_text(
        json.dumps(_BRIEF), encoding="utf-8")

    results = iter_dir / "experiments" / "results" / "full"

    def _scores(cid: str, scope: str, base: float) -> None:
        cdir = results / cid
        cdir.mkdir(parents=True, exist_ok=True)
        lines = []
        for sk in ("sample_001", "sample_002"):
            for i, mid in enumerate(("method_a", "method_b")):
                lines.append(json.dumps({
                    "sample_key": sk, "method_id": mid,
                    "score": base + 0.1 * i, "sub_scores": {}, "scope": scope,
                    "ok": True,
                }))
        (cdir / "scores.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _scores("vlm-pointwise-72b", "pointwise-1to5", 0.7)
    _scores("region-metric-baseline", "region", -0.3)

    (iter_dir / "experiments" / "results" / "summary.json").write_text(
        json.dumps({
            "iteration_dir": "iter_001", "mode": "full", "config_count": 2,
            "complete": True,
            "configs": [
                {"combination_id": "vlm-pointwise-72b", "family": "A",
                 "score_mean": 0.75, "ok_count": 4},
                {"combination_id": "region-metric-baseline", "family": "B",
                 "score_mean": -0.25, "ok_count": 4},
            ],
            "totals": {},
        }), encoding="utf-8")
    return ws, iter_dir


class _FakePopen:
    """Stand-in for a detached uvicorn process."""
    instances = 0

    def __init__(self, *args, **kwargs):
        _FakePopen.instances += 1
        self.pid = 424242


def _mock_launch(monkeypatch, *, alive=True, responds=True):
    _FakePopen.instances = 0
    monkeypatch.setattr(hf.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(hf, "_pick_port", lambda host, pref: 8799)
    monkeypatch.setattr(hf, "_wait_responsive", lambda h, p, timeout: responds)
    monkeypatch.setattr(hf, "_pid_alive", lambda pid: alive)
    monkeypatch.setattr(hf, "_port_responds", lambda h, p: responds)


# ---- serve_feedback -----------------------------------------------------

def test_serve_feedback_missing_summary(tmp_path: Path):
    ws, iter_dir = make_feedback_workspace(tmp_path)
    (iter_dir / "experiments" / "results" / "summary.json").unlink()
    result = hf.serve_feedback(ws.root, iteration=1)
    assert result["error"] == "no_summary"


def test_serve_feedback_launches_and_writes_pidfile(tmp_path, monkeypatch):
    ws, iter_dir = make_feedback_workspace(tmp_path)
    _mock_launch(monkeypatch)
    result = hf.serve_feedback(ws.root, iteration=1)
    assert result["status"] == "ok"
    assert result["port"] == 8799
    assert result["url"] == "http://127.0.0.1:8799/"
    pidfile = iter_dir / "human" / "server.pid"
    assert pidfile.is_file()
    rec = json.loads(pidfile.read_text())
    assert rec["pid"] == 424242 and rec["port"] == 8799


def test_serve_feedback_idempotent(tmp_path, monkeypatch):
    ws, _ = make_feedback_workspace(tmp_path)
    _mock_launch(monkeypatch)
    first = hf.serve_feedback(ws.root, iteration=1)
    assert first["status"] == "ok"
    second = hf.serve_feedback(ws.root, iteration=1)
    assert second["status"] == "already_running"
    assert _FakePopen.instances == 1  # no second process spawned


# ---- feedback_status ----------------------------------------------------

def test_feedback_status_no_server(tmp_path: Path):
    ws, _ = make_feedback_workspace(tmp_path)
    result = hf.feedback_status(ws.root, iteration=1)
    assert result["server"]["running"] is False
    assert result["feedback"]["present"] is False


def test_feedback_status_draft_then_finalized(tmp_path: Path):
    ws, iter_dir = make_feedback_workspace(tmp_path)
    fb_path = iter_dir / "human" / "feedback.json"
    fb_path.write_text(json.dumps({"schema_version": "1.0", "status": "draft"}))
    status = hf.feedback_status(ws.root, iteration=1)
    assert status["feedback"]["present"] is True
    assert status["feedback"]["finalized"] is False

    hf.finalize_feedback(ws.root, iteration=1)
    status = hf.feedback_status(ws.root, iteration=1)
    assert status["feedback"]["finalized"] is True
    assert status["feedback"]["labels_present"] is True


# ---- finalize_feedback --------------------------------------------------

def test_finalize_feedback_derives_both_modalities(tmp_path: Path):
    ws, iter_dir = make_feedback_workspace(tmp_path)
    feedback = {
        "item_marks": [{
            "sample_key": "sample_001", "method_id": "method_a",
            "combination_id": "vlm-pointwise-72b", "label": "wrong",
            "comment": "judge over-rated this",
        }],
        "comparison_marks": [{
            "sample_key": "sample_002",
            "combination_id": "region-metric-baseline", "label": "wrong",
            "comment": "ranking looks inverted",
        }],
    }
    result = hf.finalize_feedback(ws.root, feedback=feedback, iteration=1)
    assert result["status"] == "ok"

    labels = json.loads(
        (iter_dir / "human" / "human_labels.json").read_text())
    assert labels["default_rule"] == "unmarked_is_correct"
    # 1 flagging config x 2 samples x 2 methods = 4 cells
    assert len(labels["labels"]) == 4
    wrong = [x for x in labels["labels"] if x["human_verdict"] == "wrong"]
    correct = [x for x in labels["labels"] if x["human_verdict"] == "correct"]
    assert len(wrong) == 1 and len(correct) == 3
    assert wrong[0]["comment"] == "judge over-rated this"
    # 1 comparison config x 2 samples = 2 rankings
    assert len(labels["comparison_labels"]) == 2
    cmp_wrong = [x for x in labels["comparison_labels"]
                 if x["human_verdict"] == "wrong"]
    assert len(cmp_wrong) == 1
    modalities = {c["modality"] for c in labels["config_summary"]}
    assert modalities == {"flag", "comparison"}

    fb = json.loads((iter_dir / "human" / "feedback.json").read_text())
    assert fb["status"] == "finalized" and fb["finalized_at"]


def test_finalize_feedback_idempotent(tmp_path: Path):
    ws, iter_dir = make_feedback_workspace(tmp_path)
    hf.finalize_feedback(ws.root, iteration=1)
    labels_path = iter_dir / "human" / "human_labels.json"
    first = labels_path.read_text()
    hf.finalize_feedback(ws.root, iteration=1)
    assert labels_path.read_text() == first


# ---- find_prior_human_labels -------------------------------------------

def test_find_prior_human_labels(tmp_path: Path):
    ws, _ = make_feedback_workspace(tmp_path)
    hf.finalize_feedback(ws.root, iteration=1)
    assert hf.find_prior_human_labels(ws.root, 1) is None
    found = hf.find_prior_human_labels(ws.root, 2)
    assert found is not None and found.name == "human_labels.json"


def test_stop_feedback_not_running(tmp_path: Path):
    ws, _ = make_feedback_workspace(tmp_path)
    assert hf.stop_feedback(ws.root, iteration=1)["status"] == "not_running"


# ---- CLI round-trips ----------------------------------------------------

def _run_cli(argv, stdin_obj, monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(stdin_obj)))
    code = main(argv)
    return code, json.loads(capsys.readouterr().out)


def test_cli_serve_feedback_missing_path(tmp_path, monkeypatch, capsys):
    code, result = _run_cli(["serve-feedback"], {}, monkeypatch, capsys)
    assert code == 1
    assert result["error"] == "missing_workspace_path"


def test_cli_serve_feedback_bad_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO("{not json"))
    code = main(["serve-feedback"])
    assert code == 1
    assert json.loads(capsys.readouterr().out)["error"] == "bad_stdin_json"


def test_cli_feedback_status_round_trip(tmp_path, monkeypatch, capsys):
    ws, _ = make_feedback_workspace(tmp_path)
    code, result = _run_cli(
        ["feedback-status"], {"workspace_path": str(ws.root), "iteration": 1},
        monkeypatch, capsys,
    )
    assert code == 0
    assert result["status"] == "ok"
    assert result["server"]["running"] is False


def test_cli_finalize_feedback_round_trip(tmp_path, monkeypatch, capsys):
    ws, iter_dir = make_feedback_workspace(tmp_path)
    code, result = _run_cli(
        ["finalize-feedback"],
        {"workspace_path": str(ws.root), "iteration": 1,
         "feedback": {"item_marks": [], "comparison_marks": []}},
        monkeypatch, capsys,
    )
    assert code == 0
    assert result["status"] == "ok"
    assert (iter_dir / "human" / "human_labels.json").is_file()
