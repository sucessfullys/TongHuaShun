import json
from pathlib import Path

from system_prompt_retrieval_agent.tracing import RunTraceLogger, redact_secrets


def test_redact_env_value(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-abc-xyz")
    out = redact_secrets({"msg": "using sk-abc-xyz in call"})
    assert "sk-abc-xyz" not in json.dumps(out)


def test_redact_large_base64():
    long_b64 = "A" * 2000
    out = redact_secrets({"img": long_b64})
    assert "base64 redacted" in out["img"]


def test_trace_logger_writes_jsonl(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-xyz")
    log = RunTraceLogger(tmp_path / "trace.jsonl", ["OPENAI_API_KEY"])
    log.emit("init", detail="leaking sk-secret-xyz here")
    log.emit("step", n=1)
    lines = (tmp_path / "trace.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        assert "sk-secret-xyz" not in line
