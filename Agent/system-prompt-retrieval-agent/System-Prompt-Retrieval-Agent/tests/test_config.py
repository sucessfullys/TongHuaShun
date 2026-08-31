from pathlib import Path

from conftest import write_mock_config
from system_prompt_retrieval_agent.config import load_config


def test_load_config_example(tmp_path: Path):
    cfg_path = write_mock_config(tmp_path)
    cfg = load_config(cfg_path)
    assert cfg.config_version == "0.2.1"
    assert cfg.remote.controller_base_url.endswith(":17700")
    assert cfg.rate_limits.requests_per_second == 3.0
    assert cfg.workflow.allow_partial_samples is False
    assert "dress" in cfg.evaluation.categories


def test_redacted_dump_hides_env_values(tmp_path: Path, monkeypatch):
    cfg_path = write_mock_config(tmp_path)
    cfg = load_config(cfg_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret-123")
    dumped = cfg.redacted_dump()
    import json

    blob = json.dumps(dumped)
    assert "sk-super-secret-123" not in blob
