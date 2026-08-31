import logging
import os

from system_prompt_retrieval_agent.logging_setup import RedactionFilter, setup_logging


def test_redaction_filter_strips_env_value(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-xxx-yyy-top-secret")
    logger = logging.getLogger("test-redaction-1")
    logger.setLevel(logging.INFO)
    logger.addFilter(RedactionFilter(["OPENAI_API_KEY"]))
    with caplog.at_level(logging.INFO, logger="test-redaction-1"):
        logger.info("api key was sk-xxx-yyy-top-secret leaked")
    for rec in caplog.records:
        assert "sk-xxx-yyy-top-secret" not in rec.getMessage()


def test_setup_logging_installs_filter(monkeypatch):
    monkeypatch.setenv("Google_API_KEY", "google-xxx")
    logger = setup_logging("INFO", ["Google_API_KEY"])
    assert any(isinstance(f, RedactionFilter) for f in logger.filters)
