import sys
from pathlib import Path

from conftest import write_mock_config
from system_prompt_retrieval_agent.cli import build_parser, main


def test_cli_help_exits_zero(capsys):
    rc = main([])  # no subcommand -> print help
    assert rc == 0
    captured = capsys.readouterr()
    assert "run" in captured.out


def test_cli_run_dry_run(tmp_path: Path, capsys):
    cfg = write_mock_config(tmp_path)
    rc = main(["run", "--config", str(cfg), "--dry-run", "--max-rounds", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[dry-run]" in out


def test_parser_accepts_all_flags():
    p = build_parser()
    ns = p.parse_args(["run", "--config", "/tmp/c.yaml", "--limit", "30", "--max-rounds", "2", "--dry-run"])
    assert ns.cmd == "run"
    assert ns.limit == 30
    assert ns.max_rounds == 2
    assert ns.dry_run is True
