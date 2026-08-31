import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _example_config_path() -> Path:
    return ROOT / "config.yaml.example"


def write_mock_config(tmp_path: Path, **overrides) -> Path:
    """Copy config.yaml.example to tmp_path with path overrides for tests."""
    import yaml

    raw = yaml.safe_load(_example_config_path().read_text())
    raw["paths"]["local_agent_root"] = str(tmp_path)
    raw["paths"]["output_root"] = str(tmp_path / "outputs")
    raw["paths"]["memory_root"] = str(tmp_path / "memory")
    raw["paths"]["env_file"] = str(tmp_path / ".env")
    for k, v in overrides.items():
        raw[k] = v
    out = tmp_path / "config.yaml"
    out.write_text(yaml.safe_dump(raw))
    (tmp_path / ".env").write_text("OPENAI_API_KEY=test-key-xyz\nGoogle_API_KEY=test-google-abc\n")
    return out
