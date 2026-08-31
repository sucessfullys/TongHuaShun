"""Local checkpoint probe — lists model repos under a model root."""

from __future__ import annotations

from pathlib import Path

_WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.gguf", "*.pth", "*.pt", "*.ckpt")


def probe_checkpoints(model_root: str | None) -> dict:
    """List immediate sub-directories of ``model_root`` that look like model
    repositories (contain ``config.json`` or weight files)."""
    result: dict = {
        "local_model_root": "",
        "detected": [],
        "probe_ok": False,
        "error": "",
    }

    if not model_root:
        result["error"] = "no model root provided"
        return result

    path = Path(model_root)
    if not path.is_dir():
        result["error"] = f"model root not found: {model_root}"
        return result

    result["local_model_root"] = str(path.resolve())

    try:
        entries = sorted(p for p in path.iterdir() if p.is_dir())
    except OSError as exc:
        result["error"] = f"cannot list model root: {exc}"
        return result

    result["detected"] = [d.name for d in entries if _looks_like_model(d)]
    result["probe_ok"] = True
    return result


def _looks_like_model(directory: Path) -> bool:
    if (directory / "config.json").is_file():
        return True
    for pattern in _WEIGHT_GLOBS:
        if any(directory.glob(pattern)):
            return True
    # One level deeper — e.g. HuggingFace cache snapshot layout.
    try:
        for sub in directory.iterdir():
            if sub.is_dir() and (sub / "config.json").is_file():
                return True
    except OSError:
        pass
    return False
