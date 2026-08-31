"""seed_loader.py — Dynamic loader for seed prompt strings from PROMPTS.py."""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FALLBACK_SEEDS: dict[str, str] = {
    "system_prompt_for_tryon_v5": (
        "You are a prompt engineer for AI image generation systems specializing in "
        "clothing transfer. Generate clear, structured prompts that accurately describe "
        "the clothing replacement task. Focus on garment type, replacement logic, and "
        "avoid detailed texture descriptions."
    )
}


def load_seed_prompts(prompt_file: Path) -> dict[str, str]:
    """Dynamically import PROMPTS.py and return all system_prompt_* strings.

    Returns a dict mapping attribute name -> text. Never logs text values,
    only attribute names. On import failure, returns fallback seeds with a WARN.
    """
    prompt_file = Path(prompt_file)
    if not prompt_file.exists():
        logger.warning(
            "Seed prompt file not found: %s — using inline fallback seeds.",
            prompt_file,
        )
        return dict(_FALLBACK_SEEDS)

    try:
        spec = importlib.util.spec_from_file_location("_seed_prompts_module", prompt_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {prompt_file}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning(
            "Failed to import seed prompts from %s (%s) — using inline fallback seeds.",
            prompt_file,
            exc,
        )
        return dict(_FALLBACK_SEEDS)

    result: dict[str, str] = {}
    for attr_name in dir(mod):
        if not attr_name.startswith("system_prompt_"):
            continue
        value = getattr(mod, attr_name)
        if isinstance(value, str):
            result[attr_name] = value

    if result:
        logger.info("Loaded seed prompts: %s", list(result.keys()))
    else:
        logger.warning(
            "No system_prompt_* strings found in %s — using inline fallback seeds.",
            prompt_file,
        )
        return dict(_FALLBACK_SEEDS)

    return result


def load_tryon_prompts(prompts_file: Path | None = None) -> list[dict[str, Any]]:
    """Import TRYON_PROMPTS list from PROMPTS.py.

    Resolves the file path in order:
      1. Explicit ``prompts_file`` argument (if given).
      2. Auto-detect: walk up from this file to the repo root, then look in
         ``Image-Generater-Remote/temp_wxy/PROMPTS.py``.

    Returns the TRYON_PROMPTS list (all entries, enabled and disabled).
    Never logs entry text values, only count/file path.
    On any failure returns an empty list with a WARNING.
    """
    candidate: Path | None = None

    if prompts_file is not None:
        candidate = Path(prompts_file)
    else:
        # Auto-detect: climb to repo root (the dir that contains both
        # System-Prompt-Retrieval-Agent/ and Image-Generater-Remote/).
        # __file__ is inside System-Prompt-Retrieval-Agent/src/...
        this_dir = Path(__file__).resolve()
        # Walk up to find the directory whose parent contains Image-Generater-Remote
        for ancestor in this_dir.parents:
            probe = ancestor / "Image-Generater-Remote" / "temp_wxy" / "PROMPTS.py"
            if probe.exists():
                candidate = probe
                break
        if candidate is None:
            # Fallback: try relative to current working directory
            cwd_probe = Path.cwd() / "Image-Generater-Remote" / "temp_wxy" / "PROMPTS.py"
            if cwd_probe.exists():
                candidate = cwd_probe

    if candidate is None or not candidate.exists():
        logger.warning(
            "TRYON_PROMPTS file not found (tried auto-detect). "
            "Returning empty list. Supply prompts_file explicitly or ensure "
            "Image-Generater-Remote/temp_wxy/PROMPTS.py is reachable."
        )
        return []

    try:
        spec = importlib.util.spec_from_file_location("_tryon_prompts_module", candidate)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {candidate}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as exc:
        logger.warning(
            "Failed to import TRYON_PROMPTS from %s (%s). Returning empty list.",
            candidate,
            exc,
        )
        return []

    tryon = getattr(mod, "TRYON_PROMPTS", None)
    if not isinstance(tryon, list):
        logger.warning(
            "TRYON_PROMPTS not found or not a list in %s. Returning empty list.",
            candidate,
        )
        return []

    logger.info(
        "Loaded TRYON_PROMPTS from %s: %d entries (ids: %s).",
        candidate,
        len(tryon),
        [e.get("user_prompt_id") for e in tryon if isinstance(e, dict)],
    )
    return list(tryon)


def load_tryon_prompts_names(prompts_file: Path | None = None) -> list[str]:
    """Return only the user_prompt_id names from TRYON_PROMPTS.

    Convenience wrapper around ``load_tryon_prompts`` that extracts the
    ``user_prompt_id`` field from each entry. Never logs text values.
    Returns [] on failure.
    """
    entries = load_tryon_prompts(prompts_file=prompts_file)
    return [
        e["user_prompt_id"]
        for e in entries
        if isinstance(e, dict) and "user_prompt_id" in e
    ]
