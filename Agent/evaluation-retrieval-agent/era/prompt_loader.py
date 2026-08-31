"""Load and render ERA runtime prompts/templates from ``docs/prompts/``.

Templates use ``{{PLACEHOLDER}}`` markers (double braces) so that literal
single braces — e.g. JSON examples inside a prompt — are never disturbed.
"""

from __future__ import annotations

from .paths import prompts_dir


def load_prompt(name: str) -> str:
    """Return the raw text of ``docs/prompts/<name>``.

    A missing ``.md`` suffix is added automatically.
    """
    path = prompts_dir() / name
    if not path.suffix:
        path = path.with_suffix(".md")
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **values: object) -> str:
    """Load a template and substitute ``{{KEY}}`` markers with ``values``."""
    text = load_prompt(name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text
