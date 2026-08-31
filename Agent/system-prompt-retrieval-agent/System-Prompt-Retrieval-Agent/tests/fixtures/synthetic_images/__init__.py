"""Synthetic image fixture helpers.

Images are generated on-the-fly via pytest fixtures (conftest.py) into
``tmp_path`` so that no binary PNG files need to be committed to the repository.

This module provides a factory function used by the test conftest.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


# Palette of (R, G, B) colours for the 5 sample slots.
_COLOURS: list[tuple[int, int, int]] = [
    (220, 50, 50),    # red
    (50, 200, 50),    # green
    (50, 50, 220),    # blue
    (200, 200, 50),   # yellow
    (160, 50, 200),   # purple
]


def make_synthetic_image_set(base_dir: Path, n_samples: int = 5) -> dict[str, dict[str, Path]]:
    """Generate a set of synthetic PNG images for testing.

    For each sample index ``i`` in ``range(n_samples)``:
    - ``model_{i}.png``    — 64x96  portrait  (h > w)
    - ``cloth_{i}.png``    — 96x64  landscape (w > h)
    - ``prev_{i}.png``     — 64x96  portrait
    - ``curr_{i}.png``     — 96x64  landscape

    Returns a mapping ``{sample_id: {"model": Path, "cloth": Path,
    "prev_result": Path, "curr_result": Path}}``.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    image_map: dict[str, dict[str, Path]] = {}

    for i in range(n_samples):
        colour = _COLOURS[i % len(_COLOURS)]
        # Slightly vary the shade per column to make images visually distinct.
        shades = [colour, _darken(colour, 20), _lighten(colour, 30), _darken(colour, 50)]
        sizes = [(64, 96), (96, 64), (64, 96), (96, 64)]
        col_names = ["model", "cloth", "prev_result", "curr_result"]
        fnames = [f"model_{i}.png", f"cloth_{i}.png", f"prev_{i}.png", f"curr_{i}.png"]

        sample_paths: dict[str, Path] = {}
        for col_name, fname, size, shade in zip(col_names, fnames, sizes, shades):
            p = base_dir / fname
            Image.new("RGB", size, color=shade).save(p)
            sample_paths[col_name] = p

        sample_id = f"sample_{i:02d}"
        image_map[sample_id] = sample_paths

    return image_map


def _darken(colour: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(max(0, c - amount) for c in colour)  # type: ignore[return-value]


def _lighten(colour: tuple[int, int, int], amount: int) -> tuple[int, int, int]:
    return tuple(min(255, c + amount) for c in colour)  # type: ignore[return-value]
