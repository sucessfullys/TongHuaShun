"""Compose per-round comparison grids using PIL.

V0.2 layout (backward-compatible)
----------------------------------
Each row corresponds to one sample. Columns are:
    model | cloth | prev_result | curr_result | delta_label

V0.2.1 layout (multi-phrasing)
--------------------------------
Each row corresponds to one (pair_id, sample_id). Columns are one slot per
enabled user_prompt_id (e.g. zh_001, zh_003, en_001, en_003) labelled with the
user_prompt_id and language tag.  Failed cells render a placeholder tile
containing the ``failure_reason`` text so the grid is always dense.

A ``grid_meta.yaml`` sidecar is emitted next to the output PNG containing:
    pair_id, round_id, user_prompt_ids, sample_ids, corpus_hash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# V0.2 legacy constants
# ---------------------------------------------------------------------------

COLUMNS = ["model", "cloth", "prev_result", "curr_result", "delta_label"]

# Background colour for the delta-label column
_DELTA_BG_IMPROVEMENT = (180, 240, 180)  # light green
_DELTA_BG_DECREASE = (240, 180, 180)     # light red
_DELTA_BG_NEUTRAL = (230, 230, 230)      # light grey

# Text colour for delta label
_DELTA_TEXT_COLOUR = (20, 20, 20)

# Width of the generated delta_label column (pixels)
_DELTA_LABEL_WIDTH = 160

# ---------------------------------------------------------------------------
# V0.2.1 constants
# ---------------------------------------------------------------------------

# Background for a failed-cell placeholder tile
_FAILED_BG = (200, 180, 180)       # muted red-grey
_FAILED_TEXT_COLOUR = (80, 20, 20)

# Header row height in the multi-phrasing grid
_HEADER_HEIGHT = 32
_HEADER_BG = (245, 245, 255)
_HEADER_TEXT_COLOUR = (30, 30, 80)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_resize(
    path: Path,
    portrait_h: int,
    landscape_w: int,
) -> Image.Image:
    """Load an image and resize it according to its aspect ratio.

    Portrait (h > w): resize so h == portrait_h, preserve aspect.
    Landscape/square (w >= h): resize so w == landscape_w, preserve aspect.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if h > w:
        # Portrait
        new_h = portrait_h
        new_w = max(1, round(w * portrait_h / h))
    else:
        # Landscape or square
        new_w = landscape_w
        new_h = max(1, round(h * landscape_w / w))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _make_delta_label_image(
    delta: float,
    width: int,
    height: int,
) -> Image.Image:
    """Create a text image showing the delta value."""
    if delta > 1e-9:
        text = f"+{delta:.3f}"
        bg = _DELTA_BG_IMPROVEMENT
    elif delta < -1e-9:
        text = f"{delta:.3f}"
        bg = _DELTA_BG_DECREASE
    else:
        text = "0.000"
        bg = _DELTA_BG_NEUTRAL

    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    # Use default PIL font; centre text in the image
    try:
        bbox = draw.textbbox((0, 0), text)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        # Older Pillow fallback
        text_w, text_h = draw.textsize(text)  # type: ignore[attr-defined]
    x = max(0, (width - text_w) // 2)
    y = max(0, (height - text_h) // 2)
    draw.text((x, y), text, fill=_DELTA_TEXT_COLOUR)
    return img


def _make_text_tile(
    text: str,
    width: int,
    height: int,
    bg: tuple[int, int, int],
    fg: tuple[int, int, int],
) -> Image.Image:
    """Create a solid-colour tile with centred text (single or multi-line)."""
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    # Truncate long text to fit width
    lines = text.split("\n")
    try:
        total_h = 0
        line_heights: list[int] = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line)
            lh = bbox[3] - bbox[1]
            line_heights.append(lh)
            total_h += lh
        y = max(0, (height - total_h) // 2)
        for line, lh in zip(lines, line_heights):
            bbox = draw.textbbox((0, 0), line)
            lw = bbox[2] - bbox[0]
            x = max(0, (width - lw) // 2)
            draw.text((x, y), line, fill=fg)
            y += lh
    except AttributeError:
        # Older Pillow fallback — draw first line only
        draw.text((4, height // 2 - 6), lines[0], fill=fg)
    return img


# ---------------------------------------------------------------------------
# V0.2 public API (backward-compatible)
# ---------------------------------------------------------------------------

def compose_comparison_grid(
    deltas: list[dict],
    image_map: dict[str, dict[str, Path]],
    output_path: Path,
    *,
    portrait_h: int = 1376,
    landscape_w: int = 768,
    row_gap: int = 4,
    col_gap: int = 4,
    label_height: int = 40,
) -> Path:
    """Compose a PIL grid of comparison images and save as PNG.

    Parameters
    ----------
    deltas:
        Ordered list of delta dicts (from selectors). Rows follow this order.
    image_map:
        ``{sample_id: {"model": Path, "cloth": Path, "prev_result": Path,
        "curr_result": Path}}``.
    output_path:
        Destination path for the output PNG.
    portrait_h:
        Target height for portrait images (h > w).
    landscape_w:
        Target width for landscape/square images (w >= h).
    row_gap:
        Vertical gap between rows in pixels.
    col_gap:
        Horizontal gap between columns in pixels.
    label_height:
        Height of the optional header label row (currently not drawn; reserved
        for future use).  Not used in row-height computation.

    Returns
    -------
    Path
        The ``output_path`` after the file has been written.
    """
    if not deltas:
        # Nothing to draw — write a 1×1 placeholder so the path always exists.
        img = Image.new("RGB", (1, 1), color=(0, 0, 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path

    # ------------------------------------------------------------------
    # Phase 1: load + resize all images to determine per-row dimensions.
    # ------------------------------------------------------------------
    # row_images[row_idx][col_name] = PIL Image
    row_images: list[dict[str, Image.Image]] = []
    # col_widths[col_name] = max width across all rows
    col_widths: dict[str, int] = {c: 0 for c in COLUMNS}
    row_heights: list[int] = []

    for delta in deltas:
        sid = delta["sample_id"]
        paths = image_map.get(sid, {})
        row_imgs: dict[str, Image.Image] = {}

        for col in COLUMNS[:-1]:  # skip delta_label — generated later
            img_path = paths.get(col)
            if img_path is not None and Path(img_path).exists():
                img = _load_and_resize(Path(img_path), portrait_h, landscape_w)
            else:
                # Placeholder for missing images
                img = Image.new(
                    "RGB",
                    (landscape_w // 2, portrait_h // 2),
                    color=(200, 200, 200),
                )
            row_imgs[col] = img
            col_widths[col] = max(col_widths[col], img.size[0])

        # Row height determined by the tallest image cell in the row
        row_h = max(img.size[1] for img in row_imgs.values())

        # delta_label column: same height as the tallest image in the row
        delta_img = _make_delta_label_image(delta["delta"], _DELTA_LABEL_WIDTH, row_h)
        row_imgs["delta_label"] = delta_img
        col_widths["delta_label"] = max(col_widths["delta_label"], _DELTA_LABEL_WIDTH)

        row_images.append(row_imgs)
        row_heights.append(row_h)

    # ------------------------------------------------------------------
    # Phase 2: compute canvas size and compose.
    # ------------------------------------------------------------------
    n_rows = len(row_images)
    n_cols = len(COLUMNS)

    total_w = sum(col_widths[c] for c in COLUMNS) + col_gap * (n_cols - 1)
    total_h = sum(row_heights) + row_gap * (n_rows - 1)

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))

    y_offset = 0
    for row_idx, (row_imgs, row_h) in enumerate(zip(row_images, row_heights)):
        x_offset = 0
        for col in COLUMNS:
            img = row_imgs[col]
            # Centre vertically within the row height
            y_cell = y_offset + (row_h - img.size[1]) // 2
            canvas.paste(img, (x_offset, y_cell))
            x_offset += col_widths[col] + col_gap
        y_offset += row_h + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def write_grid_metadata(
    metadata_path: Path,
    deltas: list[dict],
    grid_path: Path,
) -> None:
    """Write grid metadata as JSON.

    Schema::

        {
            "grid": "<absolute path to grid PNG>",
            "samples": [
                {"sample_id": ..., "prev": ..., "curr": ..., "delta": ...,
                 "category": ...},
                ...
            ]
        }
    """
    samples = [
        {
            "sample_id": d["sample_id"],
            "prev": d["prev"],
            "curr": d["curr"],
            "delta": d["delta"],
            "category": d.get("category"),
        }
        for d in deltas
    ]
    payload = {"grid": str(grid_path.resolve()), "samples": samples}
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# V0.2.1 multi-phrasing grid API
# ---------------------------------------------------------------------------

def compose_user_prompt_grid(
    rows: list[tuple[str, str]],
    cell_map: dict[tuple[str, str, str], dict],
    user_prompt_ids: list[str],
    output_path: Path,
    *,
    portrait_h: int = 1376,
    landscape_w: int = 768,
    row_gap: int = 4,
    col_gap: int = 4,
) -> Path:
    """Compose a multi-phrasing comparison grid and save as PNG.

    Each row corresponds to one ``(pair_id, sample_id)``.
    Each column corresponds to one ``user_prompt_id`` from ``user_prompt_ids``.
    The grid has a header row labelling each column with its ``user_prompt_id``
    and language tag.

    A failed cell (where ``cell_map[(pair_id, user_prompt_id, sample_id)]``
    contains a non-None ``failure_reason``) renders a placeholder tile with the
    failure reason text instead of an image, keeping the grid dense.

    Parameters
    ----------
    rows:
        Ordered list of ``(pair_id, sample_id)`` tuples defining the row order.
    cell_map:
        ``{(pair_id, user_prompt_id, sample_id): cell_dict}`` where
        ``cell_dict`` contains at minimum:
        - ``"image_path"``: ``Path | None`` — path to the result PNG
        - ``"failure_reason"``: ``str | None`` — set when the cell failed
        - ``"language"``: ``str | None`` — e.g. "zh", "en"
    user_prompt_ids:
        Ordered list of ``user_prompt_id`` values defining the column order
        (e.g. ``["zh_001", "zh_003", "en_001", "en_003"]``).
    output_path:
        Destination path for the output PNG.
    portrait_h:
        Target height for portrait images.
    landscape_w:
        Target width for landscape/square images.
    row_gap:
        Vertical gap between rows in pixels.
    col_gap:
        Horizontal gap between columns in pixels.

    Returns
    -------
    Path
        The ``output_path`` after the file has been written.
    """
    if not rows or not user_prompt_ids:
        img = Image.new("RGB", (1, 1), color=(0, 0, 0))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return output_path

    n_cols = len(user_prompt_ids)

    # ------------------------------------------------------------------
    # Phase 1: build cell images and measure dimensions.
    # ------------------------------------------------------------------
    # grid_imgs[row_idx][col_idx] = PIL Image
    grid_imgs: list[list[Image.Image]] = []
    col_widths: list[int] = [0] * n_cols
    row_heights: list[int] = []

    for pair_id, sample_id in rows:
        row_imgs: list[Image.Image] = []
        for col_idx, up_id in enumerate(user_prompt_ids):
            cell = cell_map.get((pair_id, up_id, sample_id))
            if cell is None:
                # Treat fully-absent cell as failed with unknown reason
                tile = _make_text_tile(
                    "missing",
                    landscape_w // 2,
                    portrait_h // 2,
                    _FAILED_BG,
                    _FAILED_TEXT_COLOUR,
                )
            elif cell.get("failure_reason"):
                reason = cell["failure_reason"] or "failed"
                tile = _make_text_tile(
                    reason,
                    landscape_w // 2,
                    portrait_h // 2,
                    _FAILED_BG,
                    _FAILED_TEXT_COLOUR,
                )
            else:
                img_path = cell.get("image_path")
                if img_path is not None and Path(img_path).exists():
                    tile = _load_and_resize(Path(img_path), portrait_h, landscape_w)
                else:
                    tile = _make_text_tile(
                        "no image",
                        landscape_w // 2,
                        portrait_h // 2,
                        (220, 220, 220),
                        (60, 60, 60),
                    )
            row_imgs.append(tile)
            col_widths[col_idx] = max(col_widths[col_idx], tile.size[0])

        row_h = max(t.size[1] for t in row_imgs)
        grid_imgs.append(row_imgs)
        row_heights.append(row_h)

    # ------------------------------------------------------------------
    # Phase 2: build header row images.
    # ------------------------------------------------------------------
    header_imgs: list[Image.Image] = []
    for col_idx, up_id in enumerate(user_prompt_ids):
        header_tile = _make_text_tile(
            up_id,
            col_widths[col_idx],
            _HEADER_HEIGHT,
            _HEADER_BG,
            _HEADER_TEXT_COLOUR,
        )
        header_imgs.append(header_tile)

    # ------------------------------------------------------------------
    # Phase 3: compose canvas.
    # ------------------------------------------------------------------
    n_rows = len(rows)
    total_w = sum(col_widths) + col_gap * (n_cols - 1)
    total_h = _HEADER_HEIGHT + row_gap + sum(row_heights) + row_gap * (n_rows - 1)

    canvas = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))

    # Draw header
    x_offset = 0
    for col_idx, header_tile in enumerate(header_imgs):
        canvas.paste(header_tile, (x_offset, 0))
        x_offset += col_widths[col_idx] + col_gap

    # Draw data rows
    y_offset = _HEADER_HEIGHT + row_gap
    for row_idx, (row_imgs, row_h) in enumerate(zip(grid_imgs, row_heights)):
        x_offset = 0
        for col_idx, tile in enumerate(row_imgs):
            y_cell = y_offset + (row_h - tile.size[1]) // 2
            canvas.paste(tile, (x_offset, y_cell))
            x_offset += col_widths[col_idx] + col_gap
        y_offset += row_h + row_gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def write_grid_meta_yaml(
    meta_path: Path,
    *,
    pair_id: str,
    round_id: int,
    user_prompt_ids: list[str],
    sample_ids: list[str],
    corpus_hash: str,
) -> None:
    """Write a ``grid_meta.yaml`` sidecar file next to the grid PNG.

    Schema::

        pair_id: str
        round_id: int
        user_prompt_ids: [str, ...]
        sample_ids: [str, ...]
        corpus_hash: str

    Parameters
    ----------
    meta_path:
        Destination path for the YAML file (conventionally ``grid_meta.yaml``
        adjacent to the grid PNG).
    pair_id:
        The prompt-pair ID whose cells are shown in the grid.
    round_id:
        The round number at which the grid was produced.
    user_prompt_ids:
        Ordered list of user_prompt_ids corresponding to grid columns.
    sample_ids:
        Ordered list of sample_ids corresponding to grid rows.
    corpus_hash:
        SHA-256 hex digest of the enabled user-prompt corpus (pinned at
        ``S00.10``); used to detect corpus drift.
    """
    payload = {
        "pair_id": pair_id,
        "round_id": round_id,
        "user_prompt_ids": user_prompt_ids,
        "sample_ids": sample_ids,
        "corpus_hash": corpus_hash,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(yaml.dump(payload, default_flow_style=False, sort_keys=False))
