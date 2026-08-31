"""Data-root probe — recursively discovers the dataset layout.

Walks each root to find **leaf sample directories** (dirs that directly hold
image files), groups them into generation **methods**, infers the relative
``sample_glob``, the co-located ``input_roles``, and each method's generated
``output_file``. The probe only *suggests* — the init slash-command confirms
ambiguous items with the operator, so it reports a ``confidence`` and ``notes``.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Keyword hints for tagging a role as input. Plain subject nouns (the depicted
# object or scene) are deliberately excluded — they are not role indicators.
_INPUT_HINTS = (
    "input", "source", "src", "origin", "orig", "ref", "reference",
    "cond", "condition", "content", "style", "mask", "prompt", "before",
)

_MAX_DEPTH = 8
_MAX_DIRS = 50000
_ROLE_SAMPLE_CAP = 64
# Files matching this are treated as intermediate artifacts, not final outputs.
_INTERMEDIATE_RE = re.compile(r".*_\d+\.[A-Za-z0-9]+$")


def probe_data_roots(
    roots: list[str], image_exts: list[str] | None = None
) -> dict:
    """Probe one or more candidate data roots.

    Returns ``data_root``, ``layout``, a ``methods`` list (each with
    ``method_id``/``path``/``output_file``/``output_candidates``/
    ``sample_count``), ``sample_glob``, ``input_roles``, ``sample_key``,
    ``confidence`` and ``notes``.
    """
    exts = {e.lower() for e in (image_exts or IMAGE_EXTS)}
    notes: list[str] = []

    resolved: list[Path] = []
    for raw in roots:
        path = Path(raw)
        if not path.is_dir():
            notes.append(f"root not found or not a directory: {raw}")
            continue
        resolved.append(path)

    result: dict = {
        "data_root": "",
        "layout": "unknown",
        "methods": [],
        "sample_glob": "",
        "sample_count": 0,
        "input_roles": {},
        "input_root": "",
        "sample_key": "relpath",
        "confidence": "needs_confirmation",
        "notes": notes,
        "scan": {"dirs_scanned": 0, "leaves_found": 0, "truncated": False},
    }

    if not resolved:
        notes.append("no usable data roots were given")
        return result

    # ---- discover leaf sample directories under every root ----
    dirs_scanned = 0
    truncated = False
    # method_groups: list of (method_dir, [(leaf, depth_within_method), ...])
    method_groups: list[tuple[Path, list[tuple[Path, int]]]] = []
    flat_root: Path | None = None

    if len(resolved) == 1:
        root = resolved[0]
        leaves, scanned, trunc = _find_leaf_dirs(root, exts)
        dirs_scanned += scanned
        truncated = truncated or trunc
        if leaves and all(depth == 0 for _, depth in leaves):
            flat_root = root
        else:
            method_groups = _group_into_methods(root, leaves, notes)
        result["data_root"] = str(root)
    else:
        # each root passed is its own method
        for root in resolved:
            leaves, scanned, trunc = _find_leaf_dirs(root, exts)
            dirs_scanned += scanned
            truncated = truncated or trunc
            within = [(leaf, depth) for leaf, depth in leaves if depth > 0]
            if within:
                method_groups.append((root, within))
            elif leaves:  # root itself holds images
                method_groups.append((root, [(root, 0)]))
        result["data_root"] = os.path.commonpath([str(r) for r in resolved])

    leaves_found = sum(len(g) for _, g in method_groups) + (
        1 if flat_root else 0
    )
    result["scan"] = {
        "dirs_scanned": dirs_scanned,
        "leaves_found": leaves_found,
        "truncated": truncated,
    }
    if truncated:
        notes.append(
            f"directory scan hit the {_MAX_DIRS}-dir cap — counts may be partial"
        )

    # ---- flat layout: a single directory of image files ----
    if flat_root is not None:
        images = [
            f for f in flat_root.iterdir()
            if f.is_file() and f.suffix.lower() in exts
        ]
        result["layout"] = "flat_files"
        result["sample_key"] = "basename"
        result["sample_count"] = len(images)
        result["methods"] = [{
            "method_id": _ident(flat_root.name),
            "path": str(flat_root),
            "output_file": "",
            "output_candidates": sorted(f.name for f in images[:20]),
            "sample_count": len(images),
        }]
        notes.append(
            "single flat directory of images — confirm the input root and how "
            "inputs pair to outputs"
        )
        return result

    if not method_groups:
        notes.append("no leaf sample directories found under the given root(s)")
        return result

    # ---- per-method role detection ----
    result["layout"] = "per_sample_dirs"
    method_files: dict[str, list[str]] = {}
    for method_dir, leaves in method_groups:
        method_files[str(method_dir)] = _method_files(
            [leaf for leaf, _ in leaves], exts
        )

    # input roles — primary signal: filename keyword hints; fallback: files
    # common to every method.
    all_files = set().union(*(set(v) for v in method_files.values()))
    keyword_inputs = sorted(
        f for f in all_files
        if _is_input_name(f) and all(f in mf for mf in method_files.values())
    )
    if keyword_inputs:
        inputs = set(keyword_inputs)
    elif len(method_files) >= 2:
        inputs = set.intersection(*(set(v) for v in method_files.values()))
        if inputs:
            notes.append(
                "input roles guessed as the files common to every method — "
                "confirm"
            )
    else:
        inputs = set()
        notes.append(
            "could not identify input images by filename — confirm input roles"
        )
    input_files = sorted(inputs)
    result["input_roles"] = {
        _ident(Path(name).stem): name for name in input_files
    }

    # depth -> sample_glob
    depths = [d for _, leaves in method_groups for _, d in leaves]
    modal_depth = Counter(depths).most_common(1)[0][0] if depths else 1
    if len(set(depths)) > 1:
        notes.append(
            f"sample directories sit at varying depths {sorted(set(depths))}; "
            f"using the most common ({modal_depth})"
        )
    result["sample_glob"] = "/".join(["*"] * modal_depth) if modal_depth else "*"

    methods: list[dict] = []
    all_single_output = True
    for method_dir, leaves in method_groups:
        files = method_files[str(method_dir)]
        candidates = [f for f in files if f not in inputs]
        output_file, single = _pick_output(candidates, notes, method_dir.name)
        all_single_output = all_single_output and single
        methods.append({
            "method_id": _ident(method_dir.name),
            "path": str(method_dir),
            "output_file": output_file,
            "output_candidates": candidates,
            "sample_count": len(leaves),
        })
    result["methods"] = methods

    counts = {m["sample_count"] for m in methods}
    result["sample_count"] = max(counts) if counts else 0
    if len(counts) > 1:
        notes.append(
            f"methods have differing sample counts: "
            f"{ {m['method_id']: m['sample_count'] for m in methods} }"
        )

    if (
        not truncated
        and input_files
        and all_single_output
        and len(set(depths)) <= 1
    ):
        result["confidence"] = "high"
    else:
        notes.append("confirm input roles and per-method output files")
    return result


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _find_leaf_dirs(
    root: Path, exts: set[str]
) -> tuple[list[tuple[Path, int]], int, bool]:
    """DFS for directories that directly contain image files.

    Returns ``(leaves, dirs_scanned, truncated)`` where each leaf is
    ``(path, depth)`` and depth is measured from ``root`` (root == 0).
    """
    leaves: list[tuple[Path, int]] = []
    scanned = 0
    truncated = False
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        scanned += 1
        if scanned > _MAX_DIRS:
            truncated = True
            break
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        has_image = any(
            e.is_file() and e.suffix.lower() in exts for e in entries
        )
        if has_image:
            leaves.append((directory, depth))
            continue  # samples do not nest — stop descending
        if depth >= _MAX_DEPTH:
            continue
        for entry in entries:
            if entry.is_dir() and not entry.name.startswith("."):
                stack.append((entry, depth + 1))
    return leaves, scanned, truncated


def _group_into_methods(
    root: Path, leaves: list[tuple[Path, int]], notes: list[str]
) -> list[tuple[Path, list[tuple[Path, int]]]]:
    """Group leaves under a single root into per-method (method_dir, leaves)."""
    if not leaves:
        return []

    direct = [(leaf, d) for leaf, d in leaves if d == 1]
    nested = [(leaf, d) for leaf, d in leaves if d >= 2]

    # samples are direct children -> single method == root
    if len(direct) >= len(nested):
        return [(root, [(leaf, 1) for leaf, _ in direct])]

    # nested: group by the first path component under root
    groups: dict[str, list[tuple[Path, int]]] = {}
    for leaf, depth in nested:
        first = leaf.relative_to(root).parts[0]
        # depth within the method dir = depth from root - 1
        groups.setdefault(first, []).append((leaf, depth - 1))

    if len(groups) >= 2:
        return [(root / name, groups[name]) for name in sorted(groups)]

    # a single nested group -> root is one method, keep full relative depth
    only = next(iter(groups))
    notes.append(
        f"one nested branch ({only}/) under the root — treating the root as a "
        f"single method"
    )
    return [(root, [(leaf, d + 1) for leaf, d in groups[only]])]


def _method_files(leaves: list[Path], exts: set[str]) -> list[str]:
    """Image filenames consistently present across a method's sample dirs."""
    per_leaf: list[list[str]] = []
    for leaf in leaves[:_ROLE_SAMPLE_CAP]:
        try:
            names = sorted(
                f.name for f in leaf.iterdir()
                if f.is_file() and f.suffix.lower() in exts
            )
        except OSError:
            names = []
        if names:
            per_leaf.append(names)
    if not per_leaf:
        return []

    n = len(per_leaf)
    freq: Counter[str] = Counter()
    for names in per_leaf:
        for name in set(names):
            freq[name] += 1
    threshold = max(1, int(0.6 * n))
    common = sorted(name for name, c in freq.items() if c >= threshold)
    if common and len(common) == len(per_leaf[0]):
        return common  # identical filename set across sample dirs

    # filenames vary across dirs -> collapse each position into a glob
    globs: list[str] = []
    max_len = max(len(names) for names in per_leaf)
    for i in range(max_len):
        column = [names[i] for names in per_leaf if len(names) > i]
        globs.append(_glob_of(column))
    return globs


def _pick_output(
    candidates: list[str], notes: list[str], method_name: str
) -> tuple[str, bool]:
    """Choose a method's output file. Returns ``(output_file, is_unambiguous)``."""
    if not candidates:
        notes.append(f"method {method_name!r}: no generated-output file found")
        return "", False
    if len(candidates) == 1:
        return candidates[0], True
    # multiple candidates — drop intermediate artifacts (trailing _<digit>)
    finals = [c for c in candidates if not _INTERMEDIATE_RE.match(c)]
    if len(finals) == 1:
        notes.append(
            f"method {method_name!r}: {len(candidates)} output candidates; "
            f"guessed {finals[0]!r} (others look like intermediates) — confirm"
        )
        return finals[0], False
    notes.append(
        f"method {method_name!r}: multiple output candidates {candidates} — "
        f"confirm which is the final output"
    )
    return "", False


def _glob_of(names: list[str]) -> str:
    """Collapse filenames into a glob by keeping common prefix/suffix."""
    if not names:
        return "*"
    if len(set(names)) == 1:
        return names[0]
    prefix = os.path.commonprefix(names)
    suffix = os.path.commonprefix([n[::-1] for n in names])[::-1]
    shortest = min(len(n) for n in names)
    if prefix and suffix and len(prefix) + len(suffix) < shortest:
        return f"{prefix}*{suffix}"
    if suffix and len(suffix) < shortest:
        return f"*{suffix}"
    return "*"


def _is_input_name(filename: str) -> bool:
    """True if a filename looks like an input image (keyword heuristic)."""
    stem = Path(filename).stem.lower()
    return any(hint in stem for hint in _INPUT_HINTS)


def _ident(text: str) -> str:
    """Normalize a directory/file stem into an identifier."""
    ident = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return ident or "item"
