"""Stage 0 annotation probe — discover pre-existing operator annotations.

When the dataset already has annotations from a prior ``/era:annotate`` run
(the central ``<data_root>/annotations/<sample_key>.json`` files plus the
per-method ``<method>/<sample_key>/annotation.json`` mirror copies), this
probe surfaces them at ``/era:init`` so:

- the operator sees how many samples are already covered (post-init guide);
- Stage 2's brainstorm sub-agents (Phase B) can read the operator's
  problem notes verbatim and design evaluators that catch *known*
  failure modes;
- Phase C's auto-validation gate knows which sample_keys form the
  annotated benchmark set.

Phase A scope: deterministic Python only — no LLM theme clustering.
The probe returns raw counts plus a small ``sample_notes_preview``
(operator-readable, capped). LLM clustering lands in Phase B alongside
the Stage 2 sub-agent dispatch, which already has the infrastructure for
sub-agent calls.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# Filenames the /era:annotate writer creates (mirrors era/annotate/store.py).
CENTRAL_DIR = "annotations"
ANNOTATION_SUFFIX = ".json"
PER_METHOD_FILENAME = "annotation.json"

# How many sample notes to include verbatim in the post-init guide preview.
# Small enough to keep the guide readable; large enough to give a sense.
_PREVIEW_SAMPLES = 5

# Cap on ``samples_with_any_note`` written to ``probe/annotations.json``.
# The full list isn't load-bearing — ``central_count`` and
# ``method_coverage`` carry the counts; downstream consumers that need
# the full set can re-walk ``<dataset>/annotations/``. Capping the disk
# artifact keeps Stage 2's context window bounded when the operator
# annotates a 10K-sample dataset.
SAMPLES_WITH_NOTE_CAP = 100


def probe_annotations(
    data_root: str | Path,
    method_paths: dict[str, Path | str] | None = None,
) -> dict:
    """Walk the dataset's annotation files; report counts + a small preview.

    ``data_root`` — the dataset root (same path /era:annotate operates on).
    ``method_paths`` — ``{method_id: method_dir}``; when supplied, the
    probe also counts the per-method mirror copies under each method dir
    so the post-init guide can flag a sync drift between central and
    per-method state.

    Returns the shape::

        {
          "data_root": str,
          "central_dir": str,                # absolute
          "central_count": int,              # # of <data_root>/annotations/**/*.json files
          "per_method_count": {method_id: int, ...},  # # of <method>/<sample>/annotation.json
          "per_method_count_total": int,
          "annotators_in_sync": bool,        # central_count * methods == sum(per_method)
          "samples_with_any_note": [sample_key, ...],   # union, sorted
          "method_coverage": {method_id: int, ...},  # # samples annotated per method
          "sample_notes_preview": [
            {"sample_key": str, "per_method": {method_id: note, ...}},
            ...
          ],
        }

    Missing annotations directory is *not* an error: every count is 0
    and the caller (init flow) reports "no pre-existing annotations".
    """
    root = Path(data_root).resolve()
    central = root / CENTRAL_DIR
    result: dict = {
        "data_root": str(root),
        "central_dir": str(central),
        "central_count": 0,
        "per_method_count": {},
        "per_method_count_total": 0,
        "annotators_in_sync": True,
        "samples_with_any_note": [],
        "samples_with_any_note_total": 0,
        "samples_with_any_note_truncated": False,
        "method_coverage": {},
        "sample_notes_preview": [],
    }

    # Empty / missing directory is the "no annotations yet" base case.
    if not central.is_dir():
        if method_paths:
            result["per_method_count"] = {
                method_id: 0 for method_id in method_paths
            }
            result["method_coverage"] = {
                method_id: 0 for method_id in method_paths
            }
        return result

    # Walk central files. Each carries the canonical per_method dict.
    samples: list[str] = []
    method_coverage: Counter = Counter()
    previews: list[dict] = []
    for path in sorted(central.rglob("*" + ANNOTATION_SUFFIX)):
        if not path.is_file():
            continue
        # Skip the cache files the annotate app may drop later.
        if path.name.startswith("__"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        sample_key = data.get("sample_key")
        per_method = data.get("per_method")
        if not sample_key or not isinstance(per_method, dict):
            continue
        # Only count samples whose per_method has at least one non-empty slot.
        non_empty = {
            method_id: text for method_id, text in per_method.items()
            if isinstance(text, str) and text.strip()
        }
        if not non_empty:
            continue
        samples.append(sample_key)
        for method_id in non_empty:
            method_coverage[method_id] += 1
        if len(previews) < _PREVIEW_SAMPLES:
            previews.append({
                "sample_key": sample_key,
                "per_method": non_empty,
            })

    result["central_count"] = len(samples)
    sorted_samples = sorted(samples)
    result["samples_with_any_note"] = sorted_samples[:SAMPLES_WITH_NOTE_CAP]
    result["samples_with_any_note_total"] = len(sorted_samples)
    result["samples_with_any_note_truncated"] = (
        len(sorted_samples) > SAMPLES_WITH_NOTE_CAP
    )
    result["method_coverage"] = dict(method_coverage)
    result["sample_notes_preview"] = previews

    # Count per-method mirror copies when the caller passed method_paths.
    # The mirror is what an agent walking <method>/<sample>/ would find.
    if method_paths:
        per_method_count: dict[str, int] = {}
        for method_id, method_dir in method_paths.items():
            mdir = Path(method_dir).resolve()
            if not mdir.is_dir():
                per_method_count[method_id] = 0
                continue
            n = sum(
                1 for p in mdir.rglob(PER_METHOD_FILENAME)
                if p.is_file() and _has_real_annotation(p)
            )
            per_method_count[method_id] = n
        result["per_method_count"] = per_method_count
        result["per_method_count_total"] = sum(per_method_count.values())
        # Sync check: total per-method copies should equal sum(method_coverage)
        # (each (sample, method) populated slot writes one mirror copy).
        result["annotators_in_sync"] = (
            result["per_method_count_total"] == sum(method_coverage.values())
        )
        # Ensure method_coverage has an entry per method (zero if unannotated).
        for method_id in method_paths:
            result["method_coverage"].setdefault(method_id, 0)
    return result


def _has_real_annotation(path: Path) -> bool:
    """True if a per-method copy file carries a non-empty ``annotation`` slot.

    Stale files (annotation cleared but file not deleted by some
    out-of-band edit) are treated as "no annotation" for counting.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    annotation = data.get("annotation")
    return isinstance(annotation, str) and bool(annotation.strip())
