"""Production ``--resume-from-run-id`` validator (S00.16a / plan §2.10c).

When the CLI is invoked with ``--resume-from-run-id RUN_ID``, the new
invocation reuses ``RUN_ID`` as its own ``run_id``. Before any remote
dispatch and before any copy-back, this module:

1. Validates ``RUN_ID`` against the canonical regex (§2.10a).
2. Confirms the prior run directory exists locally with at least one
   prior stage manifest under ``manifests/``.
3. Reads each prior stage manifest and confirms **all four** prior
   hashes match the current invocation's hashes:

   * ``config_hash``
   * ``user_prompt_corpus_hash``
   * ``prompt_pair_corpus_hash``
   * ``sample_corpus_hash``

   On the first divergence, raises :class:`ResumeDriftError` with the
   message ``"resume corpus/config drift: <hash_name> differs"``.
4. Confirms cross-manifest consistency (a single prior run cannot have
   manifests recording two different ``config_hash`` values, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .remote._vendored import canonical_paths as cp


HASH_FIELDS_TO_VALIDATE: tuple[str, ...] = (
    "config_hash",
    "user_prompt_corpus_hash",
    "prompt_pair_corpus_hash",
    "sample_corpus_hash",
)


@dataclass(frozen=True)
class CurrentRunHashes:
    """Four canonical hashes of the new invocation."""

    config_hash: str
    user_prompt_corpus_hash: str
    prompt_pair_corpus_hash: str
    sample_corpus_hash: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "config_hash": self.config_hash,
            "user_prompt_corpus_hash": self.user_prompt_corpus_hash,
            "prompt_pair_corpus_hash": self.prompt_pair_corpus_hash,
            "sample_corpus_hash": self.sample_corpus_hash,
        }


class ResumeFromRunIdError(ValueError):
    """Base class for ``--resume-from-run-id`` validation failures."""


class ResumeRunIdInvalid(ResumeFromRunIdError):
    """Supplied RUN_ID does not match the canonical regex."""


class ResumeRunDirMissing(ResumeFromRunIdError):
    """Prior run directory is absent or has no prior stage manifests."""


class ResumeDriftError(ResumeFromRunIdError):
    """One of the four canonical hashes diverged from the prior run."""


def _iter_prior_stage_manifests(manifests_dir: Path) -> Iterable[Path]:
    """Yield prior per-attempt and pointer stage manifests in stable order."""
    if not manifests_dir.is_dir():
        return iter(())
    files = [
        p
        for p in manifests_dir.iterdir()
        if p.is_file() and p.suffix == ".json" and p.name != "_canonical_paths_provenance.json"
    ]
    files.sort()
    return iter(files)


def _read_manifest_hashes(path: Path) -> dict[str, str]:
    with path.open("rb") as fh:
        data = json.load(fh)
    out: dict[str, str] = {}
    for key in HASH_FIELDS_TO_VALIDATE:
        if key in data:
            out[key] = data[key]
    return out


def validate_resume_from_run_id(
    run_id: str,
    *,
    output_root: Path,
    current_hashes: CurrentRunHashes,
) -> Path:
    """Validate a ``--resume-from-run-id`` request.

    Returns the resolved prior run directory path on success. Raises a
    :class:`ResumeFromRunIdError` subclass on any failure.
    """
    if not isinstance(run_id, str) or not cp.RUN_ID_REGEX.match(run_id):
        raise ResumeRunIdInvalid(
            f"--resume-from-run-id RUN_ID does not match {cp.RUN_ID_REGEX.pattern!r}: "
            f"{run_id!r}"
        )

    prior_run_dir = Path(output_root) / cp.OUTPUTS_ROOT.split("/", 1)[1] / run_id \
        if str(output_root).endswith("outputs") else Path(output_root) / run_id

    # Standard layout: <output_root>/<OUTPUTS_ROOT>/<run_id>/manifests/
    # We accept either:
    #   • <output_root> already pointing at OUTPUTS_ROOT, or
    #   • <output_root> being the project output root, in which case we
    #     append OUTPUTS_ROOT.
    candidate_a = Path(output_root) / run_id / "manifests"
    candidate_b = Path(output_root) / cp.OUTPUTS_ROOT / run_id / "manifests"
    if candidate_a.is_dir():
        manifests_dir = candidate_a
        prior_run_dir = candidate_a.parent
    elif candidate_b.is_dir():
        manifests_dir = candidate_b
        prior_run_dir = candidate_b.parent
    else:
        raise ResumeRunDirMissing(
            f"prior run directory for RUN_ID={run_id!r} not found under "
            f"{output_root}; expected {candidate_a} or {candidate_b}"
        )

    prior_manifests = list(_iter_prior_stage_manifests(manifests_dir))
    if not prior_manifests:
        raise ResumeRunDirMissing(
            f"prior run directory {prior_run_dir} has no stage manifests under "
            f"{manifests_dir}; resume requires at least one."
        )

    # Cross-manifest consistency + drift detection.
    seen_per_field: dict[str, str] = {}
    for manifest_path in prior_manifests:
        try:
            prior_hashes = _read_manifest_hashes(manifest_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ResumeFromRunIdError(
                f"prior stage manifest unreadable: {manifest_path} ({exc})"
            ) from exc
        for field in HASH_FIELDS_TO_VALIDATE:
            if field not in prior_hashes:
                # Tolerated: not every manifest must declare every hash
                # in older partial states; cross-manifest comparison
                # below catches drift for fields that are present.
                continue
            prior_value = prior_hashes[field]
            recorded = seen_per_field.get(field)
            if recorded is None:
                seen_per_field[field] = prior_value
            elif recorded != prior_value:
                raise ResumeDriftError(
                    f"resume corpus/config drift: {field} differs "
                    f"(prior run has inconsistent {field}: {recorded} vs {prior_value})"
                )

    current = current_hashes.as_mapping()
    for field in HASH_FIELDS_TO_VALIDATE:
        if field not in seen_per_field:
            raise ResumeDriftError(
                f"resume corpus/config drift: {field} differs "
                f"(prior run did not record {field}; cannot verify against current)"
            )
        if seen_per_field[field] != current[field]:
            raise ResumeDriftError(
                f"resume corpus/config drift: {field} differs"
            )

    return prior_run_dir
