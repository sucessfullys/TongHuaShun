"""Validate a RemoteManifest (V0.2) or StageManifest (V0.2.1).

V0.2.1 extensions:
  - ``validate_stage_manifest`` validates V0.2.1 ``StageManifest`` objects,
    including ``pairs``, ``surviving_pairs``, ``failed_pairs``,
    ``user_prompt_corpus_hash`` (64-char hex), ``lifecycle_state_after``,
    and per-``user_prompt`` rollup consistency.
  - The legacy ``validate_manifest`` function is retained unchanged for
    V0.2 back-compat.

Plan reference: §6.4, §11.5, S04.05.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from system_prompt_retrieval_agent.schemas import RemoteManifest, StageManifest

# Canonical error codes surfaced by this module.
MANIFEST_VALIDATION_ERRORS: list[str] = [
    "count_mismatch",       # ok + errors != total
    "strict_has_errors",    # allow_partial=False but errors > 0
    "missing_output_file",  # verify_existing_files=True and file not present locally
    "duplicate_sample_id",
    "worker_not_done",      # at least one worker entry without terminal status
]

_CORPUS_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")

_WORKER_TERMINAL_STATUSES = {"ok", "error", "done"}


class ManifestValidationError(RuntimeError):
    """Raised when a RemoteManifest fails validation.

    Attributes
    ----------
    code:
        One of :data:`MANIFEST_VALIDATION_ERRORS`.
    detail:
        Human-readable description of the violation.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"[{code}] {detail}")
        self.code = code
        self.detail = detail


def validate_manifest(
    m: RemoteManifest,
    *,
    allow_partial: bool = False,
    require_all_gpu_workers: bool = True,
    local_output_root: Optional[Path] = None,
    verify_existing_files: bool = False,
) -> None:
    """Validate *m* and raise :exc:`ManifestValidationError` on the first violation.

    Parameters
    ----------
    m:
        The manifest to validate.
    allow_partial:
        If ``False`` (default), raise ``strict_has_errors`` when
        ``m.errors > 0``.
    require_all_gpu_workers:
        If ``True`` (default), every entry in ``m.workers`` must have a
        terminal status (``"ok"``, ``"error"``, or ``"done"``).
    local_output_root:
        Local directory root used with *verify_existing_files*.
    verify_existing_files:
        When ``True``, check that every entry with a non-empty ``output_path``
        exists under *local_output_root*.
    """
    # 1. Count integrity.
    if m.ok + m.errors != m.total:
        raise ManifestValidationError(
            "count_mismatch",
            f"ok({m.ok}) + errors({m.errors}) = {m.ok + m.errors} != total({m.total})",
        )

    # 2. Strict-mode error check.
    if not allow_partial and m.errors > 0:
        raise ManifestValidationError(
            "strict_has_errors",
            f"allow_partial=False but manifest has {m.errors} error(s)",
        )

    # 3. Duplicate sample_id check.
    seen: set[str] = set()
    for entry in m.entries:
        if entry.sample_id in seen:
            raise ManifestValidationError(
                "duplicate_sample_id",
                f"sample_id '{entry.sample_id}' appears more than once",
            )
        seen.add(entry.sample_id)

    # 4. Worker terminal-status check.
    if require_all_gpu_workers:
        for worker in m.workers:
            status = worker.get("status", "")
            if status not in _WORKER_TERMINAL_STATUSES:
                raise ManifestValidationError(
                    "worker_not_done",
                    f"Worker {worker} has non-terminal status '{status}'",
                )

    # 5. Optional local file existence check.
    if verify_existing_files and local_output_root is not None:
        for entry in m.entries:
            if entry.output_path:
                full_path = local_output_root / entry.output_path
                if not full_path.exists():
                    raise ManifestValidationError(
                        "missing_output_file",
                        f"Expected output file not found: {full_path}",
                    )


# ---------------------------------------------------------------------------
# V0.2.1 stage manifest validator
# ---------------------------------------------------------------------------


def validate_stage_manifest(
    m: StageManifest,
    *,
    allow_partial: bool = False,
    require_corpus_hash: bool = True,
) -> None:
    """Validate a V0.2.1 ``StageManifest`` and raise on the first violation.

    Checks:
    1. ``surviving_pairs`` ⊆ ``pairs.keys()``.
    2. ``failed_pairs[*].prompt_pair_id`` ⊆ ``pairs.keys()``.
    3. ``surviving_pairs ∩ failed_pair_ids == ∅``.
    4. ``surviving_pairs ∪ failed_pair_ids == pairs.keys()``.
    5. Per-pair rollup integrity: ``ok + errors == total`` for every pair.
    6. Per-user-prompt rollup sums match pair totals for every pair.
    7. No errors on surviving pairs when ``allow_partial=False``.
    8. ``user_prompt_corpus_hash`` is present and matches
       ``/^[0-9a-fA-F]{64}$/`` when ``require_corpus_hash=True``.
    9. ``lifecycle_state_after`` is a known value when present.

    Parameters
    ----------
    m:
        The V0.2.1 stage manifest to validate.
    allow_partial:
        If ``False`` (default), any surviving pair with ``errors > 0``
        raises ``strict_has_errors``.
    require_corpus_hash:
        If ``True`` (default), ``user_prompt_corpus_hash`` must be
        present and a valid 64-char hex string.

    Raises
    ------
    ManifestValidationError
        On the first violation found.
    """
    # 8. Corpus hash format.
    if require_corpus_hash:
        ch = m.user_prompt_corpus_hash
        if not ch:
            raise ManifestValidationError(
                "missing_corpus_hash",
                "user_prompt_corpus_hash is absent but require_corpus_hash=True",
            )
        if not _CORPUS_HASH_RE.match(ch):
            raise ManifestValidationError(
                "invalid_corpus_hash_format",
                f"user_prompt_corpus_hash '{ch}' is not a 64-char hex string",
            )

    # 9. lifecycle_state_after.
    _VALID_LIFECYCLE_STATES = {
        "disk_unloaded", "cold", "cpu_prefetched", "gpu_loaded",
        "gpu_unloaded_cpu_retained",
    }
    if m.lifecycle_state_after is not None:
        if m.lifecycle_state_after not in _VALID_LIFECYCLE_STATES:
            raise ManifestValidationError(
                "invalid_lifecycle_state",
                f"lifecycle_state_after '{m.lifecycle_state_after}' is not a known state",
            )

    pair_ids = set(m.pairs.keys())
    surviving_set = set(m.surviving_pairs)
    failed_pair_ids = {fp.get("prompt_pair_id", "") for fp in m.failed_pairs}

    # 1. surviving_pairs ⊆ pairs.keys()
    unknown_surviving = surviving_set - pair_ids
    if unknown_surviving:
        raise ManifestValidationError(
            "unknown_surviving_pair",
            f"surviving_pairs contains pair(s) not in pairs dict: {unknown_surviving}",
        )

    # 2. failed_pairs ⊆ pairs.keys()
    unknown_failed = failed_pair_ids - pair_ids
    if unknown_failed:
        raise ManifestValidationError(
            "unknown_failed_pair",
            f"failed_pairs contains pair(s) not in pairs dict: {unknown_failed}",
        )

    # 3. Disjoint.
    overlap = surviving_set & failed_pair_ids
    if overlap:
        raise ManifestValidationError(
            "pair_in_both_surviving_and_failed",
            f"pair(s) appear in both surviving and failed: {overlap}",
        )

    # 4. Union covers all pairs.
    all_accounted = surviving_set | failed_pair_ids
    unaccounted = pair_ids - all_accounted
    if unaccounted:
        raise ManifestValidationError(
            "pair_not_accounted",
            f"pair(s) in pairs dict not in surviving_pairs or failed_pairs: {unaccounted}",
        )

    # 5 & 6. Per-pair and per-user-prompt rollup integrity.
    for pair_id, per_pair in m.pairs.items():
        if per_pair.ok + per_pair.errors != per_pair.total:
            raise ManifestValidationError(
                "count_mismatch",
                f"Pair '{pair_id}': ok({per_pair.ok}) + errors({per_pair.errors}) "
                f"!= total({per_pair.total})",
            )

        if per_pair.per_user_prompt:
            up_ok = sum(u.ok for u in per_pair.per_user_prompt.values())
            up_err = sum(u.errors for u in per_pair.per_user_prompt.values())
            up_tot = sum(u.total for u in per_pair.per_user_prompt.values())
            if up_ok != per_pair.ok or up_err != per_pair.errors or up_tot != per_pair.total:
                raise ManifestValidationError(
                    "per_up_rollup_mismatch",
                    f"Pair '{pair_id}': per_user_prompt sums "
                    f"ok={up_ok}, errors={up_err}, total={up_tot} "
                    f"!= pair rollup ok={per_pair.ok}, errors={per_pair.errors}, "
                    f"total={per_pair.total}",
                )

    # 7. Strict mode: no errors on surviving pairs.
    if not allow_partial:
        for pair_id in m.surviving_pairs:
            per_pair = m.pairs[pair_id]
            if per_pair.errors > 0:
                raise ManifestValidationError(
                    "strict_has_errors",
                    f"Pair '{pair_id}' is in surviving_pairs but has "
                    f"{per_pair.errors} error(s) (allow_partial=False)",
                )
