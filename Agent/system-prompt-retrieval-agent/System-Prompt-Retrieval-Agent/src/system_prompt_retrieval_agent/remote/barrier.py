"""All-workers-done barrier for remote stage validation.

Provides two entry points:
  - ``assert_all_workers_done`` — V0.2 legacy barrier (RemoteManifest).
  - ``barrier_cell_keyed``       — V0.2.1 cell-keyed barrier (StageManifest).

Plan reference: §11.5, S04.08.
"""
from __future__ import annotations

from typing import Optional

from system_prompt_retrieval_agent.schemas import RemoteManifest, StageManifest


class BarrierViolation(RuntimeError):
    """Raised when the all-workers-done barrier is not satisfied.

    Attributes
    ----------
    code:
        Short identifier for the violation type.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"[{code}] {detail}" if detail else code)
        self.code = code
        self.detail = detail


def assert_all_workers_done(
    manifest: RemoteManifest,
    *,
    allow_partial: bool = False,
    post_unload_vram_free_min_gib: Optional[float] = 70.0,
) -> None:
    """Assert that all workers completed successfully and VRAM was freed.

    This check is non-negotiable — it must be called after every stage
    response before proceeding to copy-back or scoring.

    Parameters
    ----------
    manifest:
        The manifest returned by the remote controller.
    allow_partial:
        If ``False`` (default), any ``manifest.errors > 0`` raises
        ``BarrierViolation('strict_has_errors')``.
    post_unload_vram_free_min_gib:
        Minimum free VRAM (GiB) expected on every GPU after model unload.
        Pass ``None`` to skip the VRAM check.

    Raises
    ------
    BarrierViolation
        - ``"count_mismatch"`` when ``ok + errors != total``.
        - ``"strict_has_errors"`` when ``allow_partial=False`` and ``errors > 0``.
        - ``"vram_leak_detected"`` when any GPU reports free VRAM below the
          threshold.
    """
    # 1. Count integrity (non-negotiable).
    if manifest.ok + manifest.errors != manifest.total:
        raise BarrierViolation(
            "count_mismatch",
            f"ok({manifest.ok}) + errors({manifest.errors}) = "
            f"{manifest.ok + manifest.errors} != total({manifest.total})",
        )

    # 2. Strict-mode error check.
    if not allow_partial and manifest.errors > 0:
        raise BarrierViolation(
            "strict_has_errors",
            f"allow_partial=False but manifest has {manifest.errors} error(s)",
        )

    # 3. VRAM leak check.
    if (
        post_unload_vram_free_min_gib is not None
        and manifest.vram_free_gib is not None
    ):
        for vram in manifest.vram_free_gib:
            if vram < post_unload_vram_free_min_gib:
                raise BarrierViolation(
                    "vram_leak_detected",
                    f"GPU reports only {vram:.1f} GiB free "
                    f"(threshold {post_unload_vram_free_min_gib:.1f} GiB) — "
                    "possible VRAM leak after model unload",
                )


# ---------------------------------------------------------------------------
# V0.2.1 cell-keyed barrier
# ---------------------------------------------------------------------------


def barrier_cell_keyed(
    stage_manifest: StageManifest,
    allow_partial: bool,
    enabled_user_prompt_ids: list[str],
    sample_universe_for_pair: dict[str, list[str]],
    manifest_purpose: Optional[str] = None,
) -> tuple[list[str], list[dict]]:
    """V0.2.1 cell-keyed all-workers-wait barrier (plan §11.5).

    A **cell** is ``(prompt_pair_id, user_prompt_id, sample_id)``.

    Parameters
    ----------
    stage_manifest:
        The V0.2.1 ``StageManifest`` returned by the remote controller.
    allow_partial:
        If ``False`` (strict mode, default), any enabled-cell failure moves
        the **entire pair** to ``failed_pairs[]``.  ``allow_partial=True``
        enables partial-mode where a pair survives if at least one
        ``(pair, user_prompt)`` cell succeeded.
    enabled_user_prompt_ids:
        Full list of enabled user-prompt IDs for the run.
    sample_universe_for_pair:
        Mapping ``{prompt_pair_id: [sample_id, ...]}`` of the samples that
        were dispatched for each pair.  Used to compute expected cell counts.
    manifest_purpose:
        ``None`` — no ``sample_manifest_path`` was sent; full Cartesian
        expected count is used.
        ``"resume_missing_cells"`` — expected count = number of dispatched
        manifest rows; omitted cells are assumed already valid on disk.
        ``"prior_stage_survivor_cells"`` — expected count = sum of
        surviving cells in the upstream stage manifest.

    Returns
    -------
    surviving_pairs:
        List of ``prompt_pair_id`` strings that passed the barrier.
    failed_pairs:
        List of dicts ``{prompt_pair_id, failure_reason}``.

    Raises
    ------
    BarrierViolation
        Raised only for surviving-pair violations (``errors > 0`` under
        strict mode) or lifecycle mismatches.  Non-empty ``failed_pairs``
        does **not** raise.
    """
    from system_prompt_retrieval_agent.remote.partition import (
        partition_stage_pairs,
    )

    # Phase 1: Validate the manifest's DECLARED surviving_pairs rigorously.
    # These are the pairs the remote controller claims survived.  We check
    # count integrity, rollup consistency, and strict-mode completeness.
    # If any violation is found in a declared surviving pair, we raise.
    declared_surviving = list(stage_manifest.surviving_pairs)

    for pair_id in declared_surviving:
        per_pair = stage_manifest.pairs.get(pair_id)
        if per_pair is None:
            raise BarrierViolation(
                "missing_pair_manifest",
                f"Pair '{pair_id}' listed in surviving_pairs but absent from pairs dict",
            )

        # (a) Pair-level rollup integrity.
        if per_pair.ok + per_pair.errors != per_pair.total:
            raise BarrierViolation(
                "count_mismatch",
                f"Pair '{pair_id}': ok({per_pair.ok}) + errors({per_pair.errors}) "
                f"!= total({per_pair.total})",
            )

        # (b) Per-user-prompt rollup must sum to pair totals.
        if per_pair.per_user_prompt:
            up_ok = sum(u.ok for u in per_pair.per_user_prompt.values())
            up_errors = sum(u.errors for u in per_pair.per_user_prompt.values())
            up_total = sum(u.total for u in per_pair.per_user_prompt.values())
            if up_ok != per_pair.ok or up_errors != per_pair.errors or up_total != per_pair.total:
                raise BarrierViolation(
                    "per_up_rollup_mismatch",
                    f"Pair '{pair_id}': per_user_prompt sums "
                    f"ok={up_ok}, errors={up_errors}, total={up_total} "
                    f"!= pair rollup ok={per_pair.ok}, errors={per_pair.errors}, "
                    f"total={per_pair.total}",
                )

        # (c) Strict-mode completeness; branch on manifest_purpose.
        if not allow_partial:
            expected_samples = sample_universe_for_pair.get(pair_id, [])
            n_up = len(enabled_user_prompt_ids)
            n_samp = len(expected_samples)

            if manifest_purpose == "resume_missing_cells":
                # Omitted cells assumed already valid on disk; no count check.
                # But any errors in dispatched cells are still violations.
                if per_pair.errors > 0:
                    raise BarrierViolation(
                        "strict_has_errors",
                        f"Pair '{pair_id}': resume_missing_cells but "
                        f"{per_pair.errors} error(s) in dispatched cells",
                    )

            elif manifest_purpose == "prior_stage_survivor_cells":
                # Listed rows = upstream survivor cells; all must succeed.
                if per_pair.errors > 0:
                    raise BarrierViolation(
                        "strict_has_errors",
                        f"Pair '{pair_id}': prior_stage_survivor_cells manifest "
                        f"but {per_pair.errors} error(s) found in surviving pair",
                    )

            else:
                # Full Cartesian dispatch.
                expected_total = n_up * n_samp
                if expected_total > 0 and per_pair.total < expected_total:
                    raise BarrierViolation(
                        "count_mismatch",
                        f"Pair '{pair_id}': expected {expected_total} cells "
                        f"({n_up} user_prompts × {n_samp} samples) "
                        f"but got total={per_pair.total}",
                    )
                if per_pair.errors > 0:
                    raise BarrierViolation(
                        "strict_has_errors",
                        f"Pair '{pair_id}': allow_partial=False but "
                        f"{per_pair.errors} error(s) in surviving pair",
                    )

    # Phase 2: Recompute the final surviving / failed split using the local
    # partition logic (may further split the declared-surviving set if local
    # policy is stricter, but typically matches the remote declaration after
    # Phase 1 validation passes).  Pass manifest_purpose so the partition
    # logic skips the count check for resume_missing_cells.
    surviving_pairs, failed_pairs = partition_stage_pairs(
        stage_manifest=stage_manifest,
        allow_partial=allow_partial,
        enabled_user_prompt_ids=enabled_user_prompt_ids,
        expected_samples_for_pair=sample_universe_for_pair,
        manifest_purpose=manifest_purpose,
    )

    return surviving_pairs, failed_pairs
