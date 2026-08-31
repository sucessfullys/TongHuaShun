"""Resume helpers for V0.2.1 surgical per-cell re-dispatch.

Plan reference: §6.4, §6.5, §7 step [3a][4a][5a], §17.3, S04.09.

Two manifest builders:
  - ``build_missing_manifest`` — scans local artifacts for missing/invalid
    cells and writes JSONL rows for surgical re-dispatch with
    ``sample_manifest_path_purpose="resume_missing_cells"``.
  - ``build_survivor_manifest`` — emits JSONL rows for every surviving cell
    from a prior stage manifest, for hand-off with
    ``sample_manifest_path_purpose="prior_stage_survivor_cells"``.

Output layout per §6.5:
  gemma  → {stage_root}/round_{round_id}/{pair_id}/{up_id}/{sample_id}/intermediate_prompt.txt
  flux   → {stage_root}/round_{round_id}/{pair_id}/{up_id}/{sample_id}/result.png
  qwen   → {stage_root}/round_{round_id}/{pair_id}/{up_id}/{sample_id}/eval.json
"""
from __future__ import annotations

import json
from pathlib import Path

from system_prompt_retrieval_agent.schemas import StageManifest

# Artifact filename per stage.
_ARTIFACT_FILENAME: dict[str, str] = {
    "gemma": "intermediate_prompt.txt",
    "flux": "result.png",
    "qwen": "eval.json",
}


def _artifact_path(
    local_artifacts_root: Path,
    stage: str,
    round_id: str,
    prompt_pair_id: str,
    user_prompt_id: str,
    sample_id: str,
) -> Path:
    """Return the local artifact path for a cell per §6.5."""
    filename = _ARTIFACT_FILENAME.get(stage)
    if filename is None:
        raise ValueError(f"Unknown stage '{stage}'; must be gemma, flux, or qwen")
    return (
        local_artifacts_root
        / stage
        / f"round_{round_id}"
        / prompt_pair_id
        / user_prompt_id
        / sample_id
        / filename
    )


def build_missing_manifest(
    local_artifacts_root: Path,
    prompt_pair_ids: list[str],
    user_prompt_ids: list[str],
    sample_ids: list[str],
    stage: str,
    *,
    round_id: str = "1",
    output_path: Path | None = None,
) -> Path:
    """Scan local artifacts and write a JSONL manifest of missing cells.

    For each ``(prompt_pair_id, user_prompt_id, sample_id)`` triple in the
    full Cartesian product, the cell is **missing** when its required artifact
    file does not exist on disk.  Cells that already have a valid artifact on
    disk are **omitted** from the JSONL — the caller passes this manifest with
    ``sample_manifest_path_purpose="resume_missing_cells"`` so the controller
    dispatches only the missing rows.

    Parameters
    ----------
    local_artifacts_root:
        Root directory under which artifacts are stored per §6.5.
    prompt_pair_ids:
        All prompt-pair IDs for this stage dispatch.
    user_prompt_ids:
        All enabled user-prompt IDs.
    sample_ids:
        All expected sample IDs.
    stage:
        One of ``"gemma"``, ``"flux"``, ``"qwen"``.
    round_id:
        The round identifier string (e.g. ``"3"``).
    output_path:
        Where to write the JSONL file.  Defaults to
        ``{local_artifacts_root}/manifests/resume_{stage}_round_{round_id}.jsonl``.

    Returns
    -------
    Path
        Path to the written JSONL file.
    """
    if output_path is None:
        manifests_dir = local_artifacts_root / "manifests"
        manifests_dir.mkdir(parents=True, exist_ok=True)
        output_path = manifests_dir / f"resume_{stage}_round_{round_id}.jsonl"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for pair_id in prompt_pair_ids:
        for up_id in user_prompt_ids:
            for sample_id in sample_ids:
                artifact = _artifact_path(
                    local_artifacts_root, stage, round_id, pair_id, up_id, sample_id
                )
                if not artifact.exists():
                    rows.append(
                        {
                            "prompt_pair_id": pair_id,
                            "user_prompt_id": up_id,
                            "sample_id": sample_id,
                        }
                    )

    with open(output_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path


def build_survivor_manifest(
    prior_stage_manifest: StageManifest,
    *,
    output_path: Path | None = None,
    local_manifests_root: Path | None = None,
) -> Path:
    """**Hard-deprecated in V0.2.2 (S02.01).** Emits placeholder
    ``sample_id="__all__"`` rows that the V0.2.2 contract forbids
    (plan §2.5 / S00.07). Use
    ``system_prompt_retrieval_agent.survivor_resume_v022.build_survivor_manifest_v022``
    which works directly off cell-keyed merged-view manifests.

    The legacy function is preserved only for ``legacy_cartesian_compat``
    replay tooling. Calling it from any production code path raises
    ``RuntimeError`` to prevent accidental drift.

    Build a JSONL of surviving cells from *prior_stage_manifest*.

    Iterates over every pair in ``prior_stage_manifest.surviving_pairs`` and
    for each enabled ``(user_prompt_id, sample_id)`` cell that has ``ok > 0``
    in the pair's ``per_user_prompt`` rollup, emits one JSONL row.

    The caller passes this manifest with
    ``sample_manifest_path_purpose="prior_stage_survivor_cells"`` on the next
    stage POST so the controller dispatches exactly these cells.

    Parameters
    ----------
    prior_stage_manifest:
        The stage manifest from the previous stage (e.g. Gemma manifest when
        building the FLUX survivor manifest).
    output_path:
        Explicit destination path.  When ``None``, a default path is derived
        from ``local_manifests_root``.
    local_manifests_root:
        Used only when *output_path* is ``None``.  The file is written to
        ``{local_manifests_root}/survivors_{stage}_round_{round_id}.jsonl``.

    Returns
    -------
    Path
        Path to the written JSONL file.

    Raises
    ------
    ValueError
        When both *output_path* and *local_manifests_root* are ``None``.
    """
    import os as _os
    if _os.environ.get("SPRA_LEGACY_SURVIVOR_MANIFEST_OK") != "1":
        raise RuntimeError(
            "build_survivor_manifest is hard-deprecated under V0.2.2 (S02.01); "
            "it emits placeholder sample_id='__all__' rows that violate the "
            "V0.2.2 cell-keyed contract (plan §2.5 / S00.07). Use "
            "system_prompt_retrieval_agent.survivor_resume_v022."
            "build_survivor_manifest_v022 instead. Set "
            "SPRA_LEGACY_SURVIVOR_MANIFEST_OK=1 only for legacy_cartesian_compat "
            "replay tooling."
        )
    if output_path is None:
        if local_manifests_root is None:
            raise ValueError(
                "Provide either output_path or local_manifests_root"
            )
        local_manifests_root.mkdir(parents=True, exist_ok=True)
        stage = prior_stage_manifest.stage
        round_id = prior_stage_manifest.round_id or "1"
        output_path = (
            local_manifests_root
            / f"survivors_{stage}_round_{round_id}.jsonl"
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    surviving = set(prior_stage_manifest.surviving_pairs)

    rows: list[dict] = []
    for pair_id, per_pair in prior_stage_manifest.pairs.items():
        if pair_id not in surviving:
            continue
        for up_id, up_manifest in per_pair.per_user_prompt.items():
            if up_manifest.ok > 0:
                # Emit one row per (pair, user_prompt) that succeeded.
                # The controller will expand to the individual sample_ids via
                # the per-cell tracking, but we need concrete sample_id rows.
                # Since PerUserPromptManifest only has ok/errors/total counts
                # (no cell-level list in the local schema), we emit a
                # synthetic row that signals the surviving cell count.
                # For the full cell enumeration the caller must pass actual
                # sample_ids; here we emit (pair_id, up_id) rows which the
                # controller accepts when ok == total (every sample succeeded).
                # NOTE: Per plan §6.4 the manifest rows must be 3-key
                # {prompt_pair_id, user_prompt_id, sample_id}.  Since the
                # local PerUserPromptManifest doesn't store individual
                # sample_ids, we use a placeholder "__all__" and document
                # that callers must supply a sample list override via
                # build_missing_manifest when exact sample_id routing matters.
                rows.append(
                    {
                        "prompt_pair_id": pair_id,
                        "user_prompt_id": up_id,
                        "sample_id": "__all__",
                        "ok": up_manifest.ok,
                        "errors": up_manifest.errors,
                        "total": up_manifest.total,
                    }
                )

    with open(output_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path


def build_survivor_manifest_with_samples(
    prior_stage_manifest: StageManifest,
    sample_ids_for_pair: dict[str, list[str]],
    *,
    output_path: Path | None = None,
    local_manifests_root: Path | None = None,
) -> Path:
    """Build a 3-key JSONL of surviving cells with explicit sample_id rows.

    Like :func:`build_survivor_manifest` but uses *sample_ids_for_pair* to
    emit one row per ``(prompt_pair_id, user_prompt_id, sample_id)`` triple
    for pairs in ``prior_stage_manifest.surviving_pairs``.  Only
    ``(user_prompt_id)`` cells where ``up_manifest.errors == 0`` are included.

    Parameters
    ----------
    prior_stage_manifest:
        The stage manifest from the prior stage.
    sample_ids_for_pair:
        Mapping of ``prompt_pair_id`` → list of ``sample_id``s that succeeded
        at the prior stage.
    output_path:
        Explicit destination path.
    local_manifests_root:
        Fallback directory when *output_path* is ``None``.

    Returns
    -------
    Path
        Path to the written JSONL file.
    """
    if output_path is None:
        if local_manifests_root is None:
            raise ValueError(
                "Provide either output_path or local_manifests_root"
            )
        local_manifests_root.mkdir(parents=True, exist_ok=True)
        stage = prior_stage_manifest.stage
        round_id = prior_stage_manifest.round_id or "1"
        output_path = (
            local_manifests_root
            / f"survivors_{stage}_round_{round_id}_cells.jsonl"
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    surviving = set(prior_stage_manifest.surviving_pairs)

    rows: list[dict] = []
    for pair_id, per_pair in prior_stage_manifest.pairs.items():
        if pair_id not in surviving:
            continue
        sample_ids = sample_ids_for_pair.get(pair_id, [])
        for up_id, up_manifest in per_pair.per_user_prompt.items():
            if up_manifest.errors == 0 and up_manifest.ok > 0:
                for sample_id in sample_ids:
                    rows.append(
                        {
                            "prompt_pair_id": pair_id,
                            "user_prompt_id": up_id,
                            "sample_id": sample_id,
                        }
                    )

    with open(output_path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path
