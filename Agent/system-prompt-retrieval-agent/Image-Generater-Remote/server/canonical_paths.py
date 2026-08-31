"""Canonical paths, filenames, hashes, and identifiers for V0.2.2.

This module is the single source of truth for path, filename, manifest,
hash, schema-version, run-id, attempt-id, lifecycle, and execution-mode
contracts shared between the remote service and the local agent.

It is the master source. The local agent imports a byte-identical
vendored copy at:

    System-Prompt-Retrieval-Agent/src/system_prompt_retrieval_agent/
        remote/_vendored/canonical_paths.py

Vendor sync is performed by the deterministic vendor-sync script. CI
enforces byte equality between the master and the vendored copy.

Stdlib-only by design. Never import anything outside the Python
standard library here.
"""

from __future__ import annotations

import hashlib
import json
import re
import secrets
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Schema version (S00.15)
# ---------------------------------------------------------------------------

SCHEMA_VERSION: str = "v0.2.2"

# ---------------------------------------------------------------------------
# Stage names (S00.03 / S00.04)
# ---------------------------------------------------------------------------

STAGE_GEMMA: str = "gemma"
STAGE_FLUX: str = "flux"
STAGE_QWEN: str = "qwen"

ALLOWED_STAGES: tuple[str, ...] = (STAGE_GEMMA, STAGE_FLUX, STAGE_QWEN)

# ---------------------------------------------------------------------------
# Canonical artifact filenames (S00.04 / plan §2.2)
# ---------------------------------------------------------------------------

ARTIFACT_FILENAMES: dict[str, str] = {
    STAGE_GEMMA: "intermediate_prompt.txt",
    STAGE_FLUX: "result.png",
    STAGE_QWEN: "eval.json",
}


def artifact_filename(stage: str) -> str:
    """Return the canonical artifact filename for *stage*."""
    if stage not in ARTIFACT_FILENAMES:
        raise ValueError(f"unknown stage: {stage!r}")
    return ARTIFACT_FILENAMES[stage]


# ---------------------------------------------------------------------------
# Canonical paths (S00.03 / plan §2.1, §2.3)
# ---------------------------------------------------------------------------

OUTPUTS_ROOT: str = "outputs/v02"


def run_dir(run_id: str) -> str:
    """Return ``outputs/v02/{run_id}``."""
    _validate_run_id(run_id)
    return f"{OUTPUTS_ROOT}/{run_id}"


def stage_dir(run_id: str, stage: str, round_id: int) -> str:
    """Return ``outputs/v02/{run_id}/{stage}/round_{round_id}``."""
    _validate_stage(stage)
    return f"{run_dir(run_id)}/{stage}/round_{round_id}"


def cell_dir(
    run_id: str,
    stage: str,
    round_id: int,
    prompt_pair_id: str,
    user_prompt_id: str,
    sample_id: str,
) -> str:
    """Return the canonical cell directory.

    ``outputs/v02/{run_id}/{stage}/round_{round_id}/{prompt_pair_id}/
    {user_prompt_id}/{sample_id}``.
    """
    return (
        f"{stage_dir(run_id, stage, round_id)}"
        f"/{prompt_pair_id}/{user_prompt_id}/{sample_id}"
    )


def cell_artifact_path(
    run_id: str,
    stage: str,
    round_id: int,
    prompt_pair_id: str,
    user_prompt_id: str,
    sample_id: str,
) -> str:
    """Return the canonical artifact file path for a cell."""
    return (
        f"{cell_dir(run_id, stage, round_id, prompt_pair_id, user_prompt_id, sample_id)}"
        f"/{artifact_filename(stage)}"
    )


def manifests_dir(run_id: str) -> str:
    """Return ``outputs/v02/{run_id}/manifests``."""
    return f"{run_dir(run_id)}/manifests"


def stage_manifest_attempt_path(
    run_id: str, stage: str, round_id: int, attempt_id: str
) -> str:
    """Return the per-attempt stage manifest path (never overwritten)."""
    _validate_stage(stage)
    _validate_attempt_id(attempt_id)
    return (
        f"{manifests_dir(run_id)}"
        f"/{stage}_round_{round_id}_attempt_{attempt_id}.json"
    )


def stage_manifest_pointer_path(run_id: str, stage: str, round_id: int) -> str:
    """Return the canonical manifest pointer path.

    The pointer is atomically renamed to point at the latest valid
    per-attempt file.
    """
    _validate_stage(stage)
    return f"{manifests_dir(run_id)}/{stage}_round_{round_id}_manifest.json"


def stage_manifest_merged_path(run_id: str, stage: str, round_id: int) -> str:
    """Return the locally-owned merged-view manifest path."""
    _validate_stage(stage)
    return f"{manifests_dir(run_id)}/{stage}_round_{round_id}_merged.json"


# ---------------------------------------------------------------------------
# Manifest identity field sets (S00.06 / plan §2.4)
# ---------------------------------------------------------------------------

MATCH_FIELDS: tuple[str, ...] = (
    "schema_version",
    "run_id",
    "round_id",
    "stage",
    "config_hash",
    "user_prompt_corpus_id",
    "user_prompt_corpus_hash",
    "prompt_pair_corpus_hash",
    "sample_corpus_hash",
)

INFORMATIONAL_FIELDS: tuple[str, ...] = (
    "lifecycle_mode",
    "lifecycle_state_after",
)


# ---------------------------------------------------------------------------
# Cell record shape (S00.07 / plan §2.5) — declarative constants only
# ---------------------------------------------------------------------------

CELL_STATUS_OK: str = "ok"
CELL_STATUS_ERROR: str = "error"
CELL_STATUS_MISSING: str = "missing"
CELL_STATUS_PARSE_FAILED: str = "parse_failed"
CELL_STATUS_CARRIED_OVER: str = "carried_over"

ALLOWED_CELL_STATUSES: tuple[str, ...] = (
    CELL_STATUS_OK,
    CELL_STATUS_ERROR,
    CELL_STATUS_MISSING,
    CELL_STATUS_PARSE_FAILED,
    CELL_STATUS_CARRIED_OVER,
)

SUCCESSFUL_CELL_STATUSES: frozenset[str] = frozenset(
    {CELL_STATUS_OK, CELL_STATUS_CARRIED_OVER}
)

# Required keys on every cell record (S00.07 / plan §2.5).
CELL_REQUIRED_FIELDS: tuple[str, ...] = (
    "prompt_pair_id",
    "user_prompt_id",
    "sample_id",
    "status",
    "artifact_relpath",
)

# Required only when status implies a real on-disk artifact.
CELL_ARTIFACT_VERIFICATION_FIELDS: tuple[str, ...] = (
    "artifact_size_bytes",
    "artifact_sha256",
)

# Optional / status-conditional fields.
CELL_OPTIONAL_FIELDS: tuple[str, ...] = ("error_reason",)
CELL_QWEN_ONLY_FIELDS: tuple[str, ...] = ("parse_status", "parsed")


def validate_cell_record(cell: Mapping[str, Any], stage: str | None = None) -> None:
    """Validate a single ``cells[]`` record against plan §2.5 (S00.07).

    Raises ``ValueError`` describing the first violation found. Does not
    perform on-disk size / sha-256 verification — that is the local
    artifact barrier's job (S03.04).
    """
    if not isinstance(cell, Mapping):
        raise ValueError(f"cell record is not a mapping: {type(cell).__name__}")
    for key in CELL_REQUIRED_FIELDS:
        if key not in cell:
            raise ValueError(f"cell record missing required key: {key!r}")
    status = cell["status"]
    if status not in ALLOWED_CELL_STATUSES:
        raise ValueError(f"cell record has invalid status: {status!r}")
    if status in SUCCESSFUL_CELL_STATUSES:
        for key in CELL_ARTIFACT_VERIFICATION_FIELDS:
            if key not in cell:
                raise ValueError(
                    f"cell record with status={status!r} missing required "
                    f"verification key: {key!r}"
                )
        size = cell["artifact_size_bytes"]
        if not isinstance(size, int) or size < 0:
            raise ValueError(
                f"artifact_size_bytes must be non-negative int; got {size!r}"
            )
        sha = cell["artifact_sha256"]
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError(
                f"artifact_sha256 must be 64-hex sha256; got {sha!r}"
            )
    if stage is not None and stage != STAGE_QWEN:
        for key in CELL_QWEN_ONLY_FIELDS:
            if key in cell:
                raise ValueError(
                    f"non-Qwen stage {stage!r} must not declare {key!r} on cells"
                )


# ---------------------------------------------------------------------------
# Execution-mode labels (S00.09 / plan §3.1)
# ---------------------------------------------------------------------------

EXECUTION_MODE_V022: str = "v022_stage_major"
EXECUTION_MODE_LEGACY_STAGE_ALL: str = "legacy_stage_all_compat"
EXECUTION_MODE_LEGACY_CARTESIAN: str = "legacy_cartesian_compat"

ALLOWED_EXECUTION_MODES: tuple[str, ...] = (
    EXECUTION_MODE_V022,
    EXECUTION_MODE_LEGACY_STAGE_ALL,
    EXECUTION_MODE_LEGACY_CARTESIAN,
)

PRODUCTION_EXECUTION_MODE: str = EXECUTION_MODE_V022


# ---------------------------------------------------------------------------
# Lifecycle enum + mode/state matrix (S00.10 / S00.10a / plan §3.9)
# ---------------------------------------------------------------------------

LIFECYCLE_DISK_UNLOADED: str = "disk_unloaded"
LIFECYCLE_CPU_PREFETCHED: str = "cpu_prefetched"
LIFECYCLE_GPU_LOADED: str = "gpu_loaded"
LIFECYCLE_GPU_UNLOADED_CPU_RETAINED: str = "gpu_unloaded_cpu_retained"

ALLOWED_LIFECYCLE_STATES: tuple[str, ...] = (
    LIFECYCLE_DISK_UNLOADED,
    LIFECYCLE_CPU_PREFETCHED,
    LIFECYCLE_GPU_LOADED,
    LIFECYCLE_GPU_UNLOADED_CPU_RETAINED,
)

LIFECYCLE_MODE_COLD: str = "cold"
LIFECYCLE_MODE_WARM: str = "warm"

ALLOWED_LIFECYCLE_MODES: tuple[str, ...] = (
    LIFECYCLE_MODE_COLD,
    LIFECYCLE_MODE_WARM,
)

# Mapping mode → allowed `lifecycle_state_after` values. `gpu_loaded` is
# never an allowed `lifecycle_state_after`; it may appear transiently
# during execution only.
LIFECYCLE_MODE_STATE_MATRIX: dict[str, frozenset[str]] = {
    LIFECYCLE_MODE_COLD: frozenset({LIFECYCLE_DISK_UNLOADED}),
    LIFECYCLE_MODE_WARM: frozenset(
        {LIFECYCLE_CPU_PREFETCHED, LIFECYCLE_GPU_UNLOADED_CPU_RETAINED}
    ),
}

# V0.2.2 hard-disables warm mode at runtime (plan §3.9).
WARM_MODE_HARD_DISABLED_IN_V022: bool = True
WARM_MODE_DISABLED_ERROR: str = "warm mode disabled in V0.2.2"


def validate_lifecycle_state_after(mode: str, state_after: str) -> None:
    """Raise ``ValueError`` if *state_after* is not allowed for *mode*."""
    if mode not in LIFECYCLE_MODE_STATE_MATRIX:
        raise ValueError(f"unknown lifecycle mode: {mode!r}")
    allowed = LIFECYCLE_MODE_STATE_MATRIX[mode]
    if state_after not in allowed:
        raise ValueError(
            f"lifecycle_state_after={state_after!r} not allowed under "
            f"lifecycle_mode={mode!r}; allowed={sorted(allowed)}"
        )


# ---------------------------------------------------------------------------
# Canonical JSON encoding (plan §2.6)
# ---------------------------------------------------------------------------

def canonical_json(value: Any) -> bytes:
    """Encode *value* as canonical JSON bytes.

    Rules: UTF-8, sorted keys at every level, no insignificant
    whitespace, `ensure_ascii=False`. ``json.dumps`` already orders
    list elements in declared order, so the caller is responsible for
    pre-sorting lists into deterministic order.
    """
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Corpus hashes (S00.08 / S00.19 / plan §2.4b, §2.6)
# ---------------------------------------------------------------------------

USER_PROMPT_FIELDS: tuple[str, ...] = ("user_prompt_id", "language", "text", "enabled")
PROMPT_PAIR_FIELDS: tuple[str, ...] = (
    "prompt_pair_id",
    "system_prompt",
    "user_prompt_template",
    "language",
    "enabled",
)
SAMPLE_FIELDS: tuple[str, ...] = (
    "sample_id",
    "model_path",
    "cloth_path",
    "model_sha256",
    "cloth_sha256",
)


def _project_keys(record: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        if k not in record:
            raise KeyError(f"required key missing: {k!r}")
        out[k] = record[k]
    return out


def user_prompt_corpus_hash(enabled_user_prompts: Iterable[Mapping[str, Any]]) -> str:
    """Compute the user-prompt corpus hash (plan §2.6)."""
    items = sorted(
        (_project_keys(p, USER_PROMPT_FIELDS) for p in enabled_user_prompts),
        key=lambda r: r["user_prompt_id"],
    )
    return hashlib.sha256(canonical_json(items)).hexdigest()


def prompt_pair_corpus_hash(enabled_prompt_pairs: Iterable[Mapping[str, Any]]) -> str:
    """Compute the prompt-pair corpus hash (plan §2.4b)."""
    items = sorted(
        (_project_keys(p, PROMPT_PAIR_FIELDS) for p in enabled_prompt_pairs),
        key=lambda r: r["prompt_pair_id"],
    )
    return hashlib.sha256(canonical_json(items)).hexdigest()


def sample_corpus_hash(enabled_samples: Iterable[Mapping[str, Any]]) -> str:
    """Compute the sample corpus hash (plan §2.4b)."""
    items = sorted(
        (_project_keys(s, SAMPLE_FIELDS) for s in enabled_samples),
        key=lambda r: r["sample_id"],
    )
    return hashlib.sha256(canonical_json(items)).hexdigest()


# ---------------------------------------------------------------------------
# Config hash (S00.14 / S00.14a / plan §2.4a)
# ---------------------------------------------------------------------------

CANONICAL_CONFIG_KEYS: tuple[str, ...] = (
    "models.gemma.checkpoint",
    "models.flux.checkpoint",
    "models.qwen.checkpoint",
    "models.gemma.gen_params",
    "models.flux.gen_params",
    "models.qwen.gen_params",
    "evaluation.categories",
    "evaluation.weights",
    "evaluation.partial_mode_thresholds",
    "scoring.gate",
    "scoring.aggregation",
    "execution.mode",
    "execution.lifecycle_mode",
)

REQUIRED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "models.gemma.checkpoint",
        "models.flux.checkpoint",
        "models.qwen.checkpoint",
        "models.gemma.gen_params",
        "models.flux.gen_params",
        "models.qwen.gen_params",
        "evaluation.categories",
        "evaluation.weights",
        "scoring.gate",
    }
)


def _resolve_dotted(cfg: Mapping[str, Any], dotted: str) -> tuple[bool, Any]:
    cursor: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cursor, Mapping) or part not in cursor:
            return False, None
        cursor = cursor[part]
    return True, cursor


def canonical_config_subset(effective_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical config subset keyed by ``CANONICAL_CONFIG_KEYS``.

    Required keys (``REQUIRED_CONFIG_KEYS``) absent at runtime cause a
    ``ValueError`` raised before any hash is computed. Optional keys
    absent at runtime are encoded as JSON ``null``.
    """
    missing_required: list[str] = []
    out: dict[str, Any] = {}
    for key in CANONICAL_CONFIG_KEYS:
        present, value = _resolve_dotted(effective_config, key)
        if not present:
            if key in REQUIRED_CONFIG_KEYS:
                missing_required.append(key)
            else:
                out[key] = None
        else:
            out[key] = value
    if missing_required:
        raise ValueError(
            "config_hash required keys absent: " + ", ".join(missing_required)
        )
    return out


def config_hash(effective_config: Mapping[str, Any]) -> str:
    """Compute the canonical config hash (plan §2.4a)."""
    subset = canonical_config_subset(effective_config)
    return hashlib.sha256(canonical_json(subset)).hexdigest()


# ---------------------------------------------------------------------------
# run_id and attempt_id generators (S00.16 / S00.05a / plan §2.10a, §2.3a)
# ---------------------------------------------------------------------------

RUN_ID_REGEX: re.Pattern[str] = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
ATTEMPT_ID_REGEX: re.Pattern[str] = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{4}$")


def _utc_stamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def generate_run_id() -> str:
    """Return ``"{YYYYMMDDTHHMMSSZ}-{8hex_random}"`` per §2.10a."""
    return f"{_utc_stamp()}-{secrets.token_hex(4)}"


def generate_attempt_id() -> str:
    """Return ``"{YYYYMMDDTHHMMSSZ}-{4hex_random}"`` per §2.3a."""
    return f"{_utc_stamp()}-{secrets.token_hex(2)}"


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not RUN_ID_REGEX.match(run_id):
        raise ValueError(f"invalid run_id: {run_id!r}")


def _validate_attempt_id(attempt_id: str) -> None:
    if not isinstance(attempt_id, str) or not ATTEMPT_ID_REGEX.match(attempt_id):
        raise ValueError(f"invalid attempt_id: {attempt_id!r}")


def _validate_stage(stage: str) -> None:
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"invalid stage: {stage!r}")


# ---------------------------------------------------------------------------
# Scoring missing-input policy (S00.11 / plan §2.8)
# ---------------------------------------------------------------------------

# When a surviving prompt-pair lacks a required scoring input, scoring
# moves it to ``failed_pairs[]`` with this exact reason string.
SCORING_FAILED_REASON_MISSING_INPUT: str = "missing_scoring_input"


# ---------------------------------------------------------------------------
# Qwen parse-failure policy (S00.12 / plan §2.5)
# ---------------------------------------------------------------------------

# Qwen cells whose JSON parse fails must be recorded with this status.
# In strict mode (``workflow.allow_partial_samples=false``) they count as
# errors against survival; in partial mode they are subject to category
# coverage thresholds.
QWEN_PARSE_FAILED_STATUS: str = CELL_STATUS_PARSE_FAILED


# ---------------------------------------------------------------------------
# Resume × survivor merge rule (S00.17 / plan §2.7a) — declarative spec
# ---------------------------------------------------------------------------

# The per-stage dispatch input ``D = S − L`` where:
#   S — upstream merged-view survivor cell set for the current stage
#       (Gemma uses the §2.7c base set: enabled prompt-pairs × enabled
#        user prompts × configured sample IDs).
#   L — current-stage cells whose canonical local artifacts already exist
#       and whose recorded match-fields, on-disk size, and sha-256 match
#       (validated by S04 copy-back / S04.07 ``build_missing_manifest``).
#   D — the dispatch set sent to the remote stage runner.
#
# The remote stage manifest's ``cells[]`` contains entries only for cells
# in ``D``. The local agent owns construction of the merged stage view at
# ``stage_manifest_merged_path(...)`` containing every cell in ``S``:
#   • cells in ``L`` → status ``carried_over`` (sourced from the prior
#     copied per-attempt manifest).
#   • cells in ``D`` → status from the current remote per-attempt
#     manifest.
# Empty dispatch (``D = ∅``) skips the remote call entirely (plan §2.7d).
RESUME_MERGE_RULE_DOCSTRING: str = (
    "D = S - L; remote sees only D; local agent merges S = L (carried_over) "
    "+ D (current attempt) into the merged view at stage_manifest_merged_path."
)


# ---------------------------------------------------------------------------
# Remote artifact-write ordering (S00.18 / plan §2.7b) — declarative spec
# ---------------------------------------------------------------------------

# Per-cell ordering enforced by every remote stage writer:
#   1. write artifact bytes
#   2. fsync the artifact file descriptor
#   3. normalize the on-disk filename (rename to canonical filename)
#   4. fsync the containing directory
#   5. stat the file size from disk
#   6. compute sha-256 from the on-disk bytes
#   7. append the ``cells[]`` entry with status="ok" and verified size + sha
# Any step failing prevents the cell from being declared ``ok``; the cell
# instead becomes ``error`` with an explanatory ``error_reason``.
ARTIFACT_WRITE_ORDERING_STEPS: tuple[str, ...] = (
    "write",
    "fsync_file",
    "normalize_filename",
    "fsync_dir",
    "stat_size",
    "sha256_from_disk",
    "append_cell",
)


# ---------------------------------------------------------------------------
# Long-memory legacy sentinel (plan §2.10b)
# ---------------------------------------------------------------------------

LEGACY_LONG_MEMORY_RUN_ID: str = "legacy:v021"


__all__ = [
    "SCHEMA_VERSION",
    "STAGE_GEMMA",
    "STAGE_FLUX",
    "STAGE_QWEN",
    "ALLOWED_STAGES",
    "ARTIFACT_FILENAMES",
    "artifact_filename",
    "OUTPUTS_ROOT",
    "run_dir",
    "stage_dir",
    "cell_dir",
    "cell_artifact_path",
    "manifests_dir",
    "stage_manifest_attempt_path",
    "stage_manifest_pointer_path",
    "stage_manifest_merged_path",
    "MATCH_FIELDS",
    "INFORMATIONAL_FIELDS",
    "CELL_STATUS_OK",
    "CELL_STATUS_ERROR",
    "CELL_STATUS_MISSING",
    "CELL_STATUS_PARSE_FAILED",
    "CELL_STATUS_CARRIED_OVER",
    "ALLOWED_CELL_STATUSES",
    "SUCCESSFUL_CELL_STATUSES",
    "CELL_REQUIRED_FIELDS",
    "CELL_ARTIFACT_VERIFICATION_FIELDS",
    "CELL_OPTIONAL_FIELDS",
    "CELL_QWEN_ONLY_FIELDS",
    "validate_cell_record",
    "SCORING_FAILED_REASON_MISSING_INPUT",
    "QWEN_PARSE_FAILED_STATUS",
    "RESUME_MERGE_RULE_DOCSTRING",
    "ARTIFACT_WRITE_ORDERING_STEPS",
    "EXECUTION_MODE_V022",
    "EXECUTION_MODE_LEGACY_STAGE_ALL",
    "EXECUTION_MODE_LEGACY_CARTESIAN",
    "ALLOWED_EXECUTION_MODES",
    "PRODUCTION_EXECUTION_MODE",
    "LIFECYCLE_DISK_UNLOADED",
    "LIFECYCLE_CPU_PREFETCHED",
    "LIFECYCLE_GPU_LOADED",
    "LIFECYCLE_GPU_UNLOADED_CPU_RETAINED",
    "ALLOWED_LIFECYCLE_STATES",
    "LIFECYCLE_MODE_COLD",
    "LIFECYCLE_MODE_WARM",
    "ALLOWED_LIFECYCLE_MODES",
    "LIFECYCLE_MODE_STATE_MATRIX",
    "WARM_MODE_HARD_DISABLED_IN_V022",
    "WARM_MODE_DISABLED_ERROR",
    "validate_lifecycle_state_after",
    "canonical_json",
    "USER_PROMPT_FIELDS",
    "PROMPT_PAIR_FIELDS",
    "SAMPLE_FIELDS",
    "user_prompt_corpus_hash",
    "prompt_pair_corpus_hash",
    "sample_corpus_hash",
    "CANONICAL_CONFIG_KEYS",
    "REQUIRED_CONFIG_KEYS",
    "canonical_config_subset",
    "config_hash",
    "RUN_ID_REGEX",
    "ATTEMPT_ID_REGEX",
    "generate_run_id",
    "generate_attempt_id",
    "LEGACY_LONG_MEMORY_RUN_ID",
]
