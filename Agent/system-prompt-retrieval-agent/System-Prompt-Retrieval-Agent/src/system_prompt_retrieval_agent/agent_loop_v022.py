"""V0.2.2 stage-major orchestrator (S05.01–S05.13).

One :class:`V022Orchestrator` instance owns one round of stage-major
execution: Gemma → FLUX → Qwen. Each stage flow is identical:

1. Compute the survivor set ``S`` (Gemma uses the §2.7c base set; FLUX
   uses the Gemma merged-view; Qwen uses the FLUX merged-view).
2. Compute ``L`` from prior per-attempt manifests + on-disk artifact
   verification.
3. ``D = S - L``. If empty, short-circuit (no remote call, no copy-back,
   merged view = ``L``, then run the local artifact barrier on ``L``).
4. Otherwise, POST to ``/v022/stage/{stage}`` (NEVER ``/stage/all``).
5. Run the remote stage barrier on the response manifest before copy-back.
6. Run cell-scoped copy-back into a same-filesystem temp dir; verify;
   atomic promote per cell; non-overwriting per-attempt manifest path.
7. Run the local artifact barrier on the copied manifest.
8. Build the local merged stage view; durably write it; retarget the
   canonical pointer only after the merged write succeeds.

Every dispatched ``run_id`` matches the canonical regex; production
mode rejects arbitrary user-supplied ``run_id`` outside
``--resume-from-run-id``. A runtime hard-fail kills the process if any
code path attempts ``/stage/all`` while in ``v022_stage_major`` mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .barrier_v022 import (
    BarrierViolation,
    PartialModeThresholds,
    apply_survival_policy,
    run_local_artifact_barrier,
    run_remote_stage_barrier,
)
from .copy_back_v022 import (
    CopyBackError,
    CopyBackPlan,
    PromotionResult,
    copy_back_dispatch_cells,
    finalize_pointer_after_merge,
)
from .remote._vendored import canonical_paths as cp
from .resume_from_run_id import (
    CurrentRunHashes,
    ResumeDriftError,
    validate_resume_from_run_id,
)
from .survivor_resume_v022 import (
    CellKey,
    LocalArtifactView,
    build_gemma_base_survivor_set,
    build_merged_view,
    build_survivor_manifest_v022,
    compute_dispatch_set,
    compute_local_artifact_view,
    write_merged_view,
    write_survivor_jsonl,
)


log = logging.getLogger(__name__)


CellKeyTuple = tuple[str, str, str]


def _serialize_records(
    records: Iterable[Any], id_field: str, allowed_ids: set[str]
) -> list[dict[str, Any]]:
    """Serialize Pydantic / dataclass / dict records to JSON-safe dicts.

    Filters to entries whose ``id_field`` is in ``allowed_ids``.
    """
    out: list[dict[str, Any]] = []
    for r in records:
        if hasattr(r, "model_dump"):
            d = r.model_dump(mode="json")
        elif isinstance(r, Mapping):
            d = dict(r)
        elif hasattr(r, "__dict__"):
            d = dict(r.__dict__)
        else:
            continue
        if d.get(id_field) in allowed_ids:
            out.append(d)
    return out


class V022ProductionViolation(RuntimeError):
    """Raised when production mode attempts a forbidden code path."""


# ---------------------------------------------------------------------------
# Stage dispatcher abstraction (injectable for tests)
# ---------------------------------------------------------------------------


class StageDispatcher:
    """Sends V0.2.2 stage requests to the remote controller.

    Tests inject a fake via the ``post_v022`` callable; the production
    implementation is built by :meth:`from_client` which owns the
    underlying async :class:`RemoteControllerClient` lifecycle on a
    private event loop (S05B.20–S05B.22, S05B.33). Calling
    :meth:`dispatch_stage_all` raises :class:`V022ProductionViolation`
    to enforce S05.06.
    """

    def __init__(
        self,
        post_v022: Callable[[str, dict], dict],
        *,
        production_mode: bool = True,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        self._post_v022 = post_v022
        self.production_mode = production_mode
        self.dispatched_endpoints: list[str] = []
        self._on_close = on_close
        self._closed = False

    def dispatch(self, stage: str, payload: dict) -> dict:
        endpoint = f"/v022/stage/{stage}"
        self.dispatched_endpoints.append(endpoint)
        result = self._post_v022(stage, payload)
        if asyncio.iscoroutine(result):
            raise V022ProductionViolation(
                "StageDispatcher.dispatch must return dict, not coroutine; "
                "use StageDispatcher.from_client(...) for production wiring"
            )
        if not isinstance(result, dict):
            raise V022ProductionViolation(
                f"StageDispatcher.dispatch must return dict, got {type(result).__name__}"
            )
        return result

    def dispatch_stage_all(self, *args: Any, **kwargs: Any) -> Any:
        if self.production_mode:
            raise V022ProductionViolation(
                "production mode (v022_stage_major) must not call /stage/all"
            )
        raise NotImplementedError(
            "/stage/all is only available under legacy_stage_all_compat"
        )

    def close(self) -> None:
        """Release resources owned by the dispatcher (private loop, client)."""
        if self._closed:
            return
        self._closed = True
        if self._on_close is not None:
            self._on_close()

    # ------------------------------------------------------------------
    # Production factory (S05B.20–S05B.22, S05B.33)
    # ------------------------------------------------------------------

    @classmethod
    def from_client(
        cls,
        client_factory: Callable[[], Any],
        *,
        production_mode: bool = True,
    ) -> "StageDispatcher":
        """Build a production-grade sync dispatcher around an async client.

        ``client_factory`` is invoked **on the dispatcher's private event
        loop** to construct a :class:`RemoteControllerClient` (or any
        object with ``async def post_v022_stage(stage, payload)`` and
        async-context-manager semantics). The same client instance is
        reused across all dispatches in the run; ``close()`` runs the
        client's ``__aexit__`` on the same loop and stops the loop.

        Constructing the client outside this factory and handing it in
        is forbidden — it produces silent loop-affinity bugs and
        connection leaks (plan §8.4.1).
        """
        loop = asyncio.new_event_loop()
        loop_thread_id_holder: dict[str, int] = {}
        client_holder: dict[str, Any] = {"client": None}
        lock = threading.Lock()

        def _ensure_client() -> Any:
            if client_holder["client"] is None:
                with lock:
                    if client_holder["client"] is None:
                        client = client_factory()
                        # Enter the async context on the private loop so
                        # the underlying httpx.AsyncClient binds to it.
                        loop.run_until_complete(client.__aenter__())
                        loop_thread_id_holder["loop_owner"] = threading.get_ident()
                        client_holder["client"] = client
            return client_holder["client"]

        def _post_v022(stage: str, payload: dict) -> dict:
            client = _ensure_client()
            coro = client.post_v022_stage(stage, payload)
            result = loop.run_until_complete(coro)
            if not isinstance(result, dict):
                raise V022ProductionViolation(
                    "post_v022_stage must return dict, "
                    f"got {type(result).__name__}"
                )
            return result

        def _on_close() -> None:
            client = client_holder["client"]
            if client is not None:
                # Drive __aexit__ on the dispatcher's private loop. If
                # close() is called from inside an outer event loop,
                # delegate the loop work to a worker thread so we
                # never call loop.run_until_complete() while another
                # loop is running on the same thread.
                def _run_aexit() -> None:
                    try:
                        loop.run_until_complete(
                            client.__aexit__(None, None, None)
                        )
                    except Exception:  # pragma: no cover — defensive
                        log.exception("error closing dispatcher client")

                try:
                    asyncio.get_running_loop()
                    t = threading.Thread(target=_run_aexit, daemon=True)
                    t.start()
                    t.join()
                except RuntimeError:
                    # No running loop on this thread — safe to drive
                    # the private loop directly.
                    _run_aexit()
                client_holder["client"] = None
            try:
                loop.close()
            except Exception:  # pragma: no cover
                pass

        d = cls(_post_v022, production_mode=production_mode, on_close=_on_close)
        # Expose internals for the loop-affinity test (S05B.33).
        d._private_loop = loop  # type: ignore[attr-defined]
        d._client_holder = client_holder  # type: ignore[attr-defined]
        return d


# ---------------------------------------------------------------------------
# Round configuration
# ---------------------------------------------------------------------------


@dataclass
class CorpusContext:
    enabled_prompt_pair_ids: tuple[str, ...]
    enabled_user_prompt_ids: tuple[str, ...]
    configured_sample_ids: tuple[str, ...]
    user_prompt_corpus_id: str
    user_prompt_corpus_hash: str
    prompt_pair_corpus_hash: str
    sample_corpus_hash: str
    config_hash: str

    def current_run_hashes(self) -> CurrentRunHashes:
        return CurrentRunHashes(
            config_hash=self.config_hash,
            user_prompt_corpus_hash=self.user_prompt_corpus_hash,
            prompt_pair_corpus_hash=self.prompt_pair_corpus_hash,
            sample_corpus_hash=self.sample_corpus_hash,
        )


@dataclass
class RoundCorpusRecords:
    """Full prompt-pair / user-prompt records for a round (S05B.23).

    The orchestrator passes these onto ``/v022/stage/{stage}`` so the
    remote can hydrate Gemma/FLUX/Qwen with actual prompt text — IDs
    alone are not enough for real model execution. Records must be the
    **enabled** subset, in deterministic ID order. The four corpus
    hashes in :class:`CorpusContext` must be computed over these same
    records so resume drift detection stays consistent.
    """

    prompt_pairs: tuple[Any, ...] = ()
    user_prompts: tuple[Any, ...] = ()
    samples: tuple[Any, ...] = ()


@dataclass
class RoundContext:
    run_id: str
    round_id: int
    artifact_root: Path
    remote_artifact_root: str
    corpus: CorpusContext
    lifecycle_mode: str = "cold"
    lifecycle_state_after: str = "disk_unloaded"
    allow_partial: bool = False
    partial_thresholds: PartialModeThresholds | None = None
    cell_categories: Mapping[CellKeyTuple, str] | None = None
    corpus_records: RoundCorpusRecords = field(default_factory=RoundCorpusRecords)
    dataset_root: str | None = None
    output_root: str | None = None
    # When set, collapses the (pair × user_prompt × sample) cartesian to one
    # cell per (pair, sample) using the per-sample user_prompt_id from this map.
    # See survivor_resume_v022.build_gemma_base_survivor_set.
    sample_user_prompt_map: Mapping[str, str] | None = None


# ---------------------------------------------------------------------------
# Per-stage result for the orchestrator
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    stage: str
    survivor_set: list[CellKey]
    local_view: LocalArtifactView
    dispatch_set: list[CellKey]
    attempt_id: str | None  # None when D=∅ short-circuited
    remote_manifest: dict | None  # None when D=∅
    merged_manifest: dict
    promotion_result: PromotionResult | None  # None when D=∅
    survivor_jsonl_path: Path | None = None


@dataclass
class V022OrchestratorOutcome:
    rounds: list[dict[str, Any]] = field(default_factory=list)
    aborted_reason: str | None = None


@dataclass
class RoundExecutionResult:
    """Structured return from :meth:`V022Orchestrator.run_round` (S05B.31).

    Carries each stage's full :class:`StageResult` (incl. merged
    manifest + on-disk path) so the runner can hand them to scoring
    without re-globbing the run directory. ``summary`` is the legacy
    counts dict that the orchestrator already produced.
    """

    run_id: str
    round_id: int
    stage_results: dict[str, "StageResult"] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class V022Orchestrator:
    def __init__(
        self,
        *,
        dispatcher: StageDispatcher,
        copy_back_transport: Callable | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self._copy_back_transport = copy_back_transport

    # ---- top-level round execution ----------------------------------

    def run_round(self, ctx: RoundContext) -> RoundExecutionResult:
        if not cp.RUN_ID_REGEX.match(ctx.run_id):
            raise V022ProductionViolation(
                f"production run_id does not match canonical regex: {ctx.run_id!r}"
            )
        round_summary: dict[str, Any] = {
            "run_id": ctx.run_id,
            "round_id": ctx.round_id,
            "stages": {},
        }
        result = RoundExecutionResult(
            run_id=ctx.run_id, round_id=ctx.round_id, summary=round_summary,
        )

        gemma_result = self.run_gemma_stage(ctx)
        round_summary["stages"]["gemma"] = self._stage_summary(gemma_result)
        result.stage_results["gemma"] = gemma_result

        flux_result = self.run_downstream_stage(ctx, "flux", upstream=gemma_result)
        round_summary["stages"]["flux"] = self._stage_summary(flux_result)
        result.stage_results["flux"] = flux_result

        qwen_result = self.run_downstream_stage(ctx, "qwen", upstream=flux_result)
        round_summary["stages"]["qwen"] = self._stage_summary(qwen_result)
        result.stage_results["qwen"] = qwen_result
        return result

    # ---- Gemma (§2.7c base set) -------------------------------------

    def run_gemma_stage(self, ctx: RoundContext) -> StageResult:
        survivor = build_gemma_base_survivor_set(
            enabled_prompt_pair_ids=ctx.corpus.enabled_prompt_pair_ids,
            enabled_user_prompt_ids=ctx.corpus.enabled_user_prompt_ids,
            configured_sample_ids=ctx.corpus.configured_sample_ids,
            sample_user_prompt_map=ctx.sample_user_prompt_map,
        )
        return self._execute_stage(ctx, "gemma", survivor)

    # ---- FLUX / Qwen (downstream) -----------------------------------

    def run_downstream_stage(
        self, ctx: RoundContext, stage: str, *, upstream: StageResult
    ) -> StageResult:
        survivor = build_survivor_manifest_v022(
            upstream.merged_manifest,
            enabled_prompt_pair_ids=ctx.corpus.enabled_prompt_pair_ids,
            enabled_user_prompt_ids=ctx.corpus.enabled_user_prompt_ids,
            enabled_sample_ids=ctx.corpus.configured_sample_ids,
        )
        return self._execute_stage(ctx, stage, survivor)

    # ---- Shared per-stage flow --------------------------------------

    def _execute_stage(
        self,
        ctx: RoundContext,
        stage: str,
        survivor: Sequence[CellKey],
    ) -> StageResult:
        expected_match = self._expected_match_fields(ctx, stage)
        local_view = compute_local_artifact_view(
            run_id=ctx.run_id, stage=stage, round_id=ctx.round_id,
            artifact_root=ctx.artifact_root,
            expected_match_fields=expected_match,
            survivor_set=survivor,
        )
        dispatch = compute_dispatch_set(survivor, local_view)

        if not dispatch:
            return self._handle_empty_dispatch(
                ctx, stage, survivor, local_view, expected_match
            )

        attempt_id = cp.generate_attempt_id()
        log.info(
            "stage=%s building payload — survivor=%d dispatch=%d",
            stage, len(survivor), len(dispatch),
        )
        payload = self._build_v022_payload(
            ctx, stage, attempt_id, dispatch, expected_match
        )
        try:
            payload_bytes = len(json.dumps(payload, default=str))
        except Exception:
            payload_bytes = -1
        log.info(
            "stage=%s payload built — cells=%d approx_bytes=%d",
            stage, len(dispatch), payload_bytes,
        )
        log.info(
            "dispatching stage=%s cells=%d run_id=%s round_id=%d",
            stage, len(dispatch), ctx.run_id, ctx.round_id,
        )
        _t0 = time.monotonic()
        remote_manifest = self.dispatcher.dispatch(stage, payload)
        log.info(
            "stage=%s dispatch returned — cells_in_manifest=%d elapsed=%.1fs",
            stage,
            len(remote_manifest.get("cells", []) if isinstance(remote_manifest, dict) else []),
            time.monotonic() - _t0,
        )
        # Remote stage barrier (S05.07) — runs before copy-back.
        run_remote_stage_barrier(
            remote_manifest, expected_match_fields=expected_match,
            expected_cell_keys=[c.as_tuple() for c in dispatch],
        )

        # Cell-scoped copy-back (S04 + S05.08). Strict by default: any
        # non-success cell in the per-attempt manifest aborts copy-back
        # before cell rsync. Partial mode passes through ``ctx.allow_partial``
        # so failed cells are recorded but skipped instead of fatal.
        plan = CopyBackPlan(
            run_id=ctx.run_id, stage=stage, round_id=ctx.round_id,
            attempt_id=attempt_id,
            expected_match_fields=expected_match,
            dispatch_cells=[c.as_tuple() for c in dispatch],
            artifact_root=ctx.artifact_root,
            remote_artifact_root=ctx.remote_artifact_root,
            remote_manifest_root=ctx.remote_artifact_root,
            allow_partial=ctx.allow_partial,
        )
        if self._copy_back_transport is not None:
            promotion = copy_back_dispatch_cells(
                plan, transport=self._copy_back_transport
            )
        else:
            promotion = copy_back_dispatch_cells(plan)

        # Local artifact barrier (S05.08).
        copied_manifest = json.loads(promotion.promoted_manifest_path.read_text(encoding="utf-8"))
        run_local_artifact_barrier(
            copied_manifest=copied_manifest,
            expected_match_fields=expected_match,
            artifact_root=ctx.artifact_root,
        )

        # Local merged-view construction (S05.08a).
        merged = build_merged_view(
            survivor_set=survivor, local_view=local_view,
            current_remote_manifest=copied_manifest,
            current_attempt_id=attempt_id,
        )
        write_merged_view(
            artifact_root=ctx.artifact_root, run_id=ctx.run_id,
            stage=stage, round_id=ctx.round_id, merged_payload=merged,
        )
        # Pointer retarget AFTER merged view is durably written (S04.04).
        finalize_pointer_after_merge(
            attempt_path=promotion.pending_attempt_path,
            pointer_path=promotion.pending_pointer_path,
        )

        # Survivor JSONL (S05.09).
        survivor_jsonl = ctx.artifact_root / cp.manifests_dir(ctx.run_id) / (
            f"survivors_{stage}_round_{ctx.round_id}.jsonl"
        )
        write_survivor_jsonl(survivor, survivor_jsonl)
        return StageResult(
            stage=stage, survivor_set=list(survivor),
            local_view=local_view,
            dispatch_set=list(dispatch),
            attempt_id=attempt_id,
            remote_manifest=copied_manifest,
            merged_manifest=merged,
            promotion_result=promotion,
            survivor_jsonl_path=survivor_jsonl,
        )

    def _handle_empty_dispatch(
        self,
        ctx: RoundContext,
        stage: str,
        survivor: Sequence[CellKey],
        local_view: LocalArtifactView,
        expected_match: Mapping[str, Any],
    ) -> StageResult:
        # S05.07a: skip remote, run only local artifact barrier on L,
        # write merged manifest with cells = L. Match-fields are
        # inherited from the current invocation (no remote response to
        # echo them).
        merged = build_merged_view(
            survivor_set=survivor, local_view=local_view,
            current_remote_manifest=None, current_attempt_id=None,
            match_fields_override=expected_match,
        )
        merged_path = write_merged_view(
            artifact_root=ctx.artifact_root, run_id=ctx.run_id,
            stage=stage, round_id=ctx.round_id, merged_payload=merged,
        )
        # Local artifact barrier on the L-only merged manifest.
        run_local_artifact_barrier(
            copied_manifest=merged,
            expected_match_fields=expected_match,
            artifact_root=ctx.artifact_root,
        )
        return StageResult(
            stage=stage, survivor_set=list(survivor),
            local_view=local_view, dispatch_set=[],
            attempt_id=None, remote_manifest=None,
            merged_manifest=merged, promotion_result=None,
            survivor_jsonl_path=None,
        )

    def _expected_match_fields(
        self, ctx: RoundContext, stage: str
    ) -> dict[str, Any]:
        return {
            "schema_version": cp.SCHEMA_VERSION,
            "run_id": ctx.run_id,
            "round_id": ctx.round_id,
            "stage": stage,
            "config_hash": ctx.corpus.config_hash,
            "user_prompt_corpus_id": ctx.corpus.user_prompt_corpus_id,
            "user_prompt_corpus_hash": ctx.corpus.user_prompt_corpus_hash,
            "prompt_pair_corpus_hash": ctx.corpus.prompt_pair_corpus_hash,
            "sample_corpus_hash": ctx.corpus.sample_corpus_hash,
        }

    def _build_v022_payload(
        self,
        ctx: RoundContext,
        stage: str,
        attempt_id: str,
        dispatch: Sequence[CellKey],
        expected_match: Mapping[str, Any],
    ) -> dict:
        # S05B.24: filter prompt-pair / user-prompt records to only those
        # IDs that actually appear in dispatch_cells. The remote needs
        # the full text records (system_prompt, user-prompt template,
        # negative_prompt, language); IDs alone are not enough for real
        # Gemma/FLUX/Qwen execution.
        dispatch_pids = {c.prompt_pair_id for c in dispatch}
        dispatch_upids = {c.user_prompt_id for c in dispatch}
        dispatch_sids = {c.sample_id for c in dispatch}
        pair_records = _serialize_records(
            ctx.corpus_records.prompt_pairs, "prompt_pair_id", dispatch_pids
        )
        up_records = _serialize_records(
            ctx.corpus_records.user_prompts, "user_prompt_id", dispatch_upids
        )
        sample_records = _serialize_records(
            ctx.corpus_records.samples, "sample_id", dispatch_sids
        )
        payload: dict[str, Any] = {
            **expected_match,
            "attempt_id": attempt_id,
            "lifecycle_mode": ctx.lifecycle_mode,
            "lifecycle_state_after": ctx.lifecycle_state_after,
            "dispatch_cells": [c.as_row() for c in dispatch],
        }
        if pair_records:
            payload["prompt_pairs"] = pair_records
        if up_records:
            payload["user_prompts"] = up_records
        if sample_records:
            payload["samples"] = sample_records
        if ctx.dataset_root is not None:
            payload["dataset_root"] = ctx.dataset_root
        if ctx.output_root is not None:
            payload["output_root"] = ctx.output_root
            payload["artifact_root"] = ctx.output_root
        return payload

    def _stage_summary(self, result: StageResult) -> dict[str, Any]:
        return {
            "stage": result.stage,
            "survivor_count": len(result.survivor_set),
            "local_count": len(result.local_view.cell_ids()),
            "dispatch_count": len(result.dispatch_set),
            "attempt_id": result.attempt_id,
            "remote_called": result.remote_manifest is not None,
            "merged_cell_count": len(result.merged_manifest.get("cells", [])),
        }


# ---------------------------------------------------------------------------
# Resume gate (S05 wiring of S00.16a)
# ---------------------------------------------------------------------------


def gate_resume_from_run_id(
    *, ctx: RoundContext, resume_from_run_id: str
) -> Path:
    """Run the four-hash drift gate before any dispatch / copy-back.

    Raises :class:`ResumeDriftError` etc. on drift; returns the prior
    run directory on success.
    """
    return validate_resume_from_run_id(
        resume_from_run_id,
        output_root=ctx.artifact_root,
        current_hashes=ctx.corpus.current_run_hashes(),
    )
