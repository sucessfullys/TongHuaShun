"""V0.2.2 production runner (S05B.04, S05B.16–S05B.19, S05B.26–S05B.30).

Owns the autonomous round outer loop and the post-Qwen handoff:

    CLI run
      -> resolve execution mode (default v022_stage_major)
      -> for each round:
           prompt generation, budget guard, stop-signal handling
           build RoundContext (incl. RoundCorpusRecords)
           V022Orchestrator.run_round  -> RoundExecutionResult
           build_evaluation_cells from result.stage_results[*].merged_manifest
           partition pairs (failed_pairs[] / NoRankablePairsError abort)
           per-cell eval via LocalApiEvaluator.evaluate_many_cells
           scoring_v022.build_pair_ranking_rows  (NEVER pair_row_for_csv)
           memory_v022.upsert_long_memory_rows
           persist next_round_context, append failure summary
      -> emit final stop reason

The legacy AgentLoop body remains only behind explicit
``legacy_stage_all_compat`` / ``legacy_cartesian_compat`` modes;
production V0.2.2 must never fall through into it.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from .agent_loop_v022 import (
    CorpusContext,
    RoundContext,
    RoundCorpusRecords,
    StageDispatcher,
    V022Orchestrator,
)
from .config import AppConfig
from .evaluation import (
    BudgetGuard,
    LocalApiEvaluator,
    select_worst_per_category,
    summarize_failures,
    summarize_pair_for_improvement,
)
from .evaluation.failure_summary import append_summaries_jsonl
from .execution_mode_v022 import (
    PRODUCTION_MODE,
    ExecutionModeError,
    assert_production_mode_does_not_call_stage_all,
    resolve_execution_mode,
)
from .memory.manager import MemoryManager
from .memory.shared import prune_rules
from .memory_v022 import (
    V022_FIELDNAMES,
    flatten_pair_to_row_v022,
    upsert_long_memory_rows,
)
from .prompt_generation import generate_prompt_pairs
from .rate_limiter import init_rate_limiter
from .remote._vendored import canonical_paths as cp
from .remote.client import RemoteControllerClient
from .schemas import GemmaUserPrompt, PromptPair, PromptPairRequest
from .scoring_v022 import (
    EvalCell,
    NoRankablePairsError,
    apply_failed_cell_policy,
    assert_some_pairs_rankable,
    build_evaluation_cells,
    build_next_round_context_v022,
    build_pair_ranking_rows,
    partition_pairs_by_scoring_input,
    reject_empty_evaluation_per_pair,
    FailedCellPolicy,
)
from .tracing import RunTraceLogger


log = logging.getLogger("system_prompt_retrieval_agent.runner_v022")


class V022StopReason:
    THRESHOLD_MET = "threshold_met"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    BUDGET_EXCEEDED = "budget_exceeded"
    PERSISTENT_GENERATION_FAILURE = "persistent_generation_failure"
    SIGNAL = "signal"
    NO_RANKABLE_PAIRS = "no_rankable_pairs"
    EXECUTION_MODE_VIOLATION = "execution_mode_violation"
    ORCHESTRATOR_ERROR = "orchestrator_error"
    EVAL_ERROR = "eval_error"


@dataclass
class V022RunnerOutcome:
    run_id: str
    rounds_completed: int = 0
    stop_reason: str = ""
    best_pair_id: str | None = None
    best_pair_overall: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _enabled_user_prompts(cfg: AppConfig) -> list[GemmaUserPrompt]:
    """Return the enabled GemmaUserPrompt records in deterministic ID order."""
    out: list[GemmaUserPrompt] = []
    for entry in cfg.gemma_user_prompts.library:
        if not entry.enabled:
            continue
        out.append(
            GemmaUserPrompt(
                user_prompt_id=entry.user_prompt_id,
                language=entry.language,
                text=entry.text,
                enabled=True,
            )
        )
    out.sort(key=lambda u: u.user_prompt_id)
    return out


def _build_sample_user_prompt_map(
    cfg: AppConfig, sample_ids: Sequence[str]
) -> dict[str, str]:
    """Per-sample random one-of-K mapping ``sample_id -> user_prompt_id``.

    Each (pair, sample) cell will pick its user_prompt independently from
    this map. Combined with the non-cartesian survivor set in
    :func:`survivor_resume_v022.build_gemma_base_survivor_set`, this
    collapses fan-out: each (pair, sample) gets exactly ONE cell with
    ONE randomly-chosen user_prompt_id (varying across samples), instead
    of ``K`` cells per (pair, sample).
    """
    import random

    enabled = _enabled_user_prompts(cfg)
    if not enabled:
        return {}
    return {sid: random.choice(enabled).user_prompt_id for sid in sample_ids}


def _pair_to_request(pair: PromptPair) -> PromptPairRequest:
    """Project a generated PromptPair to the canonical PromptPairRequest."""
    return PromptPairRequest(
        prompt_pair_id=pair.prompt_pair_id,
        system_prompt_id=getattr(pair, "system_prompt_id", None),
        system_prompt_text=getattr(pair, "system_prompt", None),
        negative_prompt_id=getattr(pair, "negative_prompt_id", None),
        negative_prompt=getattr(pair, "negative_prompt", None),
    )


def _load_sample_lookup(cfg: AppConfig) -> dict[str, dict[str, str]]:
    """Build sample_id -> {model_image_path, cloth_image_path} lookup.

    Discovers ``sample_manifest.json`` at the dataset root (a list of
    objects with ``sample_id``, ``model_image_path``,
    ``cloth_image_path``). Returns an empty dict when missing — the
    runner surfaces a clear failure later if scoring needs samples.
    """
    manifest_path = Path(cfg.paths.dataset_root) / "sample_manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        sid = r.get("sample_id")
        if sid:
            out[sid] = {
                "model_image_path": r.get("model_image_path", ""),
                "cloth_image_path": r.get("cloth_image_path", ""),
            }
    return out


def _configured_sample_ids(cfg: AppConfig, lookup: dict[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(lookup.keys()))


def _build_corpus_context(
    cfg: AppConfig,
    *,
    pairs: list[PromptPair],
    user_prompts: list[GemmaUserPrompt],
    sample_lookup: dict[str, dict[str, str]],
) -> CorpusContext:
    """Build the per-round CorpusContext + the four canonical hashes.

    The hashes are computed over the **enabled** records that are also
    propagated into RoundCorpusRecords, keeping resume-drift detection
    consistent with the records the remote actually receives.
    """
    enabled_pair_ids = tuple(sorted(p.prompt_pair_id for p in pairs))
    enabled_up_ids = tuple(u.user_prompt_id for u in user_prompts)
    sample_ids = _configured_sample_ids(cfg, sample_lookup)

    user_prompt_records = [u.model_dump() for u in user_prompts]
    pair_records = [
        {
            "prompt_pair_id": p.prompt_pair_id,
            "system_prompt": getattr(p, "system_prompt", "") or "",
            "user_prompt_template": getattr(p, "user_prompt_template", "") or "",
            "language": getattr(p, "language", "en"),
            "enabled": True,
        }
        for p in pairs
    ]
    sample_records = [
        {
            "sample_id": sid,
            "model_path": sample_lookup[sid].get("model_image_path", ""),
            "cloth_path": sample_lookup[sid].get("cloth_image_path", ""),
            "model_sha256": "",
            "cloth_sha256": "",
        }
        for sid in sample_ids
    ]

    up_hash = cp.user_prompt_corpus_hash(user_prompt_records)
    pp_hash = cp.prompt_pair_corpus_hash(pair_records)
    sample_hash = (
        cp.sample_corpus_hash(sample_records) if sample_records else "0" * 64
    )
    try:
        cfg_hash = cp.config_hash(cfg.model_dump())
    except ValueError:
        # Required keys may be absent in test/dev configs that are not
        # the production checkpoint config. Use a deterministic stub
        # so dev runs do not silently match a prior production run_id.
        cfg_hash = "d" * 64

    return CorpusContext(
        enabled_prompt_pair_ids=enabled_pair_ids,
        enabled_user_prompt_ids=enabled_up_ids,
        configured_sample_ids=sample_ids,
        user_prompt_corpus_id=cfg.gemma_user_prompts.corpus_id,
        user_prompt_corpus_hash=up_hash,
        prompt_pair_corpus_hash=pp_hash,
        sample_corpus_hash=sample_hash,
        config_hash=cfg_hash,
    )


_SUB_SCORE_KEYS_MEM = (
    "qwen_pass_rate",
    "edit_correctness",
    "garment_transfer_correctness",
    "preservation",
    "artifact_penalty",
)
_CATEGORIES_MEM = ("dress", "lower", "upper")


def _populate_memory_row_subscores(row: dict[str, Any], r: Any) -> None:
    """Copy mean sub-scores + per-category blocks from a PairRankingRow
    into a memory row, clearing the ``scores_missing`` sentinels that
    ``flatten_pair_to_row_v022`` pre-stamps when the source PromptPair
    has ``scores=None`` (memory/long.py:133-135).
    """
    has_top_data = False
    for k in _SUB_SCORE_KEYS_MEM:
        v = getattr(r, f"mean_{k}", None)
        if v is not None:
            row[k] = v
            has_top_data = True
    if getattr(r, "total_score", None) is not None:
        row["total_score"] = r.total_score
        has_top_data = True
    if has_top_data:
        row["missing_score_reason"] = None

    per_cat = getattr(r, "per_category", {}) or {}
    for cat in _CATEGORIES_MEM:
        block = per_cat.get(cat) or {}
        n = block.get("n_cells", 0) or 0
        if n > 0:
            if "weighted_score" in block:
                row[f"{cat}_weighted_score"] = block.get("weighted_score")
            if "total_score" in block:
                row[f"{cat}_total_score"] = block.get("total_score")
            for k in _SUB_SCORE_KEYS_MEM:
                if k in block:
                    row[f"{cat}_{k}"] = block.get(k)
            row[f"{cat}_missing_score_reason"] = None
        elif block.get("missing_score_reason"):
            row[f"{cat}_missing_score_reason"] = block["missing_score_reason"]


# ---------------------------------------------------------------------------
# Production runner
# ---------------------------------------------------------------------------


class V022Runner:
    """Owns the autonomous round outer loop for production V0.2.2."""

    def __init__(
        self,
        cfg: AppConfig,
        *,
        limit: int = 0,
        max_rounds: int | None = None,
        resume_from_run_id: str | None = None,
        client_factory: Any = None,
        local_evaluator: LocalApiEvaluator | None = None,
        budget_guard: BudgetGuard | None = None,
        memory: MemoryManager | None = None,
        dispatcher: StageDispatcher | None = None,
        copy_back_transport: Any | None = None,
        prompt_pair_factory: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.limit = limit
        self.max_rounds = max_rounds or cfg.workflow.max_rounds
        self.resume_from_run_id = resume_from_run_id

        # Wire the global rate limiter from cfg.rate_limits BEFORE any code
        # path can call get_rate_limiter() with the legacy 3 rps default.
        # Honours requests_per_second, requests_per_minute, max_concurrency.
        init_rate_limiter(
            rps=cfg.rate_limits.requests_per_second,
            rpm=cfg.rate_limits.requests_per_minute,
            max_concurrency=cfg.rate_limits.max_concurrency,
        )

        # Hard-fail if config asks for a non-production mode but we got
        # here anyway.
        mode = resolve_execution_mode(cfg.model_dump())
        if mode != PRODUCTION_MODE:
            raise ExecutionModeError(
                f"V022Runner only runs in {PRODUCTION_MODE!r}; got {mode!r}"
            )

        if resume_from_run_id is not None:
            if not cp.RUN_ID_REGEX.match(resume_from_run_id):
                raise ExecutionModeError(
                    f"--resume-from-run-id does not match canonical regex: "
                    f"{resume_from_run_id!r}"
                )
            self.run_id = resume_from_run_id
        else:
            self.run_id = cp.generate_run_id()
        self.run_root = Path(cfg.paths.output_root) / "runs" / self.run_id
        self.run_root.mkdir(parents=True, exist_ok=True)

        self.memory = memory or MemoryManager(Path(cfg.paths.memory_root))

        # Fresh-run archival: move the previous run's rebuilt-from-scratch
        # state (long_memory.csv, preeval_baseline.json) into memory/history/
        # so the new run starts clean. Cross-run memory (shared_rules.csv,
        # pairs/, failures.jsonl, schedule.json) is intentionally preserved.
        # Skipped on resume so the resumed run keeps its in-progress state.
        if resume_from_run_id is None:
            archived = self.memory.archive_run_state(self.run_id)
            if archived:
                log.info(
                    "archived per-run memory before run %s: %s",
                    self.run_id,
                    {k: str(v) for k, v in archived.items()},
                )
        self.budget = budget_guard or BudgetGuard(
            cfg.budget.daily_usd_cap, cfg.budget.per_round_usd_cap,
        )
        self.local_evaluator = local_evaluator or LocalApiEvaluator(
            cfg, openai_client=None, budget_guard=self.budget,
        )
        self.trace = RunTraceLogger(
            self.run_root / "trace_v022.jsonl", cfg.logging.redact_env_keys,
        )
        self._stop_requested = False
        self._owns_dispatcher = dispatcher is None
        self._dispatcher = dispatcher
        self._client_factory = client_factory
        self._copy_back_transport = copy_back_transport
        self._prompt_pair_factory = prompt_pair_factory

    def request_stop(self) -> None:
        self._stop_requested = True

    # ------------------------------------------------------------------
    # Dispatcher lifecycle (S05B.20–S05B.22, S05B.33)
    # ------------------------------------------------------------------

    def _make_dispatcher(self) -> StageDispatcher:
        if self._dispatcher is not None:
            return self._dispatcher
        cf = self._client_factory or self._default_client_factory
        d = StageDispatcher.from_client(cf, production_mode=True)
        self._dispatcher = d
        return d

    def _default_client_factory(self) -> RemoteControllerClient:
        return RemoteControllerClient(
            base_url=self.cfg.remote.controller_base_url,
            request_timeout_s=self.cfg.remote.request_timeout_s,
            stage_retry_max=self.cfg.remote.stage_retry_max,
            lock_contention_retry_max=self.cfg.remote.lock_contention_retry_max,
            lock_contention_wait_s=self.cfg.remote.lock_contention_wait_s,
        )

    def close(self) -> None:
        if self._owns_dispatcher and self._dispatcher is not None:
            self._dispatcher.close()
            self._dispatcher = None

    # ------------------------------------------------------------------
    # Main run loop (S05B.16–S05B.19)
    # ------------------------------------------------------------------

    async def run(self) -> V022RunnerOutcome:
        outcome = V022RunnerOutcome(run_id=self.run_id)
        sample_lookup = _load_sample_lookup(self.cfg)
        # Validate ≥1 enabled entry exists; per-round random pick happens
        # below inside the loop via _select_user_prompt_for_round().
        if not _enabled_user_prompts(self.cfg):
            raise ExecutionModeError(
                "no enabled user prompts in cfg.gemma_user_prompts.library; "
                "V0.2.2 requires at least one"
            )

        consecutive_fallback = 0
        schedule = self.memory.load_schedule()
        start_round = (
            schedule.get("current_round") or self.memory.determine_round_id()
        )

        dispatcher = self._make_dispatcher()
        # Production-mode hard-fail check at construction time (S09.05).
        assert_production_mode_does_not_call_stage_all(
            PRODUCTION_MODE, "/v022/stage/gemma"
        )
        orchestrator = V022Orchestrator(
            dispatcher=dispatcher,
            copy_back_transport=self._copy_back_transport,
        )

        stop_reason: str | None = None
        round_id = start_round - 1
        for round_id in range(start_round, self.max_rounds + 1):
            if self._stop_requested:
                stop_reason = V022StopReason.SIGNAL
                break

            round_dir = self.run_root / "rounds" / f"round_{round_id:03d}"
            for sub in ("scoring", "failures", "visualizations", "api_eval"):
                (round_dir / sub).mkdir(parents=True, exist_ok=True)

            # --- prompt generation (S05B.17) ---
            try:
                if self._prompt_pair_factory is not None:
                    pairs, fallback_used = await self._prompt_pair_factory(
                        self.cfg, self.memory, round_id,
                        self.cfg.prompt_generation.prompt_pairs_per_round, [],
                    )
                else:
                    pairs, fallback_used = await generate_prompt_pairs(
                        self.cfg,
                        self.memory,
                        round_id,
                        self.cfg.prompt_generation.prompt_pairs_per_round,
                        [],
                    )
            except Exception as exc:  # noqa: BLE001
                log.exception("prompt generation failed in round %d", round_id)
                self.trace.emit("prompt_gen_error", round_id=round_id, error=str(exc))
                consecutive_fallback += 1
                self.memory.mark_fallback_round(round_id, True)
                if consecutive_fallback >= 3:
                    stop_reason = V022StopReason.PERSISTENT_GENERATION_FAILURE
                    break
                continue

            (round_dir / "prompt_pairs.json").write_text(
                json.dumps([p.model_dump() for p in pairs], default=str, indent=2),
                encoding="utf-8",
            )
            log.info("round %d: prompt pairs written (count=%d)", round_id, len(pairs))

            # Persist memory/pairs/<run_id>/<pid>.yaml IMMEDIATELY so the
            # system_prompt text is recoverable even if the run crashes
            # mid-stage. Run-scoped subdir keeps round-scoped ids like
            # `pair_r01_000` from clobbering each other across runs.
            # The late end-of-round write is idempotent (same content).
            for _pair in pairs:
                self.memory.write_pair(_pair.prompt_pair_id, {
                    "prompt_pair_id": _pair.prompt_pair_id,
                    "system_prompt_id": _pair.system_prompt_id,
                    "negative_prompt_id": _pair.negative_prompt_id,
                    "run_id": self.run_id,
                    "round_id": round_id,
                    "selection_role": _pair.selection_role,
                    "system_prompt": _pair.system_prompt,
                    "negative_prompt": _pair.negative_prompt,
                    "rationale": _pair.rationale,
                }, run_id=self.run_id)
            if fallback_used:
                consecutive_fallback += 1
                self.memory.mark_fallback_round(round_id, True)
                if consecutive_fallback >= 3:
                    stop_reason = V022StopReason.PERSISTENT_GENERATION_FAILURE
                    break
            else:
                consecutive_fallback = 0
                self.memory.mark_fallback_round(round_id, False)

            # --- budget guard (S05B.18) ---
            if self.budget.daily_spent >= self.cfg.budget.daily_usd_cap:
                stop_reason = V022StopReason.BUDGET_EXCEEDED
                break

            # --- per-sample random user-prompt assignment ---
            # Each (pair, sample) cell picks its own user_prompt_id from the
            # enabled library. Corpus user_prompts is the full enabled set so
            # the Gemma adapter can resolve any chosen up_id.
            user_prompts = _enabled_user_prompts(self.cfg)
            sample_user_prompt_map = _build_sample_user_prompt_map(
                self.cfg, list(sample_lookup.keys())
            )
            import collections as _collections
            up_freq = _collections.Counter(sample_user_prompt_map.values())
            log.info(
                "round %d: per-sample random user_prompt assignment over %d samples "
                "from %d enabled (distribution=%s)",
                round_id, len(sample_user_prompt_map), len(user_prompts),
                dict(up_freq),
            )

            # --- build RoundContext incl. RoundCorpusRecords ---
            corpus = _build_corpus_context(
                self.cfg, pairs=pairs, user_prompts=user_prompts,
                sample_lookup=sample_lookup,
            )
            log.info(
                "round %d: corpus built — pairs=%d user_prompts=%d samples=%d",
                round_id,
                len(corpus.enabled_prompt_pair_ids),
                len(corpus.enabled_user_prompt_ids),
                len(corpus.configured_sample_ids),
            )
            remote_local_root = (
                self.cfg.remote.v022_artifact_root
                or self.cfg.paths.remote_image_service_root
            )
            rsync_remote_root = (
                f"{self.cfg.remote.ssh_alias}:{remote_local_root}"
            )
            ctx = RoundContext(
                run_id=self.run_id,
                round_id=round_id,
                artifact_root=Path(self.cfg.paths.output_root).parent.parent,  # strip outputs/v02; canonical_paths re-prepends it
                remote_artifact_root=rsync_remote_root,
                corpus=corpus,
                lifecycle_mode="cold",
                lifecycle_state_after="disk_unloaded",
                allow_partial=self.cfg.workflow.allow_partial_samples,
                corpus_records=RoundCorpusRecords(
                    prompt_pairs=tuple(_pair_to_request(p) for p in pairs),
                    user_prompts=tuple(user_prompts),
                    samples=tuple(
                        {
                            "sample_id": sid,
                            "model_path": sample_lookup[sid].get("model_image_path", ""),
                            "cloth_path": sample_lookup[sid].get("cloth_image_path", ""),
                            "category": (sample_lookup[sid].get("category") or sid.split("__", 1)[0]),
                        }
                        for sid in corpus.configured_sample_ids
                        if sid in sample_lookup
                    ),
                ),
                dataset_root=self.cfg.paths.dataset_root,
                output_root=remote_local_root,
                sample_user_prompt_map=sample_user_prompt_map,
            )

            # --- run V0.2.2 orchestrator (S05B.05) ---
            log.info("round %d: invoking orchestrator.run_round", round_id)
            try:
                orch_result = await asyncio.to_thread(
                    orchestrator.run_round, ctx
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("orchestrator failed in round %d", round_id)
                self.trace.emit("orchestrator_error", round_id=round_id, error=str(exc))
                stop_reason = V022StopReason.ORCHESTRATOR_ERROR
                break

            # --- post-Qwen scoring handoff (S05B.26, S05B.30) ---
            qwen_merged = orch_result.stage_results["qwen"].merged_manifest
            flux_merged = orch_result.stage_results["flux"].merged_manifest
            gemma_merged = orch_result.stage_results["gemma"].merged_manifest

            qwen_pairs = sorted({
                c["prompt_pair_id"] for c in qwen_merged.get("cells", [])
                if c.get("status") in cp.SUCCESSFUL_CELL_STATUSES
            })

            try:
                eval_cells = build_evaluation_cells(
                    artifact_root=Path(self.cfg.paths.output_root).parent.parent,  # strip outputs/v02; canonical_paths re-prepends it
                    run_id=self.run_id,
                    round_id=round_id,
                    qwen_merged=qwen_merged,
                    flux_merged=flux_merged,
                    gemma_merged=gemma_merged,
                    sample_lookup=sample_lookup,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("build_evaluation_cells failed")
                self.trace.emit("scoring_build_error", error=str(exc))
                continue

            try:
                cells_by_pair = reject_empty_evaluation_per_pair(
                    eval_cells, qwen_surviving_pairs=qwen_pairs,
                )
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "round %d: empty evaluation rejected for some pair: %s",
                    round_id, exc,
                )
                cells_by_pair = {}
                for c in eval_cells:
                    cells_by_pair.setdefault(c.prompt_pair_id, []).append(c)

            partition = partition_pairs_by_scoring_input(
                qwen_surviving_pairs=qwen_pairs,
                eval_cells_by_pair=cells_by_pair,
            )
            if partition.failed_pairs:
                (round_dir / "failures" / "failed_pairs.json").write_text(
                    json.dumps(partition.failed_pairs, indent=2),
                    encoding="utf-8",
                )
            try:
                assert_some_pairs_rankable(partition)
            except NoRankablePairsError:
                stop_reason = V022StopReason.NO_RANKABLE_PAIRS
                self.trace.emit(
                    "no_rankable_pairs",
                    round_id=round_id,
                    failed_pairs=partition.failed_pairs,
                )
                break

            # --- per-cell evaluation (S05B.27 partial path) ---
            # Any per-pair eval failure halts the round immediately. Never
            # substitute empty results for a real failure: a phantom "all
            # axes = 0" ranking row poisons memory + next-round prompts.
            cell_results_by_pair: dict[str, list[dict[str, Any]]] = {}
            eval_failed = False
            for pid in partition.rankable_pairs:
                cells = cells_by_pair.get(pid, [])
                # Stamp run_id / round_id on every cell so the Level-1
                # cell-key for resume includes them. Without this the
                # key collapses (run_id="", round_id=0) and we'd skip
                # cells from other runs.
                inputs = []
                for c in cells:
                    cell_dict = c.as_evaluator_input()
                    cell_dict["run_id"] = self.run_id
                    cell_dict["round_id"] = round_id
                    inputs.append(cell_dict)
                try:
                    results = await self.local_evaluator.evaluate_many_cells(
                        inputs, api_eval_root=round_dir / "api_eval",
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("local eval failed for pair %s", pid)
                    self.trace.emit("eval_error", pair_id=pid, error=str(exc))
                    stop_reason = V022StopReason.EVAL_ERROR
                    eval_failed = True
                    break
                cell_results_by_pair[pid] = apply_failed_cell_policy(
                    results, FailedCellPolicy(),
                )
            if eval_failed:
                break

            # --- V0.2.2-native ranking rows (S05B.28) ---
            ranking_rows = build_pair_ranking_rows(
                run_id=self.run_id,
                round_id=round_id,
                eval_cells_by_pair={
                    pid: cells_by_pair[pid] for pid in partition.rankable_pairs
                },
                cell_results_by_pair=cell_results_by_pair,
                score_weights=self.cfg.evaluation.weights.model_dump(),
            )
            (round_dir / "scoring" / "ranking_v022.json").write_text(
                json.dumps([r.as_dict() for r in ranking_rows], indent=2),
                encoding="utf-8",
            )

            # --- bottom-N worst cells (diagnostic for next-round prompts) ---
            worst_all: list[dict[str, Any]] = []
            for r in ranking_rows:
                for wc in r.worst_cells:
                    worst_all.append({"prompt_pair_id": r.prompt_pair_id, **wc})
            worst_all.sort(key=lambda x: x.get("overall", 1.0))
            worst_all = worst_all[:50]
            (round_dir / "failures").mkdir(parents=True, exist_ok=True)
            (round_dir / "failures" / "failed_cells.json").write_text(
                json.dumps(worst_all, indent=2), encoding="utf-8",
            )

            # --- memory upsert (S05B.29) ---
            mem_rows = []
            pair_lookup = {p.prompt_pair_id: p for p in pairs}
            for r in ranking_rows:
                pair_obj = pair_lookup.get(r.prompt_pair_id)
                if pair_obj is None:
                    continue
                row = flatten_pair_to_row_v022(
                    pair_obj, run_id=self.run_id, round_id=round_id,
                )
                row["pair_overall"] = r.pair_overall
                row["overall_score"] = r.pair_overall
                _populate_memory_row_subscores(row, r)
                mem_rows.append(row)
            if mem_rows:
                long_mem_path = Path(self.cfg.paths.memory_root) / "long_memory.csv"
                upsert_long_memory_rows(long_mem_path, mem_rows)

            # --- per-pair failure summarisation + improvement directive ---
            # Worst K_PER_CATEGORY=7 cells per (dress|lower|upper) → gpt-5.4-mini
            # per-sample failure summaries → 1 per-pair improvement directive.
            # All cost flows through self.budget; rate-limited via failure_summary module.
            failure_summaries_by_pair: dict[str, list[Any]] = {}
            improvement_summary_by_pair: dict[str, str] = {}
            failures_dir = round_dir / "failures"
            failures_dir.mkdir(parents=True, exist_ok=True)
            mem_failures_jsonl = (
                Path(self.cfg.paths.memory_root) / "failures.jsonl"
            )
            for r in ranking_rows:
                pid = r.prompt_pair_id
                cell_results = cell_results_by_pair.get(pid, [])
                if not cell_results:
                    continue
                worst = select_worst_per_category(
                    cell_results,
                    k_per_category=7,
                    score_weights=self.cfg.evaluation.weights.model_dump(),
                )
                if not worst:
                    continue
                try:
                    fs_list = await summarize_failures(
                        worst,
                        prompt_pair_id=pid,
                        cfg=self.cfg,
                        k=len(worst),  # already pre-selected; do not re-truncate
                        budget_guard=self.budget,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("summarize_failures failed for pair %s", pid)
                    self.trace.emit("summarize_failures_error", pair_id=pid, error=str(exc))
                    fs_list = []
                failure_summaries_by_pair[pid] = fs_list

                if fs_list:
                    append_summaries_jsonl(
                        failures_dir / f"{pid}.jsonl",
                        fs_list,
                        prompt_pair_id=pid,
                        round_id=round_id,
                    )
                    append_summaries_jsonl(
                        mem_failures_jsonl,
                        fs_list,
                        prompt_pair_id=pid,
                        round_id=round_id,
                    )

                pair_obj = pair_lookup.get(pid)
                sys_prompt = (
                    (pair_obj.system_prompt if pair_obj else None) or ""
                )
                try:
                    improvement_summary_by_pair[pid] = await summarize_pair_for_improvement(
                        prompt_pair_id=pid,
                        system_prompt=sys_prompt,
                        ranking_row=r,
                        failure_summaries=fs_list,
                        cfg=self.cfg,
                        budget_guard=self.budget,
                    )
                except Exception as exc:  # noqa: BLE001
                    log.exception("summarize_pair_for_improvement failed for %s", pid)
                    self.trace.emit("improvement_summary_error", pair_id=pid, error=str(exc))
                    improvement_summary_by_pair[pid] = ""

            # Persist round-level pair YAMLs so shared_rules.prompt_path resolves later.
            # Run-scoped subdir prevents cross-run id collisions (pair_r01_000 etc.).
            for pid, pair_obj in pair_lookup.items():
                self.memory.write_pair(pid, {
                    "prompt_pair_id": pid,
                    "system_prompt_id": pair_obj.system_prompt_id,
                    "negative_prompt_id": pair_obj.negative_prompt_id,
                    "run_id": self.run_id,
                    "round_id": round_id,
                    "selection_role": pair_obj.selection_role,
                    "system_prompt": pair_obj.system_prompt,
                    "negative_prompt": pair_obj.negative_prompt,
                    "rationale": pair_obj.rationale,
                }, run_id=self.run_id)

            # --- shared rules update (cross-round digest) ---
            self._update_shared_rules(
                round_id=round_id,
                ranking_rows=ranking_rows,
                pair_lookup=pair_lookup,
                failure_summaries_by_pair=failure_summaries_by_pair,
                improvement_summary_by_pair=improvement_summary_by_pair,
            )

            # --- next-round context (S05B.29) ---
            next_ctx = build_next_round_context_v022(ranking_rows, top_n=3)
            (round_dir / "next_round_context.json").write_text(
                json.dumps(next_ctx, indent=2), encoding="utf-8",
            )

            # --- stop decisions ---
            best = ranking_rows[0] if ranking_rows else None
            if best is not None:
                if (outcome.best_pair_overall is None
                        or best.pair_overall > outcome.best_pair_overall):
                    outcome.best_pair_id = best.prompt_pair_id
                    outcome.best_pair_overall = best.pair_overall
                if best.pair_overall >= self.cfg.workflow.score_threshold:
                    stop_reason = V022StopReason.THRESHOLD_MET
                    outcome.rounds_completed = round_id
                    break
            outcome.rounds_completed = round_id
            if self.budget.daily_spent >= self.cfg.budget.daily_usd_cap:
                stop_reason = V022StopReason.BUDGET_EXCEEDED
                break
            self.memory.save_schedule(
                {"current_round": round_id + 1, "next_round_context": next_ctx}
            )

        if stop_reason is None:
            stop_reason = V022StopReason.MAX_ROUNDS_REACHED
        outcome.stop_reason = stop_reason
        self.trace.emit(
            "v022_run_complete",
            run_id=self.run_id, rounds=outcome.rounds_completed,
            stop_reason=stop_reason, best_pair_id=outcome.best_pair_id,
        )

        # Finalize
        (self.run_root / "run_summary_v022.md").write_text(
            (
                f"# V0.2.2 run {self.run_id}\n\n"
                f"- stop_reason: {stop_reason}\n"
                f"- rounds_completed: {outcome.rounds_completed}\n"
                f"- best_pair: {outcome.best_pair_id or 'none'}\n"
                f"- best_pair_overall: {outcome.best_pair_overall}\n"
                f"- finalized: {datetime.utcnow().isoformat()}Z\n"
            ),
            encoding="utf-8",
        )

        self.close()
        return outcome

    def _update_shared_rules(
        self,
        *,
        round_id: int,
        ranking_rows: list,
        pair_lookup: dict[str, PromptPair] | None = None,
        failure_summaries_by_pair: dict[str, list[Any]] | None = None,
        improvement_summary_by_pair: dict[str, str] | None = None,
    ) -> None:
        """Append/refresh per-pair entries in memory/shared_rules.csv each round.

        One rule per ranked pair carries everything ChatGPT needs for the next
        round: linkage to the prompt YAML (so context_builder can inline the
        real system_prompt text), per-axis means, language split, top failure
        tags, the pair_overall delta vs. its prior row, and the gpt-5.4-mini
        improvement_summary directive. rule_id = pair__<pair_id>; re-appearances
        refresh last_seen and metrics. Stale low-confidence rules are pruned.
        """
        if not ranking_rows:
            return
        pair_lookup = pair_lookup or {}
        failure_summaries_by_pair = failure_summaries_by_pair or {}
        improvement_summary_by_pair = improvement_summary_by_pair or {}

        try:
            existing = self.memory.load_rules()
        except Exception:
            existing = []
        by_id = {r.get("rule_id"): dict(r) for r in existing if r.get("rule_id")}

        sub_keys = (
            ("qwen_pass_rate", False),
            ("edit_correctness", False),
            ("garment_transfer_correctness", False),
            ("preservation", False),
            ("artifact_penalty", True),  # higher is worse
        )
        for r in ranking_rows:
            pid = r.prompt_pair_id
            best_axis = None
            worst_axis = None
            best_val = None
            worst_val = None
            for name, inverted in sub_keys:
                v = getattr(r, f"mean_{name}", None)
                if v is None:
                    continue
                eff = -v if inverted else v
                if best_val is None or eff > best_val:
                    best_val, best_axis = eff, name
                if worst_val is None or eff < worst_val:
                    worst_val, worst_axis = eff, name

            axis_means = {
                "qwen_pass_rate": getattr(r, "mean_qwen_pass_rate", None),
                "edit_correctness": getattr(r, "mean_edit_correctness", None),
                "garment_transfer_correctness": getattr(
                    r, "mean_garment_transfer_correctness", None
                ),
                "preservation": getattr(r, "mean_preservation", None),
                "artifact_penalty": getattr(r, "mean_artifact_penalty", None),
            }

            # Language split derived from per_user_prompt_overall keys (zh*/en*)
            per_up = getattr(r, "per_user_prompt_overall", None) or {}
            zh_vals = [v for k, v in per_up.items() if k.startswith("zh")]
            en_vals = [v for k, v in per_up.items() if k.startswith("en")]
            zh_mean = (sum(zh_vals) / len(zh_vals)) if zh_vals else None
            en_mean = (sum(en_vals) / len(en_vals)) if en_vals else None
            worst_user_prompt_id = (
                min(per_up.items(), key=lambda kv: kv[1])[0] if per_up else ""
            )

            # Failure-tag frequency table (top 10)
            tag_counts: dict[str, int] = {}
            for fs in failure_summaries_by_pair.get(pid, []):
                for t in getattr(fs, "failure_tags", None) or []:
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:10]
            worst_failure_tags = ", ".join(f"{t}:{c}" for t, c in top_tags)

            # Delta vs previous round's row (if we previously rated this pair)
            prev_overall_str = (by_id.get(f"pair__{pid}") or {}).get("pair_overall")
            try:
                prev_overall = float(prev_overall_str) if prev_overall_str else None
            except (TypeError, ValueError):
                prev_overall = None
            delta_vs_prev = (
                f"{(r.pair_overall - prev_overall):+.4f}"
                if prev_overall is not None else ""
            )

            improvement = improvement_summary_by_pair.get(pid, "") or ""
            text = (
                f"Round {round_id} pair {pid}: pair_overall={r.pair_overall:.3f}, "
                f"strongest={best_axis or 'n/a'}, weakest={worst_axis or 'n/a'}."
                + (f" Improve: {improvement}" if improvement else "")
            )
            rule_id = f"pair__{pid}"
            confidence = max(0.0, min(1.0, float(r.pair_overall or 0.0)))
            tags = ",".join(filter(None, [
                f"round_{round_id}",
                f"pair_{pid}",
                best_axis and f"strong_{best_axis}",
                worst_axis and f"weak_{worst_axis}",
            ]))

            pair_obj = pair_lookup.get(pid)
            by_id[rule_id] = {
                "schema_version": "1.0",
                "rule_id": rule_id,
                "text": text,
                "source_round": str(by_id.get(rule_id, {}).get("source_round") or round_id),
                "confidence": f"{confidence:.4f}",
                "last_seen": str(round_id),
                "tags": tags,
                "system_prompt_id": (pair_obj.system_prompt_id if pair_obj else pid),
                "prompt_pair_id": pid,
                "source_run_id": self.run_id,
                "n_cells": str(getattr(r, "n_cells", "") or ""),
                "pair_overall": f"{r.pair_overall:.4f}",
                "prompt_path": f"pairs/{self.run_id}/{pid}.yaml",
                "improvement_summary": improvement,
                "axis_means_json": json.dumps(axis_means),
                "worst_failure_tags": worst_failure_tags,
                "worst_user_prompt_id": worst_user_prompt_id,
                "zh_mean_score": (f"{zh_mean:.4f}" if zh_mean is not None else ""),
                "en_mean_score": (f"{en_mean:.4f}" if en_mean is not None else ""),
                "delta_vs_prev": delta_vs_prev,
            }

        kept, _evicted = prune_rules(list(by_id.values()), current_round=round_id)
        try:
            self.memory.save_rules(kept)
        except Exception:
            log.exception("save_rules failed for round %d", round_id)
