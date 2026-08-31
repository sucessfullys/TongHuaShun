"""Top-level 7-step agent loop (plan §7)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from .config import AppConfig
from .evaluation import BudgetGuard, LocalApiEvaluator, parse_qwen_summary, summarize_failures
from .logging_setup import setup_logging
from .memory.manager import MemoryManager
from .prompt_generation import generate_prompt_pairs
from .remote import (
    RemoteControllerClient,
    assert_all_workers_done,
    rsync_copyback,
)
from .remote.manifest_validator import validate_manifest
from .schemas import FailureSummary, PromptPair, RemoteStageRequest, RemoteStageResponse
from .scoring import compute_overall_score, rank_pairs, write_ranking_artifacts
from .scoring.ranking import pair_row_for_csv, build_next_round_context
from .tracing import RunTraceLogger
from .visualization import compose_round_grids

log = logging.getLogger("system_prompt_retrieval_agent.agent_loop")


STATE_INITIALIZED = "initialized"
STATE_GENERATING = "generating_pairs"
STATE_REMOTE = "remote_running"
STATE_EVALUATING = "evaluating"
STATE_SCORING = "scoring"
STATE_SUMMARIZING = "summarizing_failures"
STATE_UPDATING_MEMORY = "updating_memory"
STATE_COMPLETED = "completed"
STATE_FAILED = "failed"


def _config_hash(cfg: AppConfig) -> str:
    blob = json.dumps(cfg.redacted_dump(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


class StopReason:
    THRESHOLD_MET = "threshold_met"
    MAX_ROUNDS_REACHED = "max_rounds_reached"
    COST_EXHAUSTED = "cost_exhausted"
    PERSISTENT_GENERATION_FAILURE = "persistent_generation_failure"
    SIGNAL = "signal"
    FATAL_REMOTE = "fatal_remote_error"


class AgentLoop:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        limit: int = 0,
        max_rounds: int | None = None,
        dry_run: bool = False,
        resume_run_id: str | None = None,
        resume_from_run_id: str | None = None,
        http_client=None,
        openai_client=None,
    ) -> None:
        self.cfg = cfg
        self.limit = limit
        self.max_rounds = max_rounds or cfg.workflow.max_rounds
        self.dry_run = dry_run
        self.http_client = http_client
        self.openai_client = openai_client
        setup_logging(cfg.logging.level, cfg.logging.redact_env_keys)

        self.memory = MemoryManager(Path(cfg.paths.memory_root))
        # S00.16a / plan §2.10c: when --resume-from-run-id is set, the new
        # invocation reuses RUN_ID as its own run_id. The four-hash drift
        # gate runs at S05 dispatch time, before any remote call. Storing
        # the parameter here is the parent-only contract surface; full
        # enforcement is wired by S05.
        self.resume_from_run_id = resume_from_run_id
        if resume_from_run_id is not None:
            self.run_id = resume_from_run_id
        else:
            self.run_id = resume_run_id or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.run_root = Path(cfg.paths.output_root) / "runs" / self.run_id
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.trace = RunTraceLogger(self.run_root / "trace.jsonl", cfg.logging.redact_env_keys)
        self.budget = BudgetGuard(cfg.budget.daily_usd_cap, cfg.budget.per_round_usd_cap)
        self._state: str = STATE_INITIALIZED
        self._stop_requested = False
        self._install_signal_handlers()

    # ── Signal handling (S09.08) ──────────────────────────────────────
    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):  # noqa: ARG001
            self._stop_requested = True
            log.warning("Signal %s received; flushing and exiting.", signum)
        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except (ValueError, OSError):
            pass

    # ── Status reader (CLI --status) ─────────────────────────────────
    @staticmethod
    def read_status(cfg: AppConfig, run_id: str | None) -> dict:
        run_root = Path(cfg.paths.output_root) / "runs" / (run_id or "")
        state_file = run_root / "run_state.json"
        if state_file.is_file():
            return json.loads(state_file.read_text())
        return {"state": "no_run_found", "run_id": run_id}

    def _transition(self, new_state: str, **fields: Any) -> None:
        self._state = new_state
        state_file = self.run_root / "run_state.json"
        payload = {"state": new_state, "run_id": self.run_id, "ts": datetime.utcnow().isoformat() + "Z", **fields}
        state_file.write_text(json.dumps(payload, default=str, indent=2))
        self.trace.emit("state_transition", **payload)

    # ── Main run ──────────────────────────────────────────────────────
    async def run(self) -> int:
        self._transition(STATE_INITIALIZED)
        snapshot = self.run_root / "run_config_snapshot.yaml"
        snapshot.write_text(yaml.safe_dump(self.cfg.redacted_dump()))

        if self.dry_run:
            self._dry_run_plan()
            return 0

        consecutive_fallback = 0
        schedule = self.memory.load_schedule()
        start_round = schedule.get("current_round") or self.memory.determine_round_id()
        best_pair: PromptPair | None = None
        stop_reason: str | None = None

        client = RemoteControllerClient(
            base_url=self.cfg.remote.controller_base_url,
            request_timeout_s=self.cfg.remote.request_timeout_s,
            stage_retry_max=self.cfg.remote.stage_retry_max,
            lock_contention_retry_max=self.cfg.remote.lock_contention_retry_max,
            lock_contention_wait_s=self.cfg.remote.lock_contention_wait_s,
            http_client=self.http_client,
        )
        local_evaluator = LocalApiEvaluator(self.cfg, openai_client=self.openai_client, budget_guard=self.budget)

        for round_id in range(start_round, self.max_rounds + 1):
            if self._stop_requested:
                stop_reason = StopReason.SIGNAL
                break
            round_dir = self.run_root / "rounds" / f"round_{round_id:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            (round_dir / "remote_manifests").mkdir(exist_ok=True)
            (round_dir / "scoring").mkdir(exist_ok=True)
            (round_dir / "failures").mkdir(exist_ok=True)
            (round_dir / "visualizations").mkdir(exist_ok=True)
            (round_dir / "api_eval").mkdir(exist_ok=True)

            # Step 2: generate pairs
            self._transition(STATE_GENERATING, round_id=round_id)
            existing: list[PromptPair] = []  # previous round's pairs loaded from memory as dupe guard
            pairs, fallback_used = await generate_prompt_pairs(
                self.cfg, self.memory, round_id, self.cfg.prompt_generation.prompt_pairs_per_round, existing,
            )
            (round_dir / "prompt_pairs.json").write_text(json.dumps([p.model_dump() for p in pairs], default=str, indent=2))
            if fallback_used:
                consecutive_fallback += 1
                self.memory.mark_fallback_round(round_id, True)
                if consecutive_fallback >= 3:
                    stop_reason = StopReason.PERSISTENT_GENERATION_FAILURE
                    break
            else:
                consecutive_fallback = 0
                self.memory.mark_fallback_round(round_id, False)

            # Step 3: per-pair remote + evaluation
            scored_pairs: list[PromptPair] = []
            for pair in pairs:
                if self._stop_requested:
                    break
                self._transition(STATE_REMOTE, round_id=round_id, pair_id=pair.prompt_pair_id)
                pair_dir = round_dir / pair.prompt_pair_id
                pair_dir.mkdir(exist_ok=True)

                req = RemoteStageRequest(
                    run_id=self.run_id,
                    round_id=round_id,
                    prompt_pair_id=pair.prompt_pair_id,
                    system_prompt_id=pair.system_prompt_id,
                    system_prompt_text=pair.system_prompt,
                    negative_prompt_id=pair.negative_prompt_id,
                    negative_prompt=pair.negative_prompt,
                    dataset_root=self.cfg.paths.dataset_root,
                    output_root=f"{self.cfg.paths.remote_image_service_root}/outputs/v02",
                    limit=self.limit,
                    allow_partial=self.cfg.workflow.allow_partial_samples,
                )

                # Resume check
                cell_id = f"{pair.system_prompt_id}__{pair.negative_prompt_id}"
                stage_summary_local = pair_dir / "remote_outputs" / "stage3_qwen" / "stage_summary.json"
                config_hash_file = pair_dir / ".config_hash"
                if (self.cfg.workflow.resume_from_artifacts and stage_summary_local.is_file()
                        and config_hash_file.is_file() and config_hash_file.read_text().strip() == _config_hash(self.cfg)):
                    log.info("Resume: skipping %s (artifacts match hash)", pair.prompt_pair_id)
                    self.trace.emit("resume_skip_pair", pair_id=pair.prompt_pair_id)
                else:
                    try:
                        resp: RemoteStageResponse = await client.run_stage("all", req)
                    except Exception as exc:  # noqa: BLE001
                        log.exception("Remote stage failed for %s", pair.prompt_pair_id)
                        self.trace.emit("remote_error", pair_id=pair.prompt_pair_id, error=str(exc))
                        pair.fallback = True
                        continue

                    if not resp.ok or resp.manifest is None:
                        log.error("Remote returned ok=False for %s: %s", pair.prompt_pair_id, resp.message)
                        self.trace.emit("remote_not_ok", pair_id=pair.prompt_pair_id, message=resp.message)
                        continue

                    (round_dir / "remote_manifests" / f"{pair.prompt_pair_id}_all.json").write_text(
                        json.dumps(resp.manifest.model_dump(), default=str, indent=2)
                    )

                    # Step 3c: copy-back. The remote controller writes to
                    # `<state.output_root>/<run_id>/<cell_id>/<prompt_id>/<stage>/...`.
                    # state.output_root on the V0.1.1 remote is `outputs/v01` (relative
                    # to the remote app dir). Prompt_id may be an inline_<hash> we don't
                    # know locally, so we copy the entire {run_id}/{cell_id}/ tree and
                    # find stage_summary.json by recursive lookup.
                    remote_pair_root = f"{self.cfg.paths.remote_image_service_root}/outputs/v01/{self.run_id}/{cell_id}"
                    try:
                        rsync_copyback(
                            remote_path=remote_pair_root,
                            local_path=pair_dir / "remote_outputs",
                            ssh_alias=self.cfg.remote.ssh_alias,
                            required_file="",  # we'll discover via glob
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.error("Copy-back failed for %s: %s", pair.prompt_pair_id, exc)
                        self.trace.emit("copyback_error", pair_id=pair.prompt_pair_id, error=str(exc))

                    # Barrier
                    try:
                        assert_all_workers_done(
                            resp.manifest,
                            allow_partial=self.cfg.workflow.allow_partial_samples,
                            post_unload_vram_free_min_gib=self.cfg.remote.post_unload_vram_free_min_gib,
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.error("Barrier violation for %s: %s", pair.prompt_pair_id, exc)
                        self.trace.emit("barrier_violation", pair_id=pair.prompt_pair_id, error=str(exc))
                        continue

                    config_hash_file.write_text(_config_hash(self.cfg))

                # Step 3d/3e: evaluation + scoring
                self._transition(STATE_EVALUATING, round_id=round_id, pair_id=pair.prompt_pair_id)
                qwen = {}
                # Discover stage_summary.json by recursion (prompt_id subdir is
                # an inline-hash we don't know locally).
                summaries = list((pair_dir / "remote_outputs").rglob("stage3_qwen/stage_summary.json"))
                if summaries:
                    try:
                        qwen = parse_qwen_summary(summaries[0])
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Qwen parse failed for %s: %s", pair.prompt_pair_id, exc)

                # local api eval (optional)
                local_results: list[dict] = []
                if self.cfg.evaluation.run_local_api_eval:
                    try:
                        local_results = await local_evaluator.evaluate_many([])  # real flux outputs discovery TBD
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Local eval skipped: %s", exc)

                # Aggregate
                from .schemas import PromptPairScoreContext, PromptPairSubScores

                sub = {
                    "qwen_pass_rate": qwen.get("qwen_pass_rate", 0.0),
                    "edit_correctness": 0.0 if not local_results else sum(r.get("edit_correctness", 0) for r in local_results) / len(local_results),
                    "garment_transfer_correctness": 0.0 if not local_results else sum(r.get("garment_transfer_correctness", 0) for r in local_results) / len(local_results),
                    "preservation": 0.0 if not local_results else sum(r.get("preservation", 0) for r in local_results) / len(local_results),
                    "artifact_penalty": 0.0 if not local_results else sum(r.get("artifact_penalty", 0) for r in local_results) / len(local_results),
                }
                overall = compute_overall_score(sub, self.cfg.evaluation.weights.model_dump())
                pair.scores = PromptPairScoreContext(
                    overall_score=overall,
                    sub_scores=PromptPairSubScores(**sub),
                )
                scored_pairs.append(pair)
                self.trace.emit("pair_scored", pair_id=pair.prompt_pair_id, overall_score=overall)

            # Step 4: summarize failures
            self._transition(STATE_SUMMARIZING, round_id=round_id)
            # (placeholder — would iterate low-score samples per pair)

            # Step 5: visual grids (round >= 2)
            prev_round = round_id - 1
            prev_scoring = self.run_root / "rounds" / f"round_{prev_round:03d}" / "scoring" / "weighted_scores.csv"
            if prev_round >= 1 and prev_scoring.is_file():
                try:
                    compose_round_grids(None, [], {}, round_dir / "visualizations")  # image_map discovery TBD
                except Exception:
                    pass

            # Step 6: update memory
            self._transition(STATE_UPDATING_MEMORY, round_id=round_id)
            ranked = rank_pairs(scored_pairs)
            write_ranking_artifacts(round_dir, ranked, round_id=round_id)
            for p in ranked:
                self.memory.append_long_memory(pair_row_for_csv(p, round_id))
                self.memory.write_pair(p.prompt_pair_id, p.model_dump())
            self.memory.prune_long_memory(50)

            # Step 7: decide stop
            best_this_round = ranked[0] if ranked else None
            if best_this_round and best_this_round.scores and best_this_round.scores.overall_score is not None:
                if best_pair is None or (best_pair.scores.overall_score or 0) < best_this_round.scores.overall_score:
                    best_pair = best_this_round
                if best_this_round.scores.overall_score >= self.cfg.workflow.score_threshold:
                    stop_reason = StopReason.THRESHOLD_MET
                    break

            if self.budget.daily_spent >= self.cfg.budget.daily_usd_cap:
                stop_reason = StopReason.COST_EXHAUSTED
                break

        if stop_reason is None and round_id >= self.max_rounds:
            stop_reason = StopReason.MAX_ROUNDS_REACHED

        self._finalize(best_pair, stop_reason or "completed")
        self._transition(STATE_COMPLETED, stop_reason=stop_reason)
        return 0

    def _finalize(self, best_pair: PromptPair | None, stop_reason: str) -> None:
        out = {
            "run_id": self.run_id,
            "stop_reason": stop_reason,
            "best_pair": best_pair.model_dump() if best_pair else None,
            "budget_spent_usd": self.budget.daily_spent,
        }
        (Path(self.cfg.paths.output_root) / "best_pair.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
        (self.run_root / "run_summary.md").write_text(
            f"# Run {self.run_id}\n\n- stop_reason: {stop_reason}\n- best_pair: {best_pair.prompt_pair_id if best_pair else 'none'}\n- usd_spent: {self.budget.daily_spent:.4f}\n"
        )

    def _dry_run_plan(self) -> None:
        print(f"[dry-run] run_id={self.run_id}")
        print(f"[dry-run] limit={self.limit} max_rounds={self.max_rounds}")
        print(f"[dry-run] controller={self.cfg.remote.controller_base_url}")
        print(f"[dry-run] dataset={self.cfg.paths.dataset_root}")
        print(f"[dry-run] memory_root={self.cfg.paths.memory_root}")
        self.trace.emit("dry_run_plan", run_id=self.run_id)
