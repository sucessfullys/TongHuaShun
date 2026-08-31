"""manager.py — MemoryManager: unified facade for all memory subsystems."""
from __future__ import annotations

import logging
import random
import re
from pathlib import Path
from typing import Callable, Optional

from system_prompt_retrieval_agent.memory.failures import (
    append_failure,
    read_all_failures,
    read_failures_for_pair,
)
from system_prompt_retrieval_agent.memory.long import (
    append_long_memory_row,
    archive_rows,
    prune_top_n,
    read_long_memory,
    read_pair_yaml,
    write_pair_yaml,
)
from system_prompt_retrieval_agent.memory.round import make_round_tree
from system_prompt_retrieval_agent.memory.schedule import (
    default_schedule,
    read_schedule,
    write_schedule,
)
from system_prompt_retrieval_agent.memory.shared import (
    read_rules,
    write_rules,
    write_markdown_digest,
)
from system_prompt_retrieval_agent.schemas import (
    PromptPair,
    PromptPairHistoryContext,
    PromptPairScoreContext,
    PromptPairSubScores,
)

logger = logging.getLogger(__name__)

_DEFAULT_NEGATIVE = (
    "distorted body, duplicated clothing, artifacts, blurry, low quality"
)


class MemoryManager:
    """Unified access point for the agent's memory storage."""

    def __init__(
        self,
        memory_root: Path,
        seed_loader: Optional[Callable] = None,
    ) -> None:
        self.memory_root = Path(memory_root)
        self.seed_loader = seed_loader

        # Canonical file/dir paths
        self.schedule_path = self.memory_root / "schedule.json"
        self.shared_csv = self.memory_root / "shared_rules.csv"
        self.shared_md = self.memory_root / "shared_rules_digest.md"
        self.long_csv = self.memory_root / "long_memory.csv"
        self.long_archive = self.memory_root / "long_memory_archive.csv"
        self.preeval_baseline_json = self.memory_root / "preeval_baseline.json"
        self.history_dir = self.memory_root / "history"
        self.pairs_dir = self.memory_root / "pairs"
        self.rounds_dir = self.memory_root / "rounds"
        self.failures_jsonl = self.memory_root / "failures.jsonl"

        # Ensure directories exist
        for d in (self.memory_root, self.pairs_dir, self.rounds_dir, self.history_dir):
            d.mkdir(parents=True, exist_ok=True)

        logger.info("MemoryManager initialised at %s", self.memory_root)

    # ------------------------------------------------------------------
    # Schedule
    # ------------------------------------------------------------------

    def load_schedule(self) -> dict:
        if not self.schedule_path.exists():
            sched = default_schedule()
            write_schedule(self.schedule_path, sched)
            logger.info("Created default schedule at %s", self.schedule_path)
            return sched
        return read_schedule(self.schedule_path)

    def save_schedule(self, data: dict) -> None:
        write_schedule(self.schedule_path, data)

    # ------------------------------------------------------------------
    # Round ID discovery
    # ------------------------------------------------------------------

    def determine_round_id(self) -> int:
        """Return max(existing round_{id}) + 1; or 1 if no rounds exist."""
        pattern = re.compile(r"^round_(\d+)$")
        max_id = 0
        found = False
        for child in self.rounds_dir.iterdir():
            m = pattern.match(child.name)
            if m and child.is_dir():
                found = True
                max_id = max(max_id, int(m.group(1)))
        return max_id + 1 if found else 1

    # ------------------------------------------------------------------
    # Long memory
    # ------------------------------------------------------------------

    def load_long_memory(self) -> list[dict]:
        return read_long_memory(self.long_csv)

    def append_long_memory(self, row: dict) -> None:
        append_long_memory_row(self.long_csv, row)

    def prune_long_memory(self, n: int = 50) -> None:
        rows = self.load_long_memory()
        kept, evicted = prune_top_n(rows, n)
        if evicted:
            archive_rows(self.long_archive, evicted)
            # Rewrite the CSV with only kept rows
            import csv, os
            tmp = self.long_csv.with_suffix(self.long_csv.suffix + ".tmp")
            from system_prompt_retrieval_agent.memory.long import FIELDNAMES
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(kept)
            os.replace(tmp, self.long_csv)
            logger.info("Pruned long memory: kept=%d evicted=%d", len(kept), len(evicted))

    # ------------------------------------------------------------------
    # Per-pair YAML
    # ------------------------------------------------------------------

    def write_pair(self, pair_id: str, data: dict, *, run_id: str | None = None) -> None:
        """Write a pair YAML.

        With ``run_id`` (the canonical case for round-generated pairs) the YAML
        lives under ``pairs/<run_id>/<pid>.yaml`` so it survives across runs
        without being overwritten by a future run that produces the same
        round-scoped ``pair_id`` (e.g. ``pair_r01_000``).

        Without ``run_id`` the file is written at ``pairs/<pid>.yaml`` — used
        for run-agnostic seed pairs.
        """
        target_dir = (self.pairs_dir / run_id) if run_id else self.pairs_dir
        write_pair_yaml(target_dir, pair_id, data)

    def read_pair(self, pair_id: str, *, run_id: str | None = None) -> dict | None:
        """Read a pair YAML. If ``run_id`` is given, look under ``pairs/<run_id>/``;
        otherwise check the top-level ``pairs/<pid>.yaml`` (seeds), then fall
        back to scanning ``pairs/<*>/<pid>.yaml`` for the most recent match.
        """
        if run_id:
            data = read_pair_yaml(self.pairs_dir / run_id, pair_id)
            if data is not None:
                return data
        data = read_pair_yaml(self.pairs_dir, pair_id)
        if data is not None:
            return data
        # Fall-through: scan run subdirs
        for sub in self.pairs_dir.iterdir() if self.pairs_dir.exists() else []:
            if sub.is_dir():
                hit = read_pair_yaml(sub, pair_id)
                if hit is not None:
                    return hit
        return None

    # ------------------------------------------------------------------
    # Round tree
    # ------------------------------------------------------------------

    def make_round_tree(self, round_id: int) -> dict[str, Path]:
        return make_round_tree(self.rounds_dir, round_id)

    # ------------------------------------------------------------------
    # Failures
    # ------------------------------------------------------------------

    def append_failure(self, record: dict) -> None:
        append_failure(self.failures_jsonl, record)

    def read_failures_for_pair(self, prompt_pair_id: str) -> list[dict]:
        return read_failures_for_pair(self.failures_jsonl, prompt_pair_id)

    def read_all_failures(self) -> list[dict]:
        return read_all_failures(self.failures_jsonl)

    # ------------------------------------------------------------------
    # Shared rules
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Per-run reset / archival
    # ------------------------------------------------------------------

    def archive_run_state(self, run_id: str) -> dict[str, Path]:
        """Archive per-run rebuilt state into ``memory/history/`` and reset.

        Moves the per-run-rebuilt artifacts into ``memory/history/`` with a
        timestamp + run_id suffix, then leaves the canonical paths absent (or
        empty in the case of ``rounds/``) so the new run starts from a clean
        slate:

          * ``long_memory.csv``       → ``history/long_memory__<ts>__<run_id>.csv``
          * ``preeval_baseline.json`` → ``history/preeval_baseline__<ts>__<run_id>.json``
          * ``schedule.json``         → ``history/schedule__<ts>__<run_id>.json``
          * ``rounds/`` (the dir)     → ``history/rounds__<ts>__<run_id>/``

        After ``rounds/`` is moved an empty ``rounds/`` directory is recreated
        so subsequent code can write into it without pre-creating it again.

        Cross-run shared memory (``shared_rules.csv``,
        ``shared_rules_digest.md``, ``pairs/``, ``failures.jsonl``) is
        deliberately NOT touched — those persist across runs.

        Returns a mapping of source-name → archived-path for any file/dir that
        was actually moved.
        """
        from datetime import datetime

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        moved: dict[str, Path] = {}

        file_targets = [
            ("long_memory.csv", self.long_csv),
            ("preeval_baseline.json", self.preeval_baseline_json),
            ("schedule.json", self.schedule_path),
        ]
        for name, src in file_targets:
            if not src.exists():
                continue
            stem, suffix = src.stem, src.suffix
            dest = self.history_dir / f"{stem}__{ts}__{run_id}{suffix}"
            try:
                src.rename(dest)
                moved[name] = dest
                logger.info("archived %s -> %s", src, dest)
            except OSError:
                logger.exception("archive_run_state: failed to move %s", src)

        # Archive memory/rounds/ as a directory if it has any contents.
        if self.rounds_dir.exists() and any(self.rounds_dir.iterdir()):
            dest_rounds = self.history_dir / f"rounds__{ts}__{run_id}"
            try:
                self.rounds_dir.rename(dest_rounds)
                moved["rounds"] = dest_rounds
                logger.info("archived %s -> %s", self.rounds_dir, dest_rounds)
            except OSError:
                logger.exception(
                    "archive_run_state: failed to move %s", self.rounds_dir
                )
            self.rounds_dir.mkdir(parents=True, exist_ok=True)
        return moved

    # ------------------------------------------------------------------

    def load_rules(self) -> list[dict]:
        """Return rules from shared_rules.csv **as persisted**.

        ``prompt_path`` is returned exactly as stored on disk (canonically
        relative to ``memory_root``). Callers that need an openable path
        must resolve via
        :func:`system_prompt_retrieval_agent.memory.shared.resolve_prompt_path`.
        This avoids in-place mutation that would leak machine-specific
        absolute paths into the CSV on the next save.
        """
        return read_rules(self.shared_csv)

    def save_rules(self, rows: list[dict]) -> None:
        write_rules(self.shared_csv, rows)
        write_markdown_digest(self.shared_md, rows)

    # ------------------------------------------------------------------
    # load_for_generation — builds PromptPairHistoryContext
    # ------------------------------------------------------------------

    def load_for_generation(self, round_id: int, N: int) -> PromptPairHistoryContext:
        """Build PromptPairHistoryContext with all 5 history blocks.

        Blocks (per plan §8.2):
          - previous_round_prompt_pairs: pairs from round == round_id-1
          - long_memory_prompt_pairs:    top-5 by overall_score excluding previous_round ids
          - previous_top_prompt_pairs:   top-3 distinct from previous_round
          - project_memory_prompt_pairs: score>0.60 and seen ≥2 times in long_memory
          - random_history_prompt_pairs: Random(round_id).sample(up to 3)
          - visual_comparisons:          *.png filenames under round_{K-1:03d}/visualizations/
        """
        rows = self.load_long_memory()

        # Gather visual comparisons from previous round's visualizations dir
        vis_paths: dict[str, str] = {}
        prev_round_id = round_id - 1
        if prev_round_id >= 1:
            vis_dir = self.rounds_dir / f"round_{prev_round_id:03d}" / "visualizations"
            if vis_dir.exists():
                for png in vis_dir.glob("*.png"):
                    vis_paths[png.name] = str(png)

        # Cross-round shared rules (always loaded; may be empty list)
        try:
            shared_rules = self.load_rules()
        except Exception:  # noqa: BLE001
            logger.exception("load_for_generation: load_rules failed")
            shared_rules = []

        # Empty memory — first round or no data
        if not rows:
            logger.info("load_for_generation: no long memory, returning empty context")
            return PromptPairHistoryContext(
                round_id=round_id,
                long_memory_prompt_pairs=None,
                project_memory_prompt_pairs=None,
                previous_top_prompt_pairs=None,
                previous_round_prompt_pairs=None,
                random_history_prompt_pairs=None,
                visual_comparisons=vis_paths if vis_paths else None,
                shared_rules=shared_rules or None,
            )

        # Build PromptPair objects from CSV rows
        pairs_by_id: dict[str, list[PromptPair]] = {}
        for row in rows:
            pp = _row_to_prompt_pair(row)
            pairs_by_id.setdefault(pp.prompt_pair_id, []).append(pp)

        # ----------------------------------------------------------
        # Block 1: previous_round_prompt_pairs
        # ----------------------------------------------------------
        prev_round_pairs: list[PromptPair] = []
        prev_pair_ids: set[str] = set()
        for row in rows:
            try:
                r = int(row.get("round", -1))
            except (ValueError, TypeError):
                r = -1
            if r == prev_round_id:
                pp = _row_to_prompt_pair(row, selection_role="previous_round")
                if pp.prompt_pair_id not in prev_pair_ids:
                    prev_pair_ids.add(pp.prompt_pair_id)
                    prev_round_pairs.append(pp)

        # ----------------------------------------------------------
        # Block 2: long_memory_prompt_pairs — top-5 excluding prev round
        # ----------------------------------------------------------
        non_prev = [r for r in rows if r.get("prompt_pair_id") not in prev_pair_ids]
        sorted_all = _sort_by_score(non_prev)
        long_mem_pairs = [
            _row_to_prompt_pair(r, selection_role="long_memory_reference")
            for r in sorted_all[:5]
        ]

        # ----------------------------------------------------------
        # Block 3: previous_top_prompt_pairs — top-3 distinct from prev round
        # ----------------------------------------------------------
        prev_top_pairs = [
            _row_to_prompt_pair(r, selection_role="previous_top")
            for r in sorted_all[:3]
        ]

        # ----------------------------------------------------------
        # Block 4: project_memory_prompt_pairs — score>0.60 and seen >=2
        # ----------------------------------------------------------
        seen_counts: dict[str, int] = {}
        for row in rows:
            pid = row.get("prompt_pair_id", "")
            seen_counts[pid] = seen_counts.get(pid, 0) + 1

        proj_rows = [
            r for r in rows
            if _float_score(r) > 0.60 and seen_counts.get(r.get("prompt_pair_id", ""), 0) >= 2
        ]
        # Deduplicate by pair id, pick highest score
        proj_best: dict[str, dict] = {}
        for r in proj_rows:
            pid = r.get("prompt_pair_id", "")
            if pid not in proj_best or _float_score(r) > _float_score(proj_best[pid]):
                proj_best[pid] = r
        proj_mem_pairs = [
            _row_to_prompt_pair(r, selection_role="project_memory")
            for r in _sort_by_score(list(proj_best.values()))
        ]

        # ----------------------------------------------------------
        # Block 5: random_history_prompt_pairs — seeded random sample 3
        # ----------------------------------------------------------
        rng = random.Random(round_id)
        pool = rows[:]
        sample_size = min(3, len(pool))
        rand_sample = rng.sample(pool, sample_size)
        rand_pairs = [
            _row_to_prompt_pair(r, selection_role="random_history")
            for r in rand_sample
        ]

        return PromptPairHistoryContext(
            round_id=round_id,
            previous_round_prompt_pairs=prev_round_pairs or None,
            long_memory_prompt_pairs=long_mem_pairs or None,
            previous_top_prompt_pairs=prev_top_pairs or None,
            project_memory_prompt_pairs=proj_mem_pairs or None,
            random_history_prompt_pairs=rand_pairs or None,
            visual_comparisons=vis_paths if vis_paths else None,
            shared_rules=shared_rules or None,
        )

    # ------------------------------------------------------------------
    # Fallback tracking
    # ------------------------------------------------------------------

    def mark_fallback_round(self, round_id: int, is_fallback: bool) -> None:
        """Update schedule for fallback status of a round."""
        sched = self.load_schedule()
        sched["fallback_prompt_generation"] = is_fallback
        if is_fallback:
            sched["consecutive_fallback_rounds"] = (
                int(sched.get("consecutive_fallback_rounds", 0)) + 1
            )
            logger.info(
                "Round %d marked as fallback (consecutive=%d)",
                round_id,
                sched["consecutive_fallback_rounds"],
            )
        else:
            sched["consecutive_fallback_rounds"] = 0
            logger.info("Round %d: fallback reset", round_id)
        self.save_schedule(sched)

    # ------------------------------------------------------------------
    # Seeding from temp_wxy
    # ------------------------------------------------------------------

    def seed_from_temp_wxy(self) -> list[PromptPair]:
        """Build 3 seed pairs using temp_wxy/PROMPTS.py names if memory empty.

        Pair names: v5 null, v5 default_neg, v5_new default_neg.
        Uses round_id=0 and selection_role='seed'.
        """
        if self.seed_loader is None:
            logger.info("seed_from_temp_wxy: no seed_loader provided, returning []")
            return []

        try:
            prompts = self.seed_loader()
        except Exception as exc:
            logger.warning("seed_from_temp_wxy: seed_loader raised %s", exc)
            return []

        seeds: list[PromptPair] = [
            PromptPair(
                prompt_pair_id="seed-v5-null-neg",
                system_prompt_id=prompts.get("v5", "v5"),
                negative_prompt_id="null_neg",
                round_id=0,
                selection_role="seed",
                system_prompt=prompts.get("v5_text", ""),
                negative_prompt=None,
                fallback=False,
            ),
            PromptPair(
                prompt_pair_id="seed-v5-default-neg",
                system_prompt_id=prompts.get("v5", "v5"),
                negative_prompt_id="default_neg",
                round_id=0,
                selection_role="seed",
                system_prompt=prompts.get("v5_text", ""),
                negative_prompt=_DEFAULT_NEGATIVE,
                fallback=False,
            ),
            PromptPair(
                prompt_pair_id="seed-v5new-default-neg",
                system_prompt_id=prompts.get("v5_new", "v5_new"),
                negative_prompt_id="default_neg",
                round_id=0,
                selection_role="seed",
                system_prompt=prompts.get("v5_new_text", ""),
                negative_prompt=_DEFAULT_NEGATIVE,
                fallback=False,
            ),
        ]
        logger.info("seed_from_temp_wxy: created %d seed pairs", len(seeds))
        return seeds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _float_score(row: dict) -> float:
    try:
        return float(row.get("overall_score") or 0.0)
    except (ValueError, TypeError):
        return 0.0


def _sort_by_score(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=_float_score, reverse=True)


def _row_to_prompt_pair(row: dict, selection_role: Optional[str] = None) -> PromptPair:
    """Convert a long-memory CSV row dict to a PromptPair with scores attached."""
    sub = PromptPairSubScores(
        qwen_pass_rate=_opt_float(row.get("qwen_pass_rate")),
        edit_correctness=_opt_float(row.get("edit_correctness")),
        garment_transfer_correctness=_opt_float(row.get("garment_transfer_correctness")),
        preservation=_opt_float(row.get("preservation")),
        artifact_penalty=_opt_float(row.get("artifact_penalty")),
    )
    scores = PromptPairScoreContext(
        overall_score=_opt_float(row.get("overall_score")),
        total_score=_opt_float(row.get("total_score")),
        sub_scores=sub,
    )

    try:
        round_id = int(row.get("round", 0))
    except (ValueError, TypeError):
        round_id = 0

    fallback_raw = row.get("fallback", "False")
    if isinstance(fallback_raw, bool):
        fallback = fallback_raw
    else:
        fallback = str(fallback_raw).lower() in ("true", "1", "yes")

    role = selection_role or row.get("selection_role") or None

    return PromptPair(
        prompt_pair_id=row.get("prompt_pair_id", ""),
        system_prompt_id=row.get("system_prompt_id", ""),
        negative_prompt_id=row.get("negative_prompt_id", ""),
        round_id=round_id,
        selection_role=role,
        system_prompt=row.get("system_prompt", ""),
        negative_prompt=row.get("negative_prompt") or None,
        fallback=fallback,
        scores=scores,
    )


def _opt_float(val) -> Optional[float]:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
