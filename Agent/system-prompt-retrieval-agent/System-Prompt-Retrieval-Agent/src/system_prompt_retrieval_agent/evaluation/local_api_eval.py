from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

from system_prompt_retrieval_agent.config import AppConfig
from system_prompt_retrieval_agent.rate_limiter import RateLimiter, get_rate_limiter
from system_prompt_retrieval_agent.evaluation.budget_guard import BudgetGuard, CostExhausted

log = logging.getLogger(__name__)

# Cost per evaluation call (fixed stub)
_USD_PER_CALL = 0.002

# Bumped whenever the cell key, prompt template, or response axis names
# change. Old JSONL rows stamped with a different value are not eligible
# for skip-on-resume — they will be re-evaluated.
EVAL_SCHEMA_VERSION = 1

# APIConnectionError-aware retry/backoff tried at the cell level (in
# addition to the SDK's own ``max_retries``). Last entry's index marks
# the budget; on exhaustion the cell raises and a single eval failure
# halts the round via EVAL_ERROR.
_API_RETRY_BACKOFF_S: tuple[float, ...] = (1.0, 4.0, 16.0, 60.0)


_REMOTE_IMG_CACHE = Path("/tmp/spra_local_eval_image_cache")
_REMOTE_FETCH_ALIAS = os.environ.get("SPRA_REMOTE_SSH_ALIAS", "3h100")


def _resolve_image_path(path: str | Path) -> Path:
    """Return a local Path for an image, fetching via SCP if the manifest
    path is a remote-only mount (e.g. /mnt/TryOn_Data/...).

    The manifest carries remote-absolute paths because supervisors on the
    GPU host open them directly. The local agent runs on a workstation
    that does not mount /mnt/TryOn_Data, so we lazily SCP referenced
    images into a local cache.
    """
    img_path = Path(path)
    if img_path.is_file():
        return img_path
    # Cache key mirrors the remote absolute path under the cache root.
    cache_path = _REMOTE_IMG_CACHE / str(img_path).lstrip("/")
    if cache_path.is_file():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    import subprocess
    rc = subprocess.run(
        ["scp", "-q", "-o", "BatchMode=yes",
         f"{_REMOTE_FETCH_ALIAS}:{img_path}", str(cache_path)],
        capture_output=True, text=True,
    )
    if rc.returncode != 0 or not cache_path.is_file():
        raise FileNotFoundError(
            f"local-eval image not found locally and remote fetch failed "
            f"({_REMOTE_FETCH_ALIAS}:{img_path}): rc={rc.returncode} "
            f"stderr={rc.stderr.strip()[:200]}"
        )
    return cache_path


def _encode_image(path: str | Path) -> str:
    """Return a base64 data URI for an image file. Fetches via SCP when
    the path is remote-only (manifest paths under /mnt/TryOn_Data/...)."""
    from PIL import Image
    import io

    img_path = _resolve_image_path(path)
    with Image.open(img_path) as img:
        buf = io.BytesIO()
        fmt = img.format or "PNG"
        img.save(buf, format=fmt)
        data = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{data}"


def _append_jsonl(path: Path, record: dict) -> None:
    """Append a single JSON record to a JSONL file, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _append_jsonl_fsync(path: Path, record: dict) -> None:
    """Append + fsync a JSON record. Used by Level-1 durable per-cell write.

    The fsync ensures the row is on stable storage before the sentinel
    rename so a crash between cannot leave a sentinel pointing at a
    truncated/missing row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Level-1: cell key, sentinel paths, durable record helpers
# ---------------------------------------------------------------------------


def _cell_key(cell: dict) -> tuple:
    """Build the canonical cell key.

    The key includes ``EVAL_SCHEMA_VERSION`` so a future schema bump
    invalidates old skip-on-resume hits without orphaning their JSONL
    rows (those rows simply won't match the current key).
    """
    return (
        str(cell.get("run_id") or ""),
        int(cell.get("round_id") or 0),
        str(cell.get("prompt_pair_id") or ""),
        str(cell.get("user_prompt_id") or ""),
        str(cell.get("sample_id") or ""),
        EVAL_SCHEMA_VERSION,
    )


def _sentinel_path(api_eval_root: Path, cell: dict) -> Path:
    """Per-cell sentinel under ``{root}/{pair_id}/{up_id}/.eval_done.{sample_id}``.

    One sentinel per cell rather than per JSONL file; the JSONL file is
    shared across all samples for a given (pair, user_prompt) phrasing.
    """
    pid = str(cell.get("prompt_pair_id") or "")
    up = str(cell.get("user_prompt_id") or "")
    sid = str(cell.get("sample_id") or "")
    return api_eval_root / pid / up / f".eval_done.{sid}"


def _jsonl_path(api_eval_root: Path, cell: dict) -> Path:
    pid = str(cell.get("prompt_pair_id") or "")
    up = str(cell.get("user_prompt_id") or "")
    return api_eval_root / pid / f"{up}.jsonl"


def _record_is_complete(rec: dict) -> bool:
    """A row is complete iff all 4 required gpt-4o axes are present and
    finite. Rows missing any axis (eg malformed JSON returned by gpt-4o)
    are dropped at load time so they never feed into ranking math.
    """
    required = (
        "edit_correctness",
        "garment_transfer_correctness",
        "preservation",
        "artifact_penalty",
    )
    if any(a not in rec for a in required):
        return False
    try:
        for a in required:
            float(rec[a])
    except (TypeError, ValueError):
        return False
    return True


def _key_for_record(rec: dict) -> tuple:
    """Reconstruct the cell key from a stored JSONL record.

    Old rows pre-Level-1 may not have ``run_id``/``round_id``/
    ``eval_schema_version`` fields. Missing fields produce a zeroed key
    that won't match the current eval's key, so they're treated as
    cache-miss (safe default — re-evaluate rather than risk stale skip).
    """
    return (
        str(rec.get("run_id") or ""),
        int(rec.get("round_id") or 0),
        str(rec.get("prompt_pair_id") or ""),
        str(rec.get("user_prompt_id") or ""),
        str(rec.get("sample_id") or ""),
        int(rec.get("eval_schema_version") or 0),
    )


def _load_prior_records(
    api_eval_root: Path, cells: list[dict]
) -> tuple[dict[tuple, dict], list[dict]]:
    """Scan all relevant JSONL files under ``api_eval_root`` and return
    ``(results_by_key, malformed)``.

    Dedupe rule: for each key keep the LATEST complete row, preferring
    higher line index within a file and higher mtime across files.
    Malformed/incomplete rows are dropped (returned in the second slot
    for diagnostics + tests).
    """
    if not api_eval_root.exists():
        return {}, []
    # Only scan paths that the cells in this batch actually reference.
    # Avoids paying for unrelated runs sharing the same root.
    relevant_files: set[Path] = set()
    for c in cells:
        relevant_files.add(_jsonl_path(api_eval_root, c))

    results_by_key: dict[tuple, dict] = {}
    rank_by_key: dict[tuple, tuple] = {}  # (mtime_ns, line_idx) per key
    malformed: list[dict] = []

    for jsonl in relevant_files:
        if not jsonl.is_file():
            continue
        try:
            mtime_ns = jsonl.stat().st_mtime_ns
        except OSError:
            continue
        try:
            with jsonl.open("r", encoding="utf-8") as fh:
                for idx, raw in enumerate(fh):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        rec = json.loads(raw)
                    except json.JSONDecodeError:
                        malformed.append({"file": str(jsonl), "line": idx, "raw": raw[:200]})
                        continue
                    if not isinstance(rec, dict) or not _record_is_complete(rec):
                        malformed.append({"file": str(jsonl), "line": idx, "rec": rec})
                        continue
                    key = _key_for_record(rec)
                    rank = (mtime_ns, idx)
                    if key not in results_by_key or rank > rank_by_key[key]:
                        results_by_key[key] = rec
                        rank_by_key[key] = rank
        except OSError:
            continue
    return results_by_key, malformed


class EvalRetryBudgetExhausted(RuntimeError):
    """Raised by a cell task when its APIConnectionError retry budget is
    exhausted. Wrapped by ``_gather_with_cancel`` and surfaced as
    ``EVAL_ERROR`` in the runner."""


class LocalApiEvaluator:
    """Evaluates generated try-on images using OpenAI vision API.

    Supports both legacy sample-keyed dicts and V0.2.1 cell-keyed dicts that
    include ``prompt_pair_id`` and ``user_prompt_id`` alongside ``sample_id``.
    When an ``api_eval_root`` path is provided to ``evaluate_many_cells()``,
    per-phrasing results are persisted under
    ``api_eval_root/{prompt_pair_id}/{user_prompt_id}.jsonl``.
    """

    def __init__(
        self,
        cfg: AppConfig,
        *,
        openai_client: Any = None,
        budget_guard: BudgetGuard | None = None,
    ) -> None:
        self._cfg = cfg
        self._budget_guard = budget_guard
        # Bound in-flight OpenAI eval calls by the lower of
        # ``evaluation.max_concurrent`` and ``rate_limits.max_concurrency``
        # so a tightly-rate-limited account never has more outstanding
        # connections than its proxy / quota allows.
        rl_cap = int(cfg.rate_limits.max_concurrency or 0)
        eval_cap = int(cfg.evaluation.max_concurrent or 0)
        caps = [c for c in (rl_cap, eval_cap) if c > 0]
        effective = min(caps) if caps else 1
        self._semaphore = asyncio.Semaphore(effective)

        # Level-1 eval-specific concurrency control. Falls back to the
        # legacy global cap if ``evaluation.api_concurrency`` is unset or
        # zero. Always bounded by ``effective`` so a misconfigured higher
        # value can't loosen the legacy ``max_concurrent`` /
        # ``rate_limits.max_concurrency`` floor.
        api_conc = int(getattr(cfg.evaluation, "api_concurrency", 0) or 0)
        if api_conc <= 0:
            api_conc = effective
        else:
            api_conc = min(api_conc, effective)
        self._api_semaphore = asyncio.Semaphore(api_conc)

        # Level-1 eval-specific RPS limiter. Independent from the global
        # rate limiter so prompt-generation and eval don't share a budget.
        api_rps = getattr(cfg.evaluation, "api_rps_limit", None)
        if api_rps is None:
            api_rps = float(cfg.rate_limits.requests_per_second)
        self._api_rate_limiter = RateLimiter(rps=float(api_rps))

        if openai_client is not None:
            self._client = openai_client
        else:
            import openai

            api_key = os.environ.get(cfg.api.openai_api_key_env)
            # Bound SDK-level retries by cfg.rate_limits.max_api_retries so
            # a transient proxy outage doesn't burn N×retries × cells calls.
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                max_retries=int(cfg.rate_limits.max_api_retries),
            )

    async def evaluate_sample(self, sample: dict) -> dict:
        """
        Evaluate a single sample using the vision model.

        sample keys: sample_id, model_image_path, cloth_image_path,
                     generated_image_path, intermediate_prompt
        Optional V0.2.1 cell keys: prompt_pair_id, user_prompt_id
        Returns dict with: edit_correctness, garment_transfer_correctness,
                           preservation, artifact_penalty, notes, sample_id,
                           prompt_pair_id (if present), user_prompt_id (if present),
                           usd_spent
        """
        sample_id = sample.get("sample_id", "")
        prompt_pair_id = sample.get("prompt_pair_id")
        user_prompt_id = sample.get("user_prompt_id")

        # Level-1: use the eval-specific concurrency + rps gates so a
        # change to ``cfg.evaluation.api_concurrency`` / ``api_rps_limit``
        # affects this path without leaking into prompt-generation rate
        # budgets owned by the global limiter.
        async with self._api_semaphore:
            await self._api_rate_limiter.acquire()
            # Also tick the global limiter's cost ledger so cross-system
            # accounting (`add_cost`) stays consistent.
            try:
                get_rate_limiter().add_cost(_USD_PER_CALL)
            except Exception:  # pragma: no cover — defensive
                pass

            # Check and charge budget BEFORE the API call
            if self._budget_guard is not None:
                self._budget_guard.charge(_USD_PER_CALL)

            model_img_uri = _encode_image(sample["model_image_path"])
            cloth_img_uri = _encode_image(sample["cloth_image_path"])
            gen_img_uri = _encode_image(sample["generated_image_path"])
            prompt_text = sample.get("intermediate_prompt", "")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are a garment try-on quality evaluator. "
                                "Evaluate the generated image based on the provided images and prompt.\n\n"
                                f"Intermediate prompt: {prompt_text}\n\n"
                                "Return a JSON object with exactly these keys:\n"
                                "  edit_correctness (0-1): how well the edit follows the prompt\n"
                                "  garment_transfer_correctness (0-1): accuracy of garment transfer\n"
                                "  preservation (0-1): how well identity/background is preserved\n"
                                "  artifact_penalty (0-1): degree of visible artifacts (higher = more artifacts)\n"
                                "  notes (string): brief evaluation notes"
                            ),
                        },
                        {
                            "type": "text",
                            "text": "Model image:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": model_img_uri},
                        },
                        {
                            "type": "text",
                            "text": "Cloth image:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": cloth_img_uri},
                        },
                        {
                            "type": "text",
                            "text": "Generated image:",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": gen_img_uri},
                        },
                    ],
                }
            ]

            response = await self._client.chat.completions.create(
                model=self._cfg.api.api_eval_model,
                messages=messages,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = json.loads(content)

        # qwen_pass_rate is read from the per-cell Qwen artifact (parsed: yes/no/parse_fail).
        # parse_fail leaves the field None so the aggregator excludes that cell from the denominator.
        qwen_pass_rate: float | None = None
        qwen_path = sample.get("qwen_eval_json_path")
        if qwen_path:
            try:
                with open(qwen_path, encoding="utf-8") as _f:
                    parsed = json.load(_f).get("parsed")
                if parsed == "yes":
                    qwen_pass_rate = 1.0
                elif parsed == "no":
                    qwen_pass_rate = 0.0
            except (OSError, ValueError, json.JSONDecodeError):
                qwen_pass_rate = None

        # Strict axis parsing: a malformed gpt-4o JSON response must raise so
        # the round halts via EVAL_ERROR rather than emit a phantom all-zero
        # cell that pollutes ranking + memory.
        required_axes = (
            "edit_correctness",
            "garment_transfer_correctness",
            "preservation",
            "artifact_penalty",
        )
        missing = [a for a in required_axes if a not in result]
        if missing:
            raise ValueError(
                f"local-eval response missing required axes {missing} for "
                f"sample_id={sample_id!r} prompt_pair_id={prompt_pair_id!r}: "
                f"keys={list(result.keys())}"
            )

        out: dict[str, Any] = {
            "sample_id": sample_id,
            "qwen_pass_rate": qwen_pass_rate,
            "edit_correctness": float(result["edit_correctness"]),
            "garment_transfer_correctness": float(result["garment_transfer_correctness"]),
            "preservation": float(result["preservation"]),
            "artifact_penalty": float(result["artifact_penalty"]),
            "notes": str(result.get("notes", "")),
            "usd_spent": _USD_PER_CALL,
            # Level-1 cache provenance — required for skip-on-resume.
            "eval_schema_version": EVAL_SCHEMA_VERSION,
        }
        if prompt_pair_id is not None:
            out["prompt_pair_id"] = prompt_pair_id
        if user_prompt_id is not None:
            out["user_prompt_id"] = user_prompt_id
        if sample.get("run_id"):
            out["run_id"] = str(sample.get("run_id"))
        if sample.get("round_id") is not None:
            out["round_id"] = int(sample.get("round_id"))
        return out

    async def evaluate_many(self, samples: list[dict]) -> list[dict]:
        """
        Evaluate all samples concurrently, respecting the semaphore limit.
        Returns empty list if cfg.evaluation.run_local_api_eval is False.

        Accepts both legacy sample-keyed dicts and V0.2.1 cell-keyed dicts
        (with ``prompt_pair_id`` and ``user_prompt_id``).
        """
        if not self._cfg.evaluation.run_local_api_eval:
            return []

        tasks = [self.evaluate_sample(s) for s in samples]
        return list(await asyncio.gather(*tasks))

    async def _evaluate_sample_with_retry(self, cell: dict) -> dict:
        """Run ``evaluate_sample`` with bounded APIConnectionError backoff.

        SDK-level retries (``max_retries``) handle the easy case. This
        outer loop adds a coarser backoff so a longer proxy outage gets
        a chance to recover before the round halts. Budget is bounded
        by ``_API_RETRY_BACKOFF_S``; on exhaustion the caller raises
        ``EvalRetryBudgetExhausted`` which surfaces as ``EVAL_ERROR``.
        """
        try:
            from openai import APIConnectionError, APITimeoutError
        except Exception:  # pragma: no cover — openai is a hard dep
            APIConnectionError = APITimeoutError = ConnectionError  # type: ignore[assignment]

        last_exc: BaseException | None = None
        for attempt, delay in enumerate(_API_RETRY_BACKOFF_S):
            try:
                return await self.evaluate_sample(cell)
            except (APIConnectionError, APITimeoutError) as exc:  # type: ignore[misc]
                last_exc = exc
                # Jitter to avoid thundering herd against a flaky proxy.
                jitter = random.uniform(0, max(delay * 0.2, 0.5))
                wait = delay + jitter
                log.warning(
                    "eval API connect/timeout (attempt %d/%d); sleeping %.1fs: %s",
                    attempt + 1, len(_API_RETRY_BACKOFF_S), wait, exc,
                )
                await asyncio.sleep(wait)
        raise EvalRetryBudgetExhausted(
            f"eval retry budget exhausted after {len(_API_RETRY_BACKOFF_S)} attempts: "
            f"{type(last_exc).__name__ if last_exc else '<unknown>'}: {last_exc}"
        )

    async def _gather_with_cancel(self, coros: list) -> list:
        """Run cell tasks concurrently. On the first exception, cancel
        the rest, drain their cancellation, then re-raise the original
        exception. Without this, a failing eval can leave sibling tasks
        still writing JSONL after the runner has decided to halt.
        """
        tasks = [asyncio.create_task(c) for c in coros]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def evaluate_many_cells(
        self,
        cells: list[dict],
        *,
        api_eval_root: Path | str | None = None,
    ) -> list[dict]:
        """
        V0.2.1 cell-keyed evaluation entry point.

        Each cell must carry ``prompt_pair_id``, ``user_prompt_id``,
        ``sample_id``, ``model_image_path``, ``cloth_image_path``,
        ``generated_image_path``, and optionally ``intermediate_prompt``.

        Level-1 durability:
          * Bulk-load prior JSONL records into ``results_by_key`` and
            return cached results for cells whose JSONL row + sentinel
            both exist; no HTTP call for them.
          * For un-cached cells, append + fsync the JSONL row, then
            atomically rename the per-cell ``.eval_done`` sentinel.
            Crash between fsync and rename → next run re-evaluates and
            aggregator dedupe takes the latest complete row.
          * In-flight dedupe via ``inflight_keys`` so a duplicate cell
            submission inside the same gather batch only triggers one
            HTTP call; both list slots receive the same result dict.
          * First-failure cancels siblings via ``_gather_with_cancel``.

        Returns empty list if cfg.evaluation.run_local_api_eval is False.
        """
        if not self._cfg.evaluation.run_local_api_eval:
            return []

        if api_eval_root is None:
            # Legacy/non-durable path: still kept so callers without an
            # eval root (e.g. one-off ad-hoc usage) keep working. No
            # idempotency / resume guarantees in this branch.
            results = await self.evaluate_many(cells)
            return results

        root = Path(api_eval_root)
        root.mkdir(parents=True, exist_ok=True)

        results_by_key, malformed = _load_prior_records(root, cells)
        if malformed:
            log.warning(
                "load_prior_records: dropped %d malformed/incomplete row(s); "
                "they will be re-evaluated this round.",
                len(malformed),
            )

        # Observable counters for the resume-validation log line. Each
        # _process call increments exactly one of these; their sum equals
        # len(cells) when the batch completes successfully.
        cache_hits = 0
        cache_hits_lock = asyncio.Lock()
        submitted = 0
        submitted_lock = asyncio.Lock()
        skipped_corrupted = 0
        sc_lock = asyncio.Lock()

        inflight_keys: set[tuple] = set()
        inflight_lock = asyncio.Lock()

        async def _process(cell: dict) -> dict | None:
            nonlocal cache_hits, submitted, skipped_corrupted
            key = _cell_key(cell)
            sentinel = _sentinel_path(root, cell)
            jsonl = _jsonl_path(root, cell)

            # 1. cached complete result + sentinel? return it untouched.
            if key in results_by_key and sentinel.is_file():
                async with cache_hits_lock:
                    cache_hits += 1
                return dict(results_by_key[key])

            # 2. sentinel exists but cached record missing → corrupted
            # state (probably crashed between rename steps in a prior
            # run, or sentinel touched by hand). Drop sentinel and let
            # the cell re-evaluate so we get a known-good record.
            if sentinel.is_file() and key not in results_by_key:
                log.warning(
                    "corrupted eval state for cell %s — sentinel without "
                    "record; re-evaluating", key,
                )
                try:
                    sentinel.unlink()
                except OSError:
                    pass
                async with sc_lock:
                    skipped_corrupted += 1

            # 3. dedupe within this gather batch.
            async with inflight_lock:
                if key in inflight_keys:
                    # A sibling task is already running this cell; let
                    # them finish, then read the on-disk record.
                    pass_through = True
                else:
                    inflight_keys.add(key)
                    pass_through = False

            if pass_through:
                # Wait for the in-flight peer to finish writing JSONL +
                # sentinel, then return the on-disk record.
                while True:
                    await asyncio.sleep(0.05)
                    if sentinel.is_file():
                        # Re-read the JSONL and pick the latest complete
                        # row matching this key.
                        recs, _ = _load_prior_records(root, [cell])
                        if key in recs:
                            return dict(recs[key])
                    async with inflight_lock:
                        if key not in inflight_keys:
                            # Peer finished but we didn't see a sentinel
                            # — peer must have raised. Surface the error
                            # by re-running ourselves; gather-with-cancel
                            # will halt the round on the second failure.
                            break

            try:
                # API call (with retry/backoff). Concurrency, rate
                # limit, and budget charge are owned by evaluate_sample;
                # don't re-acquire them here.
                result = await self._evaluate_sample_with_retry(cell)
                async with submitted_lock:
                    submitted += 1
                # Persist: JSONL append + fsync, then atomic sentinel.
                _append_jsonl_fsync(jsonl, result)
                tmp = sentinel.with_suffix(sentinel.suffix + ".tmp")
                tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text("done\n", encoding="utf-8")
                os.rename(tmp, sentinel)
                results_by_key[key] = result
                return result
            finally:
                async with inflight_lock:
                    inflight_keys.discard(key)

        coros = [_process(c) for c in cells]
        try:
            raw = await self._gather_with_cancel(coros)
        finally:
            # Emit cache-hit / submitted / skipped_corrupted counters
            # so resume validation has a hard signal independent of
            # wall time. submitted counts only cells we actually fired
            # an OpenAI request for; cache_hits counts pure on-disk
            # returns; skipped_corrupted counts sentinel-without-record
            # cases that were re-fired (already counted in submitted
            # too — diagnostic only).
            log.info(
                "api_eval cache_hit=%d submitted=%d skipped_corrupted=%d total=%d",
                cache_hits, submitted, skipped_corrupted, len(cells),
            )
        return [r for r in raw if r is not None]
