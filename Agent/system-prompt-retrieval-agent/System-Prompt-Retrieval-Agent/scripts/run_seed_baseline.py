#!/usr/bin/env python
"""Seed-baseline runner for V0.2.2.

Loads the two fixed seed prompts from
``temp_wxy/PROMPTS.py`` (``system_prompt_for_tryon_v5`` and
``system_prompt_for_tryon_v5_new``) and drives the production
``V022Runner`` against the full configured dataset, bypassing the
LLM-based prompt generator.

Identical to a real run except:

* prompt generation is replaced by an in-process factory that returns
  the two fixed seed pairs (with stable IDs);
* ``negative_prompt`` is always ``None`` and ``negative_prompt_id`` is
  ``"none"``;
* ``max_rounds`` is forced to 1 (no improvement loop — these are the
  baseline prompts).

All artifacts land under ``cfg.paths.output_root/runs/<run_id>/`` and
``cfg.paths.memory_root`` exactly like a real run, so every cell in
``ranking_v022.json``, ``long_memory.csv``, ``shared_rules.csv``, and
``memory/pairs/<run_id>/<pid>.yaml`` traces back to one of the two
seed system prompts.

Stable IDs (preserved across rebuilds):

* prompt_pair_id   = ``seed_v5`` / ``seed_v5_new``
* system_prompt_id = ``system_prompt_for_tryon_v5`` /
  ``system_prompt_for_tryon_v5_new``
* negative_prompt_id = ``none``
* sample_id and user_prompt_id come from the production manifest /
  ``cfg.gemma_user_prompts.library`` unchanged.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
from pathlib import Path

# Make the package importable when run as a plain script.
_PKG_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from system_prompt_retrieval_agent.config import load_config  # noqa: E402
from system_prompt_retrieval_agent.execution_mode_v022 import (  # noqa: E402
    PRODUCTION_MODE,
    resolve_execution_mode,
)
from system_prompt_retrieval_agent.logging_setup import setup_logging  # noqa: E402
from system_prompt_retrieval_agent.runner_v022 import V022Runner  # noqa: E402
from system_prompt_retrieval_agent.schemas import PromptPair  # noqa: E402

_DEFAULT_PROMPTS_PATH = (
    Path(__file__).resolve().parents[2] / "temp_wxy" / "PROMPTS.py"
)


def _load_seed_prompts(prompts_path: Path) -> dict[str, str]:
    """Load ``system_prompt_for_tryon_v5{,_new}`` from a Python module file.

    Imports the file directly (no package context) so we don't need to
    add ``temp_wxy`` to ``PYTHONPATH``. Raises if either symbol is
    missing — never substitutes a stub or default.
    """
    if not prompts_path.is_file():
        raise FileNotFoundError(
            f"seed-prompt source not found: {prompts_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "_seed_baseline_prompts", str(prompts_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"could not load python spec for {prompts_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out: dict[str, str] = {}
    for name in ("system_prompt_for_tryon_v5", "system_prompt_for_tryon_v5_new"):
        if not hasattr(module, name):
            raise AttributeError(
                f"{prompts_path} is missing required symbol {name!r}"
            )
        text = getattr(module, name)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"{prompts_path}:{name} is not a non-empty string"
            )
        out[name] = text
    return out


def _build_seed_pairs(seeds: dict[str, str], round_id: int) -> list[PromptPair]:
    """Build the two fixed PromptPair objects with stable IDs.

    No randomization, no rotation. Both pairs are tagged
    ``selection_role="baseline"`` and the rationale records the source
    file so downstream artifacts trace back to ``temp_wxy/PROMPTS.py``.
    """
    return [
        PromptPair(
            prompt_pair_id="seed_v5",
            system_prompt_id="system_prompt_for_tryon_v5",
            negative_prompt_id="none",
            round_id=round_id,
            selection_role="baseline",
            system_prompt=seeds["system_prompt_for_tryon_v5"],
            negative_prompt=None,
            rationale="seed baseline from temp_wxy/PROMPTS.py:system_prompt_for_tryon_v5",
        ),
        PromptPair(
            prompt_pair_id="seed_v5_new",
            system_prompt_id="system_prompt_for_tryon_v5_new",
            negative_prompt_id="none",
            round_id=round_id,
            selection_role="baseline",
            system_prompt=seeds["system_prompt_for_tryon_v5_new"],
            negative_prompt=None,
            rationale="seed baseline from temp_wxy/PROMPTS.py:system_prompt_for_tryon_v5_new",
        ),
    ]


def _make_factory(seeds: dict[str, str]):
    """Return a ``prompt_pair_factory`` compatible with V022Runner.

    The runner calls this with
    ``(cfg, memory, round_id, N, existing_pairs)`` and expects
    ``(list[PromptPair], fallback_used: bool)``. We always return both
    seeds and ``fallback_used=False`` so the runner never marks a
    fallback round.
    """
    async def factory(cfg, memory, round_id, N, existing_pairs):
        return _build_seed_pairs(seeds, round_id), False

    return factory


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_seed_baseline",
        description=(
            "Run the V0.2.2 pipeline (Gemma → FLUX → Qwen → local OpenAI eval) "
            "against the configured dataset using two fixed seed system prompts, "
            "with no LLM-based prompt generation."
        ),
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to config.yaml (same one used by the production CLI).",
    )
    p.add_argument(
        "--prompts",
        default=str(_DEFAULT_PROMPTS_PATH),
        help=f"Path to PROMPTS.py (default: {_DEFAULT_PROMPTS_PATH})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max samples (0 = full dataset).",
    )
    p.add_argument(
        "--resume-from-run-id",
        dest="resume_from_run_id",
        default=None,
        metavar="RUN_ID",
        help=(
            "Reuse RUN_ID for this invocation (Level-1 durable eval will "
            "skip cells whose JSONL row + .eval_done sentinel both exist). "
            "Validates RUN_ID against the canonical regex inside V022Runner."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cfg = load_config(args.config)
    setup_logging(
        getattr(getattr(cfg, "logging", None), "level", "INFO") or "INFO",
        env_keys=getattr(getattr(cfg, "logging", None), "redact_env_keys", None) or (),
    )

    mode = resolve_execution_mode(cfg.model_dump())
    if mode != PRODUCTION_MODE:
        print(
            f"[seed-baseline] cfg execution.mode must be {PRODUCTION_MODE!r}; "
            f"got {mode!r}",
            file=sys.stderr,
        )
        return 2

    seeds = _load_seed_prompts(Path(args.prompts))
    print(
        f"[seed-baseline] loaded {len(seeds)} seed prompts from {args.prompts}",
        flush=True,
    )

    factory = _make_factory(seeds)
    runner = V022Runner(
        cfg,
        limit=args.limit,
        max_rounds=1,
        prompt_pair_factory=factory,
        resume_from_run_id=args.resume_from_run_id,
    )
    try:
        outcome = asyncio.run(runner.run())
    finally:
        runner.close()

    print(
        f"[seed-baseline] run_id={outcome.run_id} "
        f"stop_reason={outcome.stop_reason} "
        f"rounds_completed={outcome.rounds_completed} "
        f"best_pair_id={outcome.best_pair_id} "
        f"best_pair_overall={outcome.best_pair_overall}",
        flush=True,
    )
    if outcome.stop_reason in {
        "execution_mode_violation",
        "orchestrator_error",
        "eval_error",
        "persistent_generation_failure",
    }:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
