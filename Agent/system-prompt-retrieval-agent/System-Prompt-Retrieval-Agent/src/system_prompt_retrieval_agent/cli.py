from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import AppConfig, load_config
from .execution_mode_v022 import PRODUCTION_MODE, resolve_execution_mode
from .logging_setup import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="system-prompt-retrieval-agent", description="V0.2 prompt-pair retrieval agent")
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run the agent loop")
    run.add_argument("--config", required=True)
    run.add_argument("--limit", type=int, default=0)
    run.add_argument("--max-rounds", type=int, default=None)
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--round", type=int, default=None)
    run.add_argument(
        "--resume-from-run-id",
        dest="resume_from_run_id",
        default=None,
        metavar="RUN_ID",
        help=(
            "Reuse RUN_ID as the new invocation's run_id (plan §2.10c). "
            "Validates RUN_ID matches the canonical regex, the prior run "
            "directory exists with at least one prior stage manifest, and "
            "all four canonical hashes (config, user-prompt corpus, "
            "prompt-pair corpus, sample corpus) match the current "
            "invocation. Aborts with 'resume corpus/config drift' on any "
            "mismatch before any dispatch and before any copy-back. "
            "Without this flag, every production invocation generates a "
            "fresh run_id and L = ∅ for every stage."
        ),
    )

    resume = sub.add_parser("resume", help="Resume last run")
    resume.add_argument("--config", required=True)
    resume.add_argument("--run-id", default=None)

    status = sub.add_parser("status", help="Print status")
    status.add_argument("--config", required=True)
    status.add_argument("--run-id", default=None)

    best = sub.add_parser("best-pair", help="Print best pair")
    best.add_argument("--config", required=True)

    return p


def _dispatch_run(
    cfg: AppConfig,
    *,
    limit: int = 0,
    max_rounds: int | None = None,
    dry_run: bool = False,
    resume_from_run_id: str | None = None,
    resume_run_id: str | None = None,
) -> int:
    """Branch on execution.mode (S05B.01, S05B.05, S05B.06).

    The default mode ``v022_stage_major`` enters the V0.2.2 production
    runner and bypasses the legacy ``AgentLoop`` body entirely. Legacy
    behavior is reachable only behind explicit
    ``legacy_stage_all_compat`` / ``legacy_cartesian_compat`` modes.
    """
    mode = resolve_execution_mode(cfg.model_dump())
    if mode == PRODUCTION_MODE:
        if dry_run:
            from .agent_loop import AgentLoop

            loop = AgentLoop(cfg, dry_run=True)
            return asyncio.run(loop.run())
        from .runner_v022 import V022Runner

        runner = V022Runner(
            cfg,
            limit=limit,
            max_rounds=max_rounds,
            resume_from_run_id=resume_from_run_id,
        )
        outcome = asyncio.run(runner.run())
        return 0 if outcome.stop_reason != "execution_mode_violation" else 2

    # Legacy compatibility paths only.
    from .agent_loop import AgentLoop

    loop = AgentLoop(
        cfg,
        limit=limit,
        max_rounds=max_rounds,
        dry_run=dry_run,
        resume_from_run_id=resume_from_run_id,
        resume_run_id=resume_run_id,
    )
    return asyncio.run(loop.run())


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 0
    cfg = load_config(args.config)
    setup_logging(cfg.logging.level, cfg.logging.redact_env_keys)

    if args.cmd == "run":
        return _dispatch_run(
            cfg,
            limit=args.limit,
            max_rounds=args.max_rounds,
            dry_run=args.dry_run,
            resume_from_run_id=args.resume_from_run_id,
        )

    if args.cmd == "resume":
        return _dispatch_run(cfg, resume_run_id=args.run_id)

    if args.cmd == "status":
        from .agent_loop import AgentLoop

        state = AgentLoop.read_status(cfg, args.run_id)
        print(json.dumps(state, indent=2, default=str))
        return 0

    if args.cmd == "best-pair":
        best_path = Path(cfg.paths.output_root) / "best_pair.yaml"
        if best_path.is_file():
            print(best_path.read_text())
        else:
            print("no best_pair.yaml yet", file=sys.stderr)
            return 2
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
