"""V0.2.2 aggregate backfill — re-aggregate sub-scores from existing eval.json
files into ranking_v022.json, next_round_context.json, failed_cells.json, and
long_memory.csv. Idempotent: re-running archives the prior memory row instead
of duplicating it.

Run after extending PairRankingRow with sub-score / per-category / worst_cells
fields when an existing run on disk needs the new data without re-running the
stages.

Layout (defaults match preeval_workspace):

    <artifact-root>/outputs/v02/<run_id>/<stage>/round_<n>/<pid>/<upid>/<sid>/<canonical filename>
    <run-root>/<run_id>/rounds/round_<NNN>/scoring/ranking_v022.json
    <run-root>/<run_id>/rounds/round_<NNN>/next_round_context.json
    <run-root>/<run_id>/rounds/round_<NNN>/failures/{failed_pairs,failed_cells}.json
    <memory-root>/long_memory.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(
    0, str(REPO_ROOT / "System-Prompt-Retrieval-Agent" / "src")
)

from system_prompt_retrieval_agent.config import EvaluationWeights  # noqa: E402
from system_prompt_retrieval_agent.memory_v022 import (  # noqa: E402
    upsert_long_memory_rows,
)
from system_prompt_retrieval_agent.remote._vendored import (  # noqa: E402
    canonical_paths as cp,
)
from system_prompt_retrieval_agent.scoring_v022 import (  # noqa: E402
    EvalCell,
    build_next_round_context_v022,
    build_pair_ranking_rows,
)


_ROUND_DIR_RE = re.compile(r"^round_(\d+)$")


def _round_id_from_dir(name: str) -> int | None:
    m = _ROUND_DIR_RE.match(name)
    return int(m.group(1)) if m else None


def _canonical_eval_score(qd: dict[str, Any]) -> float:
    """Mirror PreEvalLocalEvaluator.evaluate_many_cells weighting."""
    return float(
        qd.get("qwen_pass_rate", 0.0) * 0.4
        + qd.get("edit_correctness", 0.0) * 0.2
        + qd.get("garment_transfer_correctness", 0.0) * 0.15
        + qd.get("preservation", 0.0) * 0.15
        - qd.get("artifact_penalty", 0.0) * 0.1
    )


def _empty_memory_row(
    run_id: str,
    round_id: int,
    pair_id: str,
) -> dict[str, Any]:
    """Build a minimal V0.2.2 memory row without depending on a PromptPair.

    The runner normally calls ``flatten_pair_to_row_v022(pair_obj, ...)``
    which pre-stamps ``scores_missing`` sentinels. The backfill cannot
    reconstruct ``pair_obj`` from disk, so it builds a clean row with
    only the V0.2.2 keys it can fill; the upsert logic and CSV
    extrasaction='ignore' handle missing legacy columns gracefully.
    """
    return {
        "schema_version": "0.2.2",
        "run_id": run_id,
        "round_id": round_id,
        "prompt_pair_id": pair_id,
        "system_prompt_id": pair_id,
        "negative_prompt_id": "none",
        "round": round_id,
        "fallback": False,
        "timestamp": "",
    }


def _build_cells_from_disk(
    *, artifact_root: Path, run_id: str, round_id: int,
) -> list[EvalCell]:
    """Walk the qwen tree and pair each eval.json with its FLUX/Gemma siblings."""
    qwen_root = artifact_root / cp.stage_dir(run_id, "qwen", round_id)
    out: list[EvalCell] = []
    if not qwen_root.is_dir():
        return out
    for eval_path in qwen_root.rglob("eval.json"):
        rel = eval_path.relative_to(qwen_root)
        parts = rel.parts
        if len(parts) < 4:
            continue
        pid, upid, sid, _ = parts[-4], parts[-3], parts[-2], parts[-1]
        flux_path = artifact_root / cp.cell_artifact_path(
            run_id, "flux", round_id, pid, upid, sid
        )
        gemma_path = artifact_root / cp.cell_artifact_path(
            run_id, "gemma", round_id, pid, upid, sid
        )
        if not (flux_path.is_file() and gemma_path.is_file()):
            continue
        out.append(
            EvalCell(
                prompt_pair_id=pid,
                user_prompt_id=upid,
                sample_id=sid,
                intermediate_prompt="",
                generated_image_path=str(flux_path),
                qwen_eval_json_path=str(eval_path),
                model_image_path="",
                cloth_image_path="",
            )
        )
    return out


def _projected_results(cells: list[EvalCell]) -> dict[str, list[dict[str, Any]]]:
    by_pair: dict[str, list[dict[str, Any]]] = {}
    for c in cells:
        try:
            qd = json.loads(Path(c.qwen_eval_json_path).read_text(encoding="utf-8"))
        except Exception:
            qd = {}
        by_pair.setdefault(c.prompt_pair_id, []).append({
            "prompt_pair_id": c.prompt_pair_id,
            "user_prompt_id": c.user_prompt_id,
            "sample_id": c.sample_id,
            "score": _canonical_eval_score(qd),
            "qwen_pass_rate": qd.get("qwen_pass_rate"),
            "edit_correctness": qd.get("edit_correctness"),
            "garment_transfer_correctness": qd.get("garment_transfer_correctness"),
            "preservation": qd.get("preservation"),
            "artifact_penalty": qd.get("artifact_penalty"),
        })
    return by_pair


def backfill_round(
    *,
    artifact_root: Path,
    run_root: Path,
    memory_root: Path,
    run_id: str,
    round_id: int,
    round_dir: Path,
    weights: dict[str, float],
) -> dict[str, int]:
    cells = _build_cells_from_disk(
        artifact_root=artifact_root, run_id=run_id, round_id=round_id,
    )
    cells_by_pair: dict[str, list[EvalCell]] = {}
    for c in cells:
        cells_by_pair.setdefault(c.prompt_pair_id, []).append(c)
    results_by_pair = _projected_results(cells)

    rows = build_pair_ranking_rows(
        run_id=run_id, round_id=round_id,
        eval_cells_by_pair=cells_by_pair,
        cell_results_by_pair=results_by_pair,
        score_weights=weights,
    )

    (round_dir / "scoring").mkdir(parents=True, exist_ok=True)
    (round_dir / "scoring" / "ranking_v022.json").write_text(
        json.dumps([r.as_dict() for r in rows], indent=2), encoding="utf-8",
    )
    (round_dir / "next_round_context.json").write_text(
        json.dumps(build_next_round_context_v022(rows, top_n=3), indent=2),
        encoding="utf-8",
    )

    worst_all: list[dict[str, Any]] = []
    for r in rows:
        for wc in r.worst_cells:
            worst_all.append({"prompt_pair_id": r.prompt_pair_id, **wc})
    worst_all.sort(key=lambda x: x.get("overall", 1.0))
    worst_all = worst_all[:50]
    (round_dir / "failures").mkdir(parents=True, exist_ok=True)
    (round_dir / "failures" / "failed_cells.json").write_text(
        json.dumps(worst_all, indent=2), encoding="utf-8",
    )

    # Memory upsert (mirrors runner_v022._populate_memory_row_subscores).
    from system_prompt_retrieval_agent.runner_v022 import (
        _populate_memory_row_subscores,
    )
    mem_rows: list[dict[str, Any]] = []
    for r in rows:
        row = _empty_memory_row(run_id, round_id, r.prompt_pair_id)
        row["pair_overall"] = r.pair_overall
        row["overall_score"] = r.pair_overall
        _populate_memory_row_subscores(row, r)
        mem_rows.append(row)
    if mem_rows:
        memory_root.mkdir(parents=True, exist_ok=True)
        upsert_long_memory_rows(memory_root / "long_memory.csv", mem_rows)

    return {
        "n_pairs": len(rows),
        "n_cells": sum(r.n_cells for r in rows),
        "n_worst": len(worst_all),
        "n_mem_rows": len(mem_rows),
    }


def discover_run_id(run_root: Path, override: str | None) -> str:
    if override:
        return override
    candidates = [p for p in run_root.iterdir() if p.is_dir() and cp.RUN_ID_REGEX.match(p.name)]
    if not candidates:
        raise SystemExit(f"no run_id directories under {run_root}")
    if len(candidates) > 1:
        names = sorted(p.name for p in candidates)
        raise SystemExit(
            f"multiple run_ids under {run_root}; pass --run-id ({', '.join(names)})"
        )
    return candidates[0].name


def main() -> int:
    workspace_default = (
        REPO_ROOT / "System-Prompt-Retrieval-Agent" / "preeval_workspace"
    )
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifact-root", type=Path, default=workspace_default / "live",
        help="Parent of outputs/v02/. Defaults to preeval_workspace/live.",
    )
    p.add_argument(
        "--run-root", type=Path, default=workspace_default / "live" / "runs",
        help="Parent of <run_id>/rounds/. Defaults to preeval_workspace/live/runs.",
    )
    p.add_argument(
        "--memory-root", type=Path, default=workspace_default / "mem",
        help="Parent of long_memory.csv. Defaults to preeval_workspace/mem.",
    )
    p.add_argument("--run-id", type=str, default=None)
    args = p.parse_args()

    weights = EvaluationWeights().model_dump()
    run_id = discover_run_id(args.run_root, args.run_id)
    rounds_dir = args.run_root / run_id / "rounds"
    if not rounds_dir.is_dir():
        raise SystemExit(f"no rounds under {rounds_dir}")

    totals: dict[str, int] = {}
    for child in sorted(rounds_dir.iterdir()):
        if not child.is_dir():
            continue
        round_id = _round_id_from_dir(child.name)
        if round_id is None:
            continue
        stats = backfill_round(
            artifact_root=args.artifact_root,
            run_root=args.run_root,
            memory_root=args.memory_root,
            run_id=run_id,
            round_id=round_id,
            round_dir=child,
            weights=weights,
        )
        print(f"[{child.name}] {stats}")
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v

    print(f"[totals] run_id={run_id} {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
