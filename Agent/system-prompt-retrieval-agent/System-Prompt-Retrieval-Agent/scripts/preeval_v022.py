"""V0.2.2 pre-eval harness — fixed-prompt baseline validation.

Spawns an in-process simulator on http://127.0.0.1:17700 mimicking the
real /v022/stage/{stage} controller (deterministic stub bytes, no GPU
required), runs the V0.2.2 production runner against the two fixed
system prompts from temp_wxy/PROMPTS.py, and post-processes the round
with GPT-5.4-mini for failure summaries + next-round notes.

Run:

    .venv/bin/python scripts/preeval_v022.py
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException

REPO_ROOT = Path("/Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent")
LOCAL_AGENT = REPO_ROOT / "System-Prompt-Retrieval-Agent"
SRC = LOCAL_AGENT / "src"
sys.path.insert(0, str(SRC))

from system_prompt_retrieval_agent.config import (  # noqa: E402
    AppConfig, ApiConfig, BudgetConfig, EvaluationConfig, ExecutionConfig,
    GemmaUserPromptEntry, GemmaUserPromptsConfig, PathsConfig,
    PromptGenerationConfig, RemoteConfig, WorkflowConfig,
)
from system_prompt_retrieval_agent.remote._vendored import canonical_paths as cp  # noqa: E402
from system_prompt_retrieval_agent.runner_v022 import V022Runner  # noqa: E402
from system_prompt_retrieval_agent.schemas import PromptPair  # noqa: E402

# --- Load .env so OpenAI key is available --------------------------------
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv(REPO_ROOT / ".env", override=False)
except Exception:
    pass


# --- Load fixed prompts --------------------------------------------------

def _load_fixed_prompts() -> dict[str, str]:
    spec = importlib.util.spec_from_file_location(
        "_temp_prompts", REPO_ROOT / "temp_wxy" / "PROMPTS.py",
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return {
        "system_prompt_for_tryon_v5": mod.system_prompt_for_tryon_v5,
        "system_prompt_for_tryon_v5_new": mod.system_prompt_for_tryon_v5_new,
    }


# ---------------------------------------------------------------------------
# Simulator: FastAPI app implementing /v022/stage/{stage}
# ---------------------------------------------------------------------------


def make_simulator_app(remote_root: Path) -> FastAPI:
    app = FastAPI()
    state: dict[str, int] = {"calls": 0}

    @app.get("/health")
    def health():
        return {"ok": True, "calls": state["calls"]}

    @app.post("/v022/stage/{stage}")
    def v022_stage(stage: str, payload: dict):
        if stage not in ("gemma", "flux", "qwen"):
            raise HTTPException(status_code=400, detail=f"unknown stage: {stage}")
        attempt_id = payload.get("attempt_id")
        run_id = payload.get("run_id")
        round_id = int(payload.get("round_id", 0))
        if not attempt_id or not run_id:
            raise HTTPException(status_code=400, detail="missing attempt_id/run_id")

        state["calls"] += 1
        cells_out: list[dict[str, Any]] = []
        prompt_pairs = {p["prompt_pair_id"]: p for p in payload.get("prompt_pairs", [])}
        user_prompts = {u["user_prompt_id"]: u for u in payload.get("user_prompts", [])}

        for row in payload["dispatch_cells"]:
            pid, upid, sid = row["prompt_pair_id"], row["user_prompt_id"], row["sample_id"]
            relpath = cp.cell_artifact_path(run_id, stage, round_id, pid, upid, sid)
            full = remote_root / relpath
            full.parent.mkdir(parents=True, exist_ok=True)
            if stage == "gemma":
                # Deterministic intermediate prompt text traceable to the
                # source system prompt and the user prompt.
                pp = prompt_pairs.get(pid, {})
                up = user_prompts.get(upid, {})
                sp_text = (pp.get("system_prompt_text") or "")[:120]
                up_text = (up.get("text") or "")[:120]
                body = (
                    f"[PRE-EVAL Gemma intermediate prompt]\n"
                    f"system_prompt_id={pp.get('system_prompt_id')}\n"
                    f"prompt_pair_id={pid}\n"
                    f"user_prompt_id={upid}\n"
                    f"sample_id={sid}\n"
                    f"system_prompt_excerpt={sp_text!r}\n"
                    f"user_prompt_excerpt={up_text!r}\n"
                    f"attempt_id={attempt_id}\n"
                ).encode("utf-8")
            elif stage == "flux":
                # 1×1 PNG header — minimal valid PNG bytes for traceability.
                body = (
                    b"\x89PNG\r\n\x1a\n"
                    + f"PRE-EVAL FLUX|{pid}|{upid}|{sid}|{attempt_id}".encode()
                )
            else:  # qwen
                eval_payload = {
                    "qwen_pass_rate": 0.80,
                    "edit_correctness": 0.70,
                    "garment_transfer_correctness": 0.65,
                    "preservation": 0.85,
                    "artifact_penalty": 0.05,
                    "trace": {
                        "prompt_pair_id": pid,
                        "user_prompt_id": upid,
                        "sample_id": sid,
                        "attempt_id": attempt_id,
                    },
                }
                body = json.dumps(eval_payload, sort_keys=True).encode("utf-8")
            full.write_bytes(body)
            cells_out.append({
                **row,
                "status": "ok",
                "artifact_relpath": relpath,
                "artifact_size_bytes": len(body),
                "artifact_sha256": hashlib.sha256(body).hexdigest(),
            })

        per_up: dict[str, dict[str, dict[str, int]]] = {}
        pair_rollups: dict[str, dict[str, int]] = {}
        for c in cells_out:
            pair_rollups.setdefault(c["prompt_pair_id"], {"ok": 0, "errors": 0, "total": 0})
            pair_rollups[c["prompt_pair_id"]]["ok"] += 1
            pair_rollups[c["prompt_pair_id"]]["total"] += 1
            per_up.setdefault(c["prompt_pair_id"], {})
            per_up[c["prompt_pair_id"]].setdefault(c["user_prompt_id"], {"ok": 0, "errors": 0, "total": 0})
            per_up[c["prompt_pair_id"]][c["user_prompt_id"]]["ok"] += 1
            per_up[c["prompt_pair_id"]][c["user_prompt_id"]]["total"] += 1

        manifest = {
            **{k: payload[k] for k in cp.MATCH_FIELDS},
            "attempt_id": attempt_id,
            "lifecycle_mode": payload.get("lifecycle_mode", "cold"),
            "lifecycle_state_after": payload.get("lifecycle_state_after", "disk_unloaded"),
            "cells": cells_out,
            "pair_rollups": pair_rollups,
            "per_user_prompt": per_up,
        }
        relpath_mf = cp.stage_manifest_attempt_path(run_id, stage, round_id, attempt_id)
        full_mf = remote_root / relpath_mf
        full_mf.parent.mkdir(parents=True, exist_ok=True)
        full_mf.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return manifest

    return app


def _start_simulator(remote_root: Path, port: int) -> threading.Thread:
    app = make_simulator_app(remote_root)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    th = threading.Thread(target=server.run, daemon=True, name="preeval-simulator")
    th.start()
    # Wait until /health responds.
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if r.status_code == 200:
                return th
        except Exception:
            time.sleep(0.1)
    raise RuntimeError("simulator failed to start within 10s")


# ---------------------------------------------------------------------------
# Local-copy transport for pre-eval (rsync would need SSH; we copy locally)
# ---------------------------------------------------------------------------


def local_copy_transport(*, remote_root, local_temp, cell_relpaths, manifest_relpath):
    # The runner builds remote_root as "<ssh_alias>:<path>" for rsync (see
    # runner_v022._build_round_context). For the in-process pre-eval we strip
    # the "<host>:" prefix and operate on a local path.
    rr = str(remote_root)
    if ":" in rr and not rr.startswith("/"):
        rr = rr.split(":", 1)[1]
    for relpath in list(cell_relpaths) + [manifest_relpath]:
        src = Path(rr) / relpath
        if not src.is_file():
            continue
        dst = Path(local_temp) / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Local evaluator stub for the pre-eval baseline
# ---------------------------------------------------------------------------


class PreEvalLocalEvaluator:
    """Reads the deterministic Qwen-eval JSON we wrote on the simulator
    side and uses those scores as the baseline. No OpenAI calls here —
    this is a fixed-prompt baseline, not a real local-API eval."""

    async def evaluate_many_cells(self, cells, *, api_eval_root=None):  # noqa: ARG002
        out = []
        for c in cells:
            qpath = Path(c["qwen_eval_json_path"])
            try:
                qd = json.loads(qpath.read_text(encoding="utf-8"))
            except Exception:
                qd = {}
            score = (
                qd.get("qwen_pass_rate", 0.0) * 0.4
                + qd.get("edit_correctness", 0.0) * 0.2
                + qd.get("garment_transfer_correctness", 0.0) * 0.15
                + qd.get("preservation", 0.0) * 0.15
                - qd.get("artifact_penalty", 0.0) * 0.1
            )
            out.append({
                "prompt_pair_id": c["prompt_pair_id"],
                "user_prompt_id": c["user_prompt_id"],
                "sample_id": c["sample_id"],
                "score": float(score),
                "qwen_pass_rate": qd.get("qwen_pass_rate", 0.0),
                "edit_correctness": qd.get("edit_correctness", 0.0),
                "garment_transfer_correctness": qd.get("garment_transfer_correctness", 0.0),
                "preservation": qd.get("preservation", 0.0),
                "artifact_penalty": qd.get("artifact_penalty", 0.0),
            })
        return out


# ---------------------------------------------------------------------------
# Fixed prompt-pair factory (no LLM)
# ---------------------------------------------------------------------------


def make_fixed_prompt_factory(prompts: dict[str, str]):
    async def _factory(cfg, mem, round_id, n, existing):  # noqa: ARG001
        out: list[PromptPair] = []
        # n is ignored — we always emit exactly the two fixed prompts.
        for pid, text in prompts.items():
            out.append(
                PromptPair(
                    prompt_pair_id=pid,
                    round_id=round_id,
                    system_prompt_id=pid,
                    system_prompt=text,
                    negative_prompt_id="none",  # schema requires str; pre-eval rules: negative_prompt = None
                    negative_prompt=None,
                )
            )
        return out, False
    return _factory


# ---------------------------------------------------------------------------
# Post-eval: GPT-5.4-mini failure summaries + notes
# ---------------------------------------------------------------------------


def gpt54_mini_post_eval(
    round_dir: Path,
    ranking_path: Path,
    failed_pairs_path: Path,
    failed_cells_path: Path | None = None,
) -> dict:
    """Call gpt-5.4-mini for failure_summaries + notes_for_next_round.

    Falls back to a deterministic "no API key" summary when
    OPENAI_API_KEY is not set or the OpenAI SDK is unavailable.

    ``failed_cells_path`` (optional, V0.2.2 bottom-N diagnostic surface)
    carries per-cell sub-scores so the LLM can identify category ×
    sub-score weak spots even when no pair is below a hard threshold.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    ranking = json.loads(ranking_path.read_text(encoding="utf-8")) if ranking_path.is_file() else []
    failed = json.loads(failed_pairs_path.read_text(encoding="utf-8")) if failed_pairs_path.is_file() else []
    failed_cells: list = []
    if failed_cells_path is not None and failed_cells_path.is_file():
        try:
            failed_cells = json.loads(failed_cells_path.read_text(encoding="utf-8"))
        except Exception:
            failed_cells = []

    summary_payload = {
        "model": "gpt-5.4-mini",
        "ranking_excerpt": ranking[:5],
        "failed_pairs": failed,
        "failed_cells_excerpt": failed_cells[:10],
        "n_pairs": len(ranking),
        "n_failed_cells": len(failed_cells),
    }

    if not api_key:
        summary_payload["status"] = "skipped_no_api_key"
        summary_payload["failure_summaries"] = []
        summary_payload["notes_for_next_round"] = (
            f"API key not present; pre-eval LLM post-step skipped. "
            f"({len(failed_cells)} bottom-N cells available for inspection.)"
        )
        (round_dir / "failure_summaries.json").write_text(
            json.dumps([], indent=2), encoding="utf-8",
        )
        (round_dir / "notes_for_next_round.md").write_text(
            "# notes_for_next_round\n\nOPENAI_API_KEY missing — placeholder only.\n",
            encoding="utf-8",
        )
        return summary_payload

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        summary_payload["status"] = "skipped_no_sdk"
        return summary_payload

    client = OpenAI(api_key=api_key)
    prompt = (
        "You are reviewing a fixed-prompt baseline pre-eval of a try-on agent. "
        "Two fixed system prompts were used. Below are three JSON sections: "
        "Ranking (per-pair aggregates incl. per-category sub-score means), "
        "FailedPairs (pairs with zero eval input), and FailedCells (bottom-N "
        "per-cell sub-scores by category — focus next-round notes on whichever "
        "category x sub-score combination dominates the bottom of the list, "
        "even if no cell is below a hard failure threshold).\n"
        "Produce two outputs as JSON: "
        "{ \"failure_summaries\": [...], \"notes_for_next_round\": \"...\" }. "
        "Do NOT propose new system prompts. Stay within 250 words total.\n\n"
        f"Ranking: {json.dumps(ranking)[:8000]}\n"
        f"FailedPairs: {json.dumps(failed)[:2000]}\n"
        f"FailedCells: {json.dumps(failed_cells)[:6000]}\n"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"
        parsed = json.loads(text)
    except Exception as exc:  # noqa: BLE001
        summary_payload["status"] = f"llm_error: {exc}"
        parsed = {
            "failure_summaries": [],
            "notes_for_next_round": (
                f"LLM call failed ({exc}); pre-eval continued with placeholder notes."
            ),
        }

    summary_payload["status"] = summary_payload.get("status", "ok")
    summary_payload["failure_summaries"] = parsed.get("failure_summaries", [])
    summary_payload["notes_for_next_round"] = parsed.get("notes_for_next_round", "")
    (round_dir / "failure_summaries.json").write_text(
        json.dumps(summary_payload["failure_summaries"], indent=2), encoding="utf-8",
    )
    (round_dir / "notes_for_next_round.md").write_text(
        f"# notes_for_next_round\n\n{summary_payload['notes_for_next_round']}\n",
        encoding="utf-8",
    )
    return summary_payload


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    work = REPO_ROOT / "System-Prompt-Retrieval-Agent" / "preeval_workspace"
    # Preserve a pre-built full_sample_manifest.json across the wipe.
    full_manifest_src = work / "full_sample_manifest.json"
    full_manifest_data: list[dict] | None = None
    if full_manifest_src.is_file():
        full_manifest_data = json.loads(full_manifest_src.read_text())
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    if full_manifest_data is not None:
        full_manifest_src.write_text(json.dumps(full_manifest_data, indent=2))

    output_root = work / "live"
    memory_root = work / "mem"
    dataset_root = work / "data"
    remote_root_api = work / "remote"
    output_root.mkdir(); memory_root.mkdir(); dataset_root.mkdir(); remote_root_api.mkdir()
    (remote_root_api / "outputs" / "v02").mkdir(parents=True)

    # Two enabled "user prompts" — minimal, since we want the system-prompt
    # axis to be the dominant variable in the pre-eval. Use one user prompt
    # so each row is a clean system_prompt × sample combination.
    user_prompts_library = [
        GemmaUserPromptEntry(
            user_prompt_id="UP_FIXED",
            language="en",
            text="Replace the model's clothing with the reference garment.",
            enabled=True,
        ),
    ]

    if full_manifest_data:
        sample_manifest = full_manifest_data
        print(f"[preeval] using FULL test dataset: {len(sample_manifest)} samples")
    else:
        sample_manifest = [
            {"sample_id": "S1", "model_image_path": str(dataset_root / "m1.png"),
             "cloth_image_path": str(dataset_root / "c1.png")},
            {"sample_id": "S2", "model_image_path": str(dataset_root / "m2.png"),
             "cloth_image_path": str(dataset_root / "c2.png")},
        ]
        for s in sample_manifest:
            Path(s["model_image_path"]).write_bytes(b"stub-model")
            Path(s["cloth_image_path"]).write_bytes(b"stub-cloth")
        print(f"[preeval] using stub 2-sample dataset (no full_sample_manifest.json found)")
    (dataset_root / "sample_manifest.json").write_text(json.dumps(sample_manifest))

    cfg = AppConfig(
        paths=PathsConfig(
            local_project_root=str(REPO_ROOT),
            local_agent_root=str(LOCAL_AGENT),
            remote_image_service_root=str(remote_root_api),
            dataset_root=str(dataset_root),
            output_root=str(output_root),
            memory_root=str(memory_root),
            prompt_seed_file=str(REPO_ROOT / "temp_wxy" / "PROMPTS.py"),
            env_file=str(REPO_ROOT / ".env"),
        ),
        remote=RemoteConfig(
            tunnel_command="echo no-tunnel",
            controller_base_url="http://127.0.0.1:17700",
            request_timeout_s=30,
            stage_retry_max=1,
            lock_contention_retry_max=1,
            lock_contention_wait_s=1,
        ),
        api=ApiConfig(),
        workflow=WorkflowConfig(max_rounds=1, score_threshold=2.0),
        budget=BudgetConfig(daily_usd_cap=5.0, per_round_usd_cap=5.0),
        prompt_generation=PromptGenerationConfig(prompt_pairs_per_round=2),
        gemma_user_prompts=GemmaUserPromptsConfig(
            library=user_prompts_library, corpus_id="preeval_v022",
        ),
        evaluation=EvaluationConfig(run_local_api_eval=True),
        execution=ExecutionConfig(mode="v022_stage_major"),
    )

    # 1) Spawn simulator. canonical_paths.cell_artifact_path / stage_manifest_attempt_path
    # already include the `outputs/v02/` prefix, so the simulator's remote_root
    # must be the bare remote root (not nested under `outputs/v02`); otherwise
    # it writes to a doubled path that the local copy-back transport can't see.
    sim_remote = remote_root_api
    print(f"[preeval] starting simulator (remote root: {sim_remote})")
    _start_simulator(sim_remote, 17700)

    # 2) Build runner with fixed prompt factory + local copy-back
    fixed_prompts = _load_fixed_prompts()
    print(f"[preeval] loaded fixed prompts: {list(fixed_prompts)}")

    runner = V022Runner(
        cfg, max_rounds=1,
        local_evaluator=PreEvalLocalEvaluator(),
        copy_back_transport=local_copy_transport,
        prompt_pair_factory=make_fixed_prompt_factory(fixed_prompts),
    )
    print(f"[preeval] run_id = {runner.run_id}")

    # 3) Execute the round
    outcome = await runner.run()
    print(f"[preeval] runner outcome: {outcome}")

    # 4) Verify artifacts (cell IDs come from the actual sample manifest)
    round_dir = Path(cfg.paths.output_root) / "runs" / runner.run_id / "rounds" / "round_001"
    sample_ids = sorted(s["sample_id"] for s in sample_manifest)
    upids = tuple(u.user_prompt_id for u in user_prompts_library)
    expected_cells = []
    for stage in ("gemma", "flux", "qwen"):
        for pid in fixed_prompts:
            for upid in upids:
                for sid in sample_ids:
                    rel = cp.cell_artifact_path(runner.run_id, stage, 1, pid, upid, sid)
                    expected_cells.append((stage, pid, upid, sid, rel))
    missing = []
    for stage, pid, upid, sid, rel in expected_cells:
        p = Path(cfg.paths.output_root) / rel
        if not p.is_file():
            missing.append(str(p))
    if missing:
        print(f"[preeval] MISSING ARTIFACTS ({len(missing)} of {len(expected_cells)}):")
        for m in missing[:10]:
            print("  -", m)
        if len(missing) > 10:
            print(f"  ... +{len(missing) - 10} more")
        return 2

    n_intermediate = sum(1 for s, *_ in expected_cells if s == "gemma")
    n_images = sum(1 for s, *_ in expected_cells if s == "flux")

    # 5) Failure summaries + next-round notes via gpt-5.4-mini
    ranking_path = round_dir / "scoring" / "ranking_v022.json"
    failed_pairs_path = round_dir / "failures" / "failed_pairs.json"
    failed_cells_path = round_dir / "failures" / "failed_cells.json"
    summary = gpt54_mini_post_eval(
        round_dir, ranking_path, failed_pairs_path, failed_cells_path,
    )

    # 6) Verify memory initialization
    long_mem = Path(cfg.paths.memory_root) / "long_memory.csv"
    mem_rows = 0
    mem_run_id_ok = False
    if long_mem.is_file():
        import csv

        with long_mem.open() as fh:
            for row in csv.DictReader(fh):
                mem_rows += 1
                if row.get("run_id") == runner.run_id:
                    mem_run_id_ok = True

    report = {
        "status": "ok",
        "run_directory": str(work),
        "run_id": runner.run_id,
        "fixed_prompts": list(fixed_prompts),
        "n_intermediate_prompts": n_intermediate,
        "n_images_generated": n_images,
        "n_qwen_evals": sum(1 for s, *_ in expected_cells if s == "qwen"),
        "ranking_path": str(ranking_path) if ranking_path.is_file() else None,
        "ranking": json.loads(ranking_path.read_text())[:5] if ranking_path.is_file() else [],
        "failed_pairs": json.loads(failed_pairs_path.read_text()) if failed_pairs_path.is_file() else [],
        "memory": {
            "long_memory_csv": str(long_mem),
            "rows": mem_rows,
            "run_id_present": mem_run_id_ok,
        },
        "post_eval_llm": summary,
        "outcome": {
            "stop_reason": outcome.stop_reason,
            "rounds_completed": outcome.rounds_completed,
            "best_pair_id": outcome.best_pair_id,
            "best_pair_overall": outcome.best_pair_overall,
        },
    }
    (work / "preeval_report.json").write_text(json.dumps(report, indent=2))
    print("\n=== PRE-EVAL REPORT ===")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
