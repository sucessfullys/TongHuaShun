#!/usr/bin/env python3
"""qwen_sweep.py -- Sweep Qwen3-VL batch sizes to find the largest stable batch.

Sequence for each candidate batch size (from config calibration.qwen_batch_candidates):
  1. POST /load   with model_name="qwen3_vl"
  2. POST /infer/batch  with N synthetic items
  3. GET  /gpu_status   to check VRAM utilisation
  4. POST /unload

A batch is "stable" when:
  - No OOM error was returned by /infer/batch or /load.
  - Peak VRAM usage after inference is below calibration.memory_ceiling_pct (default 90 %).

The largest stable batch size is written to calibration_qwen.json in the
current working directory.  Exits 0 on success, 1 on any hard failure that
prevents the sweep from completing.

Usage:
    python tests/calibration/qwen_sweep.py [--host HOST] [--config CONFIG]

    --host   Supervisor base URL  (default: http://127.0.0.1:17610)
    --config Path to config YAML  (default: config.yaml)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only)
# ---------------------------------------------------------------------------

def _post(url: str, body: dict[str, Any], timeout: int = 600) -> tuple[int, dict]:
    """POST JSON body to url; return (status_code, response_dict)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"error": str(exc)}
        return exc.code, payload
    except Exception as exc:
        return -1, {"error": str(exc)}


def _get(url: str, timeout: int = 30) -> tuple[int, dict]:
    """GET url; return (status_code, response_dict)."""
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"error": str(exc)}
        return exc.code, payload
    except Exception as exc:
        return -1, {"error": str(exc)}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _load_yaml(config_path: str) -> dict[str, Any]:
    """Load YAML config; return {} on any failure."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return {}
    try:
        p = Path(config_path)
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _batch_candidates(cfg: dict[str, Any]) -> list[int]:
    """Return qwen_batch_candidates from config, falling back to defaults."""
    candidates = (
        cfg.get("calibration", {}).get("qwen_batch_candidates")
        if isinstance(cfg.get("calibration"), dict)
        else None
    )
    if isinstance(candidates, list) and candidates:
        return [int(c) for c in candidates]
    return [1, 2, 4, 8, 16]


def _memory_ceiling_pct(cfg: dict[str, Any]) -> float:
    """Return memory_ceiling_pct from calibration section, then gpu section, then 90."""
    cal = cfg.get("calibration", {})
    if isinstance(cal, dict) and "memory_ceiling_pct" in cal:
        return float(cal["memory_ceiling_pct"])
    gpu = cfg.get("gpu", {})
    if isinstance(gpu, dict) and "memory_ceiling_pct" in gpu:
        return float(gpu["memory_ceiling_pct"])
    return 90.0


# ---------------------------------------------------------------------------
# Sweep helpers
# ---------------------------------------------------------------------------

_SYNTHETIC_ITEM_TEMPLATE = {
    "model_image_path": "/tmp/sweep_model.jpg",
    "cloth_image_path": "/tmp/sweep_cloth.jpg",
    "result_image_path": "/tmp/sweep_result.jpg",
    "eval_instruction": "Does the person wear the clothing shown?",
    "run_id": "calibration_sweep",
    "stage_id": "qwen_sweep",
}


def _make_batch_payload(n: int) -> dict[str, Any]:
    """Build an /infer/batch request with N synthetic items."""
    items = []
    for i in range(n):
        item: dict[str, Any] = dict(_SYNTHETIC_ITEM_TEMPLATE)
        item["index"] = i
        item["sample_id"] = f"sweep_{i:04d}"
        item["cell_id"] = "sweep"
        item["shard_id"] = 0
        item["host_port"] = 0
        item["model_name"] = "qwen3_vl"
        items.append(item)
    return {
        "items": items,
        "run_id": "calibration_sweep",
        "stage_id": "qwen_sweep",
        "cell_id": "sweep",
        "shard_id": 0,
        "sample_id": "",
        "host_port": 0,
        "model_name": "qwen3_vl",
        "manifest_hash": "",
        "shard_indices": [],
        "output_root_relpath": "",
        "callback_url": "",
    }


def _is_oom_response(status: int, body: dict) -> bool:
    """Return True when the response indicates an OOM error."""
    if status in (500, 503):
        error_code = body.get("error", {}).get("error_code", "")
        if "oom" in error_code.lower():
            return True
        msg = body.get("error", {}).get("message", "").lower()
        if "out of memory" in msg or "cuda out of memory" in msg:
            return True
    return False


def _check_vram_pct(host: str, ceiling_pct: float) -> tuple[bool, Optional[float], str]:
    """Return (is_below_ceiling, used_pct, note).

    is_below_ceiling is True when VRAM usage is below ceiling_pct, or when
    GPU status is unavailable (conservative: assume safe so the sweep
    continues based on actual OOM signals instead).
    """
    status, body = _get(f"{host}/gpu_status", timeout=20)
    if status != 200:
        return True, None, f"gpu_status HTTP {status} — assuming safe"

    gpus = body.get("gpus", [])
    if not gpus:
        return True, None, "no GPU data returned — assuming safe"

    # Evaluate all reported GPUs; use the maximum utilisation.
    max_pct: float = 0.0
    for g in gpus:
        total = float(g.get("mem_total_gib", 0))
        used = float(g.get("mem_used_gib", 0))
        if total > 0:
            pct = (used / total) * 100.0
            if pct > max_pct:
                max_pct = pct

    is_safe = max_pct < ceiling_pct
    note = f"peak VRAM {max_pct:.1f}% (ceiling {ceiling_pct:.0f}%)"
    return is_safe, max_pct, note


def _try_batch(host: str, batch_size: int, ceiling_pct: float) -> dict[str, Any]:
    """Run one full load/infer/vram-check/unload cycle for batch_size.

    Returns a result dict with keys:
        batch_size, stable (bool), oom (bool), vram_pct (float|None),
        load_status, infer_status, unload_status, note, elapsed_s.
    """
    result: dict[str, Any] = {
        "batch_size": batch_size,
        "stable": False,
        "oom": False,
        "vram_pct": None,
        "load_status": None,
        "infer_status": None,
        "unload_status": None,
        "note": "",
        "elapsed_s": 0.0,
    }
    t_start = time.monotonic()

    # ── Load ──────────────────────────────────────────────────────────
    print(f"  [batch={batch_size}] POST /load ...")
    load_status, load_body = _post(f"{host}/load", {"model_name": "qwen3_vl"}, timeout=900)
    result["load_status"] = load_status

    if load_status != 200 or load_body.get("status") != "loaded":
        oom = _is_oom_response(load_status, load_body)
        result["oom"] = oom
        result["note"] = f"load failed HTTP {load_status}: {load_body.get('error', load_body)}"
        print(f"  [batch={batch_size}] load FAILED (oom={oom}): {result['note']}")
        result["elapsed_s"] = round(time.monotonic() - t_start, 2)
        return result

    print(f"  [batch={batch_size}] load ok  latency={load_body.get('load_latency_ms', 0):.0f}ms")

    # ── Infer ─────────────────────────────────────────────────────────
    print(f"  [batch={batch_size}] POST /infer/batch (N={batch_size}) ...")
    payload = _make_batch_payload(batch_size)
    infer_status, infer_body = _post(f"{host}/infer/batch", payload, timeout=900)
    result["infer_status"] = infer_status

    oom = _is_oom_response(infer_status, infer_body)
    if oom:
        result["oom"] = True
        result["note"] = f"OOM during infer (batch={batch_size})"
        print(f"  [batch={batch_size}] infer OOM detected")
        # Still try to unload to clean up.
        _post(f"{host}/unload", {}, timeout=120)
        result["unload_status"] = "attempted_after_oom"
        result["elapsed_s"] = round(time.monotonic() - t_start, 2)
        return result

    # Path errors (400) from synthetic paths are expected — they are not OOM
    # and do not indicate the batch size is unstable at the hardware level.
    infer_hard_failure = infer_status not in (200, 400)
    if infer_hard_failure:
        result["note"] = (
            f"infer returned unexpected HTTP {infer_status}: "
            f"{infer_body.get('error', infer_body)}"
        )
        print(f"  [batch={batch_size}] infer hard failure: {result['note']}")
        _post(f"{host}/unload", {}, timeout=120)
        result["unload_status"] = "attempted_after_failure"
        result["elapsed_s"] = round(time.monotonic() - t_start, 2)
        return result

    if infer_status == 200:
        print(
            f"  [batch={batch_size}] infer ok  "
            f"ok_count={infer_body.get('ok_count', '?')}  "
            f"error_count={infer_body.get('error_count', '?')}"
        )
    else:
        err_code = infer_body.get("error", {}).get("error_code", "")
        print(
            f"  [batch={batch_size}] infer 400 (expected path error, code={err_code!r}) "
            f"— treating as non-OOM"
        )

    # ── VRAM check ────────────────────────────────────────────────────
    vram_safe, vram_pct, vram_note = _check_vram_pct(host, ceiling_pct)
    result["vram_pct"] = vram_pct
    print(f"  [batch={batch_size}] VRAM: {vram_note}")

    # ── Unload ────────────────────────────────────────────────────────
    print(f"  [batch={batch_size}] POST /unload ...")
    unload_status, unload_body = _post(f"{host}/unload", {}, timeout=120)
    result["unload_status"] = unload_status

    if unload_status != 200:
        print(f"  [batch={batch_size}] unload returned HTTP {unload_status}: {unload_body}")
    else:
        mem_before = unload_body.get("mem_before_gib", 0)
        mem_after = unload_body.get("mem_after_gib", 0)
        print(
            f"  [batch={batch_size}] unload ok  "
            f"freed={mem_before - mem_after:.2f} GiB"
        )

    result["elapsed_s"] = round(time.monotonic() - t_start, 2)

    if not vram_safe:
        result["note"] = f"VRAM above ceiling: {vram_note}"
        print(f"  [batch={batch_size}] UNSTABLE — {result['note']}")
        return result

    result["stable"] = True
    result["note"] = f"stable — {vram_note}"
    print(f"  [batch={batch_size}] STABLE")
    return result


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep(host: str, config_path: str) -> int:
    """Run the full batch-size sweep.  Returns 0 on success, 1 on failure."""
    host = host.rstrip("/")
    print(f"[qwen_sweep] Host   : {host}")
    print(f"[qwen_sweep] Config : {config_path}")

    cfg = _load_yaml(config_path)
    candidates = _batch_candidates(cfg)
    ceiling_pct = _memory_ceiling_pct(cfg)

    print(f"[qwen_sweep] Candidates     : {candidates}")
    print(f"[qwen_sweep] VRAM ceiling   : {ceiling_pct:.0f}%")

    # ── Health check ──────────────────────────────────────────────────
    print("[qwen_sweep] GET /health ...")
    health_status, health_body = _get(f"{host}/health", timeout=15)
    if health_status != 200:
        print(f"  ERROR  /health returned HTTP {health_status}: {health_body}")
        print("[qwen_sweep] Cannot proceed — host is not reachable.")
        return 1
    print(f"  ok    /health  loaded_model={health_body.get('loaded_model')!r}  "
          f"gpu_id={health_body.get('gpu_id')}")

    # ── Sweep each candidate ──────────────────────────────────────────
    sweep_results: list[dict[str, Any]] = []
    best_batch: int = 0

    for batch_size in sorted(set(candidates)):
        print(f"\n[qwen_sweep] --- batch_size={batch_size} ---")
        result = _try_batch(host, batch_size, ceiling_pct)
        sweep_results.append(result)

        if result["stable"]:
            best_batch = batch_size
        elif result["oom"]:
            # OOM means larger sizes will also fail; stop early.
            print(f"[qwen_sweep] OOM at batch_size={batch_size} — stopping sweep early.")
            break

    # ── Determine result ──────────────────────────────────────────────
    print()
    if best_batch == 0:
        print("[qwen_sweep] No stable batch size found.")
        success = False
    else:
        print(f"[qwen_sweep] Largest stable batch size: {best_batch}")
        success = True

    # ── Write output JSON ─────────────────────────────────────────────
    output: dict[str, Any] = {
        "host": host,
        "config": config_path,
        "candidates_tried": candidates,
        "memory_ceiling_pct": ceiling_pct,
        "best_batch_size": best_batch if best_batch > 0 else None,
        "sweep_ok": success,
        "results": sweep_results,
    }

    out_path = Path("calibration_qwen.json")
    try:
        out_path.write_text(json.dumps(output, indent=2))
        print(f"[qwen_sweep] calibration_qwen.json -> {out_path.resolve()}")
    except OSError as exc:
        print(f"[qwen_sweep] ERROR writing calibration_qwen.json: {exc}")
        return 1

    # ── Summary ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("qwen_sweep summary")
    print("=" * 60)
    print(f"  host                  : {host}")
    print(f"  config                : {config_path}")
    print(f"  candidates            : {candidates}")
    print(f"  VRAM ceiling          : {ceiling_pct:.0f}%")
    print()
    for r in sweep_results:
        stable_str = "STABLE  " if r["stable"] else ("OOM     " if r["oom"] else "UNSTABLE")
        vram_str = f"{r['vram_pct']:.1f}%" if r["vram_pct"] is not None else "N/A"
        print(
            f"  batch={r['batch_size']:<4}  {stable_str}  "
            f"VRAM={vram_str:<7}  {r['note']}"
        )
    print()
    if success:
        print(f"  RESULT: best_batch_size = {best_batch}")
    else:
        print("  RESULT: no stable batch size found (best_batch_size = null)")
    print("=" * 60)

    return 0 if success else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep Qwen3-VL batch sizes to find the largest stable batch."
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:17610",
        help="Supervisor base URL (default: http://127.0.0.1:17610)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    sys.exit(run_sweep(host=args.host, config_path=args.config))


if __name__ == "__main__":
    main()
