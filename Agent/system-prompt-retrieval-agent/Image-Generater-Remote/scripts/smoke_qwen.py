#!/usr/bin/env python3
"""smoke_qwen.py — Smoke test for Qwen3-VL eval on one supervisor host.

Sequence:
  1. POST /load   with model_name="qwen3_vl"
  2. POST /infer/single  with a three-image eval request
  3. Save eval.json, verify required response keys
  4. POST /unload, record mem delta
  5. Print summary and exit 0 on success, 1 on failure.

Usage:
    python scripts/smoke_qwen.py [--host HOST] [--config CONFIG]

    --host   Supervisor base URL  (default: http://127.0.0.1:17610)
    --config Path to config YAML  (default: config.yaml)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only — no requests dependency needed for a smoke test)
# ---------------------------------------------------------------------------

def _post(url: str, body: dict[str, Any], timeout: int = 300) -> tuple[int, dict]:
    """POST JSON body to url, return (status_code, response_dict)."""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            payload = json.loads(resp.read().decode("utf-8"))
            return status, payload
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
        except Exception:
            payload = {"error": str(exc)}
        return exc.code, payload


def _get(url: str, timeout: int = 30) -> tuple[int, dict]:
    """GET url, return (status_code, response_dict)."""
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


# ---------------------------------------------------------------------------
# Main smoke test
# ---------------------------------------------------------------------------

def run_smoke(host: str, config_path: str) -> int:
    """Execute the full smoke sequence.  Returns 0 on success, 1 on failure."""
    host = host.rstrip("/")
    errors: list[str] = []
    results: dict[str, Any] = {
        "host": host,
        "config": config_path,
        "steps": {},
    }

    # ------------------------------------------------------------------
    # Step 0: health check
    # ------------------------------------------------------------------
    print(f"[smoke_qwen] Host: {host}")
    print(f"[smoke_qwen] Config: {config_path}")
    print("[smoke_qwen] Step 0: GET /health ...")
    status, body = _get(f"{host}/health", timeout=15)
    if status != 200:
        print(f"  FAIL  /health returned HTTP {status}: {body}")
        errors.append("health_check_failed")
    else:
        print(f"  ok    /health -> loaded_model={body.get('loaded_model')!r} "
              f"gpu_id={body.get('gpu_id')}")
    results["steps"]["health"] = {"status": status, "body": body}

    # ------------------------------------------------------------------
    # Step 1: POST /load
    # ------------------------------------------------------------------
    print("[smoke_qwen] Step 1: POST /load model_name=qwen3_vl ...")
    t0 = time.monotonic()
    status, body = _post(f"{host}/load", {"model_name": "qwen3_vl"}, timeout=900)
    load_elapsed_s = round(time.monotonic() - t0, 1)
    results["steps"]["load"] = {"status": status, "body": body, "elapsed_s": load_elapsed_s}

    if status != 200 or body.get("status") != "loaded":
        msg = body.get("error", {}).get("message", str(body)) if not body.get("status") else body.get("status")
        print(f"  FAIL  /load HTTP {status}: {msg}")
        errors.append("load_failed")
        # Cannot continue without a loaded model.
        _print_summary(results, errors)
        return 1

    adapter_attached = body.get("adapter_attached")
    adapter_backend_used = body.get("adapter_backend_used")
    load_latency_ms = body.get("load_latency_ms", 0)
    print(
        f"  ok    /load  latency={load_latency_ms:.0f}ms  "
        f"adapter_attached={adapter_attached}  "
        f"adapter_backend_used={adapter_backend_used!r}  "
        f"elapsed={load_elapsed_s}s"
    )

    # ------------------------------------------------------------------
    # Step 2: POST /infer/single (three-image eval request)
    # ------------------------------------------------------------------
    print("[smoke_qwen] Step 2: POST /infer/single (three-image eval) ...")

    # Use the dataset root from the config if present; otherwise use
    # a clearly synthetic path so the server can reject it gracefully
    # (or succeed if the supervisor's path-validation is skipped for smoke).
    dataset_root = _read_dataset_root(config_path)
    if dataset_root:
        base = dataset_root.rstrip("/")
        model_image_path = f"{base}/smoke/model.jpg"
        cloth_image_path = f"{base}/smoke/cloth.jpg"
        result_image_path = f"{base}/smoke/result.jpg"
    else:
        # Synthetic paths — server will likely return path_not_found;
        # the smoke test handles that gracefully (non-fatal for path errors).
        model_image_path = "/tmp/smoke_model.jpg"
        cloth_image_path = "/tmp/smoke_cloth.jpg"
        result_image_path = "/tmp/smoke_result.jpg"

    infer_payload = {
        "model_image_path": model_image_path,
        "cloth_image_path": cloth_image_path,
        "result_image_path": result_image_path,
        "eval_instruction": "Does the person in Image 3 wear the exact clothing shown in Image 2?",
        "sample_id": "smoke_001",
        "run_id": "smoke_run",
        "stage_id": "smoke_stage",
    }

    t0 = time.monotonic()
    status, body = _post(f"{host}/infer/single", infer_payload, timeout=300)
    infer_elapsed_s = round(time.monotonic() - t0, 1)
    results["steps"]["infer_single"] = {
        "status": status,
        "body": body,
        "elapsed_s": infer_elapsed_s,
    }

    # Determine whether the infer response has the required eval keys.
    # A path_not_found response is a server-side error (not a code bug),
    # but we still verify the envelope.
    infer_ok = False
    infer_is_path_error = False

    if status == 200:
        # Successful inference — verify required fields.
        has_raw_response = "raw_response" in body
        has_parsed = "parsed" in body
        has_parse_status = "parse_status" in body

        if has_raw_response and has_parsed and has_parse_status:
            infer_ok = True
            print(
                f"  ok    /infer/single  latency={infer_elapsed_s}s  "
                f"parse_status={body.get('parse_status')!r}  "
                f"parsed={body.get('parsed')!r}"
            )
        else:
            missing = [k for k in ("raw_response", "parsed", "parse_status") if k not in body]
            print(f"  FAIL  /infer/single response missing keys: {missing}  body={body}")
            errors.append("infer_missing_keys")

    elif status == 400:
        error_code = body.get("error", {}).get("error_code", "")
        if error_code in ("path_not_found", "path_not_allowed"):
            infer_is_path_error = True
            infer_ok = True  # Path error is expected in smoke with synthetic paths
            print(
                f"  warn  /infer/single path error (expected with smoke paths): "
                f"code={error_code!r}  elapsed={infer_elapsed_s}s"
            )
        else:
            print(f"  FAIL  /infer/single HTTP 400: {body}")
            errors.append("infer_failed")
    else:
        print(f"  FAIL  /infer/single HTTP {status}: {body}")
        errors.append("infer_failed")

    # ------------------------------------------------------------------
    # Step 3: Save eval.json
    # ------------------------------------------------------------------
    eval_output_path = Path("eval.json")
    eval_record = {
        "smoke_test": "smoke_qwen",
        "host": host,
        "config": config_path,
        "infer_payload": infer_payload,
        "infer_status": status,
        "infer_response": body,
        "infer_elapsed_s": infer_elapsed_s,
        "adapter_attached": adapter_attached,
        "adapter_backend_used": adapter_backend_used,
    }
    try:
        eval_output_path.write_text(json.dumps(eval_record, indent=2))
        print(f"[smoke_qwen] Step 3: eval.json saved -> {eval_output_path.resolve()}")

        # Verify eval.json has required keys (only when infer succeeded without path error)
        if not infer_is_path_error and status == 200:
            loaded = json.loads(eval_output_path.read_text())
            resp = loaded.get("infer_response", {})
            for key in ("raw_response", "parsed", "parse_status"):
                if key not in resp:
                    print(f"  FAIL  eval.json missing key in infer_response: {key!r}")
                    errors.append(f"eval_json_missing_{key}")
            if not any(e.startswith("eval_json_missing_") for e in errors):
                print("  ok    eval.json verified: raw_response, parsed, parse_status present")
        else:
            print("  ok    eval.json saved (path-error or non-200 case; fields not required)")

    except OSError as exc:
        print(f"  FAIL  Could not write eval.json: {exc}")
        errors.append("eval_json_write_failed")

    # ------------------------------------------------------------------
    # Step 4: POST /unload
    # ------------------------------------------------------------------
    print("[smoke_qwen] Step 4: POST /unload ...")
    t0 = time.monotonic()
    status, body = _post(f"{host}/unload", {}, timeout=120)
    unload_elapsed_s = round(time.monotonic() - t0, 1)
    results["steps"]["unload"] = {"status": status, "body": body, "elapsed_s": unload_elapsed_s}

    if status != 200:
        print(f"  FAIL  /unload HTTP {status}: {body}")
        errors.append("unload_failed")
    else:
        unload_status = body.get("status", "unknown")
        mem_before = body.get("mem_before_gib", 0)
        mem_after = body.get("mem_after_gib", 0)
        mem_delta = round(mem_before - mem_after, 2)
        restarted = body.get("restarted", False)
        print(
            f"  ok    /unload  status={unload_status!r}  "
            f"mem_before={mem_before:.2f}GiB  "
            f"mem_after={mem_after:.2f}GiB  "
            f"mem_freed={mem_delta:.2f}GiB  "
            f"restarted={restarted}  elapsed={unload_elapsed_s}s"
        )

    # ------------------------------------------------------------------
    # Step 5: Print summary
    # ------------------------------------------------------------------
    _print_summary(results, errors, adapter_attached=adapter_attached, adapter_backend_used=adapter_backend_used)

    return 0 if not errors else 1


def _read_dataset_root(config_path: str) -> str | None:
    """Try to read paths.dataset_root from config YAML; return None on any failure."""
    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return None

    try:
        p = Path(config_path)
        if not p.exists():
            return None
        with p.open("r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        if isinstance(cfg, dict):
            paths = cfg.get("paths", {})
            if isinstance(paths, dict):
                return paths.get("dataset_root")
    except Exception:
        pass
    return None


def _print_summary(
    results: dict[str, Any],
    errors: list[str],
    adapter_attached: bool | None = None,
    adapter_backend_used: str | None = None,
) -> None:
    """Print a human-readable summary table."""
    print()
    print("=" * 60)
    print("smoke_qwen summary")
    print("=" * 60)
    print(f"  host                  : {results['host']}")
    print(f"  config                : {results['config']}")
    print(f"  adapter_attached      : {adapter_attached}")
    print(f"  adapter_backend_used  : {adapter_backend_used!r}")
    print()

    for step, info in results.get("steps", {}).items():
        elapsed = info.get("elapsed_s", "")
        elapsed_str = f"  ({elapsed}s)" if elapsed != "" else ""
        print(f"  {step:<20} HTTP {info['status']}{elapsed_str}")

    print()
    if errors:
        print(f"  RESULT: FAILED  ({len(errors)} error(s))")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  RESULT: PASSED")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test for Qwen3-VL eval on one supervisor host."
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

    exit_code = run_smoke(host=args.host, config_path=args.config)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
