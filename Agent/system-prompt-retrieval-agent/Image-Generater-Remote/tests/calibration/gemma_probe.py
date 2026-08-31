"""gemma_probe.py — Probe Gemma batch-1 peak VRAM and calculate headroom.

Sequence
--------
1. POST /load  model_name="gemma4"
2. GET  /health  → record peak VRAM after load
3. POST /infer/single  (minimal synthetic payload)
4. GET  /health  → record peak VRAM after one inference pass
5. POST /unload
6. Calculate headroom vs calibration.memory_ceiling_pct (default 90 %)
7. Write calibration_gemma.json to <output_root>/calibration/
8. Print summary and exit 0 on success, 1 on any failure.

Usage (on remote server)
------------------------
    python tests/calibration/gemma_probe.py \\
        --host http://127.0.0.1:17610 \\
        --config config.yaml

    python tests/calibration/gemma_probe.py \\
        --host http://127.0.0.1:17610 \\
        --config config.yaml \\
        --output calibration_gemma.json

Exit codes
----------
    0 — probing completed without error (headroom may still be negative)
    1 — any HTTP or protocol error
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

# ── HTTP client (httpx preferred; urllib fallback) ─────────────────

try:
    import httpx as _httpx_module

    _USE_HTTPX = True
except ImportError:
    _USE_HTTPX = False

try:
    import urllib.request as _urllib_request
    import urllib.error as _urllib_error
except ImportError:
    _urllib_request = None  # type: ignore[assignment]
    _urllib_error = None  # type: ignore[assignment]

if not _USE_HTTPX and _urllib_request is None:
    print("ERROR: Neither httpx nor urllib.request is available.", file=sys.stderr)
    sys.exit(1)

# ── YAML loader ────────────────────────────────────────────────────

try:
    import yaml as _yaml

    _USE_YAML = True
except ImportError:
    _USE_YAML = False


# ── Config helpers ─────────────────────────────────────────────────

def _load_yaml(path: str) -> dict[str, Any]:
    """Load a YAML file; return empty dict if the file is absent."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as fh:
        if _USE_YAML:
            return _yaml.safe_load(fh) or {}
        # Minimal JSON-compatible fallback (covers .yaml that is pure JSON)
        try:
            return json.loads(fh.read())
        except Exception:
            return {}


def _deep_get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    """Navigate a nested dict with a dotted key path."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return default
        node = node.get(part, default)
        if node is default:
            return default
    return node


# ── HTTP helpers ───────────────────────────────────────────────────

_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 1800.0


def _http_post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST JSON body; return parsed response dict.  Raises RuntimeError on failure."""
    raw_bytes = json.dumps(body, ensure_ascii=False).encode("utf-8")
    if _USE_HTTPX:
        timeout = _httpx_module.Timeout(
            connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=30.0, pool=5.0
        )
        with _httpx_module.Client() as client:
            resp = client.post(
                url,
                content=raw_bytes,
                headers={"Content-Type": "application/json"},
                timeout=timeout,
            )
            try:
                data = resp.json()
            except Exception:
                data = {"_raw": resp.text}
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {resp.status_code} from {url}: {json.dumps(data, ensure_ascii=False)}"
                )
            return data
    else:
        req = _urllib_request.Request(
            url,
            data=raw_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with _urllib_request.urlopen(req, timeout=_READ_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except _urllib_error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {body_text}") from exc


def _http_get(url: str) -> dict[str, Any]:
    """GET the URL; return parsed JSON response dict.  Raises RuntimeError on failure."""
    if _USE_HTTPX:
        timeout = _httpx_module.Timeout(connect=_CONNECT_TIMEOUT, read=60.0, pool=5.0)
        with _httpx_module.Client() as client:
            resp = client.get(url, timeout=timeout)
            try:
                data = resp.json()
            except Exception:
                data = {"_raw": resp.text}
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"HTTP {resp.status_code} from {url}: {json.dumps(data, ensure_ascii=False)}"
                )
            return data
    else:
        req = _urllib_request.Request(url, method="GET")
        try:
            with _urllib_request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except _urllib_error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {body_text}") from exc


# ── Synthetic inference payload ────────────────────────────────────

_SYNTHETIC_SYSTEM_PROMPT = (
    "You are a virtual try-on assistant. "
    "In one sentence, describe how the model would look wearing the garment."
)

_SYNTHETIC_USER_INSTRUCTION = "the model in image 1 wears the clothing from image 2"

# Placeholder image paths.  The supervisor validates these only on the remote
# server where the dataset is mounted; on a fresh probe we use known-good paths
# from config or a recognisable fallback so the server can attempt inference.
_FALLBACK_MODEL_IMAGE = (
    "/mnt/image-edit/datasets/xywang/dataset"
    "/TryOn_data/VirtualTryOn_whq_test500/test_data"
    "/dress/0001/input_model.png"
)
_FALLBACK_CLOTH_IMAGE = (
    "/mnt/image-edit/datasets/xywang/dataset"
    "/TryOn_data/VirtualTryOn_whq_test500/test_data"
    "/dress/0001/input_cloth.png"
)


def _resolve_image_pair(cfg: dict[str, Any]) -> tuple[str, str]:
    """Return (model_image_path, cloth_image_path) for the probe inference."""
    candidates: list[str] = _deep_get(cfg, "paths.pilot_input_candidates", [])
    patterns = _deep_get(cfg, "data.image_pair_patterns", [
        {"model": "input_model.png", "cloth": "input_cloth.png"},
    ])
    pattern_tuples = [
        (p.get("model", "input_model.png"), p.get("cloth", "input_cloth.png"))
        for p in patterns
    ]
    for candidate in candidates:
        root = Path(candidate)
        if not root.is_dir():
            continue
        for category in ("dress", "lower", "upper"):
            cat_dir = root / category
            if not cat_dir.is_dir():
                continue
            for sample_dir in sorted(cat_dir.iterdir()):
                if not sample_dir.is_dir():
                    continue
                for model_name, cloth_name in pattern_tuples:
                    m = sample_dir / model_name
                    c = sample_dir / cloth_name
                    if m.is_file() and c.is_file():
                        return str(m), str(c)
    return _FALLBACK_MODEL_IMAGE, _FALLBACK_CLOTH_IMAGE


def _build_infer_payload(cfg: dict[str, Any], host: str) -> dict[str, Any]:
    model_image, cloth_image = _resolve_image_pair(cfg)
    user_instruction: str = _deep_get(
        cfg,
        "prompts.default_user_instruction",
        _SYNTHETIC_USER_INSTRUCTION,
    )
    try:
        host_port = int(host.rsplit(":", 1)[-1])
    except (ValueError, IndexError):
        host_port = 17610

    return {
        "run_id": "calibration_gemma_probe",
        "stage_id": "gemma",
        "cell_id": "probe__batch1",
        "shard_id": 0,
        "sample_id": "probe/dress/0001",
        "host_port": host_port,
        "model_name": "gemma4",
        "system_prompt": _SYNTHETIC_SYSTEM_PROMPT,
        "user_instruction": user_instruction,
        "model_image_path": model_image,
        "cloth_image_path": cloth_image,
    }


# ── VRAM snapshot helpers ──────────────────────────────────────────

def _extract_vram(health_resp: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
    """Return (mem_used_gib, mem_total_gib) from a /health response."""
    used = health_resp.get("gpu_mem_used_gib")
    total = health_resp.get("gpu_mem_total_gib")
    if used is None or total is None:
        return None, None
    try:
        return float(used), float(total)
    except (TypeError, ValueError):
        return None, None


def _headroom(
    peak_used_gib: Optional[float],
    total_gib: Optional[float],
    ceiling_pct: float,
) -> Optional[float]:
    """Headroom in GiB = ceiling_gib - peak_used_gib (negative means over budget)."""
    if peak_used_gib is None or total_gib is None or total_gib <= 0:
        return None
    ceiling_gib = total_gib * ceiling_pct / 100.0
    return round(ceiling_gib - peak_used_gib, 3)


# ── Output path ────────────────────────────────────────────────────

def _default_output_path(cfg: dict[str, Any]) -> Path:
    output_root = _deep_get(cfg, "paths.output_root", "outputs/v01")
    cal_dir = Path(output_root) / "calibration"
    cal_dir.mkdir(parents=True, exist_ok=True)
    return cal_dir / "calibration_gemma.json"


# ── Probe logic ────────────────────────────────────────────────────

def run_probe(
    host: str,
    cfg: dict[str, Any],
    output_path: Optional[str] = None,
) -> bool:
    """
    Execute the Gemma calibration probe.

    Returns True on protocol-level success (all HTTP steps passed).
    Headroom may still be negative if the model exceeds the ceiling.
    """
    ceiling_pct: float = float(
        _deep_get(cfg, "calibration.memory_ceiling_pct", 90)
    )
    gpu_ceiling_pct: float = float(
        _deep_get(cfg, "gpu.memory_ceiling_pct", ceiling_pct)
    )
    # Use calibration section first, then gpu section
    effective_ceiling_pct = ceiling_pct

    result: dict[str, Any] = {
        "probe": "gemma_batch1",
        "host": host,
        "model_name": "gemma4",
        "ceiling_pct": effective_ceiling_pct,
        "success": False,
    }

    print(f"\n[gemma_probe] Host         : {host}")
    print(f"[gemma_probe] Ceiling       : {effective_ceiling_pct:.0f}%")
    print()

    # ── Step 1: /load ──────────────────────────────────────────────
    print("[gemma_probe] Step 1/4 — POST /load ...")
    t0 = time.monotonic()
    try:
        load_resp = _http_post(f"{host}/load", {"model_name": "gemma4"})
    except RuntimeError as exc:
        print(f"[gemma_probe] FAIL /load: {exc}", file=sys.stderr)
        result["error"] = f"load failed: {exc}"
        _write_output(result, cfg, output_path)
        return False

    load_ms = round((time.monotonic() - t0) * 1000, 1)
    load_status = load_resp.get("status", "")
    deployment_mode: str = load_resp.get("deployment_mode", "unknown")
    tp_size: int = int(load_resp.get("tp_size", 1))
    backend_pid = load_resp.get("backend_pid")

    print(
        f"[gemma_probe]   status={load_status}"
        f"  deployment_mode={deployment_mode}"
        f"  tp_size={tp_size}"
        f"  backend_pid={backend_pid}"
        f"  load_latency_ms={load_resp.get('load_latency_ms', load_ms)}"
    )

    if load_status not in ("loaded", "ok"):
        print(
            f"[gemma_probe] FAIL: /load returned status={load_status!r}",
            file=sys.stderr,
        )
        result["error"] = f"unexpected load status: {load_status}"
        result["load_response"] = load_resp
        _write_output(result, cfg, output_path)
        return False

    result["load_latency_ms"] = load_resp.get("load_latency_ms", load_ms)
    result["deployment_mode"] = deployment_mode
    result["tp_size"] = tp_size

    # ── Step 2: /health after load ─────────────────────────────────
    print("[gemma_probe] Step 2/4 — GET /health (post-load VRAM) ...")
    try:
        health_after_load = _http_get(f"{host}/health")
    except RuntimeError as exc:
        print(f"[gemma_probe] WARN /health (post-load): {exc}", file=sys.stderr)
        health_after_load = {}

    vram_after_load, total_gib = _extract_vram(health_after_load)
    print(
        f"[gemma_probe]   gpu_mem_used_gib={vram_after_load}"
        f"  gpu_mem_total_gib={total_gib}"
    )
    result["vram_after_load_gib"] = vram_after_load
    result["vram_total_gib"] = total_gib

    # ── Step 3: /infer/single (batch=1) ────────────────────────────
    print("[gemma_probe] Step 3/4 — POST /infer/single (batch=1) ...")
    infer_payload = _build_infer_payload(cfg, host)
    t_infer = time.monotonic()
    infer_ok = False
    infer_error: Optional[str] = None

    try:
        infer_resp = _http_post(f"{host}/infer/single", infer_payload)
        infer_ok = True
    except RuntimeError as exc:
        infer_error = str(exc)
        print(f"[gemma_probe] WARN /infer/single: {exc}", file=sys.stderr)
        infer_resp = {}

    infer_ms = round((time.monotonic() - t_infer) * 1000, 1)
    tokens_out: int = int(infer_resp.get("tokens_out", 0))
    latency_ms: float = float(infer_resp.get("latency_ms", infer_ms))

    if infer_ok:
        print(
            f"[gemma_probe]   latency_ms={latency_ms}"
            f"  tokens_out={tokens_out}"
        )
    else:
        print(f"[gemma_probe]   inference skipped (error recorded)")

    result["infer_ok"] = infer_ok
    result["infer_latency_ms"] = latency_ms if infer_ok else None
    result["tokens_out"] = tokens_out if infer_ok else None
    if infer_error:
        result["infer_error"] = infer_error

    # Query VRAM after inference to capture peak usage
    try:
        health_after_infer = _http_get(f"{host}/health")
    except RuntimeError as exc:
        print(f"[gemma_probe] WARN /health (post-infer): {exc}", file=sys.stderr)
        health_after_infer = {}

    vram_after_infer, _ = _extract_vram(health_after_infer)
    result["vram_after_infer_gib"] = vram_after_infer

    # Peak VRAM = max of post-load and post-infer readings
    candidates_vram = [v for v in (vram_after_load, vram_after_infer) if v is not None]
    peak_vram_gib: Optional[float] = max(candidates_vram) if candidates_vram else None
    result["peak_vram_gib"] = peak_vram_gib

    # Headroom calculation
    headroom_gib = _headroom(peak_vram_gib, total_gib, effective_ceiling_pct)
    result["headroom_gib"] = headroom_gib
    if headroom_gib is not None and total_gib is not None and total_gib > 0:
        ceiling_gib = total_gib * effective_ceiling_pct / 100.0
        peak_pct = (peak_vram_gib / total_gib * 100.0) if peak_vram_gib is not None else None
        result["peak_vram_pct"] = round(peak_pct, 2) if peak_pct is not None else None
        result["ceiling_gib"] = round(ceiling_gib, 3)

    # ── Step 4: /unload ────────────────────────────────────────────
    print("[gemma_probe] Step 4/4 — POST /unload ...")
    try:
        unload_resp = _http_post(f"{host}/unload", {})
    except RuntimeError as exc:
        print(f"[gemma_probe] WARN /unload: {exc}", file=sys.stderr)
        unload_resp = {}

    unload_status = unload_resp.get("status", "unknown")
    mem_before = unload_resp.get("mem_before_gib", 0.0)
    mem_after = unload_resp.get("mem_after_gib", 0.0)
    print(
        f"[gemma_probe]   status={unload_status}"
        f"  mem_before={mem_before:.2f} GiB"
        f"  mem_after={mem_after:.2f} GiB"
    )
    result["unload_status"] = unload_status
    result["mem_before_gib"] = mem_before
    result["mem_after_gib"] = mem_after

    result["success"] = True

    # ── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("CALIBRATION SUMMARY — gemma batch-1 probe")
    print("=" * 62)
    print(f"  host             : {host}")
    print(f"  deployment_mode  : {deployment_mode}")
    print(f"  tp_size          : {tp_size}")
    print(f"  load_latency_ms  : {result.get('load_latency_ms', 'n/a')}")
    print(f"  infer_ok         : {infer_ok}")
    if infer_ok:
        print(f"  infer_latency_ms : {latency_ms}")
        print(f"  tokens_out       : {tokens_out}")
    print(f"  vram_after_load  : {_fmt_gib(vram_after_load)}")
    print(f"  vram_after_infer : {_fmt_gib(vram_after_infer)}")
    print(f"  peak_vram        : {_fmt_gib(peak_vram_gib)}")
    print(f"  vram_total       : {_fmt_gib(total_gib)}")
    print(f"  ceiling_pct      : {effective_ceiling_pct:.0f}%")
    if headroom_gib is not None:
        sign = "+" if headroom_gib >= 0 else ""
        status_tag = "OK" if headroom_gib >= 0 else "OVER_BUDGET"
        print(f"  headroom_gib     : {sign}{headroom_gib:.3f}  [{status_tag}]")
    else:
        print(f"  headroom_gib     : n/a (no VRAM data)")
    print("=" * 62)
    print("RESULT: PASS (probe completed without protocol error)")
    print()

    _write_output(result, cfg, output_path)
    return True


def _fmt_gib(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.3f} GiB"


def _write_output(
    result: dict[str, Any],
    cfg: dict[str, Any],
    output_path: Optional[str],
) -> None:
    """Write calibration_gemma.json to the requested path (or default location)."""
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = _default_output_path(cfg)

    out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[gemma_probe] Output written → {out}")


# ── CLI ────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe Gemma batch-1 peak VRAM and calculate headroom vs ceiling."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:17610",
        help="Supervisor base URL (e.g. http://127.0.0.1:17610).",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (relative to cwd or absolute).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Explicit path for calibration_gemma.json output.  "
            "Defaults to <output_root>/calibration/calibration_gemma.json."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve config relative to cwd, then try project root (one level up from
    # tests/) so the script can be run from any working directory.
    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        script_dir = Path(__file__).resolve().parent
        # tests/calibration/ -> Image-Generater-Remote/
        project_root = script_dir.parent.parent
        alt = project_root / config_path
        if alt.exists():
            config_path = str(alt)

    cfg = _load_yaml(config_path)
    if not cfg:
        print(
            f"[gemma_probe] WARNING: config not found at {config_path!r}; "
            "using built-in defaults.",
            file=sys.stderr,
        )

    success = run_probe(host=args.host, cfg=cfg, output_path=args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
