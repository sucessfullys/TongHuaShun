"""FLUX batch-size sweep calibration tool.

Sweeps batch sizes 1..max_batch on each GPU supervisor, recording peak VRAM
after each inference batch. Stops a GPU sweep when VRAM usage reaches 90%
of total, or when an OOM error is returned.

Usage:
  python tests/calibration/flux_sweep.py [--host HOST[:PORT[,PORT...]]] \
      [--config PATH] [--max-batch N] [--output PATH]

  --host      Base host URL or comma-separated list of supervisor URLs.
              If omitted, ports are read from config.yaml (ports.host_gpu*).
              Default: http://127.0.0.1:17610
  --config    Path to config.yaml.  Used to read calibration.flux_batch_max
              and ports.host_gpu*.  Default: config.yaml
  --max-batch Override calibration.flux_batch_max from config.
  --output    Output JSON file path. Default: calibration_flux.json

Exit 0 on success (even if some GPUs hit the VRAM ceiling or OOM).
Exit 1 only on a fatal error that prevents any result from being written.

Output schema:
  {
    "gpu_results": [
      {
        "host": "http://...",
        "gpu_index": 0,
        "batches_tested": [
          {
            "batch_size": 1,
            "peak_vram_used_gib": 12.3,
            "peak_vram_total_gib": 80.0,
            "peak_vram_pct": 15.4,
            "stopped_reason": null   // or "oom" / "vram_ceiling"
          },
          ...
        ],
        "max_safe_batch": 4,
        "stopped_at_batch": null     // batch size that triggered stop, or null
      }
    ],
    "safe_batch_size": 4,            // min of max_safe_batch across all GPUs
    "vram_ceiling_pct": 90
  }
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Optional
import urllib.error
import urllib.request


# ── Minimal PNG builder (no Pillow needed) ────────────────────────────────────

def _make_minimal_png(width: int = 64, height: int = 64) -> bytes:
    """Return a valid grayscale PNG with a solid grey fill."""

    def _chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    raw_rows = b""
    for _ in range(height):
        raw_rows += bytes([0]) + bytes([128] * width)
    idat = _chunk(b"IDAT", zlib.compress(raw_rows))
    iend = _chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


# ── HTTP helpers (stdlib only) ────────────────────────────────────────────────

def _post(url: str, body: dict[str, Any], timeout: float = 300.0) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return json.loads(raw)
        except Exception:
            return {"_http_error": exc.code, "_body": raw.decode(errors="replace")}
    except Exception as exc:
        return {"_request_error": str(exc)}


def _get(url: str, timeout: float = 30.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception as exc:
        return {"_request_error": str(exc)}


# ── VRAM probe ────────────────────────────────────────────────────────────────

def _probe_vram(host: str) -> Optional[dict[str, float]]:
    """Query /health for current VRAM usage.  Returns None if unavailable."""
    resp = _get(f"{host}/health")
    if "_request_error" in resp or "_http_error" in resp:
        return None
    used = resp.get("gpu_mem_used_gib")
    total = resp.get("gpu_mem_total_gib")
    if used is None or total is None or total <= 0:
        return None
    return {
        "used_gib": float(used),
        "total_gib": float(total),
        "pct": round(float(used) / float(total) * 100.0, 2),
    }


# ── Load / Unload helpers ─────────────────────────────────────────────────────

def _load_flux(host: str) -> dict[str, Any]:
    """POST /load for flux2_klein.  Returns response dict."""
    return _post(f"{host}/load", {"model_name": "flux2_klein"}, timeout=600.0)


def _unload(host: str) -> dict[str, Any]:
    return _post(f"{host}/unload", {}, timeout=120.0)


# ── Inference helpers ─────────────────────────────────────────────────────────

def _make_infer_item(
    model_image_path: str,
    cloth_image_path: str,
    index: int,
    output_root: str,
) -> dict[str, Any]:
    return {
        "run_id": "calibration_flux_sweep",
        "stage_id": "flux",
        "cell_id": f"calib_{index}",
        "shard_id": 0,
        "sample_id": f"calib/item_{index}",
        "model_name": "flux2_klein",
        "intermediate_prompt": "a person wearing a stylish garment, photorealistic",
        "model_image_path": model_image_path,
        "cloth_image_path": cloth_image_path,
        "output_root_relpath": output_root,
        "output_relpath": f"{output_root}/result_{index}.png",
        "index": index,
    }


def _infer_batch(
    host: str,
    model_image_path: str,
    cloth_image_path: str,
    batch_size: int,
    output_root: str,
) -> dict[str, Any]:
    """POST /infer/list with *batch_size* items.  Returns response dict."""
    items = [
        _make_infer_item(model_image_path, cloth_image_path, i, output_root)
        for i in range(batch_size)
    ]
    body = {
        "run_id": "calibration_flux_sweep",
        "stage_id": "flux",
        "cell_id": "calib_batch",
        "items": items,
    }
    return _post(f"{host}/infer/list", body, timeout=600.0)


# ── OOM detection ─────────────────────────────────────────────────────────────

def _is_oom(resp: dict[str, Any]) -> bool:
    """Return True if the response indicates an out-of-memory error."""
    if resp.get("ok") is False:
        # Top-level batch error
        error_code = resp.get("error", {}).get("error_code", "") if isinstance(resp.get("error"), dict) else ""
        message = resp.get("error", {}).get("message", "") if isinstance(resp.get("error"), dict) else ""
        if "oom" in error_code.lower() or "out of memory" in message.lower():
            return True
        if "_request_error" in resp or "_http_error" in resp:
            return False
    # Check individual batch items
    items = resp.get("items", [])
    if items:
        for item in items:
            err = item.get("error", "") or ""
            if "out of memory" in err.lower() or "oom" in err.lower():
                return True
    return False


def _is_fatal_error(resp: dict[str, Any]) -> bool:
    """Return True if the response is a fatal/unexpected HTTP or connection error."""
    return "_request_error" in resp or "_http_error" in resp


# ── Single-GPU sweep ──────────────────────────────────────────────────────────

def _sweep_one_gpu(
    host: str,
    max_batch: int,
    vram_ceiling_pct: int,
    tmp_dir: Path,
    gpu_index: int,
) -> dict[str, Any]:
    """Run the batch sweep for one GPU supervisor.

    Returns a result dict with keys: host, gpu_index, batches_tested,
    max_safe_batch, stopped_at_batch.
    """
    print(f"\n=== GPU {gpu_index} — {host} ===")

    # Write synthetic PNG inputs
    model_png = tmp_dir / f"calib_model_{gpu_index}.png"
    cloth_png = tmp_dir / f"calib_cloth_{gpu_index}.png"
    png_bytes = _make_minimal_png()
    model_png.write_bytes(png_bytes)
    cloth_png.write_bytes(png_bytes)
    model_image_path = str(model_png)
    cloth_image_path = str(cloth_png)
    output_root = f"calibration_sweep/gpu_{gpu_index}"

    batches_tested: list[dict[str, Any]] = []
    max_safe_batch = 0
    stopped_at_batch: Optional[int] = None

    for batch_size in range(1, max_batch + 1):
        print(f"  batch_size={batch_size} ...", end=" ", flush=True)

        # Load model
        load_resp = _load_flux(host)
        if load_resp.get("status") != "loaded":
            err_msg = load_resp.get("error", {}).get("message", str(load_resp)) if isinstance(load_resp.get("error"), dict) else str(load_resp)
            print(f"LOAD FAILED: {err_msg}")
            # Record stop and break
            batches_tested.append({
                "batch_size": batch_size,
                "peak_vram_used_gib": None,
                "peak_vram_total_gib": None,
                "peak_vram_pct": None,
                "stopped_reason": "load_failed",
                "load_error": err_msg,
            })
            stopped_at_batch = batch_size
            break

        # Probe VRAM before inference
        vram_before = _probe_vram(host)

        # Run inference batch
        infer_resp = _infer_batch(
            host, model_image_path, cloth_image_path, batch_size, output_root
        )

        # Probe VRAM after inference (peak usage during generation)
        vram_after = _probe_vram(host)

        # Unload model regardless of inference result
        _unload(host)

        # Determine peak VRAM (max of before/after; prefer after as it captures allocation)
        peak_vram: Optional[dict[str, float]] = None
        if vram_after is not None:
            peak_vram = vram_after
        elif vram_before is not None:
            peak_vram = vram_before

        # Classify result
        oom = _is_oom(infer_resp)
        fatal = _is_fatal_error(infer_resp)

        if oom:
            print(f"OOM  (stopped)")
            batches_tested.append({
                "batch_size": batch_size,
                "peak_vram_used_gib": peak_vram["used_gib"] if peak_vram else None,
                "peak_vram_total_gib": peak_vram["total_gib"] if peak_vram else None,
                "peak_vram_pct": peak_vram["pct"] if peak_vram else None,
                "stopped_reason": "oom",
            })
            stopped_at_batch = batch_size
            break

        if fatal:
            err_detail = infer_resp.get("_request_error") or str(infer_resp)
            print(f"FATAL_ERROR: {err_detail}  (stopped)")
            batches_tested.append({
                "batch_size": batch_size,
                "peak_vram_used_gib": peak_vram["used_gib"] if peak_vram else None,
                "peak_vram_total_gib": peak_vram["total_gib"] if peak_vram else None,
                "peak_vram_pct": peak_vram["pct"] if peak_vram else None,
                "stopped_reason": "fatal_error",
                "error": err_detail,
            })
            stopped_at_batch = batch_size
            break

        # Check VRAM ceiling
        vram_pct = peak_vram["pct"] if peak_vram else 0.0
        vram_used = peak_vram["used_gib"] if peak_vram else 0.0
        vram_total = peak_vram["total_gib"] if peak_vram else 0.0

        at_ceiling = peak_vram is not None and vram_pct >= vram_ceiling_pct

        print(
            f"ok  vram={vram_used:.1f}/{vram_total:.1f} GiB ({vram_pct:.1f}%)"
            + (" [AT CEILING — stopped]" if at_ceiling else "")
        )

        if at_ceiling:
            batches_tested.append({
                "batch_size": batch_size,
                "peak_vram_used_gib": vram_used,
                "peak_vram_total_gib": vram_total,
                "peak_vram_pct": vram_pct,
                "stopped_reason": "vram_ceiling",
            })
            # This batch_size is unsafe; max safe is batch_size - 1
            max_safe_batch = batch_size - 1
            stopped_at_batch = batch_size
            break

        # Batch succeeded and is below ceiling
        batches_tested.append({
            "batch_size": batch_size,
            "peak_vram_used_gib": vram_used,
            "peak_vram_total_gib": vram_total,
            "peak_vram_pct": vram_pct,
            "stopped_reason": None,
        })
        max_safe_batch = batch_size
    else:
        # Loop completed without hitting any ceiling — max_batch is safe
        pass

    print(f"  GPU {gpu_index} result: max_safe_batch={max_safe_batch}, stopped_at={stopped_at_batch}")

    return {
        "host": host,
        "gpu_index": gpu_index,
        "batches_tested": batches_tested,
        "max_safe_batch": max_safe_batch,
        "stopped_at_batch": stopped_at_batch,
    }


# ── Config loading ────────────────────────────────────────────────────────────

def _load_yaml_simple(path: str) -> dict[str, Any]:
    """Load YAML using yaml if available, else fall back to very basic parsing."""
    try:
        import yaml
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        pass
    # Fallback: stdlib json can't parse YAML, but we can extract scalar values
    # by line-scanning for the specific keys we need.
    result: dict[str, Any] = {}
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return result
    # Very limited: extract only top-level and one-level-deep scalar keys
    current_section: Optional[str] = None
    for raw_line in lines:
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and not line.startswith("\t") and ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val:
                # top-level scalar
                result[key] = _parse_scalar(val)
                current_section = None
            else:
                current_section = key
                result[key] = {}
        elif current_section and ":" in line:
            stripped = line.strip()
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip()
            if isinstance(result.get(current_section), dict):
                result[current_section][key] = _parse_scalar(val)
    return result


def _parse_scalar(val: str) -> Any:
    """Convert a YAML scalar string to an appropriate Python type."""
    if val in ("null", "~", ""):
        return None
    if val in ("true", "True", "yes"):
        return True
    if val in ("false", "False", "no"):
        return False
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val.strip('"\'')


def _deep_get(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    keys = dotted_key.split(".")
    current: Any = cfg
    for k in keys:
        if not isinstance(current, dict) or k not in current:
            return default
        current = current[k]
    return current


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep FLUX batch sizes on each GPU supervisor to find safe batch maximum."
    )
    parser.add_argument(
        "--host",
        default=None,
        help=(
            "Comma-separated list of supervisor URLs "
            "(e.g. http://127.0.0.1:17610,http://127.0.0.1:17611). "
            "If omitted, ports are derived from config ports.host_gpu*."
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml).",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=None,
        help="Override calibration.flux_batch_max from config.",
    )
    parser.add_argument(
        "--output",
        default="calibration_flux.json",
        help="Output JSON file path (default: calibration_flux.json).",
    )
    return parser.parse_args()


# ── Host resolution ───────────────────────────────────────────────────────────

def _resolve_hosts(args: argparse.Namespace, cfg: dict[str, Any]) -> list[str]:
    """Return list of supervisor URLs to sweep."""
    if args.host:
        return [h.strip().rstrip("/") for h in args.host.split(",") if h.strip()]

    # Derive from config ports section
    ports_section = cfg.get("ports", {})
    hosts: list[str] = []
    server_host = _deep_get(cfg, "server.host", "127.0.0.1")

    # Collect host_gpu0, host_gpu1, host_gpu2, ...
    i = 0
    while True:
        key = f"host_gpu{i}"
        port = ports_section.get(key)
        if port is None:
            break
        hosts.append(f"http://{server_host}:{int(port)}")
        i += 1

    if not hosts:
        # Default fallback
        hosts = ["http://127.0.0.1:17610"]

    return hosts


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()

    # Load config (best-effort; failures degrade gracefully)
    cfg: dict[str, Any] = {}
    if os.path.exists(args.config):
        try:
            cfg = _load_yaml_simple(args.config)
        except Exception as exc:
            print(f"[warn] Could not parse {args.config}: {exc}", file=sys.stderr)

    # Resolve max_batch
    if args.max_batch is not None:
        max_batch = args.max_batch
    else:
        max_batch = int(_deep_get(cfg, "calibration.flux_batch_max", 16))

    # Resolve VRAM ceiling
    vram_ceiling_pct = int(_deep_get(cfg, "calibration.memory_ceiling_pct", 90))
    if vram_ceiling_pct <= 0 or vram_ceiling_pct > 100:
        vram_ceiling_pct = 90

    # Resolve hosts
    hosts = _resolve_hosts(args, cfg)

    print("=== FLUX Batch Sweep Calibration ===")
    print(f"Config: {args.config}")
    print(f"Hosts: {hosts}")
    print(f"max_batch: {max_batch}")
    print(f"VRAM ceiling: {vram_ceiling_pct}%")
    print(f"Output: {args.output}")

    # Temporary directory for synthetic PNG inputs
    tmp_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "flux_calib"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    gpu_results: list[dict[str, Any]] = []

    for gpu_index, host in enumerate(hosts):
        try:
            result = _sweep_one_gpu(
                host=host,
                max_batch=max_batch,
                vram_ceiling_pct=vram_ceiling_pct,
                tmp_dir=tmp_dir,
                gpu_index=gpu_index,
            )
        except Exception as exc:
            print(f"[error] GPU {gpu_index} ({host}) sweep failed with exception: {exc}", file=sys.stderr)
            result = {
                "host": host,
                "gpu_index": gpu_index,
                "batches_tested": [],
                "max_safe_batch": 0,
                "stopped_at_batch": None,
                "sweep_error": str(exc),
            }
        gpu_results.append(result)

    # Compute safe_batch_size: min of max_safe_batch across GPUs that produced results
    valid_results = [r for r in gpu_results if r.get("max_safe_batch", 0) > 0]
    if valid_results:
        safe_batch_size = min(r["max_safe_batch"] for r in valid_results)
    else:
        # No GPU produced a usable result — conservative fallback
        safe_batch_size = 1

    output_data: dict[str, Any] = {
        "gpu_results": gpu_results,
        "safe_batch_size": safe_batch_size,
        "vram_ceiling_pct": vram_ceiling_pct,
        "max_batch_tested": max_batch,
        "hosts": hosts,
        "config_path": args.config,
    }

    # Write output JSON
    output_path = Path(args.output)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_data, indent=2, default=str))
        print(f"\nCalibration result written to: {output_path}")
    except OSError as exc:
        print(f"[error] Could not write {output_path}: {exc}", file=sys.stderr)
        return 1

    print(f"\nSafe batch size across all GPUs: {safe_batch_size}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
