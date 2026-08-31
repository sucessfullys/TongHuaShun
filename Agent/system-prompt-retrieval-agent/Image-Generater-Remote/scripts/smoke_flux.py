"""Smoke test for FLUX.2-klein-9B on one supervisor host.

Covers S08.02 and S08.03:
  S08.02 — Load flux2_klein, run one /infer/single generation, save PNG,
            record negative_prompt_applied in the response.
  S08.03 — Second run with negative_prompt="None"; verify that the response
            shows negative_prompt_applied=false OR negative_prompt_normalized=null
            (i.e. Python None reached the pipeline, not the string "None").

Usage:
  python scripts/smoke_flux.py [--host URL] [--config PATH]

Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import Any, Optional


# ── Minimal PNG builder (no Pillow needed on the local smoke runner) ──────────

def _make_minimal_png(width: int = 8, height: int = 8) -> bytes:
    """Return a valid grayscale PNG with a solid grey fill.

    Built from raw bytes so this script has zero third-party dependencies
    when run purely as a smoke check against a remote host.
    """
    def _chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    # IHDR: width, height, bit-depth=8, colour=0 (greyscale), comp=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # Raw pixel data: one filter-byte (0) + <width> grey pixels per row
    raw_rows = b""
    for _ in range(height):
        row = bytes([0]) + bytes([128] * width)  # filter=None, grey=128
        raw_rows += row
    compressed = zlib.compress(raw_rows)
    idat = _chunk(b"IDAT", compressed)

    iend = _chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + ihdr + idat + iend


def _write_temp_png(path: Path, width: int = 8, height: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_make_minimal_png(width, height))


# ── HTTP helpers (stdlib only) ────────────────────────────────────────────────

def _post(url: str, body: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    """POST JSON body to *url*, return parsed JSON response.

    Uses urllib (stdlib) so the script is self-contained.
    """
    import urllib.error
    import urllib.request

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


def _get(url: str, timeout: float = 30.0) -> dict[str, Any]:
    import urllib.request

    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


# ── Smoke helpers ─────────────────────────────────────────────────────────────

def _check_health(host: str) -> None:
    print(f"  GET {host}/health ...", end=" ", flush=True)
    resp = _get(f"{host}/health")
    print(f"status={resp.get('status')} loaded_model={resp.get('loaded_model')}")


def _do_load(host: str) -> dict[str, Any]:
    print(f"  POST {host}/load model_name=flux2_klein ...", end=" ", flush=True)
    resp = _post(f"{host}/load", {"model_name": "flux2_klein"})
    status = resp.get("status")
    ok_flag = resp.get("ok", True)
    if status != "loaded" or ok_flag is False:
        raise RuntimeError(f"/load failed: {json.dumps(resp)}")
    latency = resp.get("load_latency_ms", 0)
    print(f"loaded  latency_ms={latency:.0f}")
    return resp


def _do_unload(host: str) -> dict[str, Any]:
    print(f"  POST {host}/unload ...", end=" ", flush=True)
    resp = _post(f"{host}/unload", {})
    status = resp.get("status")
    if status not in ("unloaded", "no_model_loaded"):
        raise RuntimeError(f"/unload failed: {json.dumps(resp)}")
    mem_before = resp.get("mem_before_gib", 0)
    mem_after = resp.get("mem_after_gib", 0)
    delta = mem_before - mem_after
    print(f"status={status}  mem_before={mem_before:.2f} GiB  mem_after={mem_after:.2f} GiB  delta={delta:.2f} GiB")
    return resp


def _do_infer_single(
    host: str,
    model_image_path: str,
    cloth_image_path: str,
    output_dir: Path,
    tag: str,
    negative_prompt: Optional[str],
    intermediate_prompt: str = "a person wearing a stylish garment, photorealistic",
    run_id: str = "smoke_flux_s08",
) -> dict[str, Any]:
    """POST /infer/single for FLUX and return the response dict."""
    output_relpath = f"smoke_flux/{tag}/tryon_result.png"
    body: dict[str, Any] = {
        "run_id": run_id,
        "stage_id": "flux",
        "cell_id": f"smoke_flux__{tag}",
        "shard_id": 0,
        "sample_id": f"smoke/{tag}",
        "model_name": "flux2_klein",
        "intermediate_prompt": intermediate_prompt,
        "model_image_path": str(model_image_path),
        "cloth_image_path": str(cloth_image_path),
        "output_root_relpath": f"smoke_flux/{tag}",
        "output_relpath": output_relpath,
    }
    if negative_prompt is not None:
        body["negative_prompt"] = negative_prompt
    # omit negative_prompt key entirely when None (tests "missing key" branch)

    neg_display = repr(negative_prompt) if negative_prompt is not None else "<key omitted>"
    print(f"  POST {host}/infer/single  tag={tag}  negative_prompt={neg_display} ...",
          end=" ", flush=True)

    t0 = time.monotonic()
    resp = _post(f"{host}/infer/single", body, timeout=600.0)
    elapsed = (time.monotonic() - t0) * 1000

    if "output_path" not in resp and resp.get("ok") is False:
        raise RuntimeError(f"/infer/single failed: {json.dumps(resp)}")

    neg_applied = resp.get("negative_prompt_applied")
    neg_norm = resp.get("negative_prompt_normalized")
    seed_used = resp.get("seed_used")
    print(
        f"ok  latency_ms={resp.get('latency_ms', elapsed):.0f}  "
        f"seed={seed_used}  "
        f"negative_prompt_applied={neg_applied}  "
        f"negative_prompt_normalized={neg_norm!r}"
    )

    # Save a copy of the response alongside output for inspection
    local_resp_path = output_dir / f"{tag}_response.json"
    local_resp_path.parent.mkdir(parents=True, exist_ok=True)
    local_resp_path.write_text(json.dumps(resp, indent=2, default=str))
    return resp


# ── Validation helpers ────────────────────────────────────────────────────────

def _validate_s08_02(resp: dict[str, Any]) -> None:
    """S08.02: generation ran; negative_prompt_applied is present and a bool."""
    if "output_path" not in resp:
        raise AssertionError("S08.02 FAIL: response missing 'output_path'")
    if "negative_prompt_applied" not in resp:
        raise AssertionError("S08.02 FAIL: response missing 'negative_prompt_applied'")
    if not isinstance(resp["negative_prompt_applied"], bool):
        raise AssertionError(
            f"S08.02 FAIL: 'negative_prompt_applied' is {type(resp['negative_prompt_applied'])}, expected bool"
        )
    if "seed_used" not in resp:
        raise AssertionError("S08.02 FAIL: response missing 'seed_used'")
    print("  [S08.02 PASS] PNG path present, negative_prompt_applied is bool, seed_used present")


def _validate_s08_03(resp: dict[str, Any]) -> None:
    """S08.03: negative_prompt='None' (string) must NOT reach pipeline as the
    string 'None'.  The normalize_negative_prompt function must have converted
    it to Python None, which is evidenced by:
      - negative_prompt_applied == False, OR
      - negative_prompt_normalized == null (JSON null / Python None)

    Both conditions should be true when the kw is supported but normalized to
    None; if the pipeline lacks the kw, applied==False and normalized may be
    anything.
    """
    applied = resp.get("negative_prompt_applied")
    normalized = resp.get("negative_prompt_normalized")

    # The string "None" must have been normalised to Python None
    if normalized == "None":
        raise AssertionError(
            'S08.03 FAIL: negative_prompt_normalized == "None" (string) — '
            'normalize_negative_prompt did NOT convert it to Python None'
        )

    # After normalisation to None, applied must be False
    if applied is True:
        raise AssertionError(
            "S08.03 FAIL: negative_prompt_applied=True after sending negative_prompt='None' — "
            "the string 'None' was passed to the pipeline instead of Python None"
        )

    # Confirm normalized is actually null
    if normalized is not None:
        raise AssertionError(
            f"S08.03 FAIL: negative_prompt_normalized={normalized!r}, expected null "
            "(string 'None' should normalize to Python None)"
        )

    print(
        f"  [S08.03 PASS] negative_prompt='None' → applied={applied}, "
        f"normalized={normalized!r} (Python None reached pipeline)"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for FLUX.2-klein-9B supervisor (S08.02 + S08.03)"
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:17610",
        help="Supervisor host URL (default: http://127.0.0.1:17610)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Config file path (default: config.yaml) — used for output_root resolution",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local directory for saving smoke artefacts (default: outputs/smoke_flux)",
    )
    parser.add_argument(
        "--keep-loaded",
        action="store_true",
        help="Skip /unload at the end (useful for chaining smokes)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    host: str = args.host.rstrip("/")

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "smoke_flux"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create minimal synthetic input PNGs for the smoke test.
    # In a real run these would be actual try-on images; for a smoke test
    # any valid PNG is sufficient to exercise the pipeline API path.
    tmp_model_img = output_dir / "smoke_model_image.png"
    tmp_cloth_img = output_dir / "smoke_cloth_image.png"
    _write_temp_png(tmp_model_img)
    _write_temp_png(tmp_cloth_img)

    summary: list[str] = []
    failed = False

    try:
        # ── Pre-flight health check ────────────────────────────────────────
        print("\n=== FLUX Smoke Test ===")
        print(f"Host: {host}")
        print(f"Config: {args.config}")
        print()
        print("[ Health check ]")
        _check_health(host)

        # ── Load ──────────────────────────────────────────────────────────
        print("\n[ Load flux2_klein ]")
        load_resp = _do_load(host)
        summary.append(f"load_latency_ms: {load_resp.get('load_latency_ms', 0):.0f}")

        # ── S08.02: First run — positive negative_prompt ───────────────────
        print("\n[ S08.02 — infer/single with negative_prompt='blurry, low quality' ]")
        resp_02 = _do_infer_single(
            host=host,
            model_image_path=str(tmp_model_img),
            cloth_image_path=str(tmp_cloth_img),
            output_dir=output_dir,
            tag="run1_neg_real",
            negative_prompt="blurry, low quality",
            run_id="smoke_flux_s08",
        )
        _validate_s08_02(resp_02)
        summary.append(
            f"S08.02 run1: negative_prompt_applied={resp_02.get('negative_prompt_applied')}, "
            f"seed={resp_02.get('seed_used')}"
        )

        # ── S08.03: Second run — negative_prompt="None" (string) ──────────
        print('\n[ S08.03 — infer/single with negative_prompt="None" (string) ]')
        resp_03 = _do_infer_single(
            host=host,
            model_image_path=str(tmp_model_img),
            cloth_image_path=str(tmp_cloth_img),
            output_dir=output_dir,
            tag="run2_neg_none_str",
            negative_prompt="None",  # the string "None", not Python None
            run_id="smoke_flux_s08",
        )
        _validate_s08_03(resp_03)
        summary.append(
            f"S08.03 run2 (neg='None'): negative_prompt_applied={resp_03.get('negative_prompt_applied')}, "
            f"negative_prompt_normalized={resp_03.get('negative_prompt_normalized')!r}"
        )

        # ── Unload ─────────────────────────────────────────────────────────
        if not args.keep_loaded:
            print("\n[ Unload ]")
            unload_resp = _do_unload(host)
            mem_delta = unload_resp.get("mem_before_gib", 0) - unload_resp.get("mem_after_gib", 0)
            summary.append(
                f"unload mem_delta: {mem_delta:.2f} GiB  "
                f"(before={unload_resp.get('mem_before_gib', 0):.2f}, "
                f"after={unload_resp.get('mem_after_gib', 0):.2f})"
            )

    except Exception as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        failed = True

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    for line in summary:
        print(f"  {line}")
    if failed:
        print("\nResult: FAIL")
        return 1
    print("\nResult: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
