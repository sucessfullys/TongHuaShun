"""
smoke_gemma.py — Smoke test for the Gemma-4 supervisor on one host.

Usage (on remote server):
    python scripts/smoke_gemma.py --host http://127.0.0.1:17610 --config config.yaml

Exit codes:
    0 — all steps passed
    1 — any step failed

Steps
-----
1. Load config (for image paths and system prompt).
2. POST /load  model_name="gemma4"
3. POST /infer/single  with two image paths and a system prompt.
4. Save intermediate_prompt.txt to <output_root>/smoke/gemma/.
5. POST /unload; record mem_before/after.
6. Print summary table.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# httpx is the required HTTP client per task spec
try:
    import httpx
except ImportError:
    print("ERROR: httpx is not installed.  pip install httpx", file=sys.stderr)
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is not installed.  pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ── Config helpers ─────────────────────────────────────────────────

def load_config(path: str) -> dict[str, Any]:
    """Load YAML config; fall back gracefully if file is missing."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open() as fh:
        return yaml.safe_load(fh) or {}


def deep_get(cfg: dict, key_path: str, default: Any = None) -> Any:
    """Dot-separated key lookup into a nested dict."""
    parts = key_path.split(".")
    node: Any = cfg
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part, default)
        if node is default:
            return default
    return node


# ── Image path resolution ──────────────────────────────────────────

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
_PAIR_PATTERNS = [
    ("input_model.png", "input_cloth.png"),
    ("model.png", "cloth.png"),
]


def _probe_pair(candidate_root: str) -> tuple[str, str] | None:
    """Return the first valid (model_image, cloth_image) pair under candidate_root."""
    root = Path(candidate_root)
    if not root.is_dir():
        return None
    for category in ("dress", "lower", "upper"):
        cat_dir = root / category
        if not cat_dir.is_dir():
            continue
        for sample_dir in sorted(cat_dir.iterdir()):
            if not sample_dir.is_dir():
                continue
            for model_name, cloth_name in _PAIR_PATTERNS:
                m = sample_dir / model_name
                c = sample_dir / cloth_name
                if m.is_file() and c.is_file():
                    return str(m), str(c)
    return None


def resolve_image_pair(cfg: dict[str, Any]) -> tuple[str, str]:
    """
    Find a usable (model_image, cloth_image) pair.

    Priority:
    1. Walk each candidate in paths.pilot_input_candidates.
    2. Fall back to hardcoded paths if none of the candidates are accessible.
    """
    candidates: list[str] = deep_get(cfg, "paths.pilot_input_candidates", [])
    for candidate in candidates:
        pair = _probe_pair(candidate)
        if pair is not None:
            print(f"[smoke] Resolved image pair from candidate: {candidate}")
            return pair

    # Hardcoded fallback — always present on 3H100 if dataset is mounted
    print(f"[smoke] No config candidates accessible; using hardcoded fallback paths.")
    return _FALLBACK_MODEL_IMAGE, _FALLBACK_CLOTH_IMAGE


# ── Smoke output dir ───────────────────────────────────────────────

def smoke_output_dir(cfg: dict[str, Any]) -> Path:
    """Return (and create) the smoke output directory for Gemma."""
    output_root = deep_get(cfg, "paths.output_root", "outputs/v01")
    smoke_dir = Path(output_root) / "smoke" / "gemma"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    return smoke_dir


# ── HTTP helpers ───────────────────────────────────────────────────

_TIMEOUT = httpx.Timeout(connect=10.0, read=1800.0, write=30.0, pool=5.0)


def post(client: httpx.Client, url: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST JSON body and return parsed response dict.  Raises on HTTP error."""
    resp = client.post(url, json=body, timeout=_TIMEOUT)
    try:
        data = resp.json()
    except Exception:
        data = {"_raw": resp.text}
    if resp.status_code >= 400:
        raise RuntimeError(
            f"HTTP {resp.status_code} from {url}: {json.dumps(data, ensure_ascii=False)}"
        )
    return data


# ── System prompt (minimal stand-in) ──────────────────────────────

_MINIMAL_SYSTEM_PROMPT = (
    "You are a virtual try-on assistant. "
    "Describe in one paragraph how the model would look wearing the garment shown in image 2, "
    "focusing on fit, drape, colour, and any visible design details."
)


def get_system_prompt(cfg: dict[str, Any]) -> str:
    """Return the system prompt to use for the smoke.  Uses config if present."""
    # config.yaml.example does not store system prompt text directly;
    # the agent_sim/prompts.py module holds them.  For smoke purposes
    # we use a minimal stand-in that exercises the two-image VL path.
    return _MINIMAL_SYSTEM_PROMPT


# ── Main smoke logic ───────────────────────────────────────────────

def run_smoke(host: str, cfg: dict[str, Any]) -> bool:
    """
    Execute the Gemma smoke sequence.

    Returns True on success, False on any failure.
    """
    model_image, cloth_image = resolve_image_pair(cfg)
    system_prompt = get_system_prompt(cfg)
    out_dir = smoke_output_dir(cfg)

    print(f"\n[smoke] Host          : {host}")
    print(f"[smoke] model_image   : {model_image}")
    print(f"[smoke] cloth_image   : {cloth_image}")
    print(f"[smoke] output_dir    : {out_dir}")
    print()

    summary: dict[str, Any] = {
        "host": host,
        "model_name": "gemma4",
        "model_image": model_image,
        "cloth_image": cloth_image,
    }

    with httpx.Client() as client:
        # ── Step 1: /load ──────────────────────────────────────────
        print("[smoke] Step 1/3 — POST /load ...")
        t_load_start = time.monotonic()
        try:
            load_resp = post(client, f"{host}/load", {"model_name": "gemma4"})
        except RuntimeError as exc:
            print(f"[smoke] FAIL /load: {exc}", file=sys.stderr)
            return False
        load_ms = (time.monotonic() - t_load_start) * 1000

        load_status = load_resp.get("status", "")
        backend_pid = load_resp.get("backend_pid")
        deployment_mode = load_resp.get("deployment_mode", "unknown")
        print(
            f"[smoke]   status={load_status}  pid={backend_pid}"
            f"  deployment_mode={deployment_mode}"
            f"  load_latency_ms={load_resp.get('load_latency_ms', round(load_ms, 1))}"
        )
        summary["load_status"] = load_status
        summary["load_latency_ms"] = load_resp.get("load_latency_ms", round(load_ms, 1))
        summary["deployment_mode"] = deployment_mode

        if load_status not in ("loaded", "ok"):
            print(f"[smoke] FAIL: /load returned status={load_status!r}", file=sys.stderr)
            return False

        # ── Step 2: /infer/single ──────────────────────────────────
        print("[smoke] Step 2/3 — POST /infer/single ...")
        infer_body = {
            # Correlation IDs (minimal for smoke)
            "run_id": "smoke_gemma_001",
            "stage_id": "gemma",
            "cell_id": "smoke__neg_none",
            "shard_id": 0,
            "sample_id": "smoke/dress/0001",
            "host_port": int(host.rsplit(":", 1)[-1]) if ":" in host else 17610,
            "model_name": "gemma4",
            # Gemma-specific
            "system_prompt": system_prompt,
            "user_instruction": deep_get(
                cfg,
                "prompts.default_user_instruction",
                "the model in image 1 wears the clothing from image 2",
            ),
            "model_image_path": model_image,
            "cloth_image_path": cloth_image,
        }

        t_infer_start = time.monotonic()
        try:
            infer_resp = post(client, f"{host}/infer/single", infer_body)
        except RuntimeError as exc:
            print(f"[smoke] FAIL /infer/single: {exc}", file=sys.stderr)
            # Try to unload before returning failure
            try:
                post(client, f"{host}/unload", {"force": True})
            except Exception:
                pass
            return False
        infer_ms = (time.monotonic() - t_infer_start) * 1000

        intermediate_prompt: str = infer_resp.get("intermediate_prompt", "")
        tokens_out: int = infer_resp.get("tokens_out", 0)
        reported_ms: float = infer_resp.get("latency_ms", round(infer_ms, 1))

        print(
            f"[smoke]   latency_ms={reported_ms}"
            f"  tokens_out={tokens_out}"
            f"  prompt_len={len(intermediate_prompt)}"
        )
        summary["infer_latency_ms"] = reported_ms
        summary["tokens_out"] = tokens_out
        summary["prompt_preview"] = intermediate_prompt[:120].replace("\n", " ")

        if not intermediate_prompt:
            print("[smoke] FAIL: empty intermediate_prompt in response", file=sys.stderr)
            try:
                post(client, f"{host}/unload", {"force": True})
            except Exception:
                pass
            return False

        # Save intermediate_prompt.txt
        prompt_path = out_dir / "intermediate_prompt.txt"
        prompt_path.write_text(intermediate_prompt, encoding="utf-8")
        print(f"[smoke]   Saved intermediate_prompt.txt → {prompt_path}")

        # ── Step 3: /unload ────────────────────────────────────────
        print("[smoke] Step 3/3 — POST /unload ...")
        t_unload_start = time.monotonic()
        try:
            unload_resp = post(client, f"{host}/unload", {})
        except RuntimeError as exc:
            print(f"[smoke] FAIL /unload: {exc}", file=sys.stderr)
            return False
        unload_ms = (time.monotonic() - t_unload_start) * 1000

        mem_before = unload_resp.get("mem_before_gib", 0.0)
        mem_after = unload_resp.get("mem_after_gib", 0.0)
        mem_delta = round(mem_before - mem_after, 2)
        unload_status = unload_resp.get("status", "")
        restarted = unload_resp.get("restarted", False)

        print(
            f"[smoke]   status={unload_status}"
            f"  mem_before={mem_before:.2f} GiB"
            f"  mem_after={mem_after:.2f} GiB"
            f"  delta={mem_delta:+.2f} GiB"
            f"  restarted={restarted}"
        )
        summary["unload_status"] = unload_status
        summary["mem_before_gib"] = mem_before
        summary["mem_after_gib"] = mem_after
        summary["mem_delta_gib"] = mem_delta
        summary["restarted"] = restarted

    # ── Summary ────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("SMOKE SUMMARY — gemma4")
    print("=" * 60)
    print(f"  host             : {summary['host']}")
    print(f"  deployment_mode  : {summary.get('deployment_mode', 'n/a')}")
    print(f"  load_latency_ms  : {summary.get('load_latency_ms', 'n/a')}")
    print(f"  infer_latency_ms : {summary.get('infer_latency_ms', 'n/a')}")
    print(f"  tokens_out       : {summary.get('tokens_out', 'n/a')}")
    print(f"  mem_before_gib   : {summary.get('mem_before_gib', 'n/a')}")
    print(f"  mem_after_gib    : {summary.get('mem_after_gib', 'n/a')}")
    print(f"  mem_delta_gib    : {summary.get('mem_delta_gib', 'n/a')}")
    print(f"  restarted        : {summary.get('restarted', False)}")
    print(f"  prompt_preview   : {summary.get('prompt_preview', '')!r}")
    print("=" * 60)
    print("RESULT: PASS")
    print()

    # Save summary JSON alongside the prompt
    summary_path = out_dir / "smoke_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[smoke] Summary JSON → {summary_path}")

    return True


# ── CLI ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test for Gemma-4 on one supervisor host.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="http://127.0.0.1:17610",
        help="Supervisor base URL.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (relative to cwd or absolute).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve config path relative to script location so the script can be
    # invoked from any directory (common on the remote server).
    config_path = args.config
    if not os.path.isabs(config_path):
        # Try relative to cwd first, then relative to the project root
        # (one level up from scripts/).
        if not os.path.exists(config_path):
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent
            alt_path = project_root / config_path
            if alt_path.exists():
                config_path = str(alt_path)

    cfg = load_config(config_path)
    if not cfg:
        print(
            f"[smoke] WARNING: config not found at {config_path!r}; "
            "proceeding with defaults.",
            file=sys.stderr,
        )

    success = run_smoke(host=args.host, cfg=cfg)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
