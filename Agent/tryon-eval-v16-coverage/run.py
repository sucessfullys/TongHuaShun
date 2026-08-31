#!/usr/bin/env python3
"""Standalone virtual try-on evaluator — v16-coverage (single entrypoint).

Ships ONE proven ERA tryon-eval evaluator as a fully self-contained app with NO
dependency on the evaluation-retrieval-agent repo:

    combination_id: hybrid-editscope-zoomgate-122b-v16-coverage-1read
    scorer:         score_v16_coverage  (byte-frozen iter_038 recipe)
    judge:          Qwen3.5-122B-A10B (vLLM, tp4, served locally or attached)

Full-dataset vs operator marks: 90.9% agreement, precision 0.818, recall 0.847,
accuracy 0.908.

From one command it:
  1. auto-creates its OWN virtualenv at <pkg>/.venv and re-execs inside it;
  2. serves (or attaches to) the Qwen3.5-122B-A10B judge;
  3. scores every (sample, method) in the configured datasets (ces_old + whq_new);
  4. writes PASS / NOT PASS + reason to REPORT.md + verdicts.jsonl (+ raw
     scores.jsonl), and optionally opens the review web app.

Examples::

    bash setup.sh                              # create <pkg>/.venv (one time)
    python3 run.py --detect-only               # dataset counts, no GPU
    python3 run.py                             # serve judge + score everything
    python3 run.py --dataset ces_old --limit 3 \
        --endpoint http://127.0.0.1:8011/v1 --served Qwen3.5-122B-A10B --no-webapp
    python3 run.py --review-only --run runs/<ts>
"""
from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
VENV = PKG / ".venv"
RUNS = PKG / "runs"
SCORER_DIR = PKG / "scorer"
CONFIG_YAML = PKG / "config.yaml"

# ---- this package's pinned method -----------------------------------------
COMBINATION_ID = "hybrid-editscope-zoomgate-122b-v16-coverage-1read"
METHOD_TITLE = ("v16-coverage — one consolidated coverage read on the "
                "reference-anchored edit-scope spine")
ACCURACY_LINE = ("Full-dataset vs operator marks: 90.9% agreement, "
                 "precision 0.818, recall 0.847, accuracy 0.908.")
SCOPE_LABEL = ("Pointwise atomic Y/N defect-flag, single consolidated coverage "
               "read across the full spec surface (cloth preservation, "
               "edit-scope / non-target region, model preservation, quality).")
# scorer modules to import (priority order for SPEC resolution)
SCORER_MODULE_NAMES = ["_v16"]
DEFAULT_PORT = 8741
def _dataset_ids_from_config():
    try:
        import yaml
        _c = yaml.safe_load(open(CONFIG_YAML))
        _ids = [d["dataset_id"] for d in (_c.get("data", {}).get("datasets") or [])]
        return _ids or ["ces_old", "whq_new"]
    except Exception:
        return ["ces_old", "whq_new"]
DATASET_IDS = _dataset_ids_from_config()

# runtime flag toggled by --light (read by init_metrics, which the harness calls)
LIGHT = False


# ---------------------------------------------------------------------------
# bootstrap (stdlib only — runs before any third-party import)
# ---------------------------------------------------------------------------
def _req_hash(light: bool) -> str:
    h = hashlib.sha256()
    files = [PKG / "requirements.txt"]
    if not light:
        files.append(PKG / "requirements-metrics.txt")
    for f in files:
        h.update(f.read_bytes() if f.is_file() else b"")
    h.update(b"light" if light else b"full")
    return h.hexdigest()[:16]


def bootstrap(light: bool) -> None:
    """Create <pkg>/.venv (--system-site-packages) and re-exec inside it."""
    venv_python = VENV / "bin" / "python3"
    in_venv = Path(sys.prefix).resolve() == VENV.resolve()
    sentinel = VENV / ".bootstrapped"
    want = _req_hash(light)
    fresh = not (sentinel.is_file() and want in sentinel.read_text())

    if in_venv and not fresh:
        return
    if not venv_python.is_file():
        print(f"creating virtualenv at {VENV} ...")
        subprocess.run([sys.executable, "-m", "venv", "--system-site-packages",
                        str(VENV)], check=True)
    if fresh:
        reqs = [PKG / "requirements.txt"]
        if not light:
            reqs.append(PKG / "requirements-metrics.txt")
        for req in reqs:
            print(f"pip install -r {req.name} ...")
            r = subprocess.run([str(venv_python), "-m", "pip", "install", "-q",
                                "-r", str(req)])
            if r.returncode != 0:
                if "metrics" in req.name:
                    print(f"! installing {req.name} failed — continuing; metrics "
                          f"will degrade (report-only in this recipe). "
                          f"Re-run with --light to silence.")
                    continue
                raise SystemExit(f"pip install -r {req.name} failed")
        sentinel.write_text(want + "\n")
    if not in_venv:
        os.execv(str(venv_python),
                 [str(venv_python), str(PKG / "run.py"), *sys.argv[1:]])


# ---------------------------------------------------------------------------
# metrics hook the harness calls (CUDA_VISIBLE_DEVICES='' -> CPU only)
# ---------------------------------------------------------------------------
def init_metrics() -> None:
    if LIGHT:
        return
    import _ntmetric as NT
    NT.init_metrics()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=f"Standalone try-on evaluator ({COMBINATION_ID}, "
                    f"Qwen3.5-122B-A10B judge) — PASS/NOT PASS output.")
    p.add_argument("--dataset", choices=DATASET_IDS + ["all"], default="all",
                   help="which configured dataset to score (default: all)")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N samples PER METHOD (smoke)")
    p.add_argument("--endpoint", help="attach to a running OpenAI-compatible "
                   "judge (http://HOST:PORT[/v1]) instead of launching vLLM")
    p.add_argument("--served", help="model name at --endpoint "
                   "(default: the first model the endpoint serves)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"vLLM serve port when launching (default {DEFAULT_PORT})")
    p.add_argument("--model-path", help="judge weights dir (default: auto-scan)")
    p.add_argument("--light", action="store_true",
                   help="skip torch/timm metric deps — pure-VLM judging "
                   "(metrics are report-only in this recipe, so verdicts hold)")
    p.add_argument("--no-webapp", action="store_true",
                   help="stop after REPORT.md/verdicts.jsonl (no review server)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--run", help="existing runs/<ts> dir to review")
    p.add_argument("--review-only", action="store_true",
                   help="serve the web app for a finished run (needs --run)")
    p.add_argument("--detect-only", action="store_true",
                   help="print per-dataset sample counts and exit (no GPU)")
    p.add_argument("--bootstrap-only", action="store_true",
                   help="create the venv + install deps, then exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def resolve_run_dir(run_arg) -> Path:
    if run_arg:
        run_dir = Path(run_arg)
        if not run_dir.is_absolute():
            run_dir = (PKG / run_dir).resolve()
        if not run_dir.is_dir():
            raise SystemExit(f"--run dir does not exist: {run_dir}")
        return run_dir
    run_dir = RUNS / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_datasets() -> dict:
    import yaml
    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    return {d["dataset_id"]: d
            for d in (cfg.get("data") or {}).get("datasets") or []}


def detect_report(dataset_ids) -> None:
    dsmap = load_datasets()
    print(f"config: {CONFIG_YAML}")
    for did in dataset_ids:
        d = dsmap.get(did)
        if not d:
            print(f"  {did}: (not in config.yaml)")
            continue
        glob = d.get("sample_glob", "*")
        print(f"  dataset {did}  (sample_glob {glob!r})")
        for m in d.get("methods", []):
            base = Path(m["path"])
            n = sum(1 for p in base.glob(glob) if p.is_dir()) if base.is_dir() else 0
            flag = "" if base.is_dir() else "  [PATH MISSING]"
            print(f"    - {m['method_id']:<24} n={n:<5} {base}{flag}")


def print_tunnel_help(host: str, port: int) -> None:
    import getpass
    import socket
    user = getpass.getuser()
    hostname = socket.gethostname()
    print("\n" + "=" * 72)
    print(f"  Review web app:  http://127.0.0.1:{port}/")
    print(f"  From your laptop, open the SSH tunnel first:")
    print(f"    ssh -N -L {port}:127.0.0.1:{port} {user}@{hostname}")
    print(f"  Then browse http://127.0.0.1:{port}/ — Ctrl-C here stops it.")
    print("=" * 72 + "\n")


def serve_webapp(run_dir: Path, host: str, port: int) -> None:
    sys.path.insert(0, str(PKG))
    from webapp.server import serve as serve_web
    print_tunnel_help(host, port)
    serve_web(run_dir, host=host, port=port)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score_dataset(C, spec, dataset_id: str, run_dir: Path, base_url: str,
                  served: str, limit) -> None:
    """Set the __main__ globals the harness reads, then run one dataset."""
    g = globals()
    g["TASK_ID"] = f"score-{COMBINATION_ID}-{dataset_id}"
    g["COMBINATION_ID"] = COMBINATION_ID
    g["MODE"] = "full"
    g["DATASET_ID"] = dataset_id
    g["JUDGE_TASK_ID"] = None
    g["NEEDS_JUDGE"] = True
    g["SAMPLES_SUBSET"] = None
    g["SAMPLES_PER_METHOD"] = limit          # None => all samples (glob)
    g["SCOPE_LABEL"] = SCOPE_LABEL
    g["EXPECTED_OUTPUT"] = str(run_dir / "results" / "full" / COMBINATION_ID
                               / f"scores.{dataset_id}.jsonl")
    # endpoint -> the harness scans sys.argv for --endpoint/--served
    sys.argv = [sys.argv[0], "--endpoint", base_url, "--served", served]
    print(f"\n=== scoring dataset {dataset_id} -> {g['EXPECTED_OUTPUT']}")
    C.run(score_one=spec["fn"], needs_metrics=spec["needs_metrics"])


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    args = parse_args()
    bootstrap(args.light)
    if args.bootstrap_only:
        print(f"venv ready: {VENV}")
        return

    global LIGHT
    LIGHT = bool(args.light)

    # metrics (DINOv2 / color-EMD) run in-process on CPU — the judge is remote.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TRYON_CONFIG_YAML"] = str(CONFIG_YAML)

    dataset_ids = DATASET_IDS if args.dataset == "all" else [args.dataset]

    if args.detect_only:
        detect_report(dataset_ids)
        return

    # ---- review-only fast path ----------------------------------------
    if args.review_only:
        run_dir = resolve_run_dir(args.run)
        if not (run_dir / "human" / "review_model.json").is_file():
            raise SystemExit(f"--review-only needs a finished run; "
                             f"{run_dir}/human/review_model.json is missing")
        serve_webapp(run_dir, args.host, args.port)
        return

    run_dir = resolve_run_dir(args.run)
    os.environ["TRYON_LOGS_DIR"] = str(run_dir / "logs")
    print(f"method:  {COMBINATION_ID}")
    print(f"run dir: {run_dir}")

    # ---- import the scoring closure -----------------------------------
    sys.path.insert(0, str(SCORER_DIR))
    import era_eval_common as C   # noqa: E402
    scorer_mods = []
    for name in SCORER_MODULE_NAMES:
        try:
            scorer_mods.append(__import__(name))
        except Exception as e:
            print(f"! could not import scorer module {name}: {e}")
    # resolve the SPEC for this package's combination_id
    spec = None
    for mod in scorer_mods:
        if COMBINATION_ID in getattr(mod, "SPEC", {}):
            spec = mod.get(COMBINATION_ID)
            break
    if spec is None:
        raise SystemExit(f"no scorer SPEC for {COMBINATION_ID} in "
                         f"{SCORER_MODULE_NAMES}")
    for mod in scorer_mods:
        if hasattr(mod, "set_mode"):
            try:
                mod.set_mode("full")
            except Exception as e:
                print(f"! set_mode failed on {mod.__name__}: {e}")

    # ---- resolve the judge endpoint -----------------------------------
    sys.path.insert(0, str(SCORER_DIR))
    import serve_judge  # noqa: E402
    judge_proc = None
    attached = bool(args.endpoint)
    if attached:
        base_url, served = serve_judge.probe_endpoint(args.endpoint, args.served)
        print(f"attached to judge at {base_url} (model {served})")
    else:
        sys.path.insert(0, str(PKG))
        from evaluator import gpu
        if not gpu.kill_watchdog():
            raise SystemExit("aborting: the GPU watchdog could not be killed "
                             "(it breaks vLLM's free-memory check)")
        gpu_ids = gpu.pick_gpus(serve_judge.TENSOR_PARALLEL)
        ep = serve_judge.launch(gpu_ids, args.port, model_path=args.model_path,
                                log_file=run_dir / "vllm.log")
        judge_proc = ep["proc"]
        base_url, served = ep["base_url"], ep["served_model_name"]
        (run_dir / "endpoint.json").write_text(
            json.dumps({k: v for k, v in ep.items() if k != "proc"}, indent=2))

        def _teardown():
            if judge_proc is not None and judge_proc.poll() is None:
                serve_judge.shutdown_judge(judge_proc, served)
                gpu.restart_watchdog()
        atexit.register(_teardown)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    # ---- score every requested dataset --------------------------------
    for did in dataset_ids:
        score_dataset(C, spec, did, run_dir, base_url, served, args.limit)

    # ---- verdicts (PASS / NOT PASS + reason) --------------------------
    sys.path.insert(0, str(PKG))
    from evaluator.verdicts import build_verdicts
    vsum = build_verdicts(run_dir, COMBINATION_ID, METHOD_TITLE, ACCURACY_LINE)
    print(f"\nverdicts: {vsum['pass']} PASS / {vsum['not_pass']} NOT PASS"
          + (f" / {vsum['error']} ERROR" if vsum['error'] else "")
          + f"  (of {vsum['total']})")
    print(f"  {vsum['report_md']}")
    print(f"  {vsum['verdicts_jsonl']}")

    # ---- teardown the judge before the (long) review ------------------
    if judge_proc is not None:
        from evaluator import gpu
        serve_judge.shutdown_judge(judge_proc, served)
        judge_proc = None
        gpu.restart_watchdog()

    # ---- optional review web app (best-effort; REPORT.md is primary) --
    if args.no_webapp:
        return
    try:
        from evaluator.review_model import build as build_review
        rm = build_review(run_dir, CONFIG_YAML, COMBINATION_ID, METHOD_TITLE)
        if rm is None:
            print("no scored rows to review — skipping web app")
            return
        serve_webapp(run_dir, args.host, args.port)
    except Exception as e:
        print(f"! review web app unavailable ({e}); the headless REPORT.md is "
              f"the primary deliverable:\n    {run_dir / 'REPORT.md'}")


if __name__ == "__main__":
    main()
