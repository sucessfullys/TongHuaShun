#!/usr/bin/env python3
"""ERA Stage-6 SHARED runner module for iter_029 (multi-dataset: ces_old + whq_new).

Implements the _experiment_protocol.md contract ONCE, for every per-config
runner in this iter:

  - §1 path isolation: everything resolves under <iter>/experiments/...; <iter>
        is resolved from the workspace `current` symlink, NEVER hard-coded.
  - §2 PID marker first: write <iter>/experiments/logs/<task_id>.pid before
        anything else.
  - §3 read <task_id>.override.json (endpoint, batch_size/max_concurrency, port).
  - §4 progress heartbeat: a daemon thread refreshes
        <task_id>.progress.json every 10 s (file mtime = the runner heartbeat),
        plus a write on every sample completion — never goes stale mid-batch.
  - §5 samples_subset is AUTHORITATIVE: when the eval task carries
        samples_subset score EXACTLY those keys (resolved against the task's
        dataset_id entry in config.yaml data.datasets[]); otherwise the legacy
        sorted(sample_glob)[:samples_per_method] fallback.
  - §6 append one JSON line per (sample, method) to the task's expected_output
        (canonical per-dataset path
        results/<mode>/<combination_id>/scores.<dataset_id>.jsonl) with EXACT
        fields sample_key, method_id, dataset_id, score (number), sub_scores
        (object), scope, ok (bool) (+ error on failure).
  - §7 done.json LAST, status success|failure only (never skipped), on success
        AND failure (try/except wraps the body).
  - §8 Family-A calls the ALREADY-SERVED judge via an OpenAI-compatible client
        read from the endpoint (override `endpoint` / argv / the serve task's
        recorded done.json) — it never loads a model.
  - §11 never fabricate: a per-sample failure is a logged ok:false row, never
        an aborted task or an invented number.

A per-config runner imports this module and calls:

    era_eval_common.run(score_one=<callable>, needs_metrics=<bool>)

where score_one(ep, dataset_id, sample_key, cloth, model_img, result) returns
(score: float, sub_scores: dict, ok: bool, defect: bool). `ep` is a (base_url,
served_model_name) tuple OR a thread-safe picker callable returning that tuple
(both shapes are accepted by the v13 / editscope reads, which call ep()).

The per-config runner declares module-level constants the loop reads:
    TASK_ID, COMBINATION_ID, MODE, DATASET_ID, JUDGE_TASK_ID, NEEDS_JUDGE,
    SAMPLES_SUBSET, SAMPLES_PER_METHOD, SCOPE_LABEL
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))


# --------------------------------------------------------------------------- #
# workspace / iteration resolution (NEVER hard-code <iter>)
# --------------------------------------------------------------------------- #
def _resolve_paths():
    """configs/ lives under <iter>/experiments/configs/.  Resolve every dir from
    THIS file's location, then locate the workspace root and config.yaml."""
    script = Path(__file__).resolve()
    configs_dir = script.parent
    # --- standalone-delivery severing edit -------------------------------- #
    # When packaged as a standalone deliverable (no ERA workspace above this
    # file), the run.py driver exports TRYON_CONFIG_YAML / TRYON_LOGS_DIR so we
    # never resolve the repo workspace from __file__. Honor them here; the
    # original body below stays as the no-env fallback so the in-repo iter_038
    # runners keep working unchanged.
    env_cfg = os.environ.get("TRYON_CONFIG_YAML")
    env_logs = os.environ.get("TRYON_LOGS_DIR")
    if env_cfg or env_logs:
        config_yaml = Path(env_cfg).resolve() if env_cfg else (configs_dir / "config.yaml")
        logs_dir = Path(env_logs).resolve() if env_logs else (configs_dir / "logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        return {"configs_dir": configs_dir, "experiments_dir": configs_dir,
                "iter_dir": configs_dir, "workspace_dir": config_yaml.parent,
                "logs_dir": logs_dir, "config_yaml": config_yaml}
    experiments_dir = configs_dir.parent
    iter_dir = experiments_dir.parent          # <iter>
    workspace_dir = iter_dir.parent            # the workspace root
    logs_dir = experiments_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # sanity: the resolved iter should match the `current` symlink target when present.
    cur = workspace_dir / "current"
    try:
        if cur.is_symlink() or cur.exists():
            target = Path(os.path.realpath(cur))
            if target != iter_dir:
                # not fatal — path isolation already keys off THIS file's iter_dir,
                # but log the divergence for the operator.
                print(f"[era_eval_common] note: current -> {target} but runner is under {iter_dir}")
    except Exception:
        pass
    return {
        "configs_dir": configs_dir,
        "experiments_dir": experiments_dir,
        "iter_dir": iter_dir,
        "workspace_dir": workspace_dir,
        "logs_dir": logs_dir,
        "config_yaml": workspace_dir / "config.yaml",
    }


def _load_dataset_entry(config_yaml: Path, dataset_id: str) -> dict:
    """Resolve THIS task's dataset entry from config.yaml data.datasets[].

    Multi-dataset (v0.1.7.2): resolve data_root / methods / input_roles /
    sample_glob against the dataset_id entry, never the top-level data.*.
    """
    cfg = yaml.safe_load(config_yaml.read_text())
    data = cfg.get("data", {})
    datasets = data.get("datasets") or []
    for d in datasets:
        if d.get("dataset_id") == dataset_id:
            return d
    # single-dataset legacy fallback: the top-level data block IS the dataset.
    if not datasets:
        return data
    raise RuntimeError(f"dataset_id {dataset_id!r} not in config.yaml data.datasets")


# --------------------------------------------------------------------------- #
# §8 endpoint resolution: override `endpoint` / argv / serve done.json
# --------------------------------------------------------------------------- #
def _endpoint_from_obj(obj):
    """Normalize an endpoint spec (str base_url OR object) -> (base_url, served).

    Accepts:
      - "http://127.0.0.1:8000/v1"
      - {"base_url": ..., "served_model_name": ...}  (serve done.json shape)
      - {"endpoint": "...", "served_model_name": ...}
      - {"url": ..., "model": ...}
    Returns (base_url or None, served_model_name or None).
    """
    if obj is None:
        return None, None
    if isinstance(obj, str):
        return obj.rstrip("/"), None
    if isinstance(obj, dict):
        base = (obj.get("base_url") or obj.get("endpoint") or obj.get("url")
                or obj.get("base") or "")
        served = (obj.get("served_model_name") or obj.get("model")
                  or obj.get("served_model") or obj.get("model_name"))
        base = base.rstrip("/") if isinstance(base, str) else None
        return base, served
    return None, None


def _resolve_endpoint(paths, task_id, judge_task_id, override):
    """Endpoint priority: override.json `endpoint` > argv (--endpoint / --served)
    > the serve task's recorded done.json endpoint block. Returns
    (base_url, served_model_name). Raises if nothing resolves and a judge is
    required (the caller turns that into a failure done.json)."""
    # 1) override file `endpoint`
    base, served = _endpoint_from_obj(override.get("endpoint"))
    if not served:
        served = override.get("served_model_name") or served
    if base:
        return base, served

    # 2) argv: --endpoint <url> [--served <name>]  (also bare positional url)
    argv = sys.argv[1:]
    cli_base, cli_served = None, None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--endpoint", "--base-url", "--base_url") and i + 1 < len(argv):
            cli_base = argv[i + 1].rstrip("/"); i += 2; continue
        if a in ("--served", "--served-model-name", "--served_model_name", "--model") and i + 1 < len(argv):
            cli_served = argv[i + 1]; i += 2; continue
        if a.startswith("http") and cli_base is None:
            cli_base = a.rstrip("/")
        i += 1
    if cli_base:
        return cli_base, (cli_served or served)

    # 3) the serve task's recorded done.json endpoint
    if judge_task_id:
        p = paths["logs_dir"] / f"{judge_task_id}.done.json"
        if p.is_file():
            try:
                d = json.loads(p.read_text())
                if d.get("status") == "success":
                    b, s = _endpoint_from_obj(d.get("endpoint"))
                    if b:
                        return b, (s or served)
            except Exception:
                pass
        raise RuntimeError(
            f"no endpoint: override.json has no `endpoint`, no --endpoint argv, "
            f"and serve done.json {p} is absent/unsuccessful")
    raise RuntimeError("no endpoint resolved and no judge_task_id to read from")


def make_pick_ep(base_url, served):
    """Thread-safe round-robin endpoint picker (single endpoint here, but the
    v13 reads call ep() expecting a picker, so keep that shape)."""
    ctr = itertools.count()
    lock = threading.Lock()
    repl = [(base_url, served)]

    def pick_ep():
        with lock:
            i = next(ctr)
        return repl[i % len(repl)]
    return pick_ep


# --------------------------------------------------------------------------- #
# §5 sample selection: samples_subset authoritative, else sorted-glob first-N
# --------------------------------------------------------------------------- #
def _strip_ds_prefix(key, dataset_id):
    """A `<dataset_id>::<sample_key>` token belongs to this task's own dataset —
    strip the prefix before lookup (it is always this dataset)."""
    pre = f"{dataset_id}::"
    return key[len(pre):] if isinstance(key, str) and key.startswith(pre) else key


def _select(method_path: Path, dataset_entry, samples_subset, samples_per_method,
            dataset_id):
    """Return ([(sample_key, sample_dir), ...], missing_count).

    samples_subset set -> score EXACTLY those keys (method_path/<key>/ must be a
    dir; missing dirs are skipped + counted). Else sorted(glob(sample_glob))[:N].
    """
    base = Path(method_path)
    if samples_subset:
        out, missing = [], 0
        for raw in samples_subset:
            k = _strip_ds_prefix(raw, dataset_id)
            d = base / k
            if d.is_dir():
                out.append((k, d))
            else:
                missing += 1
        return out, missing
    glob = dataset_entry.get("sample_glob", "*")
    cands = sorted(base.glob(glob))
    keys = [str(p.relative_to(base)) for p in cands if p.is_dir()]
    n = samples_per_method if samples_per_method else len(keys)
    return [(k, base / k) for k in keys[:n]], 0


# --------------------------------------------------------------------------- #
# the run loop
# --------------------------------------------------------------------------- #
def run(*, score_one, needs_metrics=False, max_workers=16):
    """Drive one eval task. The caller's module supplies the config constants and
    a score_one(ep, dataset_id, sample_key, cloth, model_img, result) callable.

    score_one returns (score, sub_scores, ok, defect). A per-sample exception is
    caught -> ok:false row with an `error` string (never aborts the task).
    """
    caller = sys.modules["__main__"]
    TASK_ID = caller.TASK_ID
    COMBINATION_ID = caller.COMBINATION_ID
    MODE = caller.MODE
    DATASET_ID = caller.DATASET_ID
    JUDGE_TASK_ID = getattr(caller, "JUDGE_TASK_ID", None)
    NEEDS_JUDGE = getattr(caller, "NEEDS_JUDGE", True)
    SAMPLES_SUBSET = getattr(caller, "SAMPLES_SUBSET", None)
    SAMPLES_PER_METHOD = getattr(caller, "SAMPLES_PER_METHOD", None)
    SCOPE_LABEL = getattr(caller, "SCOPE_LABEL", "pointwise garment+pose category-free")
    EXPECTED_OUTPUT = getattr(caller, "EXPECTED_OUTPUT", None)

    paths = _resolve_paths()
    logs_dir = paths["logs_dir"]
    iter_dir = paths["iter_dir"]
    configs_dir = paths["configs_dir"]

    pid_marker = logs_dir / f"{TASK_ID}.pid"
    done_marker = logs_dir / f"{TASK_ID}.done.json"
    progress_marker = logs_dir / f"{TASK_ID}.progress.json"
    override_path = configs_dir / f"{TASK_ID}.override.json"

    # §2 PID marker FIRST.
    pid_marker.write_text(str(os.getpid()))

    # canonical per-dataset scores path. Prefer the task's expected_output when
    # the caller passed it (it is the authoritative Stage-5-pinned path); else
    # build the canonical results/<mode>/<combination_id>/scores.<dataset_id>.jsonl.
    if EXPECTED_OUTPUT:
        ep_path = Path(EXPECTED_OUTPUT)
        scores_path = ep_path if ep_path.is_absolute() else (paths["workspace_dir"] / ep_path)
    else:
        scores_path = (paths["experiments_dir"] / "results" / MODE / COMBINATION_ID
                       / f"scores.{DATASET_ID}.jsonl")
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    def write_progress(done, total, **extra):
        p = {"task_id": TASK_ID, "done": done, "total": total,
             "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        p.update(extra)
        try:
            progress_marker.write_text(json.dumps(p))
        except Exception:
            pass

    def write_done(status, summary, **extra):
        p = {"task_id": TASK_ID, "status": status,
             "exit_code": 0 if status == "success" else 1, "summary": summary[:8000],
             "result_dir": str(scores_path.parent),
             "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        p.update(extra)
        try:
            done_marker.write_text(json.dumps(p, indent=2))
        except Exception:
            pass

    st = {"done": 0, "ok": 0, "failed": 0, "defect": 0, "total": 0, "missing": 0}
    wl = threading.Lock()
    stop_beat = threading.Event()

    def _heartbeat():
        # §4 refresh the heartbeat marker every 10 s even mid-batch.
        while not stop_beat.wait(10):
            with wl:
                write_progress(st["done"], st["total"], ok=st["ok"],
                               failed=st["failed"], defect=st["defect"],
                               missing_samples=st["missing"])

    try:
        # §3 override file
        override = {}
        if override_path.is_file():
            try:
                override = json.loads(override_path.read_text())
            except Exception:
                override = {}
        mw = int(override.get("max_concurrency", override.get("batch_size", max_workers)) or max_workers)

        dataset_entry = _load_dataset_entry(paths["config_yaml"], DATASET_ID)
        roles = dataset_entry.get("input_roles", {})
        cloth_name = roles.get("input_cloth", "input_cloth.png")
        model_name = roles.get("input_model", "input_model.png")
        methods = dataset_entry.get("methods", [])

        # §8 endpoint (Family-A). Family-B (needs_judge False) skips it.
        pick_ep = None
        endpoint_block = None
        if NEEDS_JUDGE:
            try:
                base_url, served = _resolve_endpoint(paths, TASK_ID, JUDGE_TASK_ID, override)
            except Exception as e:
                write_done("failure", f"endpoint not available: {e}", exit_code=2)
                raise SystemExit(2)
            pick_ep = make_pick_ep(base_url, served)
            endpoint_block = {"base_url": base_url, "served_model_name": served}

        # optional Family-B metric init (H2 DINOv2). The caller owns the import;
        # we only forward the flag so the per-config module can prep its metric.
        if needs_metrics and hasattr(caller, "init_metrics"):
            try:
                caller.init_metrics()
            except Exception as e:
                print(f"[era_eval_common] metric init failed (degrading): {e}")

        # build the (method, sample_key, sample_dir) work list for THIS dataset.
        work = []
        missing_total = 0
        for method in methods:
            mpath = Path(method["path"])
            if not mpath.is_dir():
                continue
            sel, missing = _select(mpath, dataset_entry, SAMPLES_SUBSET,
                                   SAMPLES_PER_METHOD, DATASET_ID)
            missing_total += missing
            for sk, sd in sel:
                work.append((method, sk, sd))
        total = len(work)
        st["total"] = total
        st["missing"] = missing_total
        write_progress(0, total, missing_samples=missing_total)

        # fresh scores each run for THIS dataset's file (append-only within a run;
        # a healed re-run starts clean so rows are never double-counted).
        if scores_path.exists():
            scores_path.unlink()

        def _one(item):
            method, sk, sd = item
            row = {"sample_key": sk, "method_id": method["method_id"],
                   "dataset_id": DATASET_ID, "score": 0.0, "sub_scores": {},
                   "scope": SCOPE_LABEL, "ok": False}
            try:
                cloth = sd / cloth_name
                model_img = sd / model_name
                result = sd / method["output_file"]
                if not cloth.is_file():
                    raise FileNotFoundError(f"missing cloth {cloth}")
                if not result.is_file():
                    raise FileNotFoundError(f"missing result {result}")
                mi = model_img if model_img.is_file() else None
                score, sub, ok, defect = score_one(
                    pick_ep, DATASET_ID, sk, cloth, mi, result)
                row["score"] = float(score)
                row["sub_scores"] = dict(sub)
                # §6 the binary verdict the review + aggregator read.
                row["sub_scores"]["defect"] = bool(defect)
                row["ok"] = bool(ok)
            except Exception as e:
                row["ok"] = False
                row["error"] = str(e)
            return row

        beat = threading.Thread(target=_heartbeat, daemon=True)
        beat.start()
        try:
            with scores_path.open("a") as f:
                with ThreadPoolExecutor(max_workers=mw) as ex:
                    futs = [ex.submit(_one, it) for it in work]
                    for fut in as_completed(futs):
                        row = fut.result()
                        with wl:
                            f.write(json.dumps(row) + "\n")
                            f.flush()
                            st["done"] += 1
                            st["ok" if row.get("ok") else "failed"] += 1
                            if row.get("sub_scores", {}).get("defect"):
                                st["defect"] += 1
                            if (st["done"] % 4 == 0) or st["done"] == total:
                                write_progress(st["done"], total, ok=st["ok"],
                                               failed=st["failed"], defect=st["defect"],
                                               missing_samples=missing_total)
        finally:
            stop_beat.set()
        write_progress(st["done"], total, ok=st["ok"], failed=st["failed"],
                       defect=st["defect"], missing_samples=missing_total)

        extra = {"ok_count": st["ok"], "failed_count": st["failed"],
                 "defect_count": st["defect"], "missing_samples": missing_total,
                 "dataset_id": DATASET_ID, "scores_path": str(scores_path)}
        if endpoint_block:
            extra["endpoint"] = endpoint_block
        # §7 success done.json LAST. A task that scored 0 ok rows out of >0 work
        # is a real failure (judge unreachable / all inputs missing).
        if total > 0 and st["ok"] == 0:
            write_done("failure",
                       f"{MODE} {COMBINATION_ID} [{DATASET_ID}] scored 0/{total} ok "
                       f"({st['failed']} failed, {missing_total} subset keys missing)",
                       **extra)
            raise SystemExit(1)
        write_done("success",
                   f"{MODE} {COMBINATION_ID} [{DATASET_ID}] scored {st['ok']}/{total} ok, "
                   f"{st['failed']} fail, {st['defect']} defect-flagged, "
                   f"{missing_total} subset keys missing",
                   **extra)
    except SystemExit:
        raise
    except Exception:
        stop_beat.set()
        tb = traceback.format_exc()
        write_done("failure", f"runner exception: {tb}")
        raise SystemExit(1)


def main_wrap(run_main):
    """Wrap a per-config main() so a setup/import crash STILL lands a failure
    done.json (the runner's last action on ANY exit path — §7)."""
    paths = _resolve_paths()
    try:
        run_main()
    except SystemExit:
        raise
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        try:
            caller = sys.modules["__main__"]
            tid = getattr(caller, "TASK_ID", "unknown-task")
            done = paths["logs_dir"] / f"{tid}.done.json"
            if not done.is_file():
                done.write_text(json.dumps({
                    "task_id": tid, "status": "failure", "exit_code": 1,
                    "summary": ("runner setup exception: " + tb)[:8000],
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))
        except Exception:
            pass
        raise SystemExit(1)
