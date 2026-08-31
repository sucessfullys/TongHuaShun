#!/usr/bin/env python3
"""Optional review-web-app model builder (best-effort — REPORT.md is primary).

Assembles, from the trimmed config.yaml + the scorer's scores.jsonl:
  * <run_dir>/detection.json     — image paths per (sample, method, role)
  * <run_dir>/human/review_model.json — the shape the copied React front-end
    consumes, with each judge cell's `display` stamped with the PASS / NOT PASS
    verdict (so the web review shows verdicts, not the cryptic scorecard string).

Also provides the two tiny helpers the copied webapp imports
(load_review_model, write_json_atomic). NO era dependency.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from evaluator.verdicts import verdict_for

JUDGE_NAME = "Qwen3.5-122B-A10B"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def review_model_path(run_dir) -> Path:
    return Path(run_dir) / "human" / "review_model.json"


def load_review_model(run_dir):
    path = review_model_path(run_dir)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) and data.get("schema_version") else None


def _read_rows(path: Path) -> list:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


def _display(cid: str, sub: dict, ok: bool, error: str) -> str:
    verdict, reason, _ = verdict_for(cid, sub, ok, error)
    if verdict == "PASS":
        return "PASS"
    if verdict == "NOT PASS":
        return f"NOT PASS — {reason}" if reason else "NOT PASS"
    return f"{verdict} — {reason}" if reason else verdict


def build(run_dir, config_yaml, combination_id: str,
          method_title: str = "") -> Path | None:
    """Build detection.json + review_model.json. Returns the review_model path,
    or None if there is nothing scored to review."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load(Path(config_yaml).read_text())
    datasets = (cfg.get("data") or {}).get("datasets") or []

    # dataset_id -> {"input_roles": {...}, "methods": {mid: {path, output_file}}}
    dmap = {}
    for d in datasets:
        did = d.get("dataset_id")
        methods = {m["method_id"]: {"path": m["path"],
                                    "output_file": m.get("output_file", "")}
                   for m in d.get("methods", [])}
        dmap[did] = {"input_roles": d.get("input_roles", {}), "methods": methods}

    cfg_dir = run_dir / "results" / "full" / combination_id
    # (dataset_id, sample_key, method_id) -> row
    scored = {}
    for sf in sorted(cfg_dir.glob("scores.*.jsonl")):
        ds_name = sf.name[len("scores."):-len(".jsonl")]
        for row in _read_rows(sf):
            ds = row.get("dataset_id") or ds_name
            scored[(ds, row.get("sample_key"), row.get("method_id"))] = row
    if not scored:
        return None

    # detection samples keyed by sample_key (webapp key)
    det_samples: dict = {}
    all_methods: dict = {}
    input_roles: dict = {}
    for (ds, sk, mid), _row in scored.items():
        dinfo = dmap.get(ds) or {}
        input_roles.update(dinfo.get("input_roles", {}))
        minfo = (dinfo.get("methods") or {}).get(mid) or {}
        mpath = minfo.get("path")
        if not mpath:
            continue
        all_methods[mid] = {"method_id": mid,
                            "output_file": minfo.get("output_file", "")}
        sample_dir = Path(mpath) / sk
        out_path = sample_dir / minfo.get("output_file", "")
        inputs = {role: str(sample_dir / fname)
                  for role, fname in dinfo.get("input_roles", {}).items()}
        entry = det_samples.setdefault(sk, {"sample_key": sk, "methods": {},
                                            "inputs": {}})
        entry["methods"][mid] = {"output": str(out_path), "inputs": inputs}
        # sample-level inputs fall back to this method's copy
        for role, p in inputs.items():
            entry["inputs"].setdefault(role, p)

    detection = {
        "input_roles": dict(input_roles),
        "methods": list(all_methods.values()),
        "samples": list(det_samples.values()),
    }
    write_json_atomic(run_dir / "detection.json", detection)

    # review model
    method_ids = list(all_methods)
    configs = [{
        "combination_id": combination_id,
        "family": "hybrid",
        "judge": JUDGE_NAME,
        "slot": "ship",
        "title": method_title or combination_id,
        "hypothesis_id": "",
        "description": method_title or combination_id,
        "score_display": {"kind": "verdict", "factors": []},
    }]

    samples = []
    for sk in sorted(det_samples):
        det_s = det_samples[sk]
        input_images = [{"role": role, "method_id": next(iter(det_s["methods"]), ""),
                         "available": Path(p).is_file()}
                        for role, p in det_s["inputs"].items()]
        cells = []
        for mid in method_ids:
            if mid not in det_s["methods"]:
                continue
            # find the scored row for this (sk, mid) across datasets
            row = None
            for (ds, s2, m2), r in scored.items():
                if s2 == sk and m2 == mid:
                    row = r
                    break
            if row is None:
                continue
            sub = row.get("sub_scores") or {}
            judges = [{
                "combination_id": combination_id,
                "family": "hybrid",
                "score": row.get("score"),
                "score_kind": "verdict",
                "display": _display(combination_id, sub, row.get("ok", True),
                                    row.get("error", "")),
                "sub_scores": sub,
            }]
            out_path = det_s["methods"][mid].get("output")
            cells.append({
                "method_id": mid,
                "output_image": {"method_id": mid, "sample_key": sk},
                "judges": judges,
                "output_available": bool(out_path and Path(out_path).is_file()),
            })
        samples.append({
            "sample_key": sk,
            "input": {"images": input_images},
            "cells": cells,
            "family_b_rankings": [],
        })

    model = {
        "iteration": 1,
        "iteration_dir": run_dir.name,
        "mode": "full",
        "task_family": "editing",
        "input_roles": dict(input_roles),
        "methods": list(all_methods.values()),
        "configs": configs,
        "samples": samples,
        "schema_version": "1.0",
        "generated_at": _now_iso(),
        "task_adapter": "virtual_tryon",
        "warnings": [],
        "adapter": {"deterministic_ok": True, "subagent_ran": False,
                    "subagent_notes": ""},
    }
    write_json_atomic(review_model_path(run_dir), model)
    return review_model_path(run_dir)
