#!/usr/bin/env python3
"""PASS / NOT PASS verdicts from the scorer's per-sample scores.jsonl.

The scoring closure emits, per (sample, method), a row::

    {"sample_key", "method_id", "dataset_id", "score",
     "sub_scores": {..., "defect": bool, "defect_modes": [...]}, "ok": bool}

The shipped verdict is a readability transform of that row — NO re-scoring:

    sub_scores.defect == False  -> PASS
    sub_scores.defect == True   -> NOT PASS  (+ plain-English reason)
    ok == False / no defect flag -> ERROR    (the per-sample failure is surfaced,
                                              never silently turned into a verdict)

The reason mapping is ported verbatim from the project's relabel_verdicts.py.
This module has NO era dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

# defect_modes -> plain English. OR-union configs prefix modes with cov:/graft:;
# the prefix is stripped before mapping, then reasons are de-duplicated.
MODE_MAP = {
    "edit_scope":   "edit-scope: a non-target region changed",
    "nontarget":    "a non-target garment changed",
    "fidelity":     "target garment not faithfully reproduced",
    "color":        "color changed",
    "pattern":      "pattern / motif changed",
    "pose":         "body pose changed",
    "underlayer":   "underlayer changed",
    "garment_swap": "garment swapped / replaced",
    "print_source": "print / logo source mismatch",
}
# the 0/1/2 preservation-scorecard dims (fallback reason when defect_modes empty)
SCORECARD_DIMS = ["color", "pattern", "material", "garment_type", "structure",
                  "fine_details", "logo", "pose", "garment_swap"]
DIM_LABEL = {
    "garment_type": "garment type", "fine_details": "fine details",
    "garment_swap": "garment swap",
}


def _is_anchor(cid: str) -> bool:
    c = (cid or "").lower()
    return "anchor" in c or "metric" in c or "-72b-" in c or c.endswith("scale")


def verdict_for(cid: str, sub: dict, ok: bool = True, error: str = "") -> tuple:
    """Return (verdict, reason, defect_modes) for one row.

    verdict is one of "PASS", "NOT PASS", "ERROR", "BASELINE".
    """
    if not ok:
        return "ERROR", (error or "scoring failed for this sample"), []
    if not isinstance(sub, dict) or "defect" not in sub:
        if _is_anchor(cid):
            return "BASELINE", "not a verdict (calibration baseline)", []
        return "ERROR", (error or "no defect flag in row"), []
    if _is_anchor(cid):
        return "BASELINE", "not a verdict (calibration baseline)", []
    if not bool(sub.get("defect")):
        return "PASS", "", []
    # NOT PASS -> reason (strip cov:/graft: prefix, map, de-dupe)
    reasons, modes, seen = [], [], set()
    for m in (sub.get("defect_modes") or []):
        if not m:
            continue
        base = m.split(":", 1)[1] if ":" in m else m
        modes.append(base)
        txt = MODE_MAP.get(base, base.replace("_", " "))
        if txt not in seen:
            seen.add(txt)
            reasons.append(txt)
    if not reasons:  # fallback: name the scorecard dims that are 0
        zero = []
        for d in SCORECARD_DIMS:
            v = sub.get(d)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
                zero.append(DIM_LABEL.get(d, d.replace("_", " ")))
        reasons = [", ".join(zero)] if zero else ["flagged"]
    # de-dupe modes preserving order
    modes = list(dict.fromkeys(modes))
    return "NOT PASS", "; ".join(reasons), modes


def _read_rows(path: Path) -> list:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def build_verdicts(run_dir, combination_id: str, method_title: str = "",
                   accuracy_line: str = "") -> dict:
    """Read every results/full/<cid>/scores.<ds>.jsonl, write verdicts.jsonl +
    REPORT.md under run_dir. Returns a summary dict."""
    run_dir = Path(run_dir)
    cfg_dir = run_dir / "results" / "full" / combination_id
    score_files = sorted(cfg_dir.glob("scores.*.jsonl"))

    verdict_rows = []
    # summary[(dataset_id, method_id)] = {"PASS":n,"NOT PASS":n,"ERROR":n,"BASELINE":n}
    summary: dict = {}
    for sf in score_files:
        # scores.<dataset_id>.jsonl
        ds_from_name = sf.name[len("scores."):-len(".jsonl")]
        for row in _read_rows(sf):
            sk = row.get("sample_key")
            mid = row.get("method_id")
            ds = row.get("dataset_id") or ds_from_name
            sub = row.get("sub_scores") or {}
            ok = row.get("ok", True)
            err = row.get("error", "")
            verdict, reason, modes = verdict_for(combination_id, sub, ok, err)
            verdict_rows.append({
                "sample_key": sk, "method_id": mid, "dataset_id": ds,
                "verdict": verdict, "reason": reason, "defect_modes": modes,
            })
            key = (ds, mid)
            d = summary.setdefault(key, {"PASS": 0, "NOT PASS": 0,
                                         "ERROR": 0, "BASELINE": 0})
            d[verdict] = d.get(verdict, 0) + 1

    # ---- verdicts.jsonl -------------------------------------------------
    vpath = run_dir / "verdicts.jsonl"
    with vpath.open("w", encoding="utf-8") as f:
        for r in verdict_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- REPORT.md ------------------------------------------------------
    lines = []
    lines.append(f"# Try-on evaluation report — {method_title or combination_id}")
    lines.append("")
    lines.append(f"> combination_id: `{combination_id}`")
    if accuracy_line:
        lines.append(">")
        lines.append(f"> {accuracy_line}")
    lines.append("")
    total = len(verdict_rows)
    n_pass = sum(1 for r in verdict_rows if r["verdict"] == "PASS")
    n_np = sum(1 for r in verdict_rows if r["verdict"] == "NOT PASS")
    n_err = sum(1 for r in verdict_rows if r["verdict"] == "ERROR")
    lines.append(f"**{total} results scored** — {n_pass} PASS, {n_np} NOT PASS"
                 + (f", {n_err} ERROR" if n_err else "") + ".")
    lines.append("")
    lines.append("## Pass rate per try-on method")
    lines.append("")
    lines.append("| dataset | method | n | PASS | NOT PASS | ERROR | pass-rate |")
    lines.append("|---------|--------|---|------|----------|-------|-----------|")
    for (ds, mid) in sorted(summary):
        d = summary[(ds, mid)]
        scored = d["PASS"] + d["NOT PASS"]
        n = scored + d["ERROR"] + d["BASELINE"]
        pr = f"{100 * d['PASS'] / scored:.1f}%" if scored else "—"
        lines.append(f"| {ds} | {mid} | {n} | {d['PASS']} | {d['NOT PASS']} "
                     f"| {d['ERROR']} | {pr} |")
    lines.append("")
    lines.append("## Per-sample verdicts")
    lines.append("")
    cur = None
    for r in sorted(verdict_rows, key=lambda x: (x["dataset_id"] or "",
                                                 x["method_id"] or "",
                                                 x["sample_key"] or "")):
        head = (r["dataset_id"], r["method_id"])
        if head != cur:
            cur = head
            lines.append("")
            lines.append(f"### {r['dataset_id']} / {r['method_id']}")
            lines.append("")
        if r["verdict"] == "PASS":
            lines.append(f"- PASS — `{r['sample_key']}`")
        elif r["verdict"] == "NOT PASS":
            lines.append(f"- NOT PASS — {r['reason']} — `{r['sample_key']}`")
        else:
            lines.append(f"- {r['verdict']} — {r['reason']} — `{r['sample_key']}`")
    lines.append("")
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    return {"total": total, "pass": n_pass, "not_pass": n_np, "error": n_err,
            "verdicts_jsonl": str(vpath), "report_md": str(run_dir / "REPORT.md"),
            "per_method": {f"{ds}/{mid}": summary[(ds, mid)]
                           for (ds, mid) in summary}}
