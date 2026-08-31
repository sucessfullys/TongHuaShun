#!/usr/bin/env python3
"""iter_030 v15 scorer-dispatch — the 9 chosen configs as ADDITIVE deltas on the
BYTE-FROZEN iter_029 spine (_editscope.py / _ntmetric.py / _v13_eval/*).

Nothing here mutates the frozen modules; every v15 config is composed from their
public atoms so H1 (the v14b elitism control) is reproduced EXACTLY and the only
moving variable per config is the documented delta (confound-control req).

Dispatch contract (read by the generated runners + era_eval_common):
    SPEC[combination_id] = {
        "fn": score_one(pick_ep, dataset_id, sample_key, cloth, model_img, result)
              -> (score: float, sub: dict, ok: bool, defect: bool),
        "needs_judge": bool,   # Family-A/hybrid True; Family-B anchor False
        "needs_metrics": bool, # loads DINOv2 (H1/H2/H3/H8/H9 + the H4 anchor)
    }

config -> hypothesis (iter_030 brief):
  H1 hybrid-editscope-zoomgate-step0-122b-v14b-ntmetric          = NT.score_h2 EXACT
  H2 hybrid-editscope-zoomgate-confirm-step0-122b-v15a           = H1 + borderline confirm (P1)
  H3 hybrid-editscope-zoomgate-confirm-colorcal-step0-122b-v15b  = H2 + color-EMD ch + per-ds cal (P2)
  H4 nontarget-trigger-calibration-anchor-b-v1                   = Family-B per-ds threshold anchor
  H5 editscope-purevlm-thinkprobe-235b-vl-v15a                   = pure-VLM 1-call (235B, native think)
  H6 editscope-purevlm-thinkprobe-122b-v15a-ctrl                 = pure-VLM 1-call (122B, scale ctrl)
  H7 editscope-purevlm-pose-underlayer-122b-v15b                 = pure-VLM + pose/identity + underlayer
  H8 hybrid-editscope-zoomgate-122b-v15a-lean3call              = v14b minus the v13 fidelity-zoom
  H9 hybrid-editscope-singleread-122b-v15c-1call                = score_h3 (1 call) + metric zoom-gate
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PIL import Image

from _v13_eval import gf_common as G
from _v13_eval import crossover as X
from _v13_eval import metric_common as M
import _editscope as ES
import _ntmetric as NT

DEFAULT_SEEDS = [42]

# ---- combination ids ----
H1_ID = "hybrid-editscope-zoomgate-step0-122b-v14b-ntmetric"
H2_ID = "hybrid-editscope-zoomgate-confirm-step0-122b-v15a"
H3_ID = "hybrid-editscope-zoomgate-confirm-colorcal-step0-122b-v15b"
H4_ID = "nontarget-trigger-calibration-anchor-b-v1"
H5_ID = "editscope-purevlm-thinkprobe-235b-vl-v15a"
H6_ID = "editscope-purevlm-thinkprobe-122b-v15a-ctrl"
H7_ID = "editscope-purevlm-pose-underlayer-122b-v15b"
H8_ID = "hybrid-editscope-zoomgate-122b-v15a-lean3call"
H9_ID = "hybrid-editscope-singleread-122b-v15c-1call"


# --------------------------------------------------------------------------- #
# per-dataset calibration thresholds (written by the H4 anchor; H3 reads them).
# Lives at results/<mode>/<H4_ID>/thresholds.json relative to this iter. H3
# falls back to module defaults (global single threshold) when absent.
# --------------------------------------------------------------------------- #
_THRESH_CACHE = {}
_THRESH_LOCK = threading.Lock()
DEFAULT_DINO_THRESHOLD = NT.NT_COS_THRESHOLD       # 0.92 global
DEFAULT_EMD_THRESHOLD = 0.060                      # WB-norm color-EMD global cut


def _iter_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent   # configs/../../ = <iter>


def _load_thresholds(mode: str) -> dict:
    """Read the H4 anchor's per-dataset thresholds for this mode (cached).
    Shape: {dataset_id: {"dino": float, "emd": float}}. {} when the anchor has
    not run yet (H3 then uses the global defaults)."""
    key = mode
    with _THRESH_LOCK:
        if key in _THRESH_CACHE:
            return _THRESH_CACHE[key]
        out = {}
        # the anchor writes its mode's thresholds; also accept an annotated-mode
        # anchor as the calibration source for a later full run.
        for m in (mode, "annotated"):
            p = _iter_dir() / "experiments" / "results" / m / H4_ID / "thresholds.json"
            if p.is_file():
                try:
                    out = json.loads(p.read_text()).get("per_dataset", {}) or {}
                    if out:
                        break
                except Exception:
                    out = {}
        _THRESH_CACHE[key] = out
        return out


def _thresholds_for(dataset_id: str, mode: str):
    per = _load_thresholds(mode)
    d = per.get(dataset_id) or {}
    return (float(d.get("dino", DEFAULT_DINO_THRESHOLD)),
            float(d.get("emd", DEFAULT_EMD_THRESHOLD)))


# --------------------------------------------------------------------------- #
# color-EMD non-target channel (P2). WB-normalized 1-D per-channel Wasserstein.
# --------------------------------------------------------------------------- #
def _wb_norm(arr):
    import numpy as np
    a = arr.astype("float64")
    mean = a.reshape(-1, a.shape[-1]).mean(axis=0) + 1e-6
    a = a / mean[None, None, :]          # divide out global white-balance
    return a


def _channel_emd(a, b, bins=32):
    """Mean per-channel 1-D Wasserstein (EMD) between two WB-normalized crops,
    range-normalized to ~[0,1]. Robust to size (histograms, not pixels)."""
    import numpy as np
    a = _wb_norm(a); b = _wb_norm(b)
    lo = float(min(a.min(), b.min())); hi = float(max(a.max(), b.max()))
    if hi <= lo:
        return 0.0
    emds = []
    for c in range(a.shape[-1]):
        ha, _ = np.histogram(a[..., c].ravel(), bins=bins, range=(lo, hi), density=True)
        hb, _ = np.histogram(b[..., c].ravel(), bins=bins, range=(lo, hi), density=True)
        ha = ha / (ha.sum() + 1e-9); hb = hb / (hb.sum() + 1e-9)
        emds.append(float(np.abs(np.cumsum(ha) - np.cumsum(hb)).sum()) / bins)
    return float(sum(emds) / len(emds))


def _color_emd_half(model_img, result, nt_category):
    """WB-normalized color-EMD on the NON-target half-crop (model vs result)."""
    import numpy as np
    try:
        mim = Image.open(model_img).convert("RGB")
        rim = Image.open(result).convert("RGB")
        mc = NT._crop_half(mim, nt_category)
        rc = NT._crop_half(rim, nt_category)
        return _channel_emd(np.asarray(mc), np.asarray(rc))
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# borderline confirmation re-read (P1) — "is there a CLEAR, unambiguous change?"
# --------------------------------------------------------------------------- #
def _build_confirm_messages(model_crop, result_crop, nt_label, swap=False):
    instr = (
        "You are double-checking ONE body region of a virtual try-on. This region "
        "(the " + str(nt_label).upper() + " half) is NOT the try-on target. A metric "
        "flagged a possible change here, but it may be a false alarm.\n"
        "You are shown the SAME region cropped from two images:\n"
        "  [ORIGINAL " + str(nt_label).upper() + "] = the person BEFORE try-on.\n"
        "  [RESULT " + str(nt_label).upper() + "]   = the generated result, SAME region.\n"
        "Decide STRICTLY: is there a CLEAR, UNAMBIGUOUS garment change here (a different "
        "garment, or an obvious recolor/retexture)? Answer 1 ONLY if the change is clear "
        "to the naked eye. Answer 0 if it is subtle/ambiguous or explained by pose, crop, "
        "lighting/shadow, JPEG artifacts, face/skin/hair regen, or a region merely "
        "occluded/revealed by the OTHER region's new garment drape.\n"
        "ANTI-BIAS: crops may be in either order; rely on the [ORIGINAL]/[RESULT] labels.\n"
        "Reason in ONE short phrase, then finish with ONE final JSON object on the last "
        'line with EXACTLY this key (integer 0/1):\n{"clear_change":_}'
    )
    m_url = G.img_to_data_url(model_crop)
    r_url = G.img_to_data_url(result_crop)
    pair = [("[ORIGINAL " + str(nt_label).upper() + "]:", m_url),
            ("[RESULT " + str(nt_label).upper() + "]:", r_url)]
    if swap:
        pair = [pair[1], pair[0]]
    content = []
    for lab, url in pair:
        content.append({"type": "text", "text": lab})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instr})
    return [{"role": "user", "content": content}]


def _confirm_clear_change(pick_ep, model_img, result, nt_category, nt_label, seeds):
    """A focused confirmation re-read of the suspect non-target crop. Returns
    True (clear change confirmed) / False (borderline -> do NOT veto) / None
    (unparseable -> caller defers to the original verdict)."""
    try:
        mim = Image.open(model_img).convert("RGB")
        rim = Image.open(result).convert("RGB")
    except Exception:
        return None
    m_crop = NT._crop_half(mim, nt_category)
    r_crop = NT._crop_half(rim, nt_category)
    votes = []
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        try:
            msgs = _build_confirm_messages(m_crop, r_crop, nt_label, swap=(i % 2 == 1))
            txt = G.call_judge(base_url, model, msgs, seed=seed,
                               enable_thinking=False, max_tokens=512)
            parsed = G.extract_last_json(txt)
            if not parsed:
                continue
            v = parsed.get("clear_change")
            try:
                v = int(v)
            except (TypeError, ValueError):
                v = None
            if v in (0, 1):
                votes.append(v)
        except Exception:
            continue
    if not votes:
        return None
    return votes.count(1) >= votes.count(0)


# --------------------------------------------------------------------------- #
# shared hybrid core: v13 base + editscope read + (metric|color) zoom-gate,
# with an optional borderline-confirmation pass and an optional color channel.
# H1/H8 use confirm=False; H2 confirm=True; H3 confirm=True + color=True.
# --------------------------------------------------------------------------- #
def _hybrid_core(pick_ep, dataset_id, cloth, model_img, result, *, mode,
                 base_id, confirm, color_channel):
    base_score, sub, ok, base_defect = X.evaluate_sample(
        base_id, pick_ep, None, cloth, model_img, result,
        gf_seeds=DEFAULT_SEEDS, pres_seeds=DEFAULT_SEEDS)
    edit_scope, escope_sub = ES.editscope_read(
        pick_ep, cloth, model_img, result, DEFAULT_SEEDS)

    nt_sub = {"dino_ok": bool(NT._DINO_OK), "nt_zoom_triggered": False,
              "nt_dino_cos": None, "nt_color_emd": None, "nt_zoom_changed": None,
              "nt_confirmed": None, "confirm_pass": bool(confirm),
              "color_channel": bool(color_channel)}

    base_read_flagged = (edit_scope == 0)   # confident STEP-A/B/C veto -> direct

    if (edit_scope == 1 and model_img is not None and Path(model_img).is_file()
            and escope_sub.get("escope_ok")):
        targets = escope_sub.get("escope_target_regions") or []
        single = len(targets) == 1 and targets[0] in ("upper", "lower")
        if single and NT._ensure_dino():
            nt_cat, nt_label = NT._nontarget_half(targets[0])
            if nt_cat:
                dino_thr, emd_thr = _thresholds_for(dataset_id, mode)
                try:
                    mim = Image.open(model_img).convert("RGB")
                    rim = Image.open(result).convert("RGB")
                    cos = NT._dino_cos(NT._crop_half(mim, nt_cat),
                                       NT._crop_half(rim, nt_cat))
                    nt_sub["nt_dino_cos"] = cos
                    trip = cos is not None and cos < dino_thr
                    if color_channel:
                        emd = _color_emd_half(model_img, result, nt_cat)
                        nt_sub["nt_color_emd"] = emd
                        if emd is not None and emd > emd_thr:
                            trip = True
                    if trip:
                        nt_sub["nt_zoom_triggered"] = True
                        changed = NT._nt_zoom_reread(
                            pick_ep, model_img, result, nt_cat, nt_label, DEFAULT_SEEDS)
                        nt_sub["nt_zoom_changed"] = changed
                        flip = (changed == 1)
                        # P1: a metric-triggered (borderline) change is gated on a
                        # confirmation re-read; a confident base-read veto is direct.
                        if flip and confirm and not base_read_flagged:
                            conf = _confirm_clear_change(
                                pick_ep, model_img, result, nt_cat, nt_label, DEFAULT_SEEDS)
                            nt_sub["nt_confirmed"] = conf
                            if conf is False:
                                flip = False     # borderline -> keep as a PASS
                        if flip:
                            edit_scope = 0
                            escope_sub = dict(escope_sub)
                            escope_sub["edit_scope"] = 0
                            escope_sub["escope_reason"] = "nontarget_changed_zoom"
                            escope_sub["target_regions"] = targets
                            escope_sub["changed_regions"] = sorted(
                                set(escope_sub.get("escope_changed_regions") or []) | {nt_label})
                except Exception as e:
                    nt_sub["nt_zoom_err"] = str(e)

    score, sub, defect = ES.apply_editscope_veto(
        base_score, sub, base_defect, edit_scope, escope_sub)
    sub.update(nt_sub)
    return score, sub, ok, defect


# --------------------------------------------------------------------------- #
# H1 — v14b elitism control (EXACT NT.score_h2)
# --------------------------------------------------------------------------- #
def score_h1(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    return NT.score_h2(pick_ep, cloth, model_img, result,
                       gf_seeds=DEFAULT_SEEDS, pres_seeds=DEFAULT_SEEDS,
                       escope_seeds=DEFAULT_SEEDS)


# H2 — v15a borderline confirmation pass (P1)
def score_h2(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    return _hybrid_core(pick_ep, dataset_id, cloth, model_img, result,
                        mode=_MODE(), base_id=X.V13_FIDELITYZOOM_COVERAGE_FIRST_ID,
                        confirm=True, color_channel=False)


# H3 — v15b confirm + color-EMD + per-dataset calibration (P2)
def score_h3(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    return _hybrid_core(pick_ep, dataset_id, cloth, model_img, result,
                        mode=_MODE(), base_id=X.V13_FIDELITYZOOM_COVERAGE_FIRST_ID,
                        confirm=True, color_channel=True)


# H8 — lean3call: v14b machinery on the V11_GSZOOM base (drops the v13 fidelity zoom)
def score_h8_lean(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    return _hybrid_core(pick_ep, dataset_id, cloth, model_img, result,
                        mode=_MODE(), base_id=X.V11_GSZOOM_COVERAGE_FIRST_ID,
                        confirm=False, color_channel=False)


# --------------------------------------------------------------------------- #
# H5 / H6 — pure-VLM thinkprobe (single editscope+fidelity call, NO metric).
# Byte-identical prompt (ES.score_h3); the only difference is the served judge
# (235B-VL vs 122B). 235B-VL thinks natively; 122B uses prompt-CoT (env note).
# --------------------------------------------------------------------------- #
def score_h5_purevlm_235b(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    # native_think=False: byte-identical prompt+mode to H6 (clean Rule-4 scale
    # A/B; the ONLY moving variable is judge size). The 235B "-Thinking" chat
    # template defaults thinking ON and would truncate before the JSON
    # (env-vllm-serve-gotchas) — prompt-CoT keeps it tractable, and call_judge
    # forces enable_thinking OFF for both judges.
    return ES.score_h3(pick_ep, cloth, model_img, result,
                       seeds=DEFAULT_SEEDS, native_think=False)


def score_h6_purevlm_122b(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    return ES.score_h3(pick_ep, cloth, model_img, result,
                       seeds=DEFAULT_SEEDS, native_think=False)


# --------------------------------------------------------------------------- #
# H7 — pure-VLM editscope + garment-MASKED pose/identity + underlayer atoms.
# OR-veto over DISTINCT modes (edit_scope | pose/identity | underlayer) so no
# double-count. Recovers the 12 known non-edit-scope FN.
# --------------------------------------------------------------------------- #
def _build_pose_underlayer_messages(cloth_path, model_path, result_path, swap=False):
    instr = (
        "You are a virtual try-on judge checking TWO things that are SEPARATE from "
        "garment edit-scope. You are shown THREE images:\n"
        "  [CLOTH]  = the reference garment.\n"
        "  [ORIGINAL PERSON] = the person BEFORE try-on.\n"
        "  [RESULT] = the generated try-on.\n"
        "Ignore whether the right garment was applied (a different check handles that). "
        "Judge ONLY:\n"
        "  identity_preserved: is it the SAME PERSON — face identity, skin tone, hair — as "
        "the [ORIGINAL PERSON], not regenerated into a different-looking person? 1=yes, 0=no.\n"
        "  pose_preserved: is the body POSE / shape / proportions / limb arrangement the same "
        "as the [ORIGINAL PERSON] (allowing the new garment)? Judge body geometry, NOT clothing. "
        "1=yes (natural, preserved), 0=no (warped/re-posed/distorted anatomy).\n"
        "  underlayer_ok: if the reference is an OPEN garment (jacket/cardigan/coat) revealing "
        "an inner top, is the inner underlayer rendered plausibly (not erased/garbled/duplicated)? "
        "1=yes OR not-applicable (no open layer), 0=clear underlayer defect.\n"
        "ANTI-BIAS: images may be in either order; rely on the labels.\n"
        "Reason in ONE short phrase each, then finish with ONE final JSON object on the last "
        'line with EXACTLY these keys (integers 0/1):\n'
        '{"identity_preserved":_, "pose_preserved":_, "underlayer_ok":_}'
    )
    cloth_url = G.img_to_data_url(cloth_path)
    model_url = G.img_to_data_url(model_path) if model_path else None
    result_url = G.img_to_data_url(result_path)
    triples = [("[CLOTH]:", cloth_url)]
    if model_url:
        triples.append(("[ORIGINAL PERSON]:", model_url))
    triples.append(("[RESULT]:", result_url))
    if swap and model_url:
        triples = [triples[0], triples[2], triples[1]]
    content = []
    for lab, url in triples:
        content.append({"type": "text", "text": lab})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instr})
    return [{"role": "user", "content": content}]


def _pose_underlayer_read(pick_ep, cloth, model_img, result, seeds):
    """Returns dict {identity:0/1/None, pose:0/1/None, underlayer:0/1/None}."""
    has_model = model_img is not None and Path(model_img).is_file()
    vi, vp, vu = [], [], []
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        try:
            msgs = _build_pose_underlayer_messages(
                cloth, model_img if has_model else None, result, swap=(i % 2 == 1))
            txt = G.call_judge(base_url, model, msgs, seed=seed,
                               enable_thinking=False, max_tokens=900)
            parsed = G.extract_last_json(txt)
            if not parsed:
                continue
            for key, bucket in (("identity_preserved", vi),
                                ("pose_preserved", vp),
                                ("underlayer_ok", vu)):
                try:
                    val = int(parsed.get(key))
                    if val in (0, 1):
                        bucket.append(val)
                except (TypeError, ValueError):
                    pass
        except Exception:
            continue

    def _maj(bucket):
        if not bucket:
            return None
        return 1 if bucket.count(1) >= bucket.count(0) else 0
    return {"identity": _maj(vi), "pose": _maj(vp), "underlayer": _maj(vu)}


def score_h7_pose_underlayer(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    # base = pure-VLM editscope (ES.score_h3); then OR-veto distinct modes.
    score, sub, ok, defect = ES.score_h3(
        pick_ep, cloth, model_img, result, seeds=DEFAULT_SEEDS, native_think=False)
    pu = _pose_underlayer_read(pick_ep, cloth, model_img, result, DEFAULT_SEEDS)
    sub = dict(sub)
    sub["identity_preserved"] = pu["identity"]
    sub["pose_preserved"] = pu["pose"]
    sub["underlayer_ok"] = pu["underlayer"]
    modes = []
    if sub.get("edit_scope_veto"):
        modes.append("edit_scope")
    pose_bad = (pu["identity"] == 0) or (pu["pose"] == 0)
    under_bad = (pu["underlayer"] == 0)
    if pose_bad:
        modes.append("pose")
        score = min(float(score), ES.EDIT_SCOPE_CAP)
        defect = True
    if under_bad:
        modes.append("underlayer")
        score = min(float(score), ES.EDIT_SCOPE_CAP)
        defect = True
    sub["defect_modes"] = modes
    return float(score), sub, ok, bool(defect)


# --------------------------------------------------------------------------- #
# H9 — single editscope+fidelity read (ES.score_h3) + the metric zoom-gate as
# the ONLY second call. edit_scope is the hard veto; the metric trip can flip it.
# --------------------------------------------------------------------------- #
def score_h9_1call(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    score, sub, ok, defect = ES.score_h3(
        pick_ep, cloth, model_img, result, seeds=DEFAULT_SEEDS, native_think=False)
    sub = dict(sub)
    nt_sub = {"dino_ok": bool(NT._DINO_OK), "nt_zoom_triggered": False,
              "nt_dino_cos": None, "nt_zoom_changed": None}
    edit_scope = int(sub.get("edit_scope", 1))
    if (edit_scope == 1 and model_img is not None and Path(model_img).is_file()):
        targets = ES._coerce_region_list(sub.get("target_regions"))
        single = len(targets) == 1 and next(iter(targets)) in ("upper", "lower")
        if single and NT._ensure_dino():
            tgt = next(iter(targets))
            nt_cat, nt_label = NT._nontarget_half(tgt)
            if nt_cat:
                try:
                    mim = Image.open(model_img).convert("RGB")
                    rim = Image.open(result).convert("RGB")
                    cos = NT._dino_cos(NT._crop_half(mim, nt_cat),
                                       NT._crop_half(rim, nt_cat))
                    nt_sub["nt_dino_cos"] = cos
                    if cos is not None and cos < NT.NT_COS_THRESHOLD:
                        nt_sub["nt_zoom_triggered"] = True
                        changed = NT._nt_zoom_reread(
                            pick_ep, model_img, result, nt_cat, nt_label, DEFAULT_SEEDS)
                        nt_sub["nt_zoom_changed"] = changed
                        if changed == 1:
                            edit_scope = 0
                            sub["edit_scope"] = 0
                            sub["escope_reason"] = "nontarget_changed_zoom"
                            score = min(float(score), ES.EDIT_SCOPE_CAP)
                            sub["edit_scope_veto"] = True
                            defect = True
                except Exception as e:
                    nt_sub["nt_zoom_err"] = str(e)
    sub.update(nt_sub)
    return float(score), sub, ok, bool(defect)


# --------------------------------------------------------------------------- #
# H4 — Family-B per-dataset calibration anchor (NO judge, NOT C-2 scored).
# For each sample computes the half-crop nt_dino_cos + nt_color_emd on BOTH
# halves (model vs result) and reports the most-changed half's margin. The
# runner-side finalize aggregates per-dataset percentile thresholds into
# thresholds.json. score is the cosine margin (diagnostic), defect always False.
# --------------------------------------------------------------------------- #
def score_h4_anchor(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    import numpy as np  # noqa: F401
    sub = {"anchor": True, "dataset_id": dataset_id, "nt_dino_cos": None,
           "nt_color_emd": None, "by_half": {}}
    if model_img is None or not Path(model_img).is_file():
        return 1.0, {**sub, "skip": "no_model_image"}, True, False
    NT._ensure_dino()
    try:
        mim = Image.open(model_img).convert("RGB")
        rim = Image.open(result).convert("RGB")
    except Exception as e:
        return 1.0, {**sub, "error": str(e)}, False, False
    cos_vals, emd_vals = [], []
    for half in ("upper", "lower"):
        cos = NT._dino_cos(NT._crop_half(mim, half), NT._crop_half(rim, half)) \
            if NT._DINO_OK else None
        emd = _color_emd_half(model_img, result, half)
        sub["by_half"][half] = {"dino_cos": cos, "color_emd": emd}
        if cos is not None:
            cos_vals.append(cos)
        if emd is not None:
            emd_vals.append(emd)
    # most-changed half = MIN cosine / MAX emd (the calibration margin).
    sub["nt_dino_cos"] = min(cos_vals) if cos_vals else None
    sub["nt_color_emd"] = max(emd_vals) if emd_vals else None
    score = float(sub["nt_dino_cos"]) if sub["nt_dino_cos"] is not None else 1.0
    return score, sub, True, False


# --------------------------------------------------------------------------- #
# MODE plumbing — the generated runner sets _CURRENT_MODE so _hybrid_core can
# read the right per-dataset calibration file (annotated anchor feeds full).
# --------------------------------------------------------------------------- #
_CURRENT_MODE = "pilot"


def set_mode(mode):
    global _CURRENT_MODE
    _CURRENT_MODE = str(mode or "pilot")


def _MODE():
    return _CURRENT_MODE


SPEC = {
    H1_ID: {"fn": score_h1, "needs_judge": True, "needs_metrics": True},
    H2_ID: {"fn": score_h2, "needs_judge": True, "needs_metrics": True},
    H3_ID: {"fn": score_h3, "needs_judge": True, "needs_metrics": True},
    H4_ID: {"fn": score_h4_anchor, "needs_judge": False, "needs_metrics": True},
    H5_ID: {"fn": score_h5_purevlm_235b, "needs_judge": True, "needs_metrics": False},
    H6_ID: {"fn": score_h6_purevlm_122b, "needs_judge": True, "needs_metrics": False},
    H7_ID: {"fn": score_h7_pose_underlayer, "needs_judge": True, "needs_metrics": False},
    H8_ID: {"fn": score_h8_lean, "needs_judge": True, "needs_metrics": True},
    H9_ID: {"fn": score_h9_1call, "needs_judge": True, "needs_metrics": True},
}


def get(combination_id):
    if combination_id not in SPEC:
        raise KeyError(f"no v15 scorer for {combination_id!r}; known={list(SPEC)}")
    return SPEC[combination_id]
