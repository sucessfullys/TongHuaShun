#!/usr/bin/env python3
"""iter_031 v16 scorer-dispatch — the 8 chosen configs as an ATTRIBUTION LADDER
on the BYTE-FROZEN v14b reference-anchored edit-scope + non-target zoom-gate
spine (_editscope.py / _ntmetric.py / _v13_eval/*), reusing the iter_030 v14b /
v15a scorers BYTE-IDENTICAL via `_v15_reference`.

Nothing here mutates the frozen modules; every v16 config is composed from their
public atoms so H1 (v14b) and H2 (v15a) reproduce iter_030 EXACTLY and the only
moving variable per new config is the documented additive channel (confound
control). New channels (P1 garment-fidelity VLM read / P2 target-region metric
assist / P3 pose+underlayer atoms / the consolidated coverage read) are appended
on top of the frozen spine and OR-unioned, with per-sample `defect_modes` +
per-channel `pass_flags` so the next Stage 9 can prune.

Dispatch contract (read by the generated runners + era_eval_common):
    SPEC[combination_id] = {
        "fn": score(pick_ep, dataset_id, sample_key, cloth, model_img, result)
              -> (score: float, sub: dict, ok: bool, defect: bool),
        "needs_judge": bool,   # Family-A/hybrid True; Family-B anchor False
        "needs_metrics": bool, # loads DINO (every hybrid spine + the anchor + P2)
    }

config -> hypothesis (iter_031 brief):
  H1 hybrid-editscope-zoomgate-step0-122b-v14b-ntmetric           = V15.score_h1 EXACT (FROZEN)
  H2 hybrid-editscope-zoomgate-confirm-step0-122b-v15a            = V15.score_h2 EXACT (FROZEN)
  H3 hybrid-editscope-zoomgate-fidelity-only-step0-122b-v16-lean  = v14b spine + P1 fidelity VLM read
  H4 hybrid-...-fidelity-targetmetric-step0-122b-v16-fidtm        = v16-lean + P2 target-metric escalate
  H5 hybrid-...-fidelity-pose-underlayer-union-step0-122b-v16     = v16-fidtm + P3 pose + underlayer (OR-union)
  H6 hybrid-editscope-zoomgate-122b-v16-coverage-1read            = v14b spine + ONE consolidated coverage read
  H7 hybrid-...-fidelity-step0-35b-v16-cheapleg                   = same fn as v16-fidtm (judge = 35B endpoint)
  H8 target-fidelity-calibration-anchor-b-v1                      = Family-B metric-only calibration anchor

Metric subprocesses (DINO cosine + Lab color-EMD) run IN-PROCESS but the
generated runners force CUDA_VISIBLE_DEVICES='' so torch.cuda is invisible and
the metric stays strictly on CPU, OFF the judge GPUs 4-7 (brief MF / debiasing).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from PIL import Image

from _v13_eval import gf_common as G
from _v13_eval import metric_common as M  # noqa: F401  (geometry helpers / crop)
import _editscope as ES
import _ntmetric as NT
import _v15_reference as V15            # BYTE-FROZEN v14b (score_h1) + v15a (score_h2) spine

DEFAULT_SEEDS = [42]

# ---- combination ids (iter_031 brief candidate_configs) ----
V14B_ID    = "hybrid-editscope-zoomgate-step0-122b-v14b-ntmetric"
V15A_ID    = "hybrid-editscope-zoomgate-confirm-step0-122b-v15a"
LEAN_ID    = "hybrid-editscope-zoomgate-fidelity-only-step0-122b-v16-lean"
FIDTM_ID   = "hybrid-editscope-zoomgate-fidelity-targetmetric-step0-122b-v16-fidtm"
UNION_ID   = "hybrid-editscope-zoomgate-fidelity-pose-underlayer-union-step0-122b-v16"
COVERAGE_ID = "hybrid-editscope-zoomgate-122b-v16-coverage-1read"
CHEAP35_ID = "hybrid-editscope-zoomgate-fidelity-step0-35b-v16-cheapleg"
ANCHOR_ID  = "target-fidelity-calibration-anchor-b-v1"


# --------------------------------------------------------------------------- #
# MODE plumbing — the generated runner sets the mode so the P2 trigger can read
# the right per-dataset percentile-rank calibration file (the anchor's
# thresholds.json feeds the fidtm/union P2 escalate decision). Propagated to
# _v15_reference too so the frozen v15a path reads the correct mode.
# --------------------------------------------------------------------------- #
_CURRENT_MODE = "pilot"


def set_mode(mode):
    global _CURRENT_MODE
    _CURRENT_MODE = str(mode or "pilot")
    try:
        V15.set_mode(_CURRENT_MODE)   # keep the frozen v15a path's mode in sync
    except Exception:
        pass


def _MODE():
    return _CURRENT_MODE


# --------------------------------------------------------------------------- #
# TARGET-region metric thresholds (P2). Written by the H8 anchor's per-dataset
# percentile-rank finalize (finalize_anchor_thresholds below). v16-fidtm/union
# read them; absent -> conservative module defaults so escalation is bounded but
# non-zero (the brief's percentile-rank-NOT-fixed-tau is realized through the
# anchor calibration; defaults are only a no-anchor fallback). FOLD-IN to
# report-only downstream if the realized escalation rate is unhelpful.
# --------------------------------------------------------------------------- #
_TGT_THRESH_CACHE = {}
_TGT_THRESH_LOCK = threading.Lock()
DEFAULT_TGT_EMD_THRESHOLD = 0.120      # WB-norm color-EMD: above -> escalate
DEFAULT_TGT_DINO_THRESHOLD = 0.55      # DINO cosine: below -> escalate


def _iter_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent   # configs/../../ = <iter>


def _load_target_thresholds(mode: str) -> dict:
    """Per-dataset TARGET-region percentile cuts from the H8 anchor (cached).
    Shape: {dataset_id: {"emd": float, "dino": float}}. {} when the anchor has
    not finalized yet (P2 then uses the module defaults)."""
    key = mode
    with _TGT_THRESH_LOCK:
        if key in _TGT_THRESH_CACHE:
            return _TGT_THRESH_CACHE[key]
        out = {}
        for m in (mode, "annotated"):  # an annotated-mode anchor can calibrate a later full run
            p = _iter_dir() / "experiments" / "results" / m / ANCHOR_ID / "thresholds.json"
            if p.is_file():
                try:
                    out = json.loads(p.read_text()).get("per_dataset", {}) or {}
                    if out:
                        break
                except Exception:
                    out = {}
        _TGT_THRESH_CACHE[key] = out
        return out


def _target_thresholds_for(dataset_id: str, mode: str):
    per = _load_target_thresholds(mode)
    d = per.get(dataset_id) or {}
    return (float(d.get("emd", DEFAULT_TGT_EMD_THRESHOLD)),
            float(d.get("dino", DEFAULT_TGT_DINO_THRESHOLD)))


# --------------------------------------------------------------------------- #
# image-derived TARGET-garment crop (STEP-A target band; NO GroundingDINO+SAM2)
# --------------------------------------------------------------------------- #
def _target_cat_from_sub(sub) -> str:
    """The STEP-A target category the frozen spine already classified
    (image-derived; never the path). Single upper/lower -> that band; dress or
    multi/uncertain -> the widest dress box (so the crop never clips)."""
    tr = sub.get("escope_target_regions") or sub.get("target_regions") or []
    if isinstance(tr, str):
        tr = [tr]
    norm = ES._coerce_region_list(tr)
    if len(norm) == 1:
        c = next(iter(norm))
        if c in ("upper", "lower", "dress"):
            return c
    if "dress" in norm:
        return "dress"
    return "dress"   # widest fallback — never clips the garment


def _garment_crop(img_path_or_pil, target_cat):
    """Crop the TARGET garment band out of a full-person frame by STEP-A category."""
    im = img_path_or_pil if isinstance(img_path_or_pil, Image.Image) \
        else Image.open(img_path_or_pil).convert("RGB")
    im = im.convert("RGB")
    w, h = im.size
    cat = target_cat if target_cat in ("upper", "lower", "dress") else "dress"
    return M.crop(im, G._target_box(w, h, cat))


def _target_metrics(cloth_path, result_path, target_cat):
    """Masked TARGET-region change-detection margins: WB-norm Lab color-EMD +
    DINO cosine between the OUTPUT garment crop (STEP-A band) and input_cloth.
    Returns (emd or None, dino_cos or None). CPU-only (the runner pins
    CUDA_VISIBLE_DEVICES=''). A paired SAME-region trigger source, never a grade."""
    import numpy as np
    emd = cos = None
    try:
        cloth = Image.open(cloth_path).convert("RGB")
        rcrop = _garment_crop(result_path, target_cat)
        emd = V15._channel_emd(np.asarray(cloth), np.asarray(rcrop))
    except Exception:
        emd = None
    try:
        if NT._ensure_dino():
            cloth = Image.open(cloth_path).convert("RGB")
            rcrop = _garment_crop(result_path, target_cat)
            cos = NT._dino_cos(cloth, rcrop)
    except Exception:
        cos = None
    return emd, cos


# --------------------------------------------------------------------------- #
# P1 — reference-anchored garment-FIDELITY VLM read (output garment vs cloth).
# DEFAULT-PASS unless a CLEAR mismatch. Anchored boundary rubric in the prompt.
# --------------------------------------------------------------------------- #
def _build_fidelity_messages(cloth_path, garment_crop, swap=False, evidence=None):
    instr = (
        "You are a virtual try-on GARMENT-FIDELITY judge. You are shown TWO images:\n"
        "  [REFERENCE CLOTH] = the flat reference garment the result was SUPPOSED to wear.\n"
        "  [RESULT GARMENT]  = a crop of the SAME garment as actually WORN in the try-on result.\n"
        "Compare the WORN garment against the REFERENCE on these aspects and flag ONLY a "
        "CLEAR, nameable mismatch (DEFAULT to NO-mismatch when unsure):\n"
        "  color    — a nameable HUE-FAMILY shift (e.g. red->blue, navy->black, white->beige).\n"
        "  pattern  — a PATTERN-CLASS change (solid<->striped<->floral<->checked; print swapped).\n"
        "  logo     — a logo/printed-text clearly PRESENT in the reference is MISSING or REPLACED.\n"
        "  closure  — a different CLOSURE/belt/tie type (belt removed/added, buttons->zip, knot gone).\n"
        "  trim     — cuffs / collar / neckline / hem TRIM clearly different in style.\n"
        "  material — a different MATERIAL CLASS (denim<->silk, knit<->leather, matte<->shiny).\n"
        "STRICT ANTI-FALSE-ALARM: lighting, shadow, fabric drape, folds, wrinkles, worn-on-body "
        "scale, crop, JPEG artifacts and minor saturation/brightness shifts are NOT mismatches "
        "— grade 0 (no mismatch) for any of those. Flag 1 ONLY for a clear, obvious, nameable "
        "difference.\n"
    )
    if evidence:
        instr += "HINT: " + str(evidence) + "\n"
    instr += (
        "Reason in ONE short phrase, then finish with ONE final JSON object on the last line "
        "with EXACTLY these keys (each integer 0=match / 1=CLEAR mismatch):\n"
        '{"color":_, "pattern":_, "logo":_, "closure":_, "trim":_, "material":_, '
        '"fidelity_defect":_}'
    )
    cloth_url = G.img_to_data_url(cloth_path)
    crop_url = G.img_to_data_url(garment_crop, max_side=(1152 if evidence else 768))
    pair = [("[REFERENCE CLOTH]:", cloth_url), ("[RESULT GARMENT]:", crop_url)]
    if swap:
        pair = [pair[1], pair[0]]
    content = []
    for lab, url in pair:
        content.append({"type": "text", "text": lab})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instr})
    return [{"role": "user", "content": content}]


_FID_SUBKEYS = ("color", "pattern", "logo", "closure", "trim", "material")


def _fidelity_read(pick_ep, cloth, result, target_cat, seeds, hires=False, evidence=None):
    """One reference-anchored fidelity read on the STEP-A target crop. Returns
    (fidelity_defect: bool|None, detail: dict). None on total parse failure ->
    DEFAULT-PASS (the caller treats None as no defect, never fabricates one)."""
    try:
        crop = _garment_crop(result, target_cat)
    except Exception as e:
        return None, {"fidelity_read_err": str(e)}
    sub_votes = {k: [] for k in _FID_SUBKEYS}
    defect_votes = []
    ev = evidence if hires else None
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        try:
            msgs = _build_fidelity_messages(cloth, crop, swap=(i % 2 == 1), evidence=ev)
            txt = G.call_judge(base_url, model, msgs, seed=seed,
                               enable_thinking=False, max_tokens=1024)
            parsed = G.extract_last_json(txt)
            if not parsed:
                continue
            for k in _FID_SUBKEYS:
                v = parsed.get(k)
                try:
                    v = int(v)
                except (TypeError, ValueError):
                    v = None
                if v in (0, 1):
                    sub_votes[k].append(v)
            fd = parsed.get("fidelity_defect")
            try:
                fd = int(fd)
            except (TypeError, ValueError):
                fd = None
            # default-PASS aggregation: a defect iff the model's final flag is 1
            # OR any named sub-aspect is a CLEAR mismatch.
            any_sub = any(sub_votes[k][-1:] == [1] for k in _FID_SUBKEYS)
            if fd in (0, 1) or any_sub:
                defect_votes.append(1 if (fd == 1 or any_sub) else 0)
        except Exception:
            continue
    detail = {k: (1 if v.count(1) >= max(1, v.count(0)) and 1 in v else 0) if v else None
              for k, v in sub_votes.items()}
    detail["hires"] = bool(hires)
    if not defect_votes:
        return None, detail
    fidelity_defect = defect_votes.count(1) > defect_votes.count(0)
    return bool(fidelity_defect), detail


# --------------------------------------------------------------------------- #
# P3a — garment-MASKED pose/identity atom (mask = STEP-A target band complement,
# painted out so a legit garment change can't read as identity drift). Fires
# ONLY on a CLEAR posture/limb change NOT attributable to garment fit/silhouette.
# --------------------------------------------------------------------------- #
def _mask_target_band(pil_img, target_cat):
    """Paint OUT the STEP-A target garment band so the read sees only the
    complement (head / limbs / background) for the pose+identity judgement."""
    w, h = pil_img.size
    cat = target_cat if target_cat in ("upper", "lower", "dress") else "dress"
    return M.paint_out(pil_img.copy(), G._target_box(w, h, cat))


def _build_pose_messages(model_masked, result_masked, swap=False):
    instr = (
        "You are checking ONLY the PERSON's body in a virtual try-on; the GARMENT region has "
        "been GREYED OUT so you cannot see the clothing. You are shown the SAME person twice "
        "with the garment masked:\n"
        "  [ORIGINAL PERSON] = before try-on.\n"
        "  [RESULT PERSON]   = after try-on.\n"
        "Judge body geometry ONLY (head/face identity, limb arrangement, stance, proportions). "
        "IGNORE the greyed garment region entirely.\n"
        "  identity_changed = 1 ONLY if it became a CLEARLY DIFFERENT-LOOKING person "
        "(face identity / skin tone / hair regenerated into someone else), else 0.\n"
        "  pose_changed = 1 ONLY for a CLEAR posture / limb / anatomy change (warped torso, "
        "extra/missing/melted limbs or hands, grossly re-posed body) that is NOT explained by the "
        "new garment's fit or silhouette, else 0. A different natural stance/turn/crop is 0.\n"
        "DEFAULT to 0 (preserved) unless the change is clear and obvious.\n"
        "ANTI-BIAS: images may be in either order; rely on the labels.\n"
        "Reason in ONE short phrase, then finish with ONE final JSON object on the last line "
        'with EXACTLY these keys (integers 0/1):\n{"identity_changed":_, "pose_changed":_}'
    )
    m_url = G.img_to_data_url(model_masked)
    r_url = G.img_to_data_url(result_masked)
    pair = [("[ORIGINAL PERSON]:", m_url), ("[RESULT PERSON]:", r_url)]
    if swap:
        pair = [pair[1], pair[0]]
    content = []
    for lab, url in pair:
        content.append({"type": "text", "text": lab})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instr})
    return [{"role": "user", "content": content}]


def _pose_identity_read(pick_ep, model_img, result, target_cat, seeds):
    """Garment-masked pose/identity atom. Returns pose_defect bool (None-safe ->
    default-PASS). Requires the ORIGINAL PERSON image."""
    if model_img is None or not Path(model_img).is_file():
        return None
    try:
        mim = _mask_target_band(Image.open(model_img).convert("RGB"), target_cat)
        rim = _mask_target_band(Image.open(result).convert("RGB"), target_cat)
    except Exception:
        return None
    votes = []
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        try:
            msgs = _build_pose_messages(mim, rim, swap=(i % 2 == 1))
            txt = G.call_judge(base_url, model, msgs, seed=seed,
                               enable_thinking=False, max_tokens=768)
            parsed = G.extract_last_json(txt)
            if not parsed:
                continue
            bad = 0
            for k in ("identity_changed", "pose_changed"):
                try:
                    if int(parsed.get(k)) == 1:
                        bad = 1
                except (TypeError, ValueError):
                    pass
            votes.append(bad)
        except Exception:
            continue
    if not votes:
        return None
    return votes.count(1) > votes.count(0)


# --------------------------------------------------------------------------- #
# P3b — UNDERLAYER atom (full frame). Fires ONLY when the inner garment
# (neckline / hem / open-jacket reveal) is VISIBLY different vs input_model.
# --------------------------------------------------------------------------- #
def _build_underlayer_messages(model_path, result_path, swap=False):
    instr = (
        "You are checking ONE thing in a virtual try-on: the INNER UNDERLAYER garment "
        "(the top/shirt visible UNDER an open jacket/cardigan/coat, or at the neckline/hem). "
        "You are shown:\n"
        "  [ORIGINAL PERSON] = before try-on.\n"
        "  [RESULT] = after try-on.\n"
        "underlayer_changed = 1 ONLY if an INNER underlayer garment that is visible in the "
        "[ORIGINAL PERSON] is CLEARLY different (recolored / replaced / erased / garbled / "
        "duplicated) in the [RESULT]. If there is NO visible inner layer (no open outer garment), "
        "or the inner layer looks the same, grade 0. DEFAULT to 0 unless the change is clear.\n"
        "IGNORE the OUTER target garment (a different check handles that), pose, crop, lighting.\n"
        "ANTI-BIAS: images may be in either order; rely on the labels.\n"
        "Reason in ONE short phrase, then finish with ONE final JSON object on the last line "
        'with EXACTLY this key (integer 0/1):\n{"underlayer_changed":_}'
    )
    m_url = G.img_to_data_url(model_path)
    r_url = G.img_to_data_url(result_path)
    pair = [("[ORIGINAL PERSON]:", m_url), ("[RESULT]:", r_url)]
    if swap:
        pair = [pair[1], pair[0]]
    content = []
    for lab, url in pair:
        content.append({"type": "text", "text": lab})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instr})
    return [{"role": "user", "content": content}]


def _underlayer_read(pick_ep, model_img, result, seeds):
    """Full-frame underlayer atom. Returns underlayer_defect bool (None-safe ->
    default-PASS)."""
    if model_img is None or not Path(model_img).is_file():
        return None
    votes = []
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        try:
            msgs = _build_underlayer_messages(model_img, result, swap=(i % 2 == 1))
            txt = G.call_judge(base_url, model, msgs, seed=seed,
                               enable_thinking=False, max_tokens=640)
            parsed = G.extract_last_json(txt)
            if not parsed:
                continue
            try:
                votes.append(1 if int(parsed.get("underlayer_changed")) == 1 else 0)
            except (TypeError, ValueError):
                continue
        except Exception:
            continue
    if not votes:
        return None
    return votes.count(1) > votes.count(0)


# --------------------------------------------------------------------------- #
# v16-coverage-1read — ONE consolidated structured read decomposing fidelity
# {color/pattern/belt-cuffs/material} + pose + underlayer into named atomic Y/N
# in a SINGLE judge call (keeps the SEPARATE frozen edit-scope read + zoom-gate).
# --------------------------------------------------------------------------- #
def _build_coverage_messages(cloth_path, model_path, result_path, swap=False):
    instr = (
        "You are a virtual try-on coverage judge. You are shown THREE images:\n"
        "  [REFERENCE CLOTH] = the flat reference garment.\n"
        "  [ORIGINAL PERSON] = the person BEFORE try-on.\n"
        "  [RESULT] = the generated try-on.\n"
        "Answer these atomic checks. Flag 1 ONLY for a CLEAR, obvious, nameable defect; DEFAULT "
        "to 0 when unsure. IGNORE lighting / shadow / drape / folds / pose / crop / scale / "
        "minor saturation shifts.\n"
        "FIDELITY of the WORN target garment vs [REFERENCE CLOTH]:\n"
        "  fid_color      = 1 if a nameable HUE-FAMILY shift, else 0.\n"
        "  fid_pattern    = 1 if a PATTERN-CLASS change (solid/striped/floral/print swap), else 0.\n"
        "  fid_belt_cuffs = 1 if a belt/tie/closure or cuff/collar/trim detail clearly present in "
        "the reference is missing/replaced/different, else 0.\n"
        "  fid_material   = 1 if a different MATERIAL CLASS (denim/silk/knit/leather), else 0.\n"
        "PERSON (vs [ORIGINAL PERSON], judge body NOT clothing):\n"
        "  pose_changed = 1 ONLY for a CLEAR posture/limb/anatomy change NOT attributable to the "
        "new garment's fit/silhouette (or a clearly different-looking person), else 0.\n"
        "  underlayer_changed = 1 ONLY if a visible INNER underlayer garment (under an open "
        "jacket / at the neckline/hem) is clearly recolored/replaced/erased vs the original, else 0.\n"
        "ANTI-BIAS: images may be in either order; rely on the labels.\n"
        "Reason BRIEFLY (one short phrase per check), then finish with ONE final JSON object on the "
        "last line with EXACTLY these keys (integers 0/1):\n"
        '{"fid_color":_, "fid_pattern":_, "fid_belt_cuffs":_, "fid_material":_, '
        '"pose_changed":_, "underlayer_changed":_}'
    )
    cloth_url = G.img_to_data_url(cloth_path)
    model_url = G.img_to_data_url(model_path) if model_path else None
    result_url = G.img_to_data_url(result_path)
    triples = [("[REFERENCE CLOTH]:", cloth_url)]
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


_COV_FID_KEYS = ("fid_color", "fid_pattern", "fid_belt_cuffs", "fid_material")


def _coverage_read(pick_ep, cloth, model_img, result, seeds):
    """ONE consolidated coverage read. Returns
    {fidelity_defect, pose_defect, underlayer_defect, detail} (each defect
    bool|None; None -> default-PASS for that bucket)."""
    has_model = model_img is not None and Path(model_img).is_file()
    fid_votes, pose_votes, under_votes = [], [], []
    detail = {k: [] for k in _COV_FID_KEYS}
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        try:
            msgs = _build_coverage_messages(
                cloth, model_img if has_model else None, result, swap=(i % 2 == 1))
            txt = G.call_judge(base_url, model, msgs, seed=seed,
                               enable_thinking=False, max_tokens=1024)
            parsed = G.extract_last_json(txt)
            if not parsed:
                continue
            fid_bad = 0
            for k in _COV_FID_KEYS:
                try:
                    v = int(parsed.get(k))
                except (TypeError, ValueError):
                    v = None
                if v in (0, 1):
                    detail[k].append(v)
                    if v == 1:
                        fid_bad = 1
            fid_votes.append(fid_bad)
            try:
                pose_votes.append(1 if int(parsed.get("pose_changed")) == 1 else 0)
            except (TypeError, ValueError):
                pass
            try:
                under_votes.append(1 if int(parsed.get("underlayer_changed")) == 1 else 0)
            except (TypeError, ValueError):
                pass
        except Exception:
            continue

    def _maj(votes):
        if not votes:
            return None
        return votes.count(1) > votes.count(0)
    return {
        "fidelity_defect": _maj(fid_votes),
        "pose_defect": _maj(pose_votes),
        "underlayer_defect": _maj(under_votes),
        "detail": {k: (1 if v.count(1) >= max(1, v.count(0)) and 1 in v else 0) if v else None
                   for k, v in detail.items()},
    }


# --------------------------------------------------------------------------- #
# the v14b frozen spine (BYTE-IDENTICAL via V15.score_h1 == NT.score_h2)
# --------------------------------------------------------------------------- #
def _v14b_spine(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    """Run the byte-frozen v14b reference-anchored edit-scope read + non-target
    DINO zoom-gate. Returns (score, sub, ok, defect) — sub carries the STEP-A
    target_regions every new channel crops against."""
    return V15.score_h1(pick_ep, dataset_id, sample_key, cloth, model_img, result)


def _aggregate_union(score, sub, ok, spine_defect, *, fidelity_defect=None,
                     pose_defect=None, underlayer_defect=None, extra=None):
    """OR-union the frozen-spine verdict (labelled `edit_scope`) with the new
    high-precision channels; emit defect_modes + per-channel pass_flags so the
    next Stage 9 can prune. score is capped at the edit-scope veto cap on any
    defect (a hard parser-side min, never trusted to the model)."""
    sub = dict(sub)
    if extra:
        sub.update(extra)
    es_defect = bool(spine_defect)
    fid = bool(fidelity_defect)
    pose = bool(pose_defect)
    under = bool(underlayer_defect)
    modes = []
    if es_defect:
        modes.append("edit_scope")
    if fid:
        modes.append("fidelity")
    if pose:
        modes.append("pose")
    if under:
        modes.append("underlayer")
    defect = es_defect or fid or pose or under
    sub["fidelity_defect"] = (None if fidelity_defect is None else fid)
    sub["pose_defect"] = (None if pose_defect is None else pose)
    sub["underlayer_defect"] = (None if underlayer_defect is None else under)
    sub["defect_modes"] = modes
    sub["pass_flags"] = {
        "edit_scope": not es_defect,
        "fidelity": (None if fidelity_defect is None else (not fid)),
        "pose": (None if pose_defect is None else (not pose)),
        "underlayer": (None if underlayer_defect is None else (not under)),
    }
    if defect:
        score = min(float(score), ES.EDIT_SCOPE_CAP)
    return float(score), sub, bool(ok), bool(defect)


# --------------------------------------------------------------------------- #
# H1 / H2 — FROZEN baselines (byte-identical to iter_030 v14b / v15a)
# --------------------------------------------------------------------------- #
def score_v14b(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    return V15.score_h1(pick_ep, dataset_id, sample_key, cloth, model_img, result)


def score_v15a(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    return V15.score_h2(pick_ep, dataset_id, sample_key, cloth, model_img, result)


# --------------------------------------------------------------------------- #
# H3 — v16-lean: v14b spine + P1 garment-fidelity VLM read (single channel)
# --------------------------------------------------------------------------- #
def score_v16_lean(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    score, sub, ok, spine_defect = _v14b_spine(
        pick_ep, dataset_id, sample_key, cloth, model_img, result)
    tgt = _target_cat_from_sub(sub)
    fid_defect, fid_detail = _fidelity_read(pick_ep, cloth, result, tgt, DEFAULT_SEEDS)
    return _aggregate_union(
        score, sub, ok, spine_defect, fidelity_defect=fid_defect,
        extra={"fidelity_detail": fid_detail, "target_cat": tgt})


# --------------------------------------------------------------------------- #
# H4 / H7 — v16-fidtm: v16-lean + P2 TARGET-region metric escalate-only re-read.
# (H7 cheapleg shares this fn; only the served judge differs — set by the runner
# endpoint.) The metric NEVER grades; it only routes a higher-res P1 re-read.
# --------------------------------------------------------------------------- #
def score_v16_fidtm(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    score, sub, ok, spine_defect = _v14b_spine(
        pick_ep, dataset_id, sample_key, cloth, model_img, result)
    tgt = _target_cat_from_sub(sub)
    # P1 base fidelity read.
    fid_defect, fid_detail = _fidelity_read(pick_ep, cloth, result, tgt, DEFAULT_SEEDS)
    # P2 TARGET-region metric trigger (CPU). Percentile-rank cut from the H8
    # anchor (else conservative defaults). Escalate -> higher-res P1 re-read.
    emd, cos = _target_metrics(cloth, result, tgt)
    emd_cut, dino_cut = _target_thresholds_for(dataset_id, _MODE())
    escalate = ((emd is not None and emd > emd_cut)
                or (cos is not None and cos < dino_cut))
    p2_reread_used = False
    if escalate:
        ev = (f"An automated TARGET-region color/texture metric ranked this garment in the "
              f"HIGH-CHANGE band vs the reference (color_emd={emd}, dino_cos={cos}); look "
              f"closely at hue family, pattern class, logo, closure and material.")
        fid2, fid_detail2 = _fidelity_read(
            pick_ep, cloth, result, tgt, DEFAULT_SEEDS, hires=True, evidence=ev)
        if fid2 is not None:
            fid_defect, fid_detail = fid2, fid_detail2
            p2_reread_used = True
    extra = {"fidelity_detail": fid_detail, "target_cat": tgt,
             "target_color_emd": emd, "target_dino_cos": cos,
             "p2_emd_cut": emd_cut, "p2_dino_cut": dino_cut,
             "p2_escalated": bool(escalate), "p2_reread_used": bool(p2_reread_used)}
    return _aggregate_union(
        score, sub, ok, spine_defect, fidelity_defect=fid_defect, extra=extra)


# --------------------------------------------------------------------------- #
# H5 — v16-union FLAGSHIP: v16-fidtm + P3 pose/identity + underlayer (OR-union)
# --------------------------------------------------------------------------- #
def score_v16_union(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    score, sub, ok, spine_defect = _v14b_spine(
        pick_ep, dataset_id, sample_key, cloth, model_img, result)
    tgt = _target_cat_from_sub(sub)
    fid_defect, fid_detail = _fidelity_read(pick_ep, cloth, result, tgt, DEFAULT_SEEDS)
    emd, cos = _target_metrics(cloth, result, tgt)
    emd_cut, dino_cut = _target_thresholds_for(dataset_id, _MODE())
    escalate = ((emd is not None and emd > emd_cut)
                or (cos is not None and cos < dino_cut))
    p2_reread_used = False
    if escalate:
        ev = (f"An automated TARGET-region color/texture metric ranked this garment in the "
              f"HIGH-CHANGE band vs the reference (color_emd={emd}, dino_cos={cos}); look "
              f"closely at hue family, pattern class, logo, closure and material.")
        fid2, fid_detail2 = _fidelity_read(
            pick_ep, cloth, result, tgt, DEFAULT_SEEDS, hires=True, evidence=ev)
        if fid2 is not None:
            fid_defect, fid_detail = fid2, fid_detail2
            p2_reread_used = True
    # P3 high-precision pose/identity (garment-masked) + underlayer (full frame).
    pose_defect = _pose_identity_read(pick_ep, model_img, result, tgt, DEFAULT_SEEDS)
    underlayer_defect = _underlayer_read(pick_ep, model_img, result, DEFAULT_SEEDS)
    extra = {"fidelity_detail": fid_detail, "target_cat": tgt,
             "target_color_emd": emd, "target_dino_cos": cos,
             "p2_emd_cut": emd_cut, "p2_dino_cut": dino_cut,
             "p2_escalated": bool(escalate), "p2_reread_used": bool(p2_reread_used)}
    return _aggregate_union(
        score, sub, ok, spine_defect, fidelity_defect=fid_defect,
        pose_defect=pose_defect, underlayer_defect=underlayer_defect, extra=extra)


# --------------------------------------------------------------------------- #
# H6 — v16-coverage-1read: v14b spine + ONE consolidated coverage read (fidelity
# decomposed + pose + underlayer) + a REPORT-ONLY target-region color-EMD.
# --------------------------------------------------------------------------- #
def score_v16_coverage(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    score, sub, ok, spine_defect = _v14b_spine(
        pick_ep, dataset_id, sample_key, cloth, model_img, result)
    tgt = _target_cat_from_sub(sub)
    cov = _coverage_read(pick_ep, cloth, model_img, result, DEFAULT_SEEDS)
    # report-only target-region color-EMD (degrades to whole-image inside
    # _garment_crop's dress fallback; never gates).
    emd, _cos = _target_metrics(cloth, result, tgt)
    extra = {"coverage_detail": cov["detail"], "target_cat": tgt,
             "report_target_color_emd": emd, "consolidated_1read": True}
    return _aggregate_union(
        score, sub, ok, spine_defect,
        fidelity_defect=cov["fidelity_defect"],
        pose_defect=cov["pose_defect"],
        underlayer_defect=cov["underlayer_defect"], extra=extra)


# --------------------------------------------------------------------------- #
# H8 — Family-B TARGET-region metric calibration anchor (NO judge, NOT C-2
# scored as a shippable). Deterministic center/dress-band crop heuristic (no VLM
# STEP-A region available in a pure-metric config). sub_scores carry the metric
# values so the auto-validator can scope-gate it; defect via a simple threshold.
# --------------------------------------------------------------------------- #
ANCHOR_EMD_DEFECT = 0.120        # color-EMD above this -> defect (calibration baseline)
ANCHOR_DINO_DEFECT = 0.55        # dino cosine below this -> defect


def score_anchor(pick_ep, dataset_id, sample_key, cloth, model_img, result):
    sub = {"anchor": True, "dataset_id": dataset_id,
           "target_color_emd": None, "target_dino_cos": None,
           "crop_heuristic": "dress_center_band"}
    # pure-metric: no STEP-A region -> widest dress band center crop heuristic.
    emd, cos = _target_metrics(cloth, result, "dress")
    sub["target_color_emd"] = emd
    sub["target_dino_cos"] = cos
    defect = bool((emd is not None and emd > ANCHOR_EMD_DEFECT)
                  or (cos is not None and cos < ANCHOR_DINO_DEFECT))
    sub["defect_modes"] = ["fidelity_metric"] if defect else []
    # diagnostic score = dino cosine (1.0 when unavailable); ok unless both metrics failed.
    score = float(cos) if cos is not None else 1.0
    ok = (emd is not None) or (cos is not None)
    return score, sub, bool(ok), bool(defect)


# --------------------------------------------------------------------------- #
# H8 percentile-rank finalize — turn the anchor's per-sample TARGET-region
# margins into per-dataset escalate cuts for the P2 trigger (top ~PCT% escalate).
# Optional: the Stage-6 skill may call this between the anchor and v16-fidtm; P2
# falls back to module defaults when thresholds.json is absent.
# --------------------------------------------------------------------------- #
def finalize_anchor_thresholds(mode, emd_pct=70.0, dino_pct=30.0):
    """Read the anchor's per-sample scores for `mode`, compute per-dataset
    percentile cuts (EMD high-tail / DINO low-tail so ~top 30% escalate), and
    write results/<mode>/<ANCHOR_ID>/thresholds.json. Pure bookkeeping."""
    import statistics  # noqa: F401
    base = _iter_dir() / "experiments" / "results" / mode / ANCHOR_ID
    rows = []
    if base.is_dir():
        for f in sorted(base.glob("scores.*.jsonl")):
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    def _pct(vals, p):
        vals = sorted(v for v in vals if v is not None)
        if not vals:
            return None
        if len(vals) == 1:
            return vals[0]
        k = (len(vals) - 1) * (p / 100.0)
        lo = int(k)
        hi = min(lo + 1, len(vals) - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)

    by_ds = {}
    for r in rows:
        if not r.get("ok"):
            continue
        ss = r.get("sub_scores") or {}
        ds = r.get("dataset_id", "?")
        d = by_ds.setdefault(ds, {"emd": [], "dino": []})
        if ss.get("target_color_emd") is not None:
            d["emd"].append(float(ss["target_color_emd"]))
        if ss.get("target_dino_cos") is not None:
            d["dino"].append(float(ss["target_dino_cos"]))
    per_dataset = {}
    for ds, d in by_ds.items():
        emd_cut = _pct(d["emd"], emd_pct)
        dino_cut = _pct(d["dino"], dino_pct)
        per_dataset[ds] = {
            "emd": emd_cut if emd_cut is not None else DEFAULT_TGT_EMD_THRESHOLD,
            "dino": dino_cut if dino_cut is not None else DEFAULT_TGT_DINO_THRESHOLD,
            "n": max(len(d["emd"]), len(d["dino"])),
        }
    out = {"mode": mode, "anchor_id": ANCHOR_ID,
           "emd_pct": emd_pct, "dino_pct": dino_pct, "per_dataset": per_dataset}
    base.mkdir(parents=True, exist_ok=True)
    (base / "thresholds.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
SPEC = {
    V14B_ID:     {"fn": score_v14b,        "needs_judge": True,  "needs_metrics": True},
    V15A_ID:     {"fn": score_v15a,        "needs_judge": True,  "needs_metrics": True},
    LEAN_ID:     {"fn": score_v16_lean,    "needs_judge": True,  "needs_metrics": True},
    FIDTM_ID:    {"fn": score_v16_fidtm,   "needs_judge": True,  "needs_metrics": True},
    UNION_ID:    {"fn": score_v16_union,   "needs_judge": True,  "needs_metrics": True},
    COVERAGE_ID: {"fn": score_v16_coverage, "needs_judge": True, "needs_metrics": True},
    CHEAP35_ID:  {"fn": score_v16_fidtm,   "needs_judge": True,  "needs_metrics": True},
    ANCHOR_ID:   {"fn": score_anchor,      "needs_judge": False, "needs_metrics": True},
}


def get(combination_id):
    if combination_id not in SPEC:
        raise KeyError(f"no v16 scorer for {combination_id!r}; known={list(SPEC)}")
    return SPEC[combination_id]


if __name__ == "__main__":
    # optional: `_v16.py finalize-anchor <mode>` — compute the P2 percentile cuts.
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "finalize-anchor":
        print(json.dumps(finalize_anchor_thresholds(sys.argv[2]), indent=2))
    else:
        print(json.dumps(sorted(SPEC.keys()), indent=2))
