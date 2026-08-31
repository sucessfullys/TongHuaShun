#!/usr/bin/env python3
"""iter_029 H2 — H1 PLUS a DINOv2 non-target-region cosine as an ESCALATE-ONLY
zoom TRIGGER (never a vote).

The reference-anchored STEP-A/B/C read (H1) catches obvious non-target garment
changes, but a SUBTLE non-target collateral edit (a small recolor / texture
change on the half the reference cloth does NOT cover) can read as "unchanged"
at full-frame resolution. H2 adds a cheap objective trigger:

  When STEP-A's target is a SINGLE region (upper OR lower — not a dress, which
  covers both), compute a DINOv2 cosine between the input_model and the result on
  the NON-target half-crop (the half OUTSIDE the STEP-A target band). If that
  cosine < a percentile-calibrated threshold (default 0.92), the non-target half
  changed enough to warrant a closer look, so RE-READ STEP-B on a zoomed crop of
  that non-target half before grading edit_scope.

The metric is a TRIGGER, never a grade: a low cosine NEVER directly flips
edit_scope; it only escalates a second, zoomed VLM read whose verdict is what
counts (§ "metric is a trigger, never a vote"). On a high cosine (or a dress
target, or a DINOv2 load failure) H2 == H1 exactly.

DINOv2 backbone: timm vit_base_patch14_dinov2.lvd142m at the local HF hub path.
If timm/torch or the checkpoint cannot load, _ensure_dino() logs it and H2
degrades to plain H1 behavior (dino_ok=False stamped in sub_scores).
"""
from __future__ import annotations

import threading
from pathlib import Path

from PIL import Image

from _v13_eval import gf_common as G
from _v13_eval import metric_common as M
import _editscope as ES

DINO_PATH = "/mnt/image-edit/models/hf/hub/models--timm--vit_base_patch14_dinov2.lvd142m"
NT_COS_THRESHOLD = 0.92        # below -> escalate the zoomed non-target re-read

_DINO = None
_DINO_OK = False
_DINO_ERR = None
_DINO_TX = None
_DINO_DEVICE = "cpu"
_INIT_LOCK = threading.Lock()
_INITED = False


def _ensure_dino():
    """Lazy, idempotent DINOv2 load from the local timm checkpoint. On ANY
    failure sets _DINO_OK=False and records the error (H2 degrades to H1)."""
    global _DINO, _DINO_OK, _DINO_ERR, _DINO_TX, _DINO_DEVICE, _INITED
    with _INIT_LOCK:
        if _INITED:
            return _DINO_OK
        _INITED = True
        try:
            import torch
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform
            _DINO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            snap = None
            sd = Path(DINO_PATH) / "snapshots"
            if sd.is_dir():
                subs = sorted([p for p in sd.iterdir() if p.is_dir()])
                snap = subs[0] if subs else None
            # Prefer the local snapshot dir; timm reads model.safetensors from a
            # pretrained_cfg file path when given.
            model = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                pretrained_cfg_overlay=(
                    {"file": str(snap / "model.safetensors")} if snap else None),
            )
            model.eval().to(_DINO_DEVICE)
            cfg = resolve_data_config({}, model=model)
            _DINO_TX = create_transform(**cfg)
            _DINO = (torch, model)
            _DINO_OK = True
        except Exception as e:
            _DINO_ERR = f"{type(e).__name__}: {e}"
            _DINO_OK = False
    return _DINO_OK


def init_metrics():
    """Called by era_eval_common when needs_metrics=True. Non-fatal."""
    ok = _ensure_dino()
    if not ok:
        print(f"[_ntmetric] DINOv2 unavailable, H2 degrades to H1: {_DINO_ERR}")
    return ok


def _dino_cos(img_a, img_b):
    if not _DINO_OK:
        return None
    torch, model = _DINO
    try:
        a = _DINO_TX(img_a.convert("RGB")).unsqueeze(0).to(_DINO_DEVICE)
        b = _DINO_TX(img_b.convert("RGB")).unsqueeze(0).to(_DINO_DEVICE)
        with torch.no_grad():
            fa = model(a)
            fb = model(b)
        cos = torch.nn.functional.cosine_similarity(fa, fb).item()
        return float(max(0.0, min(1.0, (cos + 1.0) / 2.0)))
    except Exception:
        return None


def _nontarget_half(category):
    """The NON-target half for a single-region target: the lower half is the
    complement of an 'upper' target, the upper half complements 'lower'.
    Returns (box_fn_category, label) or (None, None) for non-single targets."""
    if category == "upper":
        return "lower", "lower"
    if category == "lower":
        return "upper", "upper"
    return None, None


def _crop_half(pil_img, category):
    w, h = pil_img.size
    box = G._target_box(w, h, category)
    return M.crop(pil_img, box)


def _build_nt_rereread_messages(model_crop, result_crop, nt_label, swap=False):
    """A zoomed STEP-B re-read on JUST the non-target half: did this non-target
    region's garment CHANGE (model -> result)? Trigger-gated; the VLM verdict is
    what counts, not the metric."""
    instr = (
        "You are checking ONE body region of a virtual try-on for a NON-TARGET edit. "
        "This region (the " + str(nt_label).upper() + " half) is NOT supposed to change — the "
        "try-on garment belongs to the OTHER region. You are shown the SAME region cropped from "
        "two images:\n"
        "  [ORIGINAL " + str(nt_label).upper() + "] = the person BEFORE try-on, this region.\n"
        "  [RESULT " + str(nt_label).upper() + "]   = the generated result, the SAME region.\n"
        "Decide: did the GARMENT in this region CLEARLY change identity (a different garment, "
        "or a clear recolor/retexture)? IGNORE pose/crop/lighting, face/skin/hair regen, and a "
        "region merely occluded or revealed by the OTHER region's new garment drape/length.\n"
        "Emit nt_changed = 1 if this non-target region's garment CLEARLY changed, else 0.\n"
        "ANTI-BIAS: the crops may be in either order; rely on the [ORIGINAL]/[RESULT] labels.\n"
        "Reason BRIEFLY (one short phrase). Finish with ONE final JSON object on the last line "
        "with EXACTLY this key (integer 0/1):\n"
        '{"nt_changed":_}'
    )
    m_url = G.img_to_data_url(model_crop)
    r_url = G.img_to_data_url(result_crop)
    if not swap:
        block = [
            {"type": "text", "text": "[ORIGINAL " + str(nt_label).upper() + "]:"},
            {"type": "image_url", "image_url": {"url": m_url}},
            {"type": "text", "text": "[RESULT " + str(nt_label).upper() + "]:"},
            {"type": "image_url", "image_url": {"url": r_url}},
        ]
    else:
        block = [
            {"type": "text", "text": "[RESULT " + str(nt_label).upper() + "]:"},
            {"type": "image_url", "image_url": {"url": r_url}},
            {"type": "text", "text": "[ORIGINAL " + str(nt_label).upper() + "]:"},
            {"type": "image_url", "image_url": {"url": m_url}},
        ]
    return [{"role": "user", "content": block + [{"type": "text", "text": instr}]}]


def _nt_zoom_reread(pick_ep, model_img, result, nt_category, nt_label, seeds):
    """Zoomed non-target STEP-B re-read. Returns 1 (changed) / 0 (unchanged) /
    None (unparseable)."""
    mim = Image.open(model_img).convert("RGB")
    rim = Image.open(result).convert("RGB")
    m_crop = _crop_half(mim, nt_category)
    r_crop = _crop_half(rim, nt_category)
    votes = []
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        swap = (i % 2 == 1)
        try:
            msgs = _build_nt_rereread_messages(m_crop, r_crop, nt_label, swap=swap)
            txt = G.call_judge(base_url, model, msgs, seed=seed,
                               enable_thinking=False, max_tokens=768)
            parsed = G.extract_last_json(txt)
            if not parsed:
                continue
            v = parsed.get("nt_changed")
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
    return 1 if votes.count(1) >= votes.count(0) else 0


def score_h2(pick_ep, cloth, model_img, result, *, gf_seeds, pres_seeds,
             escope_seeds, native_think=False, max_tokens=2048,
             nt_threshold=NT_COS_THRESHOLD):
    """H2 = H1 + the DINOv2-gated non-target zoom re-read.

    Pipeline: run the v13 base + the H1 STEP-A/B/C edit-scope read. If the read's
    target is a SINGLE region and DINOv2 cosine on the non-target half-crop is
    below nt_threshold, escalate a zoomed STEP-B re-read on that half; if that
    re-read CONFIRMS a non-target change, force edit_scope=0 (the VLM verdict
    decides, not the metric). On a high cosine / dress target / DINOv2 failure
    H2 == H1. Returns (score, sub, ok, defect)."""
    # 1) v13 base
    base_score, sub, ok, base_defect = ES.X.evaluate_sample(
        ES.X.V13_FIDELITYZOOM_COVERAGE_FIRST_ID, pick_ep, None,
        cloth, model_img, result, gf_seeds=gf_seeds, pres_seeds=pres_seeds)
    # 2) H1 edit-scope read
    edit_scope, escope_sub = ES.editscope_read(
        pick_ep, cloth, model_img, result, escope_seeds,
        max_tokens=max_tokens, native_think=native_think)

    nt_sub = {"dino_ok": bool(_DINO_OK), "nt_zoom_triggered": False,
              "nt_dino_cos": None, "nt_zoom_changed": None}

    # 3) escalate-only DINOv2 zoom trigger — only when edit_scope currently reads
    #    CLEAN (1) on a SINGLE-region target with a model image (otherwise the
    #    veto already fired, or there is no non-target half to check).
    if (edit_scope == 1 and model_img is not None and Path(model_img).is_file()
            and escope_sub.get("escope_ok")):
        targets = escope_sub.get("escope_target_regions") or []
        single = len(targets) == 1 and targets[0] in ("upper", "lower")
        if single:
            nt_cat, nt_label = _nontarget_half(targets[0])
            if nt_cat and _ensure_dino():
                try:
                    mim = Image.open(model_img).convert("RGB")
                    rim = Image.open(result).convert("RGB")
                    cos = _dino_cos(_crop_half(mim, nt_cat), _crop_half(rim, nt_cat))
                    nt_sub["nt_dino_cos"] = cos
                    if cos is not None and cos < nt_threshold:
                        nt_sub["nt_zoom_triggered"] = True
                        changed = _nt_zoom_reread(
                            pick_ep, model_img, result, nt_cat, nt_label, escope_seeds)
                        nt_sub["nt_zoom_changed"] = changed
                        if changed == 1:
                            # the zoomed VLM verdict (NOT the metric) flips edit_scope.
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
