#!/usr/bin/env python3
"""The iter_022 category-free FP-fixed crossover recipe (STEP-0 3-call) with the
ADDITIVE suit / two-piece garment_swap branch, extended for iter_026 with the
ADDITIVE per-piece REGION-ZOOM garment_swap read (gs_zoom).

Frozen from the ERA tryon-eval calibration. The configs are all
`_step0_champion` (STEP-0-in-read-1 champion + iter_020 atom fix + iter_021 FP
fix); they differ only in coverage_first on the PRES read and the suit / gs_zoom
branches:

  - hybrid-crossover-step0-122b-v6-fpfix-3call         : atom_fix, fp_fix
  - hybrid-crossover-step0-122b-v6-coverage-first      : atom_fix, fp_fix, coverage_first
  - hybrid-crossover-step0-122b-v7-suit-fpfix-3call    : atom_fix, fp_fix, suit (PR4 inline)
  - hybrid-crossover-step0-122b-v9-suit-relaxed-coverage-first :
        atom_fix, fp_fix, coverage_first, suit, suit_sharp, suit_relaxed
  - hybrid-crossover-step0-122b-v11-gszoom-coverage-first :
        the v9-suit-relaxed base + gs_zoom=True (iter_026 SHIP)

suit=False is BYTE-BEHAVIOUR-IDENTICAL to iter_021, so the v6 configs reproduce
iter_021 exactly; the suit branch only fires when STEP-0 returns
is_two_piece=true and never relaxes single-garment grading. gs_zoom=False is
byte-identical to the v9-suit-relaxed recipe; the per-piece zoom only fires when
gs_zoom=True AND STEP-0 returned is_two_piece is True.

PREFIX RULE: the sample-path category prefix is FORBIDDEN as an evaluator input —
every category signal (incl. is_two_piece, and the per-piece crop boxes) is
image-derived inside the read. There is no category_of and no path-prefix oracle.

The frozen iter_018 crossover fusion (_fuse) is carried verbatim. The only
signature change vs the source score_fns is that the input_model image path is
passed in explicitly as ``model_img`` (a Path or None) instead of being derived
from ``sd / L.INPUT_MODEL``; when model_img is None / not a file the PRES read is
skipped, exactly as the original.
"""
from pathlib import Path

from . import gf_common as G
from . import metric_common as M  # noqa: F401  (parity with the source import surface)
from . import CONFIGS, DEFAULT_CONFIG

GF_SEEDS = [42]    # the shipped full-round setting
PRES_SEEDS = [42]

COVERAGE_FIRST_ID = "hybrid-crossover-step0-122b-v6-coverage-first"
FPFIX_3CALL_ID = "hybrid-crossover-step0-122b-v6-fpfix-3call"
SUIT_COVERAGE_FIRST_ID = "hybrid-crossover-step0-122b-v7-suit-coverage-first"
SUIT_FPFIX_3CALL_ID = "hybrid-crossover-step0-122b-v7-suit-fpfix-3call"
# iter_023 suit-sharpened v8 ids (suit + suit_sharp=True): PR2 overlap-vs-suit
# gate + PR1 per-piece enumeration, added on the byte-frozen single-garment path.
SUIT_SHARP_FPFIX_3CALL_ID = "hybrid-crossover-step0-122b-v8-suit-sharp-fpfix-3call"
SUIT_SHARP_COVERAGE_FIRST_ID = "hybrid-crossover-step0-122b-v8-suit-sharp-coverage-first"
# iter_024 suit-RELAXED v9 ids (suit + suit_sharp + suit_relaxed=True): confirm-
# correct-first clear-defect-only PR1 (kills the suit garment_swap false positives);
# the -icl variant adds a generic contrastive anti-pattern (suit_icl=True).
SUIT_RELAXED_COVERAGE_FIRST_ID = "hybrid-crossover-step0-122b-v9-suit-relaxed-coverage-first"
SUIT_RELAXED_ICL_COVERAGE_FIRST_ID = "hybrid-crossover-step0-122b-v9-suit-relaxed-icl-coverage-first"
# iter_026 SHIP id (the v9-suit-relaxed base + gs_zoom=True): the ADDITIVE
# per-piece REGION-ZOOM garment_swap read. Lenient on suit garment_swap — it
# eliminates the suit false-alarms by checking each reference piece present-and-
# correct in its OWN zoomed crop; a positive 0/2 overrides the full-frame grade,
# uncertainty defers to the frozen full-frame grade (no recall collapse).
V11_GSZOOM_COVERAGE_FIRST_ID = "hybrid-crossover-step0-122b-v11-gszoom-coverage-first"
# iter_028 SHIP id (the v11-gszoom base + fidelity_zoom=True): the per-piece
# REGION-ZOOM read is re-framed from a LENIENT presence check to a STRICT
# per-region FIDELITY/DEFECT check (region_defect 0|1). Catches dress per-region
# defects (wrong/missing/distorted garment) the lenient presence read passed
# over, while keeping suits clean — the best-balanced suit+dress evaluator.
V13_FIDELITYZOOM_COVERAGE_FIRST_ID = "hybrid-crossover-step0-122b-v13-fidelityzoom-coverage-first"


def _gev(gm):
    return {"color_delta": gm.get("color_delta"), "dino_cos": gm.get("dino_cos"),
            "lpips": gm.get("lpips")}


def _ntev(nt):
    return {"nt_dino_cos": nt.get("nt_dino_cos"), "nt_lpips": nt.get("nt_lpips")}


def _declared_label(worn_region, region_conf):
    """Map a judge STEP-0 declaration to a category label; non-high confidence
    -> 'uncertain'."""
    if str(region_conf or "").strip().lower() != "high":
        return "uncertain"
    return G.map_worn_region(worn_region)


def _fuse(a_atoms, a_def, b_atoms, b_def, p_atoms, p_def, gm, nt, suppress_nt):
    """The frozen iter_018 crossover fusion (carried verbatim from iter_020).
    Returns (score, sub, defect)."""
    g_ok, g_mdef = G.metric_vetoes(gm) if gm else ({}, False)
    nt_ok, nt_raw = G.nontarget_veto(nt) if nt else ({}, False)
    vlm_swap = bool(p_atoms and p_atoms.get("garment_swap") == 0)
    nt_mdef = bool(nt_raw and vlm_swap and not suppress_nt)
    av = [v for v in a_atoms.values() if v is not None]
    a_unc = any(v <= 1 for v in av)
    veto_path = bool(a_def or (g_mdef and a_unc))
    gf_cross_def = bool(veto_path and b_def)
    defect = bool(gf_cross_def or p_def or nt_mdef)
    base = G.merge_overall(b_atoms, p_atoms)
    score = base if not (gf_cross_def or nt_mdef) else min(base, 0.40)
    sub = dict(b_atoms)
    if p_atoms:
        sub.update(p_atoms)
    if nt:
        sub.update(_ntev(nt))
    sub.update({"veto_path": veto_path, "grounded_read_def": bool(b_def),
                "gf_cross_def": gf_cross_def, "nontarget_veto": nt_mdef,
                "nontarget_veto_raw": bool(nt_raw), "pres_defect": bool(p_def)})
    return score, sub, defect


def _champion(ctx, pick_ep, cloth, model_img, result, label, atom_fix=False,
              fp_fix=False, coverage_first=False):
    """Run the frozen fusion at a resolved category label ('uncertain' allowed;
    optionally force_merged via label '__widecrop__'). atom_fix passes the
    iter_020 ADDITIVE dress/lower blocks into BOTH GF reads; fp_fix passes the
    iter_021 ADDITIVE FP-fix B (GF fine_details discipline) into the GF reads and
    FP-fix A (PRES garment_swap coverage-compensation) into the PRES read;
    coverage_first prepends the PRES coverage-map step (implies fp_fix). Returns
    (score, sub, defect)."""
    force_merged = False
    if label == "__widecrop__":
        cat, suppress_nt, emphasis = "dress", False, "dress"
        force_merged = True
    else:
        cat, suppress_nt, emphasis = G.resolve_region(label)
    gm = G.garment_metrics(cloth, result, cat)
    nt = G.nontarget_metrics(model_img, result, cat) if (model_img and model_img.is_file()) else {}
    a = G.run_gf_judge(pick_ep, cloth, result, ctx["gf_seeds"],
                       inferred_category=emphasis, force_merged=force_merged,
                       atom_fix=atom_fix, fp_fix=fp_fix)
    a_atoms, _, a_def, a_ok, a_err = a
    if a_atoms is None:
        raise RuntimeError(f"all GF read-A passes failed; {a_err}")
    b = G.run_gf_judge(pick_ep, cloth, result, ctx["gf_seeds"], evidence=_gev(gm),
                       inferred_category=emphasis, force_merged=force_merged,
                       atom_fix=atom_fix, fp_fix=fp_fix)
    b_atoms, _, b_def, b_ok, b_err = b
    if b_atoms is None:
        raise RuntimeError(f"all GF read-B passes failed; {b_err}")
    p_atoms, p_def = None, False
    if model_img and model_img.is_file():
        pev = _ntev(nt) if (nt and not suppress_nt) else None
        p_atoms, p_def, p_ok, p_err = G.run_pres_judge(
            pick_ep, model_img, result, ctx["pres_seeds"], evidence=pev,
            fp_fix=fp_fix, coverage_first=coverage_first)
    score, sub, defect = _fuse(a_atoms, a_def, b_atoms, b_def, p_atoms, p_def,
                               gm, nt, suppress_nt)
    sub["resolved_category"] = cat if label != "__widecrop__" else "dress-wide"
    sub["atom_fix"] = bool(atom_fix)
    sub["fp_fix"] = bool(fp_fix)
    sub["coverage_first"] = bool(coverage_first)
    # keep PRES atoms + the resolved cat exposed for the escalation re-read.
    sub["_p_atoms"] = p_atoms
    sub["_a_atoms"] = a_atoms
    sub["_b_atoms"] = b_atoms
    sub["_resolved_cat"] = cat
    sub["_suppress_nt"] = suppress_nt
    sub["_gm"] = gm
    sub["_nt"] = nt
    return score, sub, defect


def _step0_champion(ctx, pick_ep, cloth, model_img, result, atom_fix=True,
                    fp_fix=False, coverage_first=False, suit=False, suit_sharp=False,
                    suit_relaxed=False, suit_icl=False, gs_zoom=False,
                    fidelity_zoom=False):
    """STEP-0-in-read-1 champion (iter_019 incat body) with the iter_020 atom
    fix + the iter_021 FP fix, plus the iter_022 ADDITIVE suit branch. read-A is
    the STEP-0 full-frame declaration (when suit=True it also emits is_two_piece /
    two_piece); the declared region resolves the crop; read-B is the crop-scoped
    grounded read; PRES + frozen fusion as the champion. The single-garment path
    is byte-identical to suit=False; the suit branch only fires when STEP-0
    returns is_two_piece=true.

    iter_026 FLAG A (gs_zoom): when gs_zoom=True AND the STEP-0 read returned
    is_two_piece is True, an ADDITIVE per-piece REGION-ZOOM garment_swap read runs
    (G.run_pres_zoom): the input_model + result are cropped to the upper and lower
    boxes (via _target_box; image-derived) and each reference piece is checked
    present-and-correct in its OWN crop. A positive 0/2 OVERRIDES the full-frame
    garment_swap grade (grade 2 if both pieces present-correct; grade 0 if a piece
    is positively missing/substituted); on the defer sentinel the frozen
    full-frame grade is kept (no recall collapse). The FIXED gate fires on
    `is_two_piece is True` ALONE — it does NOT require two_piece_confidence=="high"
    (this evaluator never emits that field; requiring it was the iter_026 bug that
    disabled the zoom). gs_zoom=False is byte-identical to the v9-suit-relaxed
    recipe.

    iter_028 (fidelity_zoom): when fidelity_zoom=True (with gs_zoom=True) each
    per-region zoom read is a STRICT per-region FIDELITY/DEFECT check
    (region_defect 0|1) instead of the lenient presence read — garment_swap=0 if
    ANY region has a confirmed defect (wrong/missing/distorted garment), else 2.
    This catches dress per-region defects the lenient presence read passed over
    while keeping suits clean. fidelity_zoom=False is byte-identical to the
    iter_026 gs_zoom presence behavior. Returns (score, sub, defect)."""
    if suit:
        a = G.run_gf_judge(pick_ep, cloth, result, ctx["gf_seeds"],
                           step0=True, full_frame=True, atom_fix=atom_fix,
                           fp_fix=fp_fix, suit=True)
        a_atoms, _, a_def, a_ok, a_err, worn, conf, is_two_piece, two_piece = a
    else:
        a = G.run_gf_judge(pick_ep, cloth, result, ctx["gf_seeds"],
                           step0=True, full_frame=True, atom_fix=atom_fix, fp_fix=fp_fix)
        a_atoms, _, a_def, a_ok, a_err, worn, conf = a
        is_two_piece, two_piece = None, None
    if a_atoms is None:
        raise RuntimeError(f"all GF read-A passes failed; {a_err}")
    label = _declared_label(worn, conf)
    cat, suppress_nt, emphasis = G.resolve_region(label)
    gm = G.garment_metrics(cloth, result, cat)
    nt = G.nontarget_metrics(model_img, result, cat) if (model_img and model_img.is_file()) else {}
    b = G.run_gf_judge(pick_ep, cloth, result, ctx["gf_seeds"],
                       evidence=_gev(gm), inferred_category=emphasis,
                       atom_fix=atom_fix, fp_fix=fp_fix)
    b_atoms, _, b_def, b_ok, b_err = b
    if b_atoms is None:
        raise RuntimeError(f"all GF read-B passes failed; {b_err}")
    p_atoms, p_def = None, False
    gs_zoom_applied = False
    gs_zoom_grade = None
    gs_zoom_sub = {}
    if model_img and model_img.is_file():
        pev = _ntev(nt) if (nt and not suppress_nt) else None
        p_atoms, p_def, p_ok, p_err = G.run_pres_judge(
            pick_ep, model_img, result, ctx["pres_seeds"], evidence=pev,
            fp_fix=fp_fix, coverage_first=coverage_first, suit=suit,
            suit_sharp=suit_sharp, suit_relaxed=suit_relaxed, suit_icl=suit_icl)
        # iter_026 FLAG A (gs_zoom): per-piece region-zoom garment_swap read, ONLY
        # on a two-piece suit (FIXED gate: is_two_piece is True alone — NO
        # two_piece_confidence requirement). On a positive 0/2 it overrides the
        # full-frame garment_swap grade; on the defer sentinel it keeps the frozen
        # full-frame grade (no recall collapse).
        if gs_zoom and suit and (is_two_piece is True):
            gs_zoom_grade, gs_zoom_sub = G.run_pres_zoom(
                pick_ep, model_img, result, cloth, two_piece, ctx["pres_seeds"],
                fidelity_zoom=fidelity_zoom)
            if gs_zoom_grade != G.GS_ZOOM_DEFER:
                gs_zoom_applied = True
                if p_atoms is None:
                    p_atoms = {}
                p_atoms = dict(p_atoms)
                p_atoms["garment_swap"] = int(gs_zoom_grade)
                p_def = any(v == 0 for v in p_atoms.values() if v is not None)
    score, sub, defect = _fuse(a_atoms, a_def, b_atoms, b_def, p_atoms, p_def,
                               gm, nt, suppress_nt)
    sub.update({"worn_region": worn, "region_confidence": conf,
                "resolved_category": cat, "atom_fix": bool(atom_fix),
                "fp_fix": bool(fp_fix), "coverage_first": bool(coverage_first),
                "suit": bool(suit), "suit_sharp": bool(suit_sharp),
                "suit_relaxed": bool(suit_relaxed), "suit_icl": bool(suit_icl),
                "gs_zoom": bool(gs_zoom), "fidelity_zoom": bool(fidelity_zoom),
                "gs_zoom_applied": bool(gs_zoom_applied)})
    if gs_zoom_sub:
        sub.update(gs_zoom_sub)
    if suit:
        sub["is_two_piece"] = is_two_piece
        sub["two_piece"] = two_piece
    return score, sub, defect


_INTERNAL_KEYS = ("_p_atoms", "_a_atoms", "_b_atoms", "_resolved_cat",
                  "_suppress_nt", "_gm", "_nt")


def _strip_internal(sub):
    """Drop the internal-only carriers (PIL/dicts kept for the escalation
    re-read) before the row is serialized to scores.jsonl."""
    return {k: v for k, v in sub.items() if k not in _INTERNAL_KEYS}


def evaluate_sample(combination_id, pick_ep, sample_key, cloth, model_img, result, *,
                    gf_seeds=None, pres_seeds=None):
    """Score one (sample, method) pair for one config. Returns
    (score, sub_scores, ok, defect).

    ``cloth``/``model_img``/``result`` are image paths; ``model_img`` may be
    None / missing (the preservation read is then skipped, as in the original).
    The category is image-derived inside the STEP-0 read — there is NO category
    argument and no path-prefix oracle.
    """
    ctx = {"gf_seeds": gf_seeds or GF_SEEDS,
           "pres_seeds": pres_seeds or PRES_SEEDS}
    model_img = Path(model_img) if model_img else None

    if combination_id == COVERAGE_FIRST_ID:
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=True)
    elif combination_id == FPFIX_3CALL_ID:
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=False)
    elif combination_id == SUIT_COVERAGE_FIRST_ID:
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=True, suit=True)
    elif combination_id == SUIT_FPFIX_3CALL_ID:
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=False, suit=True)
    elif combination_id == SUIT_SHARP_FPFIX_3CALL_ID:
        # iter_023 PRIMARY: v7 suit inline base + PR2/PR1 suit-sharpening (INLINE).
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=False, suit=True,
            suit_sharp=True)
    elif combination_id == SUIT_SHARP_COVERAGE_FIRST_ID:
        # iter_023 SECONDARY: coverage-first base + PR2/PR1 suit-sharpening (ORDERED).
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=True, suit=True,
            suit_sharp=True)
    elif combination_id == SUIT_RELAXED_COVERAGE_FIRST_ID:
        # iter_024 PRIMARY: coverage-first base + PR2 gate + confirm-correct-first
        # RELAXED PR1 (clear-defect-only) — kills the suit garment_swap FPs.
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=True, suit=True,
            suit_sharp=True, suit_relaxed=True)
    elif combination_id == SUIT_RELAXED_ICL_COVERAGE_FIRST_ID:
        # iter_024: same relaxed clause + a generic contrastive-ICL anti-pattern.
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=True, suit=True,
            suit_sharp=True, suit_relaxed=True, suit_icl=True)
    elif combination_id == V11_GSZOOM_COVERAGE_FIRST_ID:
        # iter_026 SHIP: the v9-suit-relaxed base + gs_zoom=True (ADDITIVE per-piece
        # REGION-ZOOM garment_swap read). Lenient — eliminates suit false-alarms.
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=True, suit=True,
            suit_sharp=True, suit_relaxed=True, gs_zoom=True)
    elif combination_id == V13_FIDELITYZOOM_COVERAGE_FIRST_ID:
        # iter_028 SHIP: the v11-gszoom base + fidelity_zoom=True. The per-piece
        # REGION-ZOOM read is a STRICT per-region FIDELITY/DEFECT check
        # (region_defect 0|1) — catches dress per-region defects while keeping
        # suits clean. Best-balanced suit+dress evaluator.
        score, sub, defect = _step0_champion(
            ctx, pick_ep, cloth, model_img, result,
            atom_fix=True, fp_fix=True, coverage_first=True, suit=True,
            suit_sharp=True, suit_relaxed=True, gs_zoom=True, fidelity_zoom=True)
    else:
        raise ValueError(f"unknown combination_id: {combination_id!r} "
                         f"(known: {list(CONFIGS)})")

    sub = _strip_internal(sub)
    return score, sub, True, defect
