#!/usr/bin/env python3
"""iter_029 ADDITIVE reference-anchored edit-scope read (the P0 fix).

The shipped v11/v13 lineage is reference-BLIND on the edit-scope question: its
preservation/garment_swap atoms compare ORIGINAL-PERSON vs RESULT only, so they
miss 41/62 (66%) of operator edit-scope defects (spec 2b: a non-target garment
region changed, OR a target region did not change). v13's garment_swap atom is
also deliberately LENIENT (tuned down to kill suit false alarms), so it cannot
be the edit-scope detector.

This module adds an ORTHOGONAL, reference-anchored STEP-A/B/C read that sees ALL
THREE images and produces a SEPARATE `edit_scope` channel:

  STEP-A — classify input_cloth's covered region(s): {upper | lower | dress}
           (image-derived from the [REFERENCE CLOTH], never the path).
  STEP-B — which body region(s) changed input_model -> output
           (ignore pose/crop/face-skin regen + occlusion-only changes).
  STEP-C — edit_scope = 0 iff changed_regions is NOT a subset of target_regions
           (a NON-target region changed), OR a target region did NOT change
           (target region untouched). Otherwise edit_scope = 1.

It is wired as a FIRST-CLASS veto, kept SEPARATE from the leniency-tuned
garment_swap atom:

  apply_editscope_veto(score, sub, defect, escope):
      edit_scope == 0  ->  defect = True  AND  score = min(score, 0.40)

The veto cap is enforced in PARSER code (a hard min), not trusted to the model's
JSON. `edit_scope`, `target_regions`, `changed_regions`, and per-FP reason codes
are surfaced in sub_scores.

NOTHING here mutates the frozen _v13_eval package — it only consumes its VLM
plumbing (call_judge / extract_last_json / img_to_data_url / map_worn_region).
So the H1 single-garment path = v13 verbatim PLUS this additive channel; on a
clean single-garment sample (edit_scope == 1) the v13 score is unchanged.
"""
from __future__ import annotations

from pathlib import Path

from _v13_eval import gf_common as G
from _v13_eval import crossover as X

EDIT_SCOPE_CAP = 0.40          # the first-class veto score-cap (kept OUT of garment_swap)
_REGIONS = ("upper", "lower", "dress")


# --------------------------------------------------------------------------- #
# region-set helpers
# --------------------------------------------------------------------------- #
def _norm_region_token(tok):
    """Map a free-text region token to {upper, lower, dress} or None."""
    s = str(tok or "").strip().lower()
    if not s or s in ("none", "n/a", "na", "no_change", "no-change", "nochange",
                       "unchanged", "nothing"):
        return None
    if "dress" in s or "full" in s or "one-piece" in s or "one piece" in s or \
            "onepiece" in s or "jumpsuit" in s or "overall" in s:
        return "dress"
    if "upper" in s or "top" in s or "torso" in s or "shirt" in s or "jacket" in s:
        return "upper"
    if "lower" in s or "leg" in s or "trouser" in s or "pant" in s or "skirt" in s or \
            "short" in s or "bottom" in s:
        return "lower"
    return None


def _coerce_region_list(v):
    """Coerce a model field (string OR list) to a normalized region set in the
    token alphabet {upper, lower, dress} (kept as-reported, for surfacing)."""
    out = set()
    if v is None:
        return out
    items = v if isinstance(v, (list, tuple)) else [v]
    for it in items:
        r = _norm_region_token(it)
        if r:
            out.add(r)
    return out


def _half_project(region_set):
    """Project a region set onto the PHYSICAL half-coverage alphabet
    {upper, lower}: a `dress`/suit-set token covers BOTH halves. This is the set
    the subset/coverage comparison runs on, so a dress target is satisfied by a
    change reported as either `dress` OR `upper`+`lower`, and an upper+lower
    change against a dress target is NOT a spurious target-unchanged veto."""
    half = set()
    for r in region_set:
        if r == "dress":
            half |= {"upper", "lower"}
        elif r in ("upper", "lower"):
            half.add(r)
    return half


def _scope_decision(tr_set, ch_set):
    """The canonical STEP-C decision on the physical-half projections.

    Returns (edit_scope, reason):
      - (None, "ok") when both projections are empty (degenerate read; the caller
        falls back to the model's edit_scope int so a verdict-only read counts).
      - edit_scope = 0 iff a changed half is NOT a target half (nontarget
        changed) OR a target half did NOT change (target unchanged). The
        target-unchanged direction only fires when at least one change was
        reported — a fully clean 'nothing changed' read DEFERS to the v13 base,
        never a target-unchanged veto (avoids penalizing identical-garment
        no-op samples in this orthogonal channel)."""
    T = _half_project(tr_set)
    C = _half_project(ch_set)
    if not (T or C):
        return None, "ok"
    nontarget = bool(C - T)
    target_unchanged = bool(T and C and (T - C))
    if nontarget and target_unchanged:
        return 0, "both"
    if nontarget:
        return 0, "nontarget_changed"
    if target_unchanged:
        return 0, "target_unchanged"
    return 1, "ok"


# --------------------------------------------------------------------------- #
# the STEP-A/B/C reference-anchored read
# --------------------------------------------------------------------------- #
def _build_editscope_messages(cloth_path, model_path, result_path, swap=False):
    """Three position-tagged images + an ordered STEP-A/B/C CoT, finishing with a
    strict final-line JSON. The judge classifies the reference cloth's region(s),
    detects which body region(s) changed model->result, and decides edit_scope."""
    instr = (
        "You are a meticulous virtual-try-on EDIT-SCOPE judge. You are shown THREE images:\n"
        "  [REFERENCE CLOTH] = the input garment the result is SUPPOSED to put on the person.\n"
        "  [ORIGINAL PERSON] = the person BEFORE try-on.\n"
        "  [TRY-ON RESULT]   = the generated image after the try-on.\n"
        "A CORRECT try-on changes ONLY the body region(s) that the [REFERENCE CLOTH] covers, "
        "and it DOES change them. Two failures violate edit-scope:\n"
        "  (i)  a NON-TARGET region changed (a body region the reference cloth does NOT cover "
        "       was altered — e.g. the reference is a top but the trousers/skirt ALSO changed);\n"
        "  (ii) a TARGET region did NOT change (the reference cloth's region still shows the "
        "       ORIGINAL garment — the edit was not applied there).\n"
        "\n"
        "STEP A — from the [REFERENCE CLOTH] ALONE, classify which body region(s) it covers:\n"
        "  \"upper\" (a torso top/shirt/jacket), \"lower\" (trousers/pants/skirt/shorts), or "
        "  \"dress\" (a full-length one-piece / suit set covering BOTH torso and legs). "
        "  Decide from the GARMENT IMAGE, never from any filename. Report target_regions as a "
        "  list (e.g. [\"upper\"]; a dress/suit set is [\"dress\"]).\n"
        "STEP B — compare [ORIGINAL PERSON] vs [TRY-ON RESULT] and list which body region(s) "
        "  CLEARLY CHANGED garment: \"upper\" and/or \"lower\" (use \"dress\" if a single "
        "  full-length garment replaced both). IGNORE changes that are NOT garment-identity "
        "  changes: pose/stance/crop/zoom, face/skin/hair regeneration, lighting/shadow, and a "
        "  region merely OCCLUDED or revealed by the new garment's drape/length (a longer top "
        "  covering the waistband is NOT a lower change; a dress hem revealing legs is NOT a "
        "  lower change). Report changed_regions as a list; [] if no garment clearly changed.\n"
        "STEP C — decide edit_scope (binary):\n"
        "  edit_scope = 0 if (a) any changed region is NOT covered by target_regions (a "
        "  non-target region changed), OR (b) any target region is NOT in changed_regions (a "
        "  target region did not change). Otherwise edit_scope = 1.\n"
        "  Give a short reason code: \"ok\", \"nontarget_changed\", \"target_unchanged\", or "
        "  \"both\".\n"
        "\n"
        "ANTI-BIAS: the images may be given in either order; rely on the [REFERENCE CLOTH] / "
        "[ORIGINAL PERSON] / [TRY-ON RESULT] labels, never on position.\n"
        "Reason BRIEFLY — one short phrase per step; do NOT write long paragraphs. You MUST "
        "finish with ONE final JSON object on the last line with EXACTLY these keys "
        "(target_regions/changed_regions are lists of region strings; edit_scope integer 0/1; "
        "scope_reason and region_confidence strings):\n"
        '{"target_regions":[_], "changed_regions":[_], "edit_scope":_, '
        '"scope_reason":_, "region_confidence":_}'
    )
    cloth_url = G.img_to_data_url(cloth_path)
    model_url = G.img_to_data_url(model_path)
    result_url = G.img_to_data_url(result_path)
    triples = [
        ("[REFERENCE CLOTH]:", cloth_url),
        ("[ORIGINAL PERSON]:", model_url),
        ("[TRY-ON RESULT]:", result_url),
    ]
    if swap:
        # swap the ORIGINAL/RESULT presentation order for debiasing; keep the
        # reference cloth first (it anchors STEP-A).
        triples = [triples[0], triples[2], triples[1]]
    content = []
    for label, url in triples:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instr})
    return [{"role": "user", "content": content}]


def editscope_read(pick_ep, cloth_path, model_path, result_path, seeds,
                   max_tokens=2048, native_think=False):
    """Run the reference-anchored STEP-A/B/C read (order-swap ensemble over
    `seeds`). Returns (edit_scope:int 0|1, escope_sub:dict). On total parse
    failure returns (1, {...}) with escope_ok=False — a FAILED read NEVER
    fabricates a defect (it defers to the v13 base; §11)."""
    if model_path is None or not Path(model_path).is_file():
        # no ORIGINAL PERSON -> the reference-anchored read cannot run; defer.
        return 1, {"edit_scope": 1, "escope_ok": False,
                   "escope_reason": "no_model_image",
                   "target_regions": [], "changed_regions": []}

    es_votes = []
    targets, changes, reasons, confs = [], [], [], []
    passes_ok = 0
    last_err = None
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        swap = (i % 2 == 1)
        try:
            msgs = _build_editscope_messages(cloth_path, model_path, result_path, swap=swap)
            txt = G.call_judge(base_url, model, msgs, seed=seed,
                               enable_thinking=bool(native_think), max_tokens=max_tokens)
            parsed = G.extract_last_json(txt)
            if not parsed:
                txt = G.call_judge(base_url, model, msgs, seed=seed,
                                   enable_thinking=False, max_tokens=1024)
                parsed = G.extract_last_json(txt)
            if not parsed:
                last_err = "no json"
                continue
            tr = _coerce_region_list(parsed.get("target_regions"))
            ch = _coerce_region_list(parsed.get("changed_regions"))
            # PARSER-side decision: do NOT trust the model's edit_scope int alone.
            # Recompute on the physical-half projections; fall back to the model's
            # int only when the sets are degenerate (both empty).
            decided, reason = _scope_decision(tr, ch)
            if decided is None:
                raw = parsed.get("edit_scope")
                try:
                    decided = 1 if int(raw) == 1 else 0
                except (TypeError, ValueError):
                    decided = None
                reason = str(parsed.get("scope_reason") or "ok")
            if decided is not None:
                es_votes.append(decided)
                targets.append(sorted(tr))
                changes.append(sorted(ch))
                reasons.append(reason)
                confs.append(str(parsed.get("region_confidence") or ""))
                passes_ok += 1
        except Exception as e:
            last_err = str(e)

    if passes_ok == 0:
        return 1, {"edit_scope": 1, "escope_ok": False,
                   "escope_reason": "read_failed", "escope_err": last_err,
                   "target_regions": [], "changed_regions": []}

    # majority vote; ties -> defect (edit_scope 0) is the conservative recall
    # choice for the dominant operator-flagged mode.
    n0 = es_votes.count(0)
    n1 = es_votes.count(1)
    edit_scope = 0 if n0 >= n1 else 1
    # pick the reason from a vote that matches the decision.
    chosen_reason = "ok"
    for v, r in zip(es_votes, reasons):
        if v == edit_scope and r != "ok":
            chosen_reason = r
            break
    sub = {
        "edit_scope": int(edit_scope),
        "escope_ok": True,
        "escope_reason": chosen_reason,
        "escope_target_regions": targets[-1] if targets else [],
        "escope_changed_regions": changes[-1] if changes else [],
        "escope_votes": es_votes,
        "escope_region_confidence": confs[-1] if confs else "",
    }
    # surface the operator-facing names the brief / pilot gate expects.
    sub["target_regions"] = sub["escope_target_regions"]
    sub["changed_regions"] = sub["escope_changed_regions"]
    return int(edit_scope), sub


def apply_editscope_veto(score, sub, defect, edit_scope, escope_sub):
    """Wire `edit_scope` as a FIRST-CLASS veto, kept SEPARATE from garment_swap.

    edit_scope == 0  ->  defect = True  AND  score = min(score, EDIT_SCOPE_CAP).
    The cap is a HARD parser-side min (the model's JSON is never trusted to honor
    it). Returns (score, sub, defect). garment_swap is left untouched."""
    sub = dict(sub)
    sub.update(escope_sub)
    if int(edit_scope) == 0:
        defect = True
        score = min(float(score), EDIT_SCOPE_CAP)
        sub["edit_scope_veto"] = True
    else:
        sub["edit_scope_veto"] = False
    return float(score), sub, bool(defect)


# --------------------------------------------------------------------------- #
# H1 — v13 control PLUS the additive edit-scope veto
# --------------------------------------------------------------------------- #
def score_h1(pick_ep, cloth, model_img, result, *, gf_seeds, pres_seeds,
             escope_seeds, native_think=False, max_tokens=2048):
    """H1 = the frozen v13 read (byte-equivalent single-garment path) + the
    additive reference-anchored edit-scope veto. On a clean sample
    (edit_scope == 1) the returned (score, defect) equal v13's exactly; only the
    edit_scope channel is added to sub_scores."""
    score, sub, ok, defect = X.evaluate_sample(
        X.V13_FIDELITYZOOM_COVERAGE_FIRST_ID, pick_ep, None,
        cloth, model_img, result, gf_seeds=gf_seeds, pres_seeds=pres_seeds)
    edit_scope, escope_sub = editscope_read(
        pick_ep, cloth, model_img, result, escope_seeds,
        max_tokens=max_tokens, native_think=native_think)
    score, sub, defect = apply_editscope_veto(score, sub, defect, edit_scope, escope_sub)
    return score, sub, ok, defect


# --------------------------------------------------------------------------- #
# H3 — single-call compact reference-anchored edit-scope + fidelity (for 35B)
# --------------------------------------------------------------------------- #
def _build_singlecall_messages(cloth_path, model_path, result_path, swap=False):
    """ONE judge call: STEP-A/B/C edit-scope (first-class veto) + a compact
    fidelity check (report-but-don't-veto secondary atoms). Tagged
    [CLOTH]<img1>[MODEL]<img2>[RESULT]<img3>, think-then-score."""
    instr = (
        "You are a meticulous virtual try-on judge. You are shown THREE images:\n"
        "  [CLOTH]  = the reference garment the result should put on the person.\n"
        "  [MODEL]  = the person BEFORE try-on.\n"
        "  [RESULT] = the generated try-on image.\n"
        "\n"
        "PRIMARY — EDIT SCOPE (spec 2b), the decisive check:\n"
        "  STEP A: from [CLOTH] alone classify its covered region(s): \"upper\", \"lower\", "
        "or \"dress\" (a full one-piece / suit set covers BOTH). Report target_regions (list).\n"
        "  STEP B: compare [MODEL] vs [RESULT]; list which body region(s) CLEARLY changed "
        "garment (\"upper\"/\"lower\"/\"dress\"). IGNORE pose/crop/face-skin regen, lighting, "
        "and a region merely occluded/revealed by the new garment's drape. Report "
        "changed_regions (list; [] if none clearly changed).\n"
        "  STEP C: edit_scope = 0 if a NON-target region changed OR a target region did NOT "
        "change; else 1. Give scope_reason: ok|nontarget_changed|target_unchanged|both.\n"
        "\n"
        "SECONDARY — FIDELITY (report only; do NOT let these flip edit_scope):\n"
        "  garment_fidelity: does the worn target garment reproduce the [CLOTH] identity "
        "(color/pattern/type/structure)? grade 0=clearly wrong, 1=mild, 2=matches.\n"
        "  pose: is the person's body pose/anatomy preserved and natural? 0=broken, 2=fine.\n"
        "\n"
        "ANTI-BIAS: images may be in either order; rely on the [CLOTH]/[MODEL]/[RESULT] labels.\n"
        "Reason BRIEFLY (one short phrase per step). Finish with ONE final JSON object on the "
        "last line with EXACTLY these keys (target_regions/changed_regions lists; edit_scope, "
        "garment_fidelity, pose integers; scope_reason/region_confidence strings):\n"
        '{"target_regions":[_], "changed_regions":[_], "edit_scope":_, "scope_reason":_, '
        '"region_confidence":_, "garment_fidelity":_, "pose":_}'
    )
    cloth_url = G.img_to_data_url(cloth_path)
    model_url = G.img_to_data_url(model_path) if model_path else None
    result_url = G.img_to_data_url(result_path)
    triples = [("[CLOTH]:", cloth_url)]
    if model_url:
        triples.append(("[MODEL]:", model_url))
    triples.append(("[RESULT]:", result_url))
    if swap and model_url:
        triples = [triples[0], triples[2], triples[1]]
    content = []
    for label, url in triples:
        content.append({"type": "text", "text": label})
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": instr})
    return [{"role": "user", "content": content}]


def score_h3(pick_ep, cloth, model_img, result, *, seeds, native_think=False,
             max_tokens=2048):
    """H3 = a SINGLE judge call doing the reference-anchored edit-scope read +
    compact fidelity. edit_scope is the first-class veto (score-cap 0.40);
    fidelity/pose are reported but never flip edit_scope. score = mean of the
    secondary atoms (0..1), capped by the veto. Returns (score, sub, ok, defect).
    """
    votes_es, votes_fid, votes_pose = [], [], []
    targets, changes, reasons, confs = [], [], [], []
    passes_ok = 0
    last_err = None
    has_model = model_img is not None and Path(model_img).is_file()
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        swap = (i % 2 == 1)
        try:
            msgs = _build_singlecall_messages(
                cloth, model_img if has_model else None, result, swap=swap)
            txt = G.call_judge(base_url, model, msgs, seed=seed,
                               enable_thinking=bool(native_think), max_tokens=max_tokens)
            parsed = G.extract_last_json(txt)
            if not parsed:
                txt = G.call_judge(base_url, model, msgs, seed=seed,
                                   enable_thinking=False, max_tokens=1024)
                parsed = G.extract_last_json(txt)
            if not parsed:
                last_err = "no json"
                continue
            tr = _coerce_region_list(parsed.get("target_regions"))
            ch = _coerce_region_list(parsed.get("changed_regions"))
            if has_model:
                es, reason = _scope_decision(tr, ch)
                if es is None:
                    raw = parsed.get("edit_scope")
                    try:
                        es = 1 if int(raw) == 1 else 0
                    except (TypeError, ValueError):
                        es = None
                    reason = str(parsed.get("scope_reason") or "ok")
            else:
                # no model image -> the reference-anchored read cannot run; defer.
                es, reason = 1, "no_model_image"
            fid = G._grade(parsed.get("garment_fidelity"))
            pose = G._grade(parsed.get("pose"))
            if es is not None:
                votes_es.append(es)
            if fid is not None:
                votes_fid.append(fid)
            if pose is not None:
                votes_pose.append(pose)
            targets.append(sorted(tr)); changes.append(sorted(ch))
            reasons.append(reason); confs.append(str(parsed.get("region_confidence") or ""))
            passes_ok += 1
        except Exception as e:
            last_err = str(e)

    if passes_ok == 0:
        raise RuntimeError(f"all H3 single-call passes failed; {last_err}")

    # edit_scope majority vote (ties -> 0, conservative for recall). If the read
    # had no model image and no edit_scope int landed, treat as edit_scope 1.
    edit_scope = 1
    if votes_es:
        edit_scope = 0 if votes_es.count(0) >= votes_es.count(1) else 1
    # secondary atom score (report-only): mean over graded atoms in 0..1.
    graded = [v for v in (votes_fid + votes_pose) if v is not None]
    base = (sum(graded) / (2.0 * len(graded))) if graded else 1.0
    fid_atom = int(round(sum(votes_fid) / len(votes_fid))) if votes_fid else None
    pose_atom = int(round(sum(votes_pose) / len(votes_pose))) if votes_pose else None

    sub = {
        "garment_fidelity": fid_atom, "pose": pose_atom,
        "edit_scope": int(edit_scope), "escope_ok": True,
        "escope_reason": next((r for v, r in zip(votes_es, reasons)
                               if v == edit_scope and r != "ok"), "ok"),
        "target_regions": targets[-1] if targets else [],
        "changed_regions": changes[-1] if changes else [],
        "escope_votes": votes_es,
        "escope_region_confidence": confs[-1] if confs else "",
        "single_call": True,
    }
    # secondary atoms report-but-don't-veto; ONLY edit_scope vetoes here.
    defect = False
    score, sub, defect = apply_editscope_veto(base, sub, defect, edit_scope, {})
    return float(score), sub, True, bool(defect)
