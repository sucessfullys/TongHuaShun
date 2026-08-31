#!/usr/bin/env python3
"""iter_019 CATEGORY-FREE garment-fidelity + preservation common helpers.

Derived from iter_018's _gf_common.py (the operator-endorsed champion recipe:
crossover agreement gate, iter_017 FP fixes, iter_018 overlap rule, prompt-CoT
think-then-mark, single-pass reads) with the iter_019 operator constraint
applied: **the sample-path category prefix (upper/lower/dress) is FORBIDDEN as
an evaluator input**. category_of(sample_key) is deleted; every category signal
is image-derived through the new category-channel functions below:

  - dino_argmax_category : DINOv2 cosine of input_cloth vs the INPUT_MODEL's
    three candidate crops (upper/lower/dress boxes); argmax + top-2 margin.
  - changeloc_category   : DINOv2 cosine of model-crop vs result-crop per box;
    the most-changed box (lowest cosine) wins.
  - silhouette_category  : numpy-only cloth-silhouette heuristic (floor channel).
  - classify_cloth_judge : one strict-JSON single-image judge call on the cloth.
  - merge_channels       : PR7 4-quadrant cross-check merge of two channels.
  - resolve_region       : label -> (crop category, nt-veto suppression,
    emphasis key); "uncertain" -> PR6 widest (dress) box + merged emphasis +
    nt-veto suppressed.

The PRES garment_swap atom + build_pres_messages are rewritten TARGET-FREE
(PR3: the garment that changed IS the target; two-garment rule; "no garment
changed" is a valid clean state) while the iter_018 OVERLAP rule and the
iter_017 UNDERWEAR rule stay byte-verbatim. build_gf_messages gains optional
step0 (in-prompt STEP-0 region declaration), full_frame, reference_anchored
(PR5) and force_merged knobs; the iter_018 LOWER/UPPER emphasis texts stay
verbatim, selected by the image-derived inferred_category, with a compact
merged block for "uncertain". build_joint_messages (3-image joint read) backs
the 2-call cost-floor wildcard.

GROUND-TRUTH EXCEPTION: the path prefix may be used ONLY as scoring ground
truth for category-inference accuracy (the catinfer baseline) — that lives in
_i19_score_fns, never here.
"""
import base64
import io
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from . import metric_common as M

# ---- the 7 garment-fidelity atoms (cloth vs result-garment-crop) ----
GF_ATOMS = [
    ("color", "COLOR — does the garment colour in the result match the REFERENCE garment (hue/shade/saturation)? Compare the WORN garment's colour against the REFERENCE-cloth colour directly. Grade 0 ONLY for a CLEAR, obvious hue change (e.g. red->blue, black->white); IGNORE shade/brightness/saturation shifts caused by lighting, shadow, fabric drape or being worn-on-body — those are NOT colour defects."),
    ("pattern", "PATTERN/TEXTURE — does the print / texture / weave match the REFERENCE?"),
    ("material", "MATERIAL — does the fabric/material look like the REFERENCE (e.g. denim vs silk)?"),
    ("garment_type", "GARMENT-TYPE — is it the same garment category/style as the REFERENCE (no skirt<->pants, no sleeve-length change)?"),
    ("structure", "STRUCTURE/FIT — hem, collar, sleeve length, silhouette, overall length/fit vs the REFERENCE? Grade 0 ONLY for a CLEAR structural mismatch; do NOT flag normal worn-on-body drape, tuck, or fold."),
    ("fine_details", "FINE-DETAILS — buttons, belt/belt-knot, cuffs, zippers, pockets vs the REFERENCE? Grade 0 ONLY when a clearly-present REFERENCE detail is clearly wrong or absent in the worn result; do NOT flag details merely hidden by pose, arms, tuck or crop."),
    ("logo", "LOGO/TEXT — does any logo/printed text match the REFERENCE? Grade 2 (matches) UNLESS a logo that is clearly present in the REFERENCE is clearly REPLACED by a different/garbled logo in the result. IMPORTANT: a logo can become MORE or LESS visible after wearing (revealed by an open jacket, hidden by a fold/arm, or shifted by drape) — that visibility change is NOT a defect, so do NOT grade 0 for a logo that merely looks 'missing'/'smaller'/'partly hidden'. Use n/a when the reference has no logo."),
]
GF_KEYS = [k for k, _ in GF_ATOMS]

# ---- the 2 preservation atoms (input_model vs result-full) ----
# pose: verbatim iter_018. garment_swap: REWRITTEN target-free (PR3 + the
# two-garment rule); the iter_018 OVERLAP rule and the iter_017 UNDERWEAR rule
# inside it are kept byte-verbatim.
PRES_ATOMS = [
    ("pose", "POSE — is the PERSON's body pose, stance and proportions preserved and ANATOMICALLY natural in the result vs the ORIGINAL PERSON (same overall pose; no warped/distorted torso, no extra/missing/melted limbs or hands, no broken anatomy)? Grade 0 ONLY for CLEARLY broken anatomy (extra/missing/melted limbs or hands, grossly warped torso). A different stance, turn, arm position, crop or natural pose change is NOT a defect — grade 2."),
    ("garment_swap", "GARMENT-SWAP & LAYERING — the garment that CHANGED between [ORIGINAL PERSON] and [TRY-ON RESULT] IS the intended try-on target; NEVER penalize the target garment for being different. Do NOT flag minor face/skin/hair regen. Grade 0 ONLY when: (a) a SECOND garment ALSO changed — if BOTH the upper-body and lower-body garments clearly changed, the one that does not correspond to the [REFERENCE GARMENT] (the intended try-on garment) is a non-target swap, grade 0; (b) an EXTRA/layered garment appears that neither input had — *** OVERLAP / EXTRA-GARMENT RULE (grade 0) *** — look CAREFULLY for an EXTRA garment hallucinated where it does not belong: above all TROUSERS / PANTS / JEANS / LEGGINGS / SHORTS visible UNDERNEATH a SKIRT or DRESS (a skirt/dress should show BARE LEGS or sheer tights, NOT a second pair of pants under it), or a shirt/top layered under a dress, or any second garment overlapping the target that neither the REFERENCE cloth nor the ORIGINAL person has. A wrong layered/overlapping garment is garment_swap=0 EVEN IF the target garment itself looks correct; (c) the person became a clearly different identity. UNDERWEAR RULE — if the ORIGINAL PERSON wore only UNDERWEAR/bra/briefs/minimal clothing on a region (it started bare), the try-on garment appearing there is the CORRECT edit (grade 2); a non-target region bare in BOTH original and result is grade 2. If NO garment changed at all, that is a VALID CLEAN state for this atom — grade 2 (do not force a target)."),
]
PRES_KEYS = [k for k, _ in PRES_ATOMS]

ALL_KEYS = GF_KEYS + PRES_KEYS

# ---- category labels are PROMPT TEXT only (never path-derived) -------------
CATEGORY_LABEL = {
    "upper": "an UPPER-BODY garment (top / shirt / jacket — the TORSO garment)",
    "lower": "a LOWER-BODY garment (trousers / pants / skirt / shorts — the LEGS garment)",
    "dress": "a DRESS / full-length outfit (covers the torso AND the lower body)",
}

# NOTE: category_of(sample_key) is intentionally DELETED (iter_019 constraint:
# the path prefix is forbidden as an evaluator input).

CROP_CATEGORIES = ("upper", "lower", "dress")


def _target_box(w, h, category):
    """Bounding box of the TARGET garment in a full-person frame, by category."""
    cat = category or "upper"
    if cat == "lower":
        # iter_014 PR1: widen + extend the lower box to the FULL leg (waistband at
        # 0.38 down to the ankle/hem at 1.0, wide enough for baggy/wide-leg) so
        # length-change trousers (long2long/short2long) are not clipped — the
        # dominant iter_013 lower-body failure was mis-judged trouser length/fit.
        return (int(w * 0.10), int(h * 0.38), int(w * 0.90), int(h * 1.0))
    if cat == "dress":
        return (int(w * 0.16), int(h * 0.20), int(w * 0.84), int(h * 0.95))
    # iter_015 PR2: widen the upper box (arms out to the sides, long sleeves reach
    # past the old 0.70 cutoff) so sleeve-length changes (long2short) and the garment
    # hem are not clipped — the dominant iter_014 upper failure was mis-judged sleeves.
    return (int(w * 0.08), int(h * 0.16), int(w * 0.92), int(h * 0.85))  # upper


def CROP_BOXES(w, h):
    """The three candidate target boxes (shared crop-box geometry block —
    _target_box stays byte-verbatim iter_018)."""
    return {c: _target_box(w, h, c) for c in CROP_CATEGORIES}


# ---------------- VLM plumbing ----------------

def img_to_data_url(path, max_side=768):
    im = path if isinstance(path, Image.Image) else Image.open(path)
    im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        s = max_side / float(max(w, h))
        im = im.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _post_chat(base_url, payload, timeout):
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


THINK_MIN_TOKENS = int(os.environ.get("ERA_THINK_MAX_TOKENS", "8000"))


def _think_on() -> bool:
    """iter_030: the operator asked to OPEN THINK MODE. ERA_ENABLE_THINKING=1
    forces native <think> ON for every judge call (it wins over the caller's
    enable_thinking kwarg). Default OFF preserves the frozen prompt-CoT behaviour
    for anyone not setting the env."""
    return os.environ.get("ERA_ENABLE_THINKING", "").strip().lower() in (
        "1", "true", "yes", "on")


def call_judge(base_url, model, messages, max_tokens=3072, temperature=0.0,
               seed=42, timeout=900, enable_thinking=False, atom_read=False):
    """THINK-THEN-MARK. Two modes:

    * Default (ERA_ENABLE_THINKING unset): PROMPT-LEVEL CoT — the prompt forces
      brief reasoning before the final JSON; native <think> OFF (Qwen3.5's native
      thinking runs to the token ceiling, intractable at scale).
    * ERA_ENABLE_THINKING=1 (iter_030 operator request): native <think> ON for
      the DECISIVE edit-scope reads; max_tokens floored to ERA_THINK_MAX_TOKENS.

    ``atom_read=True`` opts a call OUT of native thinking even when the env is on:
    the v13 garment-fidelity / preservation ATOM grades (simple 0/1/2 rubric
    reads) truncate before their multi-atom JSON under native <think> (~24% no-json
    on the hybrids), and they earned 0.973 at prompt-CoT — they do not need deep
    thinking. Only the edit-scope STEP-A/B/C decision (the iter_030 focus) thinks.
    """
    think = (enable_thinking or _think_on()) and not atom_read
    if think:
        max_tokens = max(max_tokens, THINK_MIN_TOKENS)
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature, "seed": seed}
    # ALWAYS pass enable_thinking explicitly: Qwen3.5's chat template defaults
    # thinking ON, so merely omitting the kwarg leaves native thinking running.
    # Qwen2.5-VL rejects the kwarg (HTTP 400/422) -> strip and retry plain.
    payload["chat_template_kwargs"] = {"enable_thinking": bool(think)}
    try:
        out = _post_chat(base_url, payload, timeout)
    except urllib.error.HTTPError as e:
        if e.code in (400, 422):
            payload.pop("chat_template_kwargs", None)
            out = _post_chat(base_url, payload, timeout)
        else:
            raise
    msg = out["choices"][0].get("message", {}) or {}
    content = msg.get("content") or ""
    # vLLM reasoning parsers split native thinking into reasoning_content and the
    # answer into content. Prefer content (the final JSON); if it carries no JSON
    # (e.g. truncated/empty answer), append reasoning_content so extract_last_json
    # can still recover a JSON the model emitted before the cut.
    if "{" not in content:
        reasoning = msg.get("reasoning_content") or ""
        if reasoning:
            content = (content + "\n" + reasoning) if content else reasoning
    return content


def extract_last_json(text):
    """Return the LAST {...} object parseable in the text (after the CoT).

    Think-mode hardening (iter_030): strip a complete inline <think>...</think>
    block first so a JSON the model happened to mention INSIDE its reasoning
    never wins over the final answer JSON. An UNCLOSED <think> (the answer was
    truncated mid-thought) is left as-is — there is no final JSON to find and the
    caller's no-JSON fallback fires."""
    if text and "<think>" in text and "</think>" in text:
        text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL)
    depth = 0
    start = None
    last = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                frag = text[start:i + 1]
                try:
                    last = json.loads(frag)
                except Exception:
                    pass
    return last


def _grade(v):
    """Coerce a 0/1/2 grade (or 'na') to int 0..2 or None for n/a."""
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("na", "n/a", "none"):
            return None
        try:
            v = int(float(s))
        except Exception:
            return None
    if isinstance(v, (int, float)):
        return max(0, min(2, int(round(v))))
    return None


_GRADE_RUBRIC = (
    "For EACH atom give a grade:\n"
    "  2 = matches / preserved (including when only subtle/worn-on-body differences exist),\n"
    "  1 = a genuine but mild difference,\n"
    "  0 = CLEARLY wrong/different/broken (an obvious, unambiguous defect).\n"
    "Use \"na\" only when the atom does not apply.\n"
    "Do NOT grade 0 or 1 for a difference you are unsure about or that is only subtle — "
    "when in doubt, grade 2. Only flag a problem when it is CLEAR and obvious.\n"
)


# iter_009 PR1: clear-discipline tightening block — raises the per-atom firing
# bar, gates a 0 on NAMED pixel evidence, and excuses the worn-vs-flat / dress
# false-flag cluster the 235B over-fired on at iter_008 (text anchors, not images,
# to stay within the 3-image serve cap).
CLEAR_DISCIPLINE_BLOCK = (
    "\nSTRICT PRECISION DISCIPLINE (raise the firing bar):\n"
    "  - Mark an atom 0 ONLY on a CLEAR, UNAMBIGUOUS defect. Before marking ANY atom 0 "
    "you MUST name the specific pixel-level evidence (which region, what differs). "
    "An atom with no nameable evidence defaults to 2.\n"
    "  - These are NOT defects (do not flag): benign worn-vs-flat deformation, fabric "
    "drape/fold, lighting/shadow shift, natural pose change, a small colour shift within "
    "one perceptual step, and the inherent face/skin/hair regeneration of try-on.\n"
    "  - Calibration anchors (worn-on-body deformation / lighting on DRESS samples like "
    "grey-background and white-background dresses are NOT garment defects — do not flag "
    "them as color/structure 0).\n"
)


# iter_018 LOWER-body emphasis text — VERBATIM (now selected by the image-derived
# inferred_category, never by the path prefix).
LOWER_EMPHASIS_BLOCK = (
    "LOWER-BODY LENGTH & FIT (judge these carefully under structure & garment_type): "
    "does the result reproduce the REFERENCE trouser/skirt LENGTH (full-length vs cropped "
    "vs shorts; where the hem falls — ankle / mid-calf / knee / above-knee), the LEG WIDTH "
    "and CUT (skinny / straight / wide / flared), and the RISE? A clear length mismatch "
    "(e.g. shorts reproduced as full-length trousers, or long pants shown cropped) is a "
    "structure=0 (or garment_type=0) defect. Ignore length differences that are only due to "
    "pose/stance or crop framing.\n"
)

# iter_018 UPPER-body emphasis text — VERBATIM.
UPPER_EMPHASIS_BLOCK = (
    "UPPER-BODY SLEEVE/NECKLINE & FIT (judge under structure & garment_type): "
    "does the result reproduce the REFERENCE SLEEVE LENGTH (long / 3-4 / short / "
    "sleeveless — where the cuff falls), the NECKLINE (crew / v-neck / collar / hood), "
    "the GARMENT LENGTH (cropped / regular / longline), and the overall CUT/FIT? "
    "A clear sleeve-length mismatch (e.g. long sleeves reproduced as short, or a "
    "collar/hood dropped or added) is a structure=0 (or garment_type=0) defect.\n"
    "UPPER PRECISION GUARD: on plain white/grey-background single-garment uppers, do "
    "NOT manufacture a defect — most of these reproduce the garment correctly. Flag an "
    "atom 0 ONLY when you can name the specific region that CLEARLY differs; benign "
    "worn-vs-flat drape, fold, wrinkle, sleeve-roll, tuck, and lighting on a plain "
    "background are NOT defects.\n"
)

# PR9 COMPACT MERGED emphasis block for the "uncertain" region (covers BOTH axes
# in ~6 lines — sleeve-length/neckline + trouser-length/fit cues).
MERGED_EMPHASIS_BLOCK = (
    "GARMENT LENGTH/FIT EMPHASIS (apply whichever axis matches the garment's worn region; "
    "judge under structure & garment_type):\n"
    "  - UPPER-BODY axis: does the result reproduce the REFERENCE SLEEVE LENGTH (long / 3-4 / "
    "short / sleeveless — where the cuff falls), the NECKLINE (crew / v-neck / collar / hood), "
    "the GARMENT LENGTH (cropped / regular / longline), and the overall CUT/FIT? A clear "
    "sleeve-length or collar/hood mismatch is structure=0 (or garment_type=0).\n"
    "  - LOWER-BODY axis: does the result reproduce the REFERENCE trouser/skirt LENGTH "
    "(full-length vs cropped vs shorts; where the hem falls — ankle / mid-calf / knee / "
    "above-knee), the LEG WIDTH and CUT (skinny / straight / wide / flared), and the RISE? "
    "A clear length mismatch is structure=0 (or garment_type=0).\n"
    "  - Ignore length differences that are only due to pose/stance or crop framing; do NOT "
    "manufacture a defect — benign worn-vs-flat drape, fold, wrinkle, sleeve-roll, tuck and "
    "lighting are NOT defects; flag an atom 0 ONLY when you can name the specific region that "
    "CLEARLY differs.\n"
)

# ===========================================================================
# iter_020 ADDITIVE atom-fix blocks (gated on build_gf_messages atom_fix=True).
# PURE ADDITIONS injected AFTER the frozen iter_017 FP-fix clauses and AFTER the
# existing LOWER/UPPER/MERGED/dress emphasis selection, so they can never shadow
# or rewrite any frozen FP text. They are appended to cat_line LAST.
#   - DRESS_FULLBODY_BLOCK guards the iter_009 dress-FP mode (RISK-A).
#   - LOWER_LONG_LENGTH_BLOCK strengthens long2long/short2long/long2short cues
#     (RISK-B) while keeping the iter_014 pose/crop clause VERBATIM.
# ===========================================================================

# iter_020 NEW dress full-body block (injected when inferred_category=="dress"
# OR force_merged). ADDITIVE — does NOT replace the (absent) dress block.
DRESS_FULLBODY_BLOCK = (
    "DRESS / ONE-PIECE FULL-BODY EMPHASIS (judge under structure & garment_type): "
    "this is a full-body dress / one-piece, so check the WHOLE garment carefully — "
    "the full-body SILHOUETTE, the HEM LENGTH (where the hem falls — floor / ankle / "
    "calf / knee / above-knee / mini), and the WAIST-TO-HEM PROPORTION (bodice vs "
    "skirt balance) all reproduce the REFERENCE one-piece. A clear hem-length class "
    "change (e.g. a maxi reproduced as a mini, or knee-length shown floor-length) or a "
    "clear silhouette/proportion mismatch is a structure=0 (or garment_type=0) defect. "
    "Do NOT flag legitimate drape, fabric fall, lighting, or clean-background variation "
    "as a defect — only a CLEAR garment-identity or length/structure mismatch is a defect.\n"
)

# iter_020 NEW lower long-length cues — appended to the existing LOWER block when
# atom_fix=True (injected for "lower" OR force_merged). ADDITIVE. The iter_014
# pose/crop clause is repeated VERBATIM here (RISK-B).
LOWER_LONG_LENGTH_BLOCK = (
    "LOWER LONG-LENGTH CUES (long2long / short2long / long2short — judge under "
    "structure & garment_type): compare WHERE THE HEM FALLS — ankle vs calf vs knee vs "
    "above-knee; a clear length-class mismatch (e.g. full-length trousers reproduced as "
    "cropped/shorts, or short trousers shown full-length, or a long-to-long pair whose "
    "leg length clearly changed) is a structure / garment_type defect (grade 0). "
    "Ignore length differences that are only due to pose/stance or crop framing.\n"
)

# STEP-0 in-prompt region inference (PR2; strict-JSON keys appended to the read's
# own output schema).
STEP0_BLOCK = (
    "STEP 0 (do this FIRST): state where on the body the [REFERENCE GARMENT] is worn — "
    "\"upper-body\", \"lower-body\", or \"full-body dress/one-piece\" — and your confidence "
    "\"high\" or \"low\" (low for ambiguous cases: long shirt vs short dress, tunic, overalls). "
    "Include the keys \"worn_region\" and \"region_confidence\" in your final JSON.\n"
)

# PRES-read variant of STEP-0 (the PRES read sees ORIGINAL PERSON + RESULT, not
# the reference cloth, so the target is defined as the garment that changed).
PRES_STEP0_BLOCK = (
    "STEP 0 (do this FIRST): identify the garment that CHANGED between the [ORIGINAL PERSON] "
    "and the [TRY-ON RESULT] (the intended try-on target) and state where on the body it is "
    "worn — \"upper-body\", \"lower-body\", or \"full-body dress/one-piece\" — and your "
    "confidence \"high\" or \"low\" (low for ambiguous cases: long shirt vs short dress, "
    "tunic, overalls; also low if no garment appears to have changed). Include the keys "
    "\"worn_region\" and \"region_confidence\" in your final JSON.\n"
)

# PR5 reference-anchored full-frame instruction (the nocat config's category-free
# anchoring — no crop, no category surface).
REFERENCE_ANCHOR_LINE = (
    "Locate the SINGLE garment in [TRY-ON RESULT] that corresponds to the [REFERENCE GARMENT]; "
    "judge ONLY that garment; treat all other visible garments as pre-existing.\n"
)

# ===========================================================================
# iter_021 ADDITIVE FP-fix constants (gated behind the new fp_fix / coverage_first
# kwargs). PURE ADDITIONS: appended AFTER the frozen iter_018 overlap clause +
# iter_017 underwear clause (garment_swap) and AFTER the existing fine_details
# atom text. They cannot shadow or rewrite any frozen FP text — the negative
# example inside GARMENT_SWAP_COVERAGE_COMPENSATION re-states the iter_018 overlap
# rule, reinforcing (not replacing) it. Text carried VERBATIM from the iter_021
# experiment_brief shared_constants.
# ===========================================================================

# = brief.shared_constants.fp_fix_A_garment_swap_text VERBATIM
GARMENT_SWAP_COVERAGE_COMPENSATION = (
    "When the NEW try-on garment OCCLUDES/covers/tucks/displaces parts of the ORIGINAL clothing (a longer top covering the waistband; a jacket over a shirt; a dress hem changing what's visible of the legs), that is the CORRECT edit — grade garment_swap=2. Grade 0 ONLY for a clearly DIFFERENT non-target garment that is NOT explained by the new garment's occlusion. VISIBLE-vs-OCCLUDED: a region simply HIDDEN by the new garment is fine; an EXTRA garment ADDED where the new garment does not reach is a swap. NEGATIVE EXAMPLE (still grade 0): trousers/leggings/pants visible UNDERNEATH a skirt or dress — a skirt/dress should reveal bare legs or sheer tights, NOT a second pair of pants the new garment does not explain (the iter_018 overlap rule)."
)

# = brief.shared_constants.fp_fix_B_fine_details_text VERBATIM
FINE_DETAILS_DISCIPLINE = (
    "Only flag a CLEARLY-present reference-cloth detail (button/logo/pocket/zip) clearly WRONG or ABSENT in the result; do NOT flag details hidden by pose, by the new garment's coverage/occlusion, by drape, or by crop."
)

# iter_021 coverage-first PRES restructure step (gated on coverage_first=True,
# which implies fp_fix). PREPENDED before the atom grading, AFTER the frozen
# target/underwear/overlap clauses, so the genuine-swap branch (iter_018 overlap)
# is preserved inside the grading step.
COVERAGE_FIRST_BLOCK = (
    "\nCOVERAGE MAP (do this FIRST): state which parts of the ORIGINAL person's clothing the NEW garment covers/occludes/displaces. THEN grade garment_swap using that coverage map: a region hidden by the new garment is NOT a swap; only an EXTRA garment the coverage map does not explain is a swap (pants under a skirt/dress = still a swap).\n"
)


# ===========================================================================
# iter_022 ADDITIVE SUIT / two-piece detection (gated behind the new suit=True
# kwarg threaded through build_gf_messages / build_pres_messages). PURE
# ADDITIONS: suit=False is BYTE-BEHAVIOUR-IDENTICAL to iter_021 (verified by the
# v6 / purevlm controls reproducing iter_021). When suit=True:
#   - SUIT_DETECT_STEP0_TEXT (PR1) is appended to the STEP-0 read (the GF read-A
#     STEP-0 block) so the judge emits is_two_piece + two_piece, and the demanded
#     final-JSON keys gain is_two_piece / two_piece.
#   - SUIT_GRADE_TEXT (PR2) is appended to the PRES read as an ADDITIVE branch
#     placed AFTER the frozen iter_018 overlap + iter_017 underwear + iter_021
#     coverage clauses so it cannot shadow them. For the COVERAGE-FIRST form
#     (coverage_first=True) the PR3 ordered two-piece-first CoT structure wraps it;
#     for the INLINE form (coverage_first=False) PR4 places it inline after the
#     frozen clauses.
# Text quoted VERBATIM from iter_022/design/candidates/hybrid-innovator.md PR1/PR2.
# ===========================================================================

# = hybrid-innovator.md `suit_detect_step0_text` (PR1) VERBATIM
SUIT_DETECT_STEP0_TEXT = (
    "STEP-0 (category read of the `input_cloth` reference, image-derived ONLY — never the file path): in addition to the garment category, decide whether the reference is a SINGLE-GARMENT or a TWO-PIECE / SUIT set. A TWO-PIECE/SUIT reference is a combined upper+lower outfit supplied together as one `input_cloth` image — e.g. a jacket+trousers suit, a blazer+pants set, or a top+skirt set — where BOTH an upper piece AND a lower piece are clearly present in the reference image. Emit `is_two_piece` (boolean) and, when true, `two_piece = {upper: <upper category>, lower: <lower category>}`. Decide is_two_piece from the reference image content alone (two distinct upper- and lower-body garments visible as one coordinated set); a single dress or jumpsuit covering both regions as ONE continuous garment is NOT two-piece (`is_two_piece=false`). If uncertain whether a second piece is genuinely part of the reference, default `is_two_piece=false` (conservative — a false suit-positive would relax garment_swap and risk the degenerate-recall trap)."
)

# = hybrid-innovator.md `suit_grade_text` (PR2) VERBATIM
SUIT_GRADE_TEXT = (
    "SUIT-AWARE garment_swap (applies ONLY when STEP-0 returned `is_two_piece=true`; otherwise SKIP this clause entirely and grade single-garment as the frozen iter_021 clause specifies): a TWO-PIECE/SUIT reference is worn CORRECTLY when BOTH pieces (the reference upper AND the reference lower) are present and match the reference on the model — this full two-piece application is the CORRECT edit, NOT a swap → grade garment_swap=2. The REAL defect on a suit is one of: (a) PARTIAL application — only the upper OR only the lower piece of the reference suit is applied and the OTHER reference piece is MISSING/unchanged (the model still wears its original other-region garment) → grade garment_swap=0; or (b) a clearly DIFFERENT garment substituted for either piece, not matching the reference suit → grade garment_swap=0. Wearing both reference pieces is NEVER a swap merely because two garments changed — that is the intended edit for a suit. This clause does NOT relax single-garment grading and does NOT shadow the iter_018 overlap rule: an EXTRA hallucinated layer the reference suit does not include (e.g. pants the new garment does not explain) still flags garment_swap=0."
)

# PR1 STEP-0 wrapper appended to the GF STEP-0 block (slots after STEP0_BLOCK
# without altering the frozen STEP0_BLOCK text).
SUIT_STEP0_BLOCK = (
    "TWO-PIECE / SUIT DETECTION — " + SUIT_DETECT_STEP0_TEXT + "\n"
)

# PR4 INLINE suit branch (appended to the PRES garment_swap clauses AFTER the
# frozen overlap + underwear + iter_021 coverage clauses).
SUIT_INLINE_BLOCK = (
    "\n" + SUIT_GRADE_TEXT + "\n"
)

# PR3 ordered two-piece-first wrapper for the COVERAGE-FIRST form: an explicit
# ordered CoT step placed BEFORE the coverage-first single-garment reasoning. It
# carries SUIT_GRADE_TEXT (PR2) verbatim and routes to the UNCHANGED frozen
# single-garment read when is_two_piece=false.
SUIT_ORDERED_BLOCK = (
    "\nTWO-PIECE / SUIT determination (do this BEFORE the coverage-first single-garment reasoning, as an explicit ordered step): (0) is the reference a SINGLE garment or a TWO-PIECE / SUIT set (per the STEP-0 is_two_piece determination)? → if SUIT (is_two_piece=true): check whether BOTH reference pieces are present-and-correct on the model — both present-and-correct ⇒ garment_swap=2, a partial application or a clearly different substituted piece ⇒ garment_swap=0 — then STOP the garment_swap branch. → if SINGLE (is_two_piece=false): run the UNCHANGED frozen single-garment coverage-first read below.\n"
    + SUIT_GRADE_TEXT + "\n"
)


# ===========================================================================
# iter_023 ADDITIVE SUIT-SHARPENING (gated behind the suit_sharp=True kwarg,
# meaningful ONLY when suit=True). PURE ADDITIONS: suit_sharp=False is
# BYTE-BEHAVIOUR-IDENTICAL to the iter_022 suit method (the v7-suit-fpfix-3call
# control reproduces iter_022 exactly). When suit_sharp=True the two clauses
# below are appended to the suit branch of the PRES read AFTER the frozen
# underwear + overlap + coverage clauses AND after the iter_022 SUIT_GRADE_TEXT,
# so they can never shadow the frozen single-garment path. They fire ONLY for a
# PR2-confirmed two-piece set (is_two_piece=true); single-garment grading is
# unchanged. PR2 (overlap-vs-suit) runs FIRST (gates branch entry); PR1
# (per-piece enumeration) runs for a confirmed two-piece set. The block is
# BIT-IDENTICAL across the INLINE and ORDERED forms; only the PRES form differs.
# ===========================================================================

SUIT_SHARP_PR2_TEXT = (
    "OVERLAP-vs-SUIT DISAMBIGUATION (do this FIRST, before grading a reference as TWO-PIECE/SUIT): confirm it is a genuine coordinated upper+lower SET supplied together as one input_cloth (jacket+trousers, blazer+pants, top+skirt set) intended to be worn as a pair. It is NOT a two-piece suit if: (b) it is a SINGLE garment that merely LAYERS over or covers existing clothing (the iter_018 pants-under-skirt case), or (c) it is a single long garment (dress/coat). Cases (b) and (c) MUST fall through to the byte-frozen single-garment path so the iter_018 overlap recall rule still fires (extra hallucinated layers / pants-under-skirt still flag garment_swap=0). Derive purely from the input_cloth image; never from the sample path."
)

SUIT_SHARP_PR1_TEXT = (
    "PER-PIECE PRESENT-AND-CORRECT ENUMERATION (runs ONLY for a PR2-confirmed two-piece set): enumerate the two reference pieces from the input_cloth image (e.g. {upper: jacket, lower: trousers}). For EACH piece independently decide present-and-correct vs missing/replaced on the model: BOTH pieces present and correct => garment_swap = 2 (CORRECT full-set application, NOT a swap); ANY piece missing or replaced by a different garment => garment_swap = 0 (the REAL defect: partial application — only top OR only bottom — or a different garment)."
)

SUIT_SHARP_BLOCK = (
    "\n" + SUIT_SHARP_PR2_TEXT + "\n" + SUIT_SHARP_PR1_TEXT + "\n"
)

# iter_024 ADDITIVE SUIT-RELAXATION (suit_relaxed=True, meaningful only when
# suit_sharp=True). Replaces the iter_023 strict PR1 with confirm-correct-first,
# clear-defect-only grading to KILL the suit garment_swap false-positives the
# operator flagged on CES. PR2 gate stays byte-frozen and runs first.
# suit_relaxed=False reproduces the iter_023 strict block byte-for-byte.
SUIT_RELAXED_PR1_TEXT = (
    "CONFIRM-CORRECT-FIRST PER-PIECE CHECK — RELAXED, CLEAR-DEFECT-ONLY (runs ONLY for a PR2-confirmed two-piece set; if is_two_piece=false this whole clause is SKIPPED and the byte-frozen single-garment path grades garment_swap as before). Order the suit per-piece check confirm-correct-FIRST: (1) ask FIRST 'are BOTH reference pieces (the reference upper AND the reference lower) present and RECOGNIZABLE on the model?' — if YES, this is a CORRECT full two-piece application: grade garment_swap = 2 and STOP the garment_swap branch. A recognizable full two-piece outfit is the DEFAULT correct case; do NOT down-grade it. (2) ONLY grade garment_swap = 0 when a piece is CLEARLY a DEFECT: either (a) a reference piece is CLEARLY MISSING — not worn at all on the model, the model still shows its original other-region garment in that slot (a genuine partial application, only the top OR only the bottom of the reference set applied); or (b) a CLEARLY DIFFERENT garment is worn in a reference piece's place (a substituted garment that is not the reference piece at all). (3) Do NOT flag garment_swap=0 for any of: drape or how one piece sits over/under the other; PARTIAL OCCLUSION of one piece by the other (e.g. trousers partly hidden under a long jacket — the trousers are still present, just covered, so this is grade 2); minor per-piece differences in fit, length, styling, or exact shade that do not change garment IDENTITY. When in doubt between 'a piece is occluded/draped' and 'a piece is missing', prefer grade 2 (a recognizable full application is correct). This clause does NOT relax single-garment grading and does NOT shadow the iter_018 overlap rule: an EXTRA hallucinated layer the reference suit does not include (e.g. pants the new garment does not explain) still flags garment_swap=0 via the frozen overlap clause."
)
SUIT_RELAXED_ICL_TEXT = (
    "CONTRASTIVE EXAMPLES (generic, category-free — these illustrate the clear-defect-only rule; they are NOT tied to any specific sample, file name, or path): CORRECT full two-piece applications that must grade garment_swap=2 (do NOT flag these): a jacket+trousers set where the trousers are partly hidden under a long jacket but clearly present; a blazer+pants set where one piece is slightly different in fit/length but is the same garment identity; a top+skirt set worn fully where drape or pose makes one piece sit differently. GENUINE defects that must grade garment_swap=0: only the jacket of a jacket+trousers reference is applied and the model still wears its own original lower garment (partial application, lower piece MISSING); a clearly DIFFERENT garment (e.g. a dress, or unrelated pants) is worn where a reference piece should be (substitution). Use these as the discrimination boundary: occlusion / drape / minor per-piece difference => grade 2; a clearly missing piece OR a clearly different substituted garment => grade 0. (No sample keys, no path prefixes, no dataset-specific identifiers appear here — this is a generic anti-pattern, not memorized cases.)"
)


def _suit_block(suit_relaxed=False, suit_icl=False):
    """PR2 gate FIRST (byte-frozen), then PR1 — STRICT (iter_023) when
    suit_relaxed=False, RELAXED confirm-correct-first (iter_024) when True;
    optional generic contrastive-ICL addendum. _suit_block(False,False) ==
    SUIT_SHARP_BLOCK byte-for-byte (control invariant)."""
    pr1 = SUIT_RELAXED_PR1_TEXT if suit_relaxed else SUIT_SHARP_PR1_TEXT
    block = "\n" + SUIT_SHARP_PR2_TEXT + "\n" + pr1 + "\n"
    if suit_icl:
        block = block + SUIT_RELAXED_ICL_TEXT + "\n"
    return block


def map_worn_region(wr):
    """Normalize a judge-declared worn_region string to upper/lower/dress/uncertain."""
    s = str(wr or "").strip().lower()
    if not s:
        return "uncertain"
    if "upper" in s or "top" in s or "torso" in s:
        return "upper"
    if "lower" in s or "leg" in s or "trouser" in s or "pant" in s or "skirt" in s:
        return "lower"
    if ("dress" in s or "full" in s or "one-piece" in s or "one piece" in s
            or "onepiece" in s or "jumpsuit" in s or "overall" in s):
        return "dress"
    return "uncertain"


def build_gf_messages(cloth_path, result_path, swap=False, evidence=None,
                      inferred_category="upper", clear_discipline=False,
                      step0=False, full_frame=False, reference_anchored=False,
                      force_merged=False, atom_fix=False, fp_fix=False,
                      suit=False):
    """Side-by-side per-atom cloth-vs-result prompt (iter_007 lineage). swap flips
    image order (debiasing); labels always name REFERENCE vs RESULT.

    iter_019 changes vs iter_018:
      - `inferred_category` ("upper"/"lower"/"dress"/"uncertain") is IMAGE-DERIVED
        (STEP-0 / DINO argmax / cloth pre-pass / cross-check) — never the path
        prefix. It selects the crop box AND the emphasis block: the iter_018
        LOWER/UPPER blocks verbatim, a compact MERGED block on "uncertain"
        (PR6 wide box), nothing extra on "dress" (as iter_018).
      - `step0=True` prepends the STEP-0 region-inference block and extends the
        demanded final JSON keys with worn_region/region_confidence.
      - `full_frame=True` shows the FULL result frame (no crop).
      - `reference_anchored=True` (PR5) injects the reference-anchor instruction
        and uses the FULL frame; no emphasis block unless force_merged.
      - `force_merged=True` always injects the MERGED emphasis block.
    """
    # iter_021 ADDITIVE FP-fix B: append the fine_details discipline AFTER the
    # existing frozen fine_details atom text (purely additive; gated on fp_fix).
    _gf_atoms = ([(k, (desc + " " + FINE_DETAILS_DISCIPLINE) if (fp_fix and k == "fine_details") else desc)
                  for k, desc in GF_ATOMS])
    atoms_txt = "\n".join(f"  - {k}: {desc}" for k, desc in _gf_atoms)
    cd_block = CLEAR_DISCIPLINE_BLOCK if clear_discipline else ""
    cat = (inferred_category or "uncertain").lower()
    if cat not in ("upper", "lower", "dress", "uncertain"):
        cat = "uncertain"
    # ---- region/framing line ----
    if reference_anchored:
        cat_line = REFERENCE_ANCHOR_LINE
    elif full_frame:
        cat_line = ("The [TRY-ON RESULT] image is the FULL generated frame (no crop). First "
                    "identify the garment in the result that corresponds to the [REFERENCE "
                    "GARMENT], then judge ONLY that garment.\n")
    elif cat == "uncertain":
        cat_line = ("The garment's worn region could not be determined with confidence, so the "
                    "[TRY-ON RESULT] image shows a WIDE region covering both the torso and the "
                    "legs. First identify the garment in the result that corresponds to the "
                    "[REFERENCE GARMENT], then judge ONLY that garment.\n")
    else:
        cat_line = ("This is a try-on of " + CATEGORY_LABEL.get(cat, CATEGORY_LABEL["upper"]) +
                    ". The [TRY-ON RESULT] image is cropped to that garment region.\n")
    # ---- emphasis block (selected by the inferred category) ----
    if force_merged:
        cat_line += MERGED_EMPHASIS_BLOCK
    elif reference_anchored:
        pass  # clean reference-anchored read: no extra emphasis
    elif cat == "uncertain":
        cat_line += MERGED_EMPHASIS_BLOCK
    elif cat == "lower":
        cat_line += LOWER_EMPHASIS_BLOCK
    elif cat == "upper":
        cat_line += UPPER_EMPHASIS_BLOCK
    # dress: no extra block (as iter_018)
    # iter_020 ADDITIVE atom-fix injection — placed AFTER the frozen FP-fix
    # clauses and the existing emphasis blocks so they cannot shadow frozen text.
    if atom_fix:
        if force_merged or cat == "dress":
            cat_line += DRESS_FULLBODY_BLOCK
        if force_merged or cat == "lower":
            cat_line += LOWER_LONG_LENGTH_BLOCK
    ev = ""
    if evidence:
        ev = (
            "\nMETRIC EVIDENCE (computed on the garment regions; these are NOISY "
            "ADVISORY hints from a different image domain — your own visual "
            "judgement OVERRIDES them. Do NOT flag an atom 0 on a metric value "
            "alone: a high colour_delta is usually lighting/crop/worn-on-body "
            "noise, so keep color=2 unless YOU can SEE a clear hue change; a low "
            "garment_dino_cos only SUGGESTS the pattern/structure may have changed "
            "— verify visually before grading 0):\n"
            f"  colour_hist_delta={evidence.get('color_delta')}  (0=identical, higher=more colour change)\n"
            f"  garment_dino_cos={evidence.get('dino_cos')}  (1=identical structure/pattern, lower=more change)\n"
            f"  garment_lpips={evidence.get('lpips')}  (0=identical texture, higher=more change)\n"
        )
    step0_block = STEP0_BLOCK if step0 else ""
    # iter_022 ADDITIVE: when suit=True the STEP-0 read also emits is_two_piece /
    # two_piece (PR1). Appended AFTER the frozen STEP0_BLOCK so it cannot alter it;
    # gated on step0 (the STEP-0 read is the only read that declares the category).
    if step0 and suit:
        step0_block = step0_block + SUIT_STEP0_BLOCK
    if step0 and suit:
        keys_line = ('{"worn_region":_, "region_confidence":_, "is_two_piece":_, "two_piece":_, '
                     '"color":_, "pattern":_, '
                     '"material":_, "garment_type":_, "structure":_, "fine_details":_, "logo":_}')
        keys_note = ("these keys (worn_region/region_confidence strings; is_two_piece boolean; "
                     "two_piece an object {upper,lower} or null; the rest integer 0/1/2 or \"na\")")
    elif step0:
        keys_line = ('{"worn_region":_, "region_confidence":_, "color":_, "pattern":_, '
                     '"material":_, "garment_type":_, "structure":_, "fine_details":_, "logo":_}')
        keys_note = ("these keys (worn_region/region_confidence strings; the rest integer "
                     "0/1/2 or \"na\")")
    else:
        keys_line = '{"color":_, "pattern":_, "material":_, "garment_type":_, "structure":_, "fine_details":_, "logo":_}'
        keys_note = "these keys (integer 0/1/2 or \"na\")"
    instr = (
        "You are a meticulous garment-fidelity judge for virtual try-on. You are shown TWO images:\n"
        "  [REFERENCE GARMENT] = the input cloth the result is SUPPOSED to reproduce.\n"
        "  [TRY-ON RESULT] = the generated image of a person wearing (supposedly) that garment.\n"
        + step0_block + cat_line +
        "Judge ONLY whether the garment WORN in the RESULT reproduces the REFERENCE garment's IDENTITY.\n"
        "IMPORTANT — the result is worn on a body, so IGNORE differences that come merely from being worn: "
        "drape, folds, wrinkles, pose, body shape, scale, background, and LIGHTING/SHADOW. Also IGNORE "
        "small/subtle colour-shade differences that are hard to detect (minor lighting-driven shade or "
        "saturation shifts are FINE). Only flag a problem when the difference is CLEAR and obvious.\n"
        "ANTI-BIAS: the two images may be given in either order; rely on the [REFERENCE]/[RESULT] labels, "
        "never on left/right or first/second position.\n"
        + ev + cd_block +
        "\n" + _GRADE_RUBRIC +
        "  (0 examples here: obviously different or too-white/too-dark colour; a clearly different "
        "pattern/print; wrong material; wrong garment type/style; a clearly wrong or missing structural detail.)\n"
        "Atoms:\n" + atoms_txt +
        "\n\nReason BRIEFLY — at most one short phrase per atom (ignore worn-on-body and subtle differences); "
        "do NOT write long paragraphs. You MUST finish with ONE final JSON object on the last line with exactly "
        + keys_note + ":\n" + keys_line
    )
    cloth_url = img_to_data_url(cloth_path)
    if full_frame or reference_anchored:
        result_url = img_to_data_url(result_path)
    else:
        crop_cat = "dress" if cat == "uncertain" else cat
        _res_region, _ = _result_garment_region(result_path, crop_cat)
        result_url = img_to_data_url(_res_region)
    if not swap:
        content = [
            {"type": "text", "text": "[REFERENCE GARMENT]:"},
            {"type": "image_url", "image_url": {"url": cloth_url}},
            {"type": "text", "text": "[TRY-ON RESULT]:"},
            {"type": "image_url", "image_url": {"url": result_url}},
            {"type": "text", "text": instr},
        ]
    else:
        content = [
            {"type": "text", "text": "[TRY-ON RESULT]:"},
            {"type": "image_url", "image_url": {"url": result_url}},
            {"type": "text", "text": "[REFERENCE GARMENT]:"},
            {"type": "image_url", "image_url": {"url": cloth_url}},
            {"type": "text", "text": instr},
        ]
    return [{"role": "user", "content": content}]


def build_pres_messages(model_path, result_path, swap=False, evidence=None, step0=False,
                        fp_fix=False, coverage_first=False, suit=False, suit_sharp=False,
                        suit_relaxed=False, suit_icl=False):
    """POSE + GARMENT-SWAP preservation prompt: [ORIGINAL PERSON] (input_model)
    vs [TRY-ON RESULT] (full result). Same think-then-mark + flag-only-CLEAR
    discipline. swap flips image order; labels always name ORIGINAL vs RESULT.

    iter_019: TARGET-FREE (PR3) — the category-naming target line is GONE; the
    target is defined as the garment that CHANGED, with the two-garment rule and
    the "no garment changed is a valid clean state" rule. The iter_017 UNDERWEAR
    rule and the iter_018 OVERLAP rule stay byte-verbatim. step0=True prepends
    the PRES STEP-0 region declaration (the xcheck channel)."""
    atoms_txt = "\n".join(f"  - {k}: {desc}" for k, desc in PRES_ATOMS)
    tgt_line = ("\nIMPORTANT — TARGET-FREE RULE: the garment that CHANGED between the [ORIGINAL "
                "PERSON] and the [TRY-ON RESULT] IS the intended try-on target; NEVER penalize "
                "the target garment for being different — it is SUPPOSED to change. garment_swap "
                "is about whether a SECOND, non-target garment ALSO changed, an EXTRA garment was "
                "hallucinated, or the person became a different identity: if BOTH the upper-body "
                "and the lower-body garments clearly changed, the one that does not correspond to "
                "the intended try-on garment is a non-target swap — grade 0. Also IGNORE the minor "
                "face/skin/hair regeneration that try-on inherently introduces — only flag a "
                "CLEARLY different non-target GARMENT (or a clearly different person/identity). "
                "If NO garment changed at all, that is a VALID CLEAN state — grade garment_swap=2 "
                "(do not force a target).\n"
                "CRITICAL UNDERWEAR RULE — if the [ORIGINAL PERSON] is wearing only UNDERWEAR / a "
                "bra / briefs / minimal clothing (the target region or body started largely bare), "
                "then: (a) the try-on garment APPEARING on the target region is the CORRECT intended "
                "edit — grade garment_swap=2; (b) a non-target region that was bare/underwear in the "
                "original and stays bare/underwear in the result is CORRECT — grade 2. Do NOT read "
                "'person was in underwear, now wears clothes' as a garment_swap defect. Only grade 0 "
                "when a region that was CLOTHED in the original shows a clearly DIFFERENT garment.\n"
                "*** OVERLAP / EXTRA-GARMENT RULE (grade garment_swap=0) *** — look CAREFULLY for an "
                "EXTRA garment hallucinated where it does NOT belong, most importantly TROUSERS / PANTS "
                "/ JEANS / LEGGINGS / SHORTS visible UNDERNEATH a SKIRT or DRESS in the [RESULT] (a "
                "skirt or dress should reveal BARE LEGS or sheer tights, NEVER a second pair of pants "
                "layered under it), or a shirt/top layered under a dress, or any second garment "
                "overlapping the target that the [ORIGINAL PERSON] does not have. Such a wrong "
                "layered/overlapping garment is garment_swap=0 EVEN IF the target garment looks fine — "
                "this is a common try-on failure; do not pass it as correct.\n")
    # iter_021 ADDITIVE FP-fix A: append the garment_swap coverage-compensation
    # AFTER the frozen iter_017 underwear + iter_018 overlap clauses so it is
    # purely additive and cannot shadow them (the negative example inside the
    # compensation re-states the overlap rule, reinforcing it). coverage_first
    # implies fp_fix and PREPENDS the ordered coverage-map step before grading.
    if fp_fix or coverage_first:
        tgt_line = tgt_line + GARMENT_SWAP_COVERAGE_COMPENSATION + "\n"
    # iter_022 ADDITIVE SUIT branch (PR2), placed AFTER the frozen iter_017
    # underwear + iter_018 overlap + iter_021 coverage-compensation clauses so
    # it can never shadow them. INLINE form (PR4) when coverage_first is False;
    # the coverage-first ordered form (PR3) is applied below via SUIT_ORDERED_BLOCK.
    if suit and not coverage_first:
        tgt_line = tgt_line + SUIT_INLINE_BLOCK
        # iter_023 ADDITIVE suit-sharpening (INLINE form): appended AFTER the
        # iter_022 SUIT_INLINE_BLOCK so it cannot shadow it; gated on suit_sharp.
        # PR2 (gate) then PR1 (enumeration), BIT-IDENTICAL to the ordered form.
        if suit_sharp:
            tgt_line = tgt_line + _suit_block(suit_relaxed, suit_icl)
    if coverage_first:
        tgt_line = COVERAGE_FIRST_BLOCK + tgt_line
        # PR3 ordered two-piece-first restructure: the suit determination is an
        # explicit ordered step BEFORE the coverage-first single-garment reasoning.
        if suit:
            # suit-sharpening (ORDERED form): PR1 STRICT (iter_023) or RELAXED
            # confirm-correct-first (iter_024 suit_relaxed); PR2 gate first.
            # _suit_block(False,False) == SUIT_SHARP_BLOCK byte-for-byte.
            if suit_sharp:
                tgt_line = _suit_block(suit_relaxed, suit_icl) + tgt_line
            tgt_line = SUIT_ORDERED_BLOCK + tgt_line
    ev = ""
    if evidence:
        ev = (
            "\nMETRIC EVIDENCE on the NON-TARGET region (everything OUTSIDE the "
            "central garment band — face, lower body, shoes, background), "
            "input_model vs result:\n"
            f"  nontarget_dino_cos={evidence.get('nt_dino_cos')}  (1=non-target unchanged, lower=more change)\n"
            f"  nontarget_lpips={evidence.get('nt_lpips')}  (0=non-target unchanged, higher=more change)\n"
            "These are NOISY ADVISORY hints — a LOW nontarget_dino_cos / HIGH "
            "nontarget_lpips can be caused merely by the new garment's hem/length "
            "overlapping the band, by pose shift, or by a bare/underwear region "
            "being filled — NOT necessarily a swap. Grade garment_swap=0 ONLY when "
            "you can SEE a clearly different non-target GARMENT; never on the metric "
            "alone.\n"
        )
    step0_block = PRES_STEP0_BLOCK if step0 else ""
    # iter_022 ADDITIVE: when suit=True the PRES read also tags is_two_piece so the
    # suit-subset FP/FN delta is computable per sample (PR2/PR5). Purely additive —
    # suit=False keeps the frozen iter_021 key set byte-identical.
    if step0 and suit:
        keys_line = '{"worn_region":_, "region_confidence":_, "is_two_piece":_, "pose":_, "garment_swap":_}'
        keys_note = ("these keys (worn_region/region_confidence strings; is_two_piece boolean; "
                     "the rest integer 0/1/2 or \"na\")")
    elif suit:
        keys_line = '{"is_two_piece":_, "pose":_, "garment_swap":_}'
        keys_note = ("these keys (is_two_piece boolean; the rest integer 0/1/2 or \"na\")")
    elif step0:
        keys_line = '{"worn_region":_, "region_confidence":_, "pose":_, "garment_swap":_}'
        keys_note = ("these keys (worn_region/region_confidence strings; the rest integer "
                     "0/1/2 or \"na\")")
    else:
        keys_line = '{"pose":_, "garment_swap":_}'
        keys_note = "these keys (integer 0/1/2 or \"na\")"
    instr = (
        "You are a meticulous virtual-try-on preservation judge. You are shown TWO images:\n"
        "  [ORIGINAL PERSON] = the input photo of the person BEFORE try-on.\n"
        "  [TRY-ON RESULT] = the generated image after putting a new garment on that person.\n"
        "A correct try-on changes ONLY the target garment; the person's POSE and all NON-TARGET "
        "items (other clothing, shoes) should stay the same.\n"
        + step0_block + tgt_line +
        "IGNORE differences that come merely from the new garment itself, and IGNORE subtle "
        "lighting/scale/crop/compression differences. Only flag a problem when it is CLEAR and obvious.\n"
        "ANTI-BIAS: the two images may be given in either order; rely on the [ORIGINAL]/[RESULT] labels, "
        "never on left/right or first/second position.\n"
        + ev +
        "\n" + _GRADE_RUBRIC +
        "  (pose 0 = the body is clearly warped / has broken, extra, missing or melted limbs/hands, or the "
        "pose is obviously different; garment_swap 0 = a clearly different NON-TARGET garment appears, or the "
        "person became a clearly different identity — NOT the intended garment changing, NOT minor face regen.)\n"
        "Atoms:\n" + atoms_txt +
        "\n\nReason BRIEFLY — at most one short phrase per atom; do NOT write long paragraphs. You MUST finish "
        "with ONE final JSON object on the last line with exactly " + keys_note + ":\n" + keys_line
    )
    model_url = img_to_data_url(model_path)
    result_url = img_to_data_url(result_path)  # FULL result (pose/non-target need the whole frame)
    if not swap:
        content = [
            {"type": "text", "text": "[ORIGINAL PERSON]:"},
            {"type": "image_url", "image_url": {"url": model_url}},
            {"type": "text", "text": "[TRY-ON RESULT]:"},
            {"type": "image_url", "image_url": {"url": result_url}},
            {"type": "text", "text": instr},
        ]
    else:
        content = [
            {"type": "text", "text": "[TRY-ON RESULT]:"},
            {"type": "image_url", "image_url": {"url": result_url}},
            {"type": "text", "text": "[ORIGINAL PERSON]:"},
            {"type": "image_url", "image_url": {"url": model_url}},
            {"type": "text", "text": instr},
        ]
    return [{"role": "user", "content": content}]


def build_joint_messages(cloth_path, model_path, result_path, evidence=None):
    """The 2-call cost-floor wildcard's joint 3-image prompt: [REFERENCE GARMENT]
    [ORIGINAL PERSON][TRY-ON RESULT] (= the serve cap of 3 images), doing STEP-0
    + all 7 GF atoms (FP-fix texts verbatim) + both PRES atoms (PR3 target-free
    garment_swap with the verbatim overlap + underwear rules) in ONE
    think-then-mark JSON. Pass-B injects escalate-only metric evidence as TEXT
    (zero image headroom) with a reconcile instruction."""
    gf_txt = "\n".join(f"  - {k}: {desc}" for k, desc in GF_ATOMS)
    pres_txt = "\n".join(f"  - {k}: {desc}" for k, desc in PRES_ATOMS)
    ev = ""
    if evidence:
        ev = (
            "\nMETRIC EVIDENCE (computed on the declared garment region; these are NOISY "
            "ADVISORY hints from a different image domain — your own visual judgement "
            "OVERRIDES them; never grade an atom 0 on a metric value alone):\n"
            f"  colour_hist_delta={evidence.get('color_delta')}  (0=identical, higher=more colour change)\n"
            f"  garment_dino_cos={evidence.get('dino_cos')}  (1=identical structure/pattern, lower=more change)\n"
            f"  garment_lpips={evidence.get('lpips')}  (0=identical texture, higher=more change)\n"
            "Reconcile: confirm or refute each flag against what you see.\n"
        )
    instr = (
        "You are a meticulous virtual try-on judge. You are shown THREE images:\n"
        "  [REFERENCE GARMENT] = the input cloth the result is SUPPOSED to reproduce.\n"
        "  [ORIGINAL PERSON] = the input photo of the person BEFORE try-on.\n"
        "  [TRY-ON RESULT] = the generated image after putting that garment on the person.\n"
        + STEP0_BLOCK +
        "Then locate the garment in [TRY-ON RESULT] that corresponds to the [REFERENCE GARMENT] "
        "and judge the GARMENT-FIDELITY atoms ONLY on that garment; judge the PRESERVATION atoms "
        "on the full frame ([ORIGINAL PERSON] vs [TRY-ON RESULT]).\n"
        "IMPORTANT — the result is worn on a body, so IGNORE differences that come merely from "
        "being worn: drape, folds, wrinkles, pose, body shape, scale, background, and "
        "LIGHTING/SHADOW. Also IGNORE small/subtle colour-shade differences that are hard to "
        "detect. Only flag a problem when the difference is CLEAR and obvious.\n"
        + MERGED_EMPHASIS_BLOCK + ev +
        "\n" + _GRADE_RUBRIC +
        "Garment-fidelity atoms (judge the located garment vs the [REFERENCE GARMENT]):\n" + gf_txt +
        "\nPreservation atoms (judge [ORIGINAL PERSON] vs [TRY-ON RESULT], full frame):\n" + pres_txt +
        "\n\nReason BRIEFLY — at most one short phrase per atom; do NOT write long paragraphs. "
        "You MUST finish with ONE final JSON object on the last line with exactly these keys "
        "(worn_region/region_confidence strings; the rest integer 0/1/2 or \"na\"):\n"
        '{"worn_region":_, "region_confidence":_, "color":_, "pattern":_, "material":_, '
        '"garment_type":_, "structure":_, "fine_details":_, "logo":_, "pose":_, "garment_swap":_}'
    )
    content = [
        {"type": "text", "text": "[REFERENCE GARMENT]:"},
        {"type": "image_url", "image_url": {"url": img_to_data_url(cloth_path)}},
        {"type": "text", "text": "[ORIGINAL PERSON]:"},
        {"type": "image_url", "image_url": {"url": img_to_data_url(model_path)}},
        {"type": "text", "text": "[TRY-ON RESULT]:"},
        {"type": "image_url", "image_url": {"url": img_to_data_url(result_path)}},
        {"type": "text", "text": instr},
    ]
    return [{"role": "user", "content": content}]


def _avg_votes(votes, keys):
    atoms = {}
    for k in keys:
        vals = [v for v in votes[k] if v is not None]
        atoms[k] = (round(sum(vals) / len(vals)) if vals else None)
    return atoms


def run_gf_judge(pick_ep, cloth_path, result_path, seeds, evidence=None,
                 inferred_category="upper", clear_discipline=False, step0=False,
                 full_frame=False, reference_anchored=False, force_merged=False,
                 atom_fix=False, fp_fix=False, suit=False):
    """Order-swap ensemble over `seeds` (single-pass when seeds=[42]). Returns
    (atoms dict 0..2/None, overall float 0..1, vlm_defect bool, passes_ok int,
    last_err) — plus (worn_region, region_confidence) from the LAST successful
    parse when step0=True. Defect = ANY applicable atom graded 0."""
    votes = {k: [] for k in GF_KEYS}
    passes_ok = 0
    last_err = None
    worn_region = None
    region_conf = None
    is_two_piece = None
    two_piece = None
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        swap = (i % 2 == 1)
        try:
            msgs = build_gf_messages(cloth_path, result_path, swap, evidence,
                                     inferred_category, clear_discipline,
                                     step0=step0, full_frame=full_frame,
                                     reference_anchored=reference_anchored,
                                     force_merged=force_merged, atom_fix=atom_fix,
                                     fp_fix=fp_fix, suit=suit)
            txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed)
            parsed = extract_last_json(txt)
            if not parsed:
                # native thinking likely overran the token budget; retry once
                # WITHOUT thinking (short, JSON-direct) so the pass still lands.
                txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed, enable_thinking=False, max_tokens=1024)
                parsed = extract_last_json(txt)
            if not parsed:
                last_err = "no json"
                continue
            for k in GF_KEYS:
                votes[k].append(_grade(parsed.get(k)))
            if step0:
                worn_region = parsed.get("worn_region", worn_region)
                region_conf = parsed.get("region_confidence", region_conf)
                if suit:
                    if parsed.get("is_two_piece") is not None:
                        is_two_piece = parsed.get("is_two_piece")
                    if parsed.get("two_piece") is not None:
                        two_piece = parsed.get("two_piece")
            passes_ok += 1
        except Exception as e:
            last_err = str(e)
    if passes_ok == 0:
        if step0 and suit:
            return None, 0.0, False, 0, last_err, None, None, None, None
        if step0:
            return None, 0.0, False, 0, last_err, None, None
        return None, 0.0, False, 0, last_err
    atoms = _avg_votes(votes, GF_KEYS)
    graded = [v for v in atoms.values() if v is not None]
    overall = (sum(graded) / (2.0 * len(graded))) if graded else 1.0
    vlm_defect = any(v == 0 for v in atoms.values() if v is not None)
    if step0 and suit:
        return (atoms, float(overall), bool(vlm_defect), passes_ok, last_err,
                worn_region, region_conf, is_two_piece, two_piece)
    if step0:
        return atoms, float(overall), bool(vlm_defect), passes_ok, last_err, worn_region, region_conf
    return atoms, float(overall), bool(vlm_defect), passes_ok, last_err


def run_pres_judge(pick_ep, model_path, result_path, seeds, evidence=None, step0=False,
                   fp_fix=False, coverage_first=False, suit=False, suit_sharp=False,
                   suit_relaxed=False, suit_icl=False):
    """POSE + GARMENT-SWAP preservation judge (input_model vs result-full),
    TARGET-FREE (PR3). Returns (atoms dict, defect bool, passes_ok int,
    last_err) — plus (worn_region, region_confidence) from the LAST successful
    parse when step0=True. Defect = pose==0 OR garment_swap==0."""
    votes = {k: [] for k in PRES_KEYS}
    passes_ok = 0
    last_err = None
    worn_region = None
    region_conf = None
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        swap = (i % 2 == 1)
        try:
            msgs = build_pres_messages(model_path, result_path, swap, evidence, step0=step0,
                                       fp_fix=fp_fix, coverage_first=coverage_first, suit=suit,
                                       suit_sharp=suit_sharp, suit_relaxed=suit_relaxed,
                                       suit_icl=suit_icl)
            txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed)
            parsed = extract_last_json(txt)
            if not parsed:
                txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed, enable_thinking=False, max_tokens=1024)
                parsed = extract_last_json(txt)
            if not parsed:
                last_err = "no json"
                continue
            for k in PRES_KEYS:
                votes[k].append(_grade(parsed.get(k)))
            if step0:
                worn_region = parsed.get("worn_region", worn_region)
                region_conf = parsed.get("region_confidence", region_conf)
            passes_ok += 1
        except Exception as e:
            last_err = str(e)
    if passes_ok == 0:
        if step0:
            return None, False, 0, last_err, None, None
        return None, False, 0, last_err
    atoms = _avg_votes(votes, PRES_KEYS)
    defect = any(v == 0 for v in atoms.values() if v is not None)
    if step0:
        return atoms, bool(defect), passes_ok, last_err, worn_region, region_conf
    return atoms, bool(defect), passes_ok, last_err


def run_joint_once(pick_ep, cloth_path, model_path, result_path, seed, evidence=None):
    """One joint 3-image pass (the wildcard). Returns (gf_atoms, pres_atoms,
    worn_region, region_confidence, err); gf/pres atoms are None on failure."""
    try:
        base_url, model = pick_ep()
        msgs = build_joint_messages(cloth_path, model_path, result_path, evidence)
        txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed)
        parsed = extract_last_json(txt)
        if not parsed:
            txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed, enable_thinking=False, max_tokens=1536)
            parsed = extract_last_json(txt)
        if not parsed:
            return None, None, None, None, "no json"
        gf = {k: _grade(parsed.get(k)) for k in GF_KEYS}
        pres = {k: _grade(parsed.get(k)) for k in PRES_KEYS}
        return gf, pres, parsed.get("worn_region"), parsed.get("region_confidence"), None
    except Exception as e:
        return None, None, None, None, str(e)


def merge_overall(gf_atoms, pres_atoms):
    """Combined 0..1 score over ALL graded atoms across both calls."""
    graded = []
    for d in (gf_atoms or {}, pres_atoms or {}):
        graded += [v for v in d.values() if v is not None]
    if not graded:
        return 1.0
    return float(sum(graded) / (2.0 * len(graded)))


# ---------------- metric helpers (garment region vs input_cloth) ----------------

# iter_009: shared disk metric cache. The configs redundantly recompute the
# SAME (sample, method) garment + non-target DINO/LPIPS metrics — cache by image
# paths + category so the first config computes and the rest read. Idempotent
# (same inputs -> same value), so concurrent writers are safe. iter_019's cache
# dir is <iter_019>/experiments/metric_cache (warm-copied from iter_018: the
# keys are absolute dataset paths + category, identical across iters).
_METRIC_CACHE_DIR = Path(__file__).resolve().parent.parent / "metric_cache"


def _cache_key(tag, *parts):
    import hashlib
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]
    return f"{tag}_{h}.json"


def _cached(tag, key_parts, compute):
    try:
        _METRIC_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        p = _METRIC_CACHE_DIR / _cache_key(tag, *key_parts)
        if p.is_file():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        val = compute()
        try:
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(val))
            tmp.replace(p)
        except Exception:
            pass
        return val
    except Exception:
        return compute()


def _result_garment_region(result_path, category="upper"):
    """Category-aware garment crop of the RESULT (upper torso / lower legs / full
    for dress). Returns (PIL_crop, mask_low_conf). No segmenter -> whole-image
    fallback. The category here is ALWAYS image-derived (resolved by a category
    channel) — never the path prefix."""
    im = result_path if isinstance(result_path, Image.Image) else Image.open(result_path)
    im = im.convert("RGB")
    w, h = im.size
    try:
        box = _target_box(w, h, category)
        crop = M.crop(im, box)
        if crop is None or min(crop.size) < 16:
            return im, True
        return crop, False
    except Exception:
        return im, True


def garment_metrics(cloth_path, result_path, category="upper"):
    """cloth-vs-result-garment evidence pack: color_delta (>=0, higher worse),
    dino_cos (0..1, higher better), lpips (>=0, higher worse), mask_low_conf.
    Disk-cached by (cloth, result, category) — shared across all configs."""
    def _compute():
        cloth = Image.open(cloth_path).convert("RGB")
        region, low = _result_garment_region(result_path, category)
        out = {"mask_low_conf": low}
        try:
            out["color_delta"] = M.color_hist_delta(cloth, region)
        except Exception:
            out["color_delta"] = None
        out["dino_cos"] = M.dino_cosine(cloth, region)
        out["lpips"] = M.lpips_dist(cloth, region)
        return out
    return _cached("garment", (cloth_path, result_path, category), _compute)


def nontarget_metrics(model_path, result_path, category="upper"):
    """NON-TARGET preservation evidence: input_model vs result on the COMPLEMENT
    of the TARGET-garment box. Disk-cached by (model, result, category)."""
    def _compute():
        model = Image.open(model_path).convert("RGB")
        result = Image.open(result_path).convert("RGB")
        out = {"nt_dino_cos": None, "nt_lpips": None}
        try:
            mw, mh = model.size
            rw, rh = result.size
            model_nt = M.paint_out(model, _target_box(mw, mh, category))
            result_nt = M.paint_out(result, _target_box(rw, rh, category))
            out["nt_dino_cos"] = M.dino_cosine(model_nt, result_nt)
            out["nt_lpips"] = M.lpips_dist(model_nt, result_nt)
        except Exception:
            pass
        return out
    return _cached("nontarget", (model_path, result_path, category), _compute)


# ---------------- iter_019 category channels (all image-derived) ----------------

def dino_argmax_category(cloth_path, model_path):
    """Channel β: DINOv2 cosine of the input_cloth vs each of the INPUT_MODEL's
    three candidate crops (upper/lower/dress boxes — the INPUT side, never the
    result, to avoid defect circularity). Returns (label, margin, cosines dict);
    label = argmax box, margin = top1-top2. OVERALLS RULE: cloth cosine to BOTH
    the upper and lower model-crops within 0.02 of each other AND both above the
    dress cosine -> ("dress", small_margin, ...) — a full-coverage garment forces
    the widest box. ("uncertain", 0.0, {}) when DINO is unavailable.
    Disk-cached (tag "catdino") by (cloth, model)."""
    if not M.DINO_OK:
        return ("uncertain", 0.0, {})

    def _compute():
        cloth = Image.open(cloth_path).convert("RGB")
        model = Image.open(model_path).convert("RGB")
        w, h = model.size
        cos = {}
        for cat in CROP_CATEGORIES:
            crop = M.crop(model, _target_box(w, h, cat))
            c = M.dino_cosine(cloth, crop)
            if c is None:
                return {"label": "uncertain", "margin": 0.0, "cos": {}}
            cos[cat] = float(c)
        ranked_vals = sorted(cos.values(), reverse=True)
        # OVERALLS RULE: full-coverage garment (similar to both body halves).
        if (abs(cos["upper"] - cos["lower"]) <= 0.02
                and cos["upper"] > cos["dress"] and cos["lower"] > cos["dress"]):
            return {"label": "dress", "margin": float(ranked_vals[0] - ranked_vals[1]),
                    "cos": cos, "overalls_rule": True}
        ranked = sorted(cos.items(), key=lambda kv: kv[1], reverse=True)
        return {"label": ranked[0][0], "margin": float(ranked[0][1] - ranked[1][1]), "cos": cos}

    out = _cached("catdino", (cloth_path, model_path), _compute)
    return (out.get("label", "uncertain"), float(out.get("margin", 0.0) or 0.0),
            out.get("cos", {}) or {})


def changeloc_category(model_path, result_path):
    """Channel γ: change localization — DINOv2 cosine of model-crop vs
    result-crop per candidate box; the box with the LOWEST cosine (most changed)
    wins. Returns (label, separation, cosines); separation = second_lowest -
    lowest. ("uncertain", 0.0, {}) when DINO is unavailable. Disk-cached
    (tag "catchange") by (model, result)."""
    if not M.DINO_OK:
        return ("uncertain", 0.0, {})

    def _compute():
        model = Image.open(model_path).convert("RGB")
        result = Image.open(result_path).convert("RGB")
        mw, mh = model.size
        rw, rh = result.size
        cos = {}
        for cat in CROP_CATEGORIES:
            mc = M.crop(model, _target_box(mw, mh, cat))
            rc = M.crop(result, _target_box(rw, rh, cat))
            c = M.dino_cosine(mc, rc)
            if c is None:
                return {"label": "uncertain", "separation": 0.0, "cos": {}}
            cos[cat] = float(c)
        ranked = sorted(cos.items(), key=lambda kv: kv[1])  # ascending: most changed first
        return {"label": ranked[0][0], "separation": float(ranked[1][1] - ranked[0][1]), "cos": cos}

    out = _cached("catchange", (model_path, result_path), _compute)
    return (out.get("label", "uncertain"), float(out.get("separation", 0.0) or 0.0),
            out.get("cos", {}) or {})


def _otsu_threshold(vals):
    """Otsu's between-class-variance threshold on a 1-D value array (numpy-only)."""
    import numpy as np
    v = vals.ravel()
    hist, edges = np.histogram(v, bins=64)
    mids = (edges[:-1] + edges[1:]) / 2.0
    total = float(v.size)
    sum_all = float((hist * mids).sum())
    sum_b = 0.0
    w_b = 0.0
    best = 0.0
    thr = float(edges[1])
    for i in range(len(hist)):
        w_b += float(hist[i])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(hist[i]) * float(mids[i])
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best = between
            thr = float(edges[i + 1])
    return thr


SILHOUETTE_DEFAULT_THR = {
    "aspect_dress_min": 1.25,
    "centroid_lower_min": 0.55,
    "coverage_min": 0.05,
}


def silhouette_category(cloth_path, thresholds=None):
    """Channel δ (FLOOR channel — accuracy measured by the catinfer baseline;
    deliberately simple, numpy-only): foreground mask via an Otsu-style threshold
    on grayscale distance from the median BORDER color (product photos have flat
    backgrounds); features = fg bbox aspect ratio (h/w), fg coverage, vertical
    centroid (0..1, 0=top). Decision (thresholds overridable from the calibration
    JSON): centroid >= centroid_lower_min and aspect < aspect_dress_min ->
    "lower"; aspect >= aspect_dress_min and centroid in the middle band ->
    "dress"; else "upper". Returns (label, features)."""
    import numpy as np
    thr = dict(SILHOUETTE_DEFAULT_THR)
    if thresholds:
        try:
            thr.update({k: float(v) for k, v in thresholds.items() if k in thr})
        except Exception:
            pass
    im = Image.open(cloth_path).convert("RGB")
    im.thumbnail((192, 192))
    arr = __import__("numpy").asarray(im, dtype="float32")
    h, w = arr.shape[:2]
    border = np.concatenate([arr[0].reshape(-1, 3), arr[-1].reshape(-1, 3),
                             arr[:, 0].reshape(-1, 3), arr[:, -1].reshape(-1, 3)])
    med = np.median(border, axis=0)
    dist = np.sqrt(((arr - med) ** 2).sum(axis=2))
    t = _otsu_threshold(dist)
    fg = dist > t
    coverage = float(fg.mean())
    feats = {"coverage": coverage}
    if coverage < thr["coverage_min"] or not fg.any():
        feats["low_fg"] = True
        return "upper", feats
    ys, xs = np.nonzero(fg)
    bh = float(ys.max() - ys.min() + 1)
    bw = float(xs.max() - xs.min() + 1)
    aspect = bh / max(bw, 1.0)
    centroid = float(ys.mean()) / max(h - 1, 1)
    feats.update({"aspect": float(aspect), "centroid_y": centroid})
    if centroid >= thr["centroid_lower_min"] and aspect < thr["aspect_dress_min"]:
        label = "lower"
    elif aspect >= thr["aspect_dress_min"] and 0.35 <= centroid < thr["centroid_lower_min"]:
        label = "dress"
    else:
        label = "upper"
    return label, feats


def classify_cloth_judge(pick_ep, cloth_path, seed=42):
    """Channel α: ONE single-image strict-JSON judge call classifying the cloth.
    Returns (label, confidence) — ("uncertain", "low") on any parse/call failure
    (never raises; never fabricates a confident label)."""
    instr = (
        'You are shown ONE garment product image. Classify where it is worn. '
        'Reply with ONLY one JSON object: '
        '{"category": "upper"|"lower"|"dress", "confidence": "high"|"low"}. '
        'Use "dress" for any one-piece/full-body (dress, jumpsuit, overalls). '
        'Use "low" for ambiguous (long shirt vs short dress, tunic).'
    )
    content = [
        {"type": "image_url", "image_url": {"url": img_to_data_url(cloth_path)}},
        {"type": "text", "text": instr},
    ]
    msgs = [{"role": "user", "content": content}]
    try:
        base_url, model = pick_ep()
        txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed, max_tokens=256)
        parsed = extract_last_json(txt)
        if not parsed:
            txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed, max_tokens=512)
            parsed = extract_last_json(txt)
        if not parsed:
            return ("uncertain", "low")
        cat = str(parsed.get("category", "")).strip().lower()
        conf = str(parsed.get("confidence", "low")).strip().lower()
        if cat not in ("upper", "lower", "dress"):
            cat = map_worn_region(cat)
            if cat == "uncertain":
                return ("uncertain", "low")
        return (cat, "high" if conf == "high" else "low")
    except Exception:
        return ("uncertain", "low")


def merge_channels(a_label, a_conf_ok, b_label, b_conf_ok):
    """PR7 4-quadrant cross-check merge of two independent category channels.
    Quadrants (channel confidence x channel confidence, given the labels):
      Q1 agree AND both strong            -> (label, "high")       narrow crop, nt active
      Q2 agree AND exactly one weak       -> (label, "low-agree")  narrow crop, nt active
                                             (two independent channels agreeing
                                              dominates one weak signal)
      Q3 agree AND both weak              -> (label, "low-agree")  same rule as Q2
      Q4 disagree OR either missing       -> ("uncertain", "disagree")
                                             PR6 widest box + nt-veto suppression
    A channel that returned "uncertain" counts as missing (Q4)."""
    a = a_label if a_label in ("upper", "lower", "dress") else "uncertain"
    b = b_label if b_label in ("upper", "lower", "dress") else "uncertain"
    if a == "uncertain" or b == "uncertain" or a != b:
        return ("uncertain", "disagree")
    if a_conf_ok and b_conf_ok:
        return (a, "high")
    return (a, "low-agree")


def resolve_region(label):
    """Map a (merged) category label to the crop/veto/emphasis triple.
    "uncertain" -> the DRESS box (PR6 widest), nt-veto SUPPRESSED, merged
    emphasis; "dress" -> dress box, nt active, no extra emphasis (as iter_018);
    "upper"/"lower" -> own box, nt active, own verbatim emphasis block.
    Returns (category_for_crop, suppress_nt, emphasis_key)."""
    if label not in ("upper", "lower", "dress"):
        return ("dress", True, "uncertain")
    if label == "dress":
        return ("dress", False, "dress")
    return (label, False, label)


# Default thresholds (calibrated on the annotated subset before the C-2 gate).
DEFAULT_THR = {
    "color_delta_max": 0.30,   # above => COLOR defect (veto)
    "dino_cos_min": 0.55,      # below => PATTERN/STRUCTURE defect (veto)
    "lpips_max": 0.55,         # above => TEXTURE/MATERIAL defect (scorecard)
    "nt_dino_cos_min": 0.60,   # below => NON-TARGET changed (garment_swap veto)
    "nt_lpips_max": 0.50,      # above => NON-TARGET changed (garment_swap veto)
}


def load_calibrated_thr(configs_dir=None):
    """Load calibrated_thr.json; fall back to DEFAULT_THR."""
    d = Path(configs_dir) if configs_dir else Path(__file__).resolve().parent
    p = d / "calibrated_thr.json"
    thr = dict(DEFAULT_THR)
    if p.is_file():
        try:
            c = json.loads(p.read_text())
            for k in list(DEFAULT_THR):
                if k in c:
                    thr[k] = float(c[k])
        except Exception:
            pass
    return thr


def metric_vetoes(metrics, thr=None):
    """One-directional garment vetoes (color/pattern/texture). Returns
    (atom_ok dict, any_metric_defect bool). A None metric is skipped."""
    thr = thr or DEFAULT_THR
    ok = {}
    cd = metrics.get("color_delta")
    dc = metrics.get("dino_cos")
    lp = metrics.get("lpips")
    if cd is not None:
        ok["color"] = cd <= thr["color_delta_max"]
    if dc is not None:
        ok["pattern_structure"] = dc >= thr["dino_cos_min"]
    if lp is not None:
        ok["texture_material"] = lp <= thr["lpips_max"]
    any_defect = any(v is False for v in ok.values())
    return ok, any_defect


def nontarget_veto(nt_metrics, thr=None):
    """Non-target preservation veto (for the garment_swap atom). Returns
    (ok dict, defect bool). A None metric is skipped."""
    thr = thr or DEFAULT_THR
    ok = {}
    dc = nt_metrics.get("nt_dino_cos")
    lp = nt_metrics.get("nt_lpips")
    if dc is not None:
        ok["nt_structure"] = dc >= thr["nt_dino_cos_min"]
    if lp is not None:
        ok["nt_texture"] = lp <= thr["nt_lpips_max"]
    defect = any(v is False for v in ok.values())
    return ok, defect


def metric_scorecard_score(metrics, thr=None):
    """Family-B conjunctive any-atom-fails score in [0,1]."""
    ok, any_defect = metric_vetoes(metrics, thr)
    if not ok:
        return 1.0, False, ok
    score = sum(1 for v in ok.values() if v) / len(ok)
    return float(score), bool(any_defect), ok


# ===========================================================================
# iter_026 FLAG A (gs_zoom) — per-piece REGION-ZOOM presence read for two-piece /
# suit garment_swap. PURE ADDITION: these functions are NEVER called from
# build_pres_messages / run_pres_judge; they are invoked ONLY from the
# crossover score path when gs_zoom=True AND is_two_piece is True. With
# gs_zoom=False the v9 PRES path (run_pres_judge) is byte-identical.
#
# FIXED GATE (the iter_026 bug fix): fire the zoom on `is_two_piece is True`
# alone — do NOT require two_piece_confidence=="high" (that field is emitted only
# by the gs_conservative flag, which this evaluator does not run, so requiring it
# disabled the zoom entirely).
#
# The per-piece prompt carries the iter_017 UNDERWEAR rule, the iter_018 OVERLAP
# rule and the iter_021 coverage-compensation clause VERBATIM (re-used from the
# frozen PRES_ATOMS[1] text + GARMENT_SWAP_COVERAGE_COMPENSATION constant) so
# layering recall is preserved at the per-piece grain.
# ===========================================================================

# the iter_017 underwear + iter_018 overlap clauses, sliced VERBATIM from the
# frozen garment_swap atom description (PRES_ATOMS[1][1]) so they are guaranteed
# byte-identical to the full-frame read.
_GS_ATOM_DESC = PRES_ATOMS[1][1]
_OVERLAP_MARK = "*** OVERLAP / EXTRA-GARMENT RULE"
_UNDERWEAR_MARK = "UNDERWEAR RULE"
PIECE_OVERLAP_CLAUSE = _GS_ATOM_DESC[_GS_ATOM_DESC.index(_OVERLAP_MARK):_GS_ATOM_DESC.index(_UNDERWEAR_MARK)].strip()
PIECE_UNDERWEAR_CLAUSE = _GS_ATOM_DESC[_GS_ATOM_DESC.index(_UNDERWEAR_MARK):].strip()


def build_pres_piece_messages(model_crop, result_crop, cloth_path, piece_label,
                              swap=False, fidelity_zoom=False):
    """ZOOMED per-piece presence read for ONE region (upper OR lower) of a
    two-piece / suit reference. Shows the reference cloth (the full two-piece set),
    the ORIGINAL-person crop for this region, and the RESULT crop for this region,
    and asks whether the reference {piece_label} piece is present-and-correctly
    applied in this region. Carries the iter_017 underwear + iter_018 overlap +
    iter_021 coverage clauses VERBATIM so layering recall is preserved.

    model_crop / result_crop are PIL.Image crops (image-derived via _target_box —
    NEVER a path). Emits {"piece_present_correct": 0|1|2}.

    iter_028 (fidelity_zoom): when fidelity_zoom=True the per-region read is
    re-framed from a LENIENT presence check ("is THIS reference piece PRESENT?")
    to a STRICT per-region FIDELITY/DEFECT check ("is the garment worn in THIS
    region the SAME garment as the reference, correctly applied and DEFECT-FREE,
    vs a DEFECT?"), leveraging the zoom resolution to catch real per-region
    defects (wrong/missing garment, major texture/pattern/color mismatch, severe
    distortion) instead of passing too readily. The fidelity read emits
    {"region_defect": 0|1}. fidelity_zoom=False is byte-identical to iter_026."""
    if fidelity_zoom:
        instr = (
            "You are shown the REFERENCE garment (input_cloth) and the WORN RESULT "
            "for the " + str(piece_label) + " region only. Compare ONLY this region. "
            "Decide if the garment worn in this region is the SAME garment as the "
            "reference, CORRECTLY applied and DEFECT-FREE, vs a DEFECT. A DEFECT "
            "(emit region_defect=1) is any of: a DIFFERENT/WRONG garment than the "
            "reference (substituted), the reference garment MISSING/not worn in this "
            "region, a major texture/pattern/color mismatch vs the reference, or "
            "severe distortion/damage of the garment. If the region shows the correct "
            "reference garment applied cleanly (minor drape/wrinkle/lighting is NOT a "
            "defect), emit region_defect=0. Be strict about WRONG/MISSING garments but "
            "do NOT flag cosmetic drape. Emit {region_defect: 0|1}.\n"
            "ANTI-BIAS: the ORIGINAL and RESULT crops may be given in either order; "
            "rely on the [ORIGINAL]/[RESULT] labels, never on position.\n"
            "Reason BRIEFLY (one short phrase). You MUST finish with ONE final JSON "
            "object on the last line with exactly this key (integer 0/1):\n"
            '{"region_defect":_}'
        )
    else:
      instr = (
        "You are a meticulous virtual-try-on garment-presence judge. The [REFERENCE "
        "GARMENT] is a TWO-PIECE / SUIT set (an upper piece AND a lower piece supplied "
        "together). You are checking ONLY the " + str(piece_label).upper() + " piece in "
        "ONE zoomed body region.\n"
        "You are shown THREE images:\n"
        "  [REFERENCE GARMENT] = the full two-piece reference set the result should reproduce.\n"
        "  [ORIGINAL " + str(piece_label).upper() + " REGION] = the ORIGINAL person cropped to this region BEFORE try-on.\n"
        "  [RESULT " + str(piece_label).upper() + " REGION] = the generated result cropped to the SAME region.\n"
        "QUESTION: is the reference " + str(piece_label) + " piece PRESENT and CORRECTLY applied in "
        "the [RESULT " + str(piece_label).upper() + " REGION]?\n"
        "Grade piece_present_correct:\n"
        "  2 = the reference " + str(piece_label) + " piece is present and matches the reference (correct edit),\n"
        "  1 = present but a mild/uncertain difference,\n"
        "  0 = the reference " + str(piece_label) + " piece is CLEARLY MISSING (this region still shows the "
        "ORIGINAL garment, or is bare/underwear when the reference piece should be worn) OR a clearly "
        "DIFFERENT, substituted garment replaces it.\n"
        + PIECE_UNDERWEAR_CLAUSE + "\n"
        + PIECE_OVERLAP_CLAUSE + "\n"
        + GARMENT_SWAP_COVERAGE_COMPENSATION + "\n"
        "IGNORE differences that come merely from being worn (drape, folds, pose, body shape, "
        "scale, crop, lighting/shadow) and minor face/skin/hair regen. Only grade 0 when the "
        "defect is CLEAR and obvious.\n"
        "ANTI-BIAS: the ORIGINAL and RESULT crops may be given in either order; rely on the "
        "[ORIGINAL]/[RESULT] labels, never on position.\n"
        "Reason BRIEFLY (one short phrase). You MUST finish with ONE final JSON object on the "
        "last line with exactly this key (integer 0/1/2):\n"
        '{"piece_present_correct":_}'
    )
    cloth_url = img_to_data_url(cloth_path)
    model_url = img_to_data_url(model_crop)
    result_url = img_to_data_url(result_crop)
    if not swap:
        region_block = [
            {"type": "text", "text": "[ORIGINAL " + str(piece_label).upper() + " REGION]:"},
            {"type": "image_url", "image_url": {"url": model_url}},
            {"type": "text", "text": "[RESULT " + str(piece_label).upper() + " REGION]:"},
            {"type": "image_url", "image_url": {"url": result_url}},
        ]
    else:
        region_block = [
            {"type": "text", "text": "[RESULT " + str(piece_label).upper() + " REGION]:"},
            {"type": "image_url", "image_url": {"url": result_url}},
            {"type": "text", "text": "[ORIGINAL " + str(piece_label).upper() + " REGION]:"},
            {"type": "image_url", "image_url": {"url": model_url}},
        ]
    content = (
        [{"type": "text", "text": "[REFERENCE GARMENT]:"},
         {"type": "image_url", "image_url": {"url": cloth_url}}]
        + region_block
        + [{"type": "text", "text": instr}]
    )
    return [{"role": "user", "content": content}]


# sentinel returned when a per-piece read is uncertain — caller defers to the
# frozen full-frame run_pres_judge garment_swap grade (no recall collapse).
GS_ZOOM_DEFER = "defer"


def _read_piece(pick_ep, model_crop, result_crop, cloth_path, piece_label,
                seeds, native_think=False, fidelity_zoom=False):
    """Order-swap ensemble of per-piece reads.

    fidelity_zoom=False (PRESENCE, byte-identical to iter_026): returns the
    averaged piece grade (0..2 int) from "piece_present_correct", or None if
    every pass failed to parse.

    fidelity_zoom=True (FIDELITY/DEFECT): returns the rounded mean of the
    per-pass "region_defect" votes (0..1 int, 1 = confirmed region defect), or
    None if every pass failed to parse."""
    votes = []
    for i, seed in enumerate(seeds):
        base_url, model = pick_ep()
        swap = (i % 2 == 1)
        try:
            msgs = build_pres_piece_messages(model_crop, result_crop, cloth_path,
                                             piece_label, swap=swap,
                                             fidelity_zoom=fidelity_zoom)
            txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed,
                             enable_thinking=bool(native_think))
            parsed = extract_last_json(txt)
            if not parsed:
                txt = call_judge(base_url, model, msgs, atom_read=True, seed=seed,
                                 enable_thinking=False, max_tokens=1024)
                parsed = extract_last_json(txt)
            if not parsed:
                continue
            if fidelity_zoom:
                rd = parsed.get("region_defect")
                try:
                    rd = int(rd)
                except (TypeError, ValueError):
                    rd = None
                if rd in (0, 1):
                    votes.append(rd)
            else:
                g = _grade(parsed.get("piece_present_correct"))
                if g is not None:
                    votes.append(g)
        except Exception:
            continue
    if not votes:
        return None
    return int(round(sum(votes) / len(votes)))


def run_pres_zoom(pick_ep, model_img, result, cloth, two_piece, seeds,
                  native_think=False, fidelity_zoom=False):
    """Per-piece REGION-ZOOM garment_swap read for a two-piece suit. Crops the
    input_model AND the result to the UPPER box and the LOWER box (via
    _target_box; image-derived, NEVER the path), runs a per-piece presence read
    on each region, and fuses a garment_swap grade:
      - 2 if BOTH pieces are present-and-correct (grade 2),
      - 0 if either piece is positively confirmed MISSING/substituted (grade 0)
            in its own crop,
      - GS_ZOOM_DEFER sentinel if any crop read is uncertain (grade 1 / None) —
            the caller defers to the frozen full-frame run_pres_judge grade so
            recall does not collapse.
    Returns (garment_swap_grade_or_sentinel, sub) where sub carries per-piece
    sub fields (upper_present, lower_present).

    iter_028 (fidelity_zoom=True): each per-region read is a STRICT per-region
    FIDELITY/DEFECT check (region_defect 0|1) instead of the lenient presence
    read. Fusion: garment_swap=0 if ANY region has region_defect=1 (a confirmed
    wrong/missing/distorted garment in that region), else garment_swap=2.
    Unparseable reads still return GS_ZOOM_DEFER. fidelity_zoom=False is
    byte-identical to iter_026 presence behavior."""
    mim = model_img if isinstance(model_img, Image.Image) else Image.open(model_img)
    mim = mim.convert("RGB")
    rim = result if isinstance(result, Image.Image) else Image.open(result)
    rim = rim.convert("RGB")
    mw, mh = mim.size
    rw, rh = rim.size
    pieces = {}
    for piece in ("upper", "lower"):
        m_crop = M.crop(mim, _target_box(mw, mh, piece))
        r_crop = M.crop(rim, _target_box(rw, rh, piece))
        pieces[piece] = _read_piece(pick_ep, m_crop, r_crop, cloth, piece,
                                    seeds, native_think=native_think,
                                    fidelity_zoom=fidelity_zoom)
    up = pieces["upper"]
    lo = pieces["lower"]
    sub = {"gs_zoom_upper_present": up, "gs_zoom_lower_present": lo,
           "upper_present": up, "lower_present": lo}
    if fidelity_zoom:
        # iter_028 FIDELITY fusion: up/lo are now per-region DEFECT votes
        # (1 = confirmed region defect, 0 = clean). Stamp them under the
        # region_defect fields, then fuse: garment_swap=0 if ANY region has a
        # confirmed defect, else garment_swap=2. Unparseable reads defer to the
        # frozen full-frame grade (no recall collapse).
        sub["fidelity_zoom"] = True
        sub["upper_region_defect"] = up
        sub["lower_region_defect"] = lo
        if up is None or lo is None:
            return GS_ZOOM_DEFER, sub
        if up == 1 or lo == 1:
            return 0, sub
        return 2, sub
    # any read failed to parse -> defer.
    if up is None or lo is None:
        return GS_ZOOM_DEFER, sub
    # positively-confirmed missing/substituted in its own crop -> swap.
    if up == 0 or lo == 0:
        return 0, sub
    # both confidently present-and-correct -> not a swap.
    if up == 2 and lo == 2:
        return 2, sub
    # otherwise (any uncertain grade-1 read) -> defer to the frozen full-frame.
    return GS_ZOOM_DEFER, sub
