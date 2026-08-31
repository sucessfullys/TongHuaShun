"""Standalone virtual try-on evaluator — a single ERA tryon-eval (iter_028)
category-free method, packaged with NO dependency on the
evaluation-retrieval-agent repo. Everything this agent needs lives under this
directory.

Method: hybrid-crossover-step0-122b-v13-fidelityzoom-coverage-first
iter_028 SHIP (the operator's chosen evaluator, "best-balanced suit+dress").
It is the iter_026 v11-gszoom-coverage-first body (the v9-suit-relaxed
confirm-correct-first RELAXED suit garment_swap + the ADDITIVE per-piece
REGION-ZOOM garment_swap read) with ONE iter_028 change: the per-piece
REGION-ZOOM read is re-framed from a LENIENT PRESENCE check ("is THIS
reference piece PRESENT?", piece_present_correct 0|1|2) to a STRICT per-region
FIDELITY / DEFECT check (fidelity_zoom). On a two-piece / suit reference the
zoom crops the input_model AND the result to the upper and lower target boxes
(image-derived via _target_box; NEVER the file path) and asks, PER crop, "is
the garment worn in THIS region the SAME garment as the reference, correctly
applied and DEFECT-FREE, vs a DEFECT?" emitting {region_defect: 0|1}. Fusion:
garment_swap = 0 if ANY region has a confirmed region_defect (a wrong/missing/
distorted garment in that region), else garment_swap = 2; an unparseable read
DEFERS to the frozen full-frame grade (no recall collapse).

Why fidelity (not presence): the lenient presence read answered "present: yes"
too readily on dresses (grade 2), missing real per-region defects. The
per-region FIDELITY discrimination catches dress defects ~5.4x better than v11
while keeping suits clean (total garment_swap=0 flags across the 500-sample CES
window: v9=205 over-flag, v11=56 lenient, v13=89 discriminating). The fidelity
read is robust to dress-vs-suit because it grades defect-fidelity per region
either way (NO tight_twopiece gate).

FIXED GATE (carried from iter_026): the zoom fires on `is_two_piece is True`
ALONE — it does NOT require two_piece_confidence=="high" (that field is not
emitted on this path; requiring it disabled the zoom). The single-garment path
is byte-frozen vs v9. is_two_piece is image-derived in the STEP-0 read. 3 judge
calls/sample on a single-garment sample; +2 per-piece fidelity-zoom calls on a
confident two-piece suit.

The v9 strict/high-recall reference (hybrid-crossover-step0-122b-v9-suit-relaxed-
coverage-first) and the v11 max-lenient reference
(hybrid-crossover-step0-122b-v11-gszoom-coverage-first) ship as separate agents
and are shown alongside this SHIP slot in the combined CES review app, for
per-garment-type comparison."""

JUDGE_NAME = "Qwen3.5-122B-A10B"
SERVED_MODEL_NAME = "qwen3p5-122b-a10b"

CONFIGS = {
    "hybrid-crossover-step0-122b-v13-fidelityzoom-coverage-first":
        "iter_028 SHIP — v11-gszoom base + per-region FIDELITY/DEFECT zoom "
        "(fidelity_zoom): the strict per-region read catches dress per-region "
        "defects (wrong/missing/distorted garment) the lenient presence read "
        "passed over, while keeping suits clean — best-balanced suit+dress "
        "evaluator; single-garment path byte-frozen vs v9",
}

DEFAULT_CONFIG = "hybrid-crossover-step0-122b-v13-fidelityzoom-coverage-first"
DEFAULT_PORT = 8731
