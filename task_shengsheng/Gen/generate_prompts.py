#!/usr/bin/env python3
"""Generate deterministic, unique hand-anomaly prompts from the Gen configs."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
SEED_PROFILES = [
    ("medium_shot_hands_visible", "polydactyly", "single", "right", "female"),
    ("three_quarter_portrait", "oligodactyly", "single", "left", "male"),
    ("three_quarter_portrait", "syndactyly", "single", "left", "female"),
    ("medium_shot_hands_visible", "polydactyly", "both", "none", "male"),
    ("medium_shot_hands_visible", "multi_hand", "single", "right", "female"),
    ("full_body_hands_visible", "arm_anomaly", "single", "right", "male"),
    ("medium_shot_hands_visible", "double_palm", "single", "right", "male"),
    ("three_quarter_portrait", "polydactyly", "single", "left", "female"),
    ("three_quarter_portrait", "syndactyly", "both", "none", "female"),
    ("full_body_hands_visible", "arm_anomaly", "single", "left", "male"),
]

SCENE_VARIANTS = [
    "with a quiet, lightly populated background",
    "with subtle everyday activity in the middle distance",
    "with layered foreground and background environmental detail",
    "seen from a natural eye-level viewpoint near the main activity area",
    "with clean, orderly surroundings and restrained background detail",
    "with a few background figures rendered in gentle shallow focus",
    "with realistic lived-in details distributed through the setting",
    "with a spacious view that clearly establishes the environment",
    "with intimate environmental framing and natural candid detail",
    "with balanced architectural or landscape depth behind the subject",
]

COMPOSITION_TOTALS = {
    "medium_shot_hands_visible": 3334,
    "three_quarter_portrait": 3333,
    "full_body_hands_visible": 3333,
}
FAMILY_TOTALS = {
    "polydactyly": 1000, "oligodactyly": 1000, "multi_hand": 1000,
    "syndactyly": 1000, "arm_anomaly": 1000, "double_palm": 1000,
    "bifid_finger": 1000, "thumb_anomaly": 1000, "ectrodactyly": 1000,
    "macrodactyly": 1000,
}


def load_yaml(path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def quota_values(totals, already):
    values = []
    used = Counter(already)
    for key, total in totals.items():
        remaining = total - used[key]
        if remaining < 0:
            raise ValueError(f"Seed prompts exceed quota for {key}")
        values.extend([key] * remaining)
    return values


def original_description(data, family, variant, scope, side):
    node = data[family][variant]
    if scope == "both":
        return node["both"]["en"]
    return node["single"][side]["en"]


def extended_description(data, family, variant, scope, side):
    text = data[family][variant][scope]
    return text.format(side=side)


def normal_structure(family, scope, side):
    if scope == "both":
        return "the corresponding abnormal structures are presented consistently on both sides"
    other = "right" if side == "left" else "left"
    if family == "arm_anomaly":
        return f"the {other} arm and {other} hand keep normal structures"
    return f"the {other} hand keeps a normal five-finger structure"


def build_prompt(scene, action, subject, clothing, composition, anomaly,
                 normal, lighting, atmosphere, quality):
    parts = [
        "Photorealistic", scene, subject, clothing, composition, action,
        anomaly, normal, lighting,
        atmosphere, *quality, "realistic skin texture", "realistic fabric detail",
        "raw photo style", "highly detailed",
    ]
    return ", ".join(parts) + "."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260707)
    parser.add_argument("--output", type=Path, default=ROOT / "prompts_10000.json")
    parser.add_argument("--include-samples", type=Path, default=ROOT / "prompt_samples_10.json")
    parser.add_argument("--skip-samples", action="store_true",
                        help="Do not reuse the ten reference prompts (recommended for later batches).")
    parser.add_argument("--history", type=Path, default=ROOT / "prompt_history_sha256.txt")
    parser.add_argument("--summary", type=Path, default=ROOT / "generation_summary.json")
    args = parser.parse_args()
    if args.count != 10000:
        raise ValueError("This quota-balanced recipe currently requires --count 10000")

    rng = random.Random(args.seed)
    taxonomy = load_yaml(ROOT / "configs/taxonomy.yaml")
    templates = load_yaml(ROOT / "configs/prompt_templates.yaml")
    extended = load_yaml(ROOT / "configs/extended_anomalies.yaml")["anomaly_descriptions"]
    original = templates["anomaly_descriptions"]
    actions = taxonomy["actions"]

    samples = [] if args.skip_samples else json.loads(args.include_samples.read_text(encoding="utf-8"))
    seed_prompts = [x["prompt"] for x in samples]
    seed_profiles = [] if args.skip_samples else SEED_PROFILES
    if len(seed_prompts) != len(seed_profiles):
        raise ValueError("The seed sample file must contain the expected 10 prompts")

    previous_hashes = set()
    if args.history.exists():
        previous_hashes = {x.strip() for x in args.history.read_text().splitlines() if x.strip()}
    # The requested seed examples are intentionally allowed even if already in history.
    seen = set(seed_prompts)

    scene_catalog = []
    for scene_key, scene in taxonomy["scenes"].items():
        for variant_id, suffix in enumerate(SCENE_VARIANTS, 1):
            scene_catalog.append((f"{scene_key}_{variant_id:02d}", f"{scene['en']}, {suffix}", scene_key))
    if len(scene_catalog) != 500:
        raise AssertionError("Expected exactly 500 expanded scenes")

    # Each expanded scene appears 20 times before reserving ten rows for seed examples.
    scene_slots = scene_catalog * 20
    del scene_slots[:len(seed_prompts)]
    comp_slots = quota_values(COMPOSITION_TOTALS, [x[0] for x in seed_profiles])
    family_slots = quota_values(FAMILY_TOTALS, [x[1] for x in seed_profiles])
    gender_slots = quota_values({"male": 5000, "female": 5000}, [x[4] for x in seed_profiles])
    scope_slots = quota_values({"single": 7000, "both": 3000}, [x[2] for x in seed_profiles])
    single_seed_sides = [x[3] for x in seed_profiles if x[2] == "single"]
    side_slots = quota_values({"left": 3500, "right": 3500}, single_seed_sides)
    for slots in (scene_slots, comp_slots, family_slots, gender_slots, scope_slots, side_slots):
        rng.shuffle(slots)

    prompts = list(seed_prompts)
    records = [dict(zip(("composition", "family", "scope", "side", "gender"), p)) for p in seed_profiles]
    side_index = 0
    for i in range(args.count - len(seed_prompts)):
        scene_id, scene_text, base_scene = scene_slots[i]
        scene = taxonomy["scenes"][base_scene]
        composition_key = comp_slots[i]
        family = family_slots[i]
        gender = gender_slots[i]
        scope = scope_slots[i]
        side = "none" if scope == "both" else side_slots[side_index]
        side_index += scope == "single"

        variants = list((original if family in original else extended)[family])
        for _ in range(200):
            variant = rng.choice(variants)
            action_key = rng.choice(scene["allowed_actions"])
            subject = rng.choice(templates["subjects"][gender]["en"])
            clothing = rng.choice(templates["clothing"][gender])["en"]
            quality = rng.sample(templates["quality_style_variants"]["en"], 2)
            anomaly = (original_description(original, family, variant, scope, side)
                       if family in original else extended_description(extended, family, variant, scope, side))
            prompt = build_prompt(
                scene_text, actions[action_key]["en"], subject, clothing,
                templates["composition_overrides"][composition_key]["en"], anomaly,
                normal_structure(family, scope, side),
                scene["lighting_en"], scene["atmosphere_en"], quality,
            )
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt not in seen and digest not in previous_hashes:
                break
        else:
            raise RuntimeError(f"Could not find a unique prompt at row {i}")
        seen.add(prompt)
        prompts.append(prompt)
        records.append({"scene": scene_id, "composition": composition_key, "family": family,
                        "variant": variant, "scope": scope, "side": side, "gender": gender})

    if len(prompts) != args.count or len(set(prompts)) != args.count:
        raise AssertionError("Prompt count or uniqueness validation failed")
    args.output.write_text(json.dumps([{"prompt": p} for p in prompts], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hashes = sorted(previous_hashes | {hashlib.sha256(p.encode()).hexdigest() for p in prompts})
    args.history.write_text("\n".join(hashes) + "\n", encoding="utf-8")

    summary = {
        "count": len(prompts), "unique_count": len(set(prompts)), "seed": args.seed,
        "included_reference_samples": len(seed_prompts), "expanded_scene_count": len(scene_catalog),
        "composition": dict(Counter(r["composition"] for r in records)),
        "anomaly_family": dict(Counter(r["family"] for r in records)),
        "scope": dict(Counter(r["scope"] for r in records)),
        "gender": dict(Counter(r["gender"] for r in records)),
        "history_hash_count": len(hashes),
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "scene_catalog_500.json").write_text(
        json.dumps([{"scene_id": x[0], "description": x[1], "base_scene": x[2]} for x in scene_catalog], indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
