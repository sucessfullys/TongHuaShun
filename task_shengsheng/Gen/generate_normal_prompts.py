#!/usr/bin/env python3
"""Generate 10,000 deterministic, unique prompts with normal hand anatomy."""

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent

SCENE_VARIANTS = [
    "with a quiet, lightly populated background",
    "with subtle everyday activity in the middle distance",
    "with layered environmental detail and natural depth",
    "seen from a natural eye-level viewpoint near the main activity area",
    "with clean, orderly surroundings and restrained background detail",
    "with a few background figures rendered in gentle shallow focus",
    "with realistic lived-in details distributed through the setting",
    "with a spacious view that clearly establishes the environment",
    "with intimate environmental framing and natural candid detail",
    "with balanced architectural or landscape depth behind the subject",
]

NORMAL_HAND_DESCRIPTIONS = [
    "both hands have normal five-finger anatomy, natural joints, balanced proportions, and relaxed poses",
    "each hand has five complete fingers with realistic proportions and natural articulation",
    "both hands show natural five-finger anatomy, realistic knuckles, natural finger spacing, and relaxed wrist alignment",
    "both hands have five naturally formed fingers with realistic joints and balanced proportions",
    "all ten fingers are complete and naturally shaped, with realistic finger lengths and joints",
    "each hand has five complete fingers with realistic proportions and a believable everyday pose",
    "both hands have normal five-finger structure, natural joints, realistic fingernails, and relaxed poses",
    "both hands have five complete fingers with natural proportions and believable everyday gestures",
    "both hands have normal five-finger anatomy, smooth joint alignment, realistic finger proportions, and graceful movement",
    "each hand has five complete fingers with natural spacing, realistic knuckles, and anatomically correct proportions",
    "both hands are naturally formed with five distinct fingers on each hand and realistic joint placement",
    "the left and right hands each have five complete fingers, balanced palm proportions, and natural finger alignment",
    "both hands show realistic five-finger anatomy with complete fingertips, natural knuckles, and proportionate palms",
    "each hand presents five naturally shaped fingers with believable lengths, joints, and wrist alignment",
    "both hands have anatomically natural palms and five complete, proportionate fingers on each side",
]

COMPOSITION_TOTALS = {
    "medium_shot_hands_visible": 3334,
    "three_quarter_portrait": 3333,
    "full_body_hands_visible": 3333,
}

SAMPLE_PROFILES = [
    ("medium_shot_hands_visible", "female"),
    ("full_body_hands_visible", "male"),
    ("three_quarter_portrait", "female"),
    ("medium_shot_hands_visible", "male"),
    ("three_quarter_portrait", "female"),
    ("full_body_hands_visible", "male"),
    ("medium_shot_hands_visible", "female"),
    ("three_quarter_portrait", "male"),
    ("full_body_hands_visible", "female"),
    ("medium_shot_hands_visible", "male"),
]


def load_yaml(path):
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def quota_values(totals, existing):
    used = Counter(existing)
    values = []
    for key, total in totals.items():
        values.extend([key] * (total - used[key]))
    return values


def build_prompt(scene, subject, clothing, composition, action, hands,
                 lighting, atmosphere, quality):
    parts = [
        "Photorealistic", scene, subject, clothing, composition, action,
        hands, lighting, atmosphere, *quality, "realistic skin texture",
        "realistic fabric detail", "raw photo style", "highly detailed",
    ]
    return ", ".join(parts) + "."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--output", type=Path, default=ROOT / "prompts_10000_normal.json")
    parser.add_argument("--samples", type=Path, default=ROOT / "prompt_samples_10_normal.json")
    parser.add_argument("--history", type=Path, default=ROOT / "normal_prompt_history_sha256.txt")
    parser.add_argument("--summary", type=Path, default=ROOT / "generation_summary_normal.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    taxonomy = load_yaml(ROOT / "configs/taxonomy.yaml")
    templates = load_yaml(ROOT / "configs/prompt_templates.yaml")
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    sample_prompts = [row["prompt"] for row in samples]
    if len(sample_prompts) != 10:
        raise ValueError("Expected exactly ten reference samples")

    scene_catalog = []
    for scene_key, scene in taxonomy["scenes"].items():
        for variant_id, suffix in enumerate(SCENE_VARIANTS, 1):
            scene_catalog.append((
                f"{scene_key}_{variant_id:02d}",
                f"{scene['en']}, {suffix}",
                scene_key,
            ))
    if len(scene_catalog) != 500:
        raise AssertionError("Expected 500 expanded scene descriptions")

    scene_slots = scene_catalog * 20
    del scene_slots[:10]
    composition_slots = quota_values(
        COMPOSITION_TOTALS, [row[0] for row in SAMPLE_PROFILES])
    gender_slots = quota_values(
        {"male": 5000, "female": 5000}, [row[1] for row in SAMPLE_PROFILES])
    for slots in (scene_slots, composition_slots, gender_slots):
        rng.shuffle(slots)

    previous_hashes = set()
    if args.history.exists():
        previous_hashes = {
            line.strip() for line in args.history.read_text().splitlines()
            if line.strip()
        }

    prompts = list(sample_prompts)
    seen = set(prompts)
    records = [
        {"composition": composition, "gender": gender, "reference_sample": True}
        for composition, gender in SAMPLE_PROFILES
    ]

    for index in range(9990):
        scene_id, scene_text, base_scene = scene_slots[index]
        scene = taxonomy["scenes"][base_scene]
        composition_key = composition_slots[index]
        gender = gender_slots[index]

        for _ in range(200):
            action_key = rng.choice(scene["allowed_actions"])
            subject = rng.choice(templates["subjects"][gender]["en"])
            clothing = rng.choice(templates["clothing"][gender])["en"]
            hands = rng.choice(NORMAL_HAND_DESCRIPTIONS)
            quality = rng.sample(templates["quality_style_variants"]["en"], 2)
            prompt = build_prompt(
                scene_text,
                subject,
                clothing,
                templates["composition_overrides"][composition_key]["en"],
                taxonomy["actions"][action_key]["en"],
                hands,
                scene["lighting_en"],
                scene["atmosphere_en"],
                quality,
            )
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            if prompt not in seen and digest not in previous_hashes:
                break
        else:
            raise RuntimeError(f"Unable to create unique prompt at index {index}")

        prompts.append(prompt)
        seen.add(prompt)
        records.append({
            "scene": scene_id,
            "composition": composition_key,
            "gender": gender,
            "action": action_key,
            "reference_sample": False,
        })

    if len(prompts) != 10000 or len(seen) != 10000:
        raise AssertionError("Count or uniqueness validation failed")

    forbidden = (
        "hand-structure abnormality", "limb-structure abnormality",
        "extra finger", "additional hand", "fused together",
        "double-palm", "v-shaped cleft", "reversed-thumb",
    )
    for prompt in prompts:
        if any(term in prompt.lower() for term in forbidden):
            raise AssertionError(f"Abnormal-hand phrase found: {prompt}")

    args.output.write_text(
        json.dumps([{"prompt": prompt} for prompt in prompts], indent=2) + "\n",
        encoding="utf-8",
    )
    hashes = previous_hashes | {
        hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts
    }
    args.history.write_text("\n".join(sorted(hashes)) + "\n", encoding="utf-8")

    summary = {
        "count": len(prompts),
        "unique_count": len(seen),
        "included_reference_samples": len(sample_prompts),
        "expanded_scene_count": len(scene_catalog),
        "composition": dict(Counter(row["composition"] for row in records)),
        "gender": dict(Counter(row["gender"] for row in records)),
        "normal_hand_only": True,
        "seed": args.seed,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
