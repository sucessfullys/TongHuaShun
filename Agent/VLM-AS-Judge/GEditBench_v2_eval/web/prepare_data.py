#!/usr/bin/env python3
"""Prepare frontend manifest and original images for the comparison web UI."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
DATA_DIR = WEB_DIR / "data"
ORIGINAL_DIR = DATA_DIR / "originals"
TRANSLATION_CACHE_PATH = DATA_DIR / "translation_cache.json"
GALLERY_DIR = ROOT / "datasets" / "GEditBench-v2-CandidatesGallery"
AGENT_ARCHIVE_DIR = GALLERY_DIR / "ImageAgent_archive"
AGENT_ARCHIVE_NAME = AGENT_ARCHIVE_DIR.name
BENCH_DIR = ROOT / "datasets" / "GEditBench-v2"
PARQUET_DIR = BENCH_DIR / "data"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_AGENT_PREFIX = "ImageAgent"
EXPLICIT_FINAL_TOOL_KEYS = (
    "final_model",
    "final_winner_tool",
    "final_tool",
    "winner_tool",
)


def load_translation_cache() -> Dict[str, str]:
    if not TRANSLATION_CACHE_PATH.exists():
        return {}
    try:
        with TRANSLATION_CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str) and v}


def metadata_search_dirs() -> List[Path]:
    dirs = [GALLERY_DIR]
    if AGENT_ARCHIVE_DIR.is_dir():
        dirs.append(AGENT_ARCHIVE_DIR)
    return dirs


def metadata_files() -> List[Path]:
    candidates: List[Path] = []
    seen: set[str] = set()
    for search_dir in metadata_search_dirs():
        for metadata_file in sorted(search_dir.glob("metadata_*.jsonl"), key=lambda p: p.name):
            if metadata_file.name in seen:
                continue
            seen.add(metadata_file.name)
            candidates.append(metadata_file)
    fallback = GALLERY_DIR / "metadata.jsonl"
    if fallback.exists() and fallback.name not in seen:
        candidates.append(fallback)
    if not candidates:
        raise FileNotFoundError("No metadata_*.jsonl or metadata.jsonl found in gallery")
    return candidates


def latest_metadata_file() -> Path:
    candidates: List[Path] = []
    for search_dir in metadata_search_dirs():
        candidates.extend(search_dir.glob("metadata_*.jsonl"))
    if candidates:
        return max(candidates, key=lambda p: p.stat().st_mtime)
    fallback = GALLERY_DIR / "metadata.jsonl"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("No metadata_*.jsonl or metadata.jsonl found in gallery")


def is_model_dir(path: Path) -> bool:
    return path.is_dir() and not path.name.startswith(".")


def model_dir_for(model: str) -> Path | None:
    direct = GALLERY_DIR / model
    if direct.is_dir():
        return direct
    archived = AGENT_ARCHIVE_DIR / model
    if archived.is_dir():
        return archived
    return None


def rel_model_dir(model: str) -> str:
    if (GALLERY_DIR / model).is_dir():
        return model
    return f"{AGENT_ARCHIVE_NAME}/{model}"


def discover_models() -> List[str]:
    models: List[str] = []
    seen: set[str] = set()
    for entry in sorted(GALLERY_DIR.iterdir()):
        if not is_model_dir(entry):
            continue
        if entry.name == AGENT_ARCHIVE_NAME:
            continue
        models.append(entry.name)
        seen.add(entry.name)
    if AGENT_ARCHIVE_DIR.is_dir():
        for entry in sorted(AGENT_ARCHIVE_DIR.iterdir()):
            if not is_model_dir(entry):
                continue
            if entry.name in seen:
                continue
            models.append(entry.name)
            seen.add(entry.name)
    if not models:
        raise RuntimeError("No model folders found in candidates gallery")
    return models


def parse_metadata(metadata_file: Path) -> List[dict]:
    records: List[dict] = []
    with metadata_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                {
                    "key": obj["key"],
                    "prompt": obj.get("instruction", ""),
                    "task": obj.get("task", ""),
                }
            )
    return records


def load_metadata_records() -> tuple[Dict[str, dict], List[Path]]:
    files = metadata_files()
    records_by_key: Dict[str, dict] = {}
    for metadata_file in files:
        for record in parse_metadata(metadata_file):
            records_by_key[record["key"]] = record
    return records_by_key, files


def is_image_agent_model(model: str) -> bool:
    return model.startswith(IMAGE_AGENT_PREFIX)


def extract_final_tool_from_prompt(data: dict) -> str | None:
    for key in EXPLICIT_FINAL_TOOL_KEYS:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    steps: List[tuple[int, dict]] = []
    for field, value in data.items():
        if not field.startswith("step_") or not field.endswith("_winner_info"):
            continue
        if not isinstance(value, dict):
            continue
        try:
            step_idx = int(field.split("_")[1])
        except (IndexError, ValueError):
            continue
        steps.append((step_idx, value))
    if not steps:
        return None

    steps.sort(key=lambda item: item[0])
    for _, winner_info in steps:
        if winner_info.get("success"):
            tool = winner_info.get("winner_tool")
            if isinstance(tool, str) and tool.strip():
                return tool.strip()

    last_winner_info = steps[-1][1]
    tool = last_winner_info.get("winner_tool")
    if isinstance(tool, str) and tool.strip():
        return tool.strip()
    return None


def load_prompt_json(model_dir: Path, key: str) -> dict | None:
    prompt_path = model_dir / f"{key}_prompt.json"
    if not prompt_path.exists():
        return None
    try:
        with prompt_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def build_agent_final_tools(key: str, models: List[str]) -> Dict[str, str]:
    tools: Dict[str, str] = {}
    for model in models:
        if not is_image_agent_model(model):
            continue
        model_dir = model_dir_for(model)
        if not model_dir:
            continue
        prompt_data = load_prompt_json(model_dir, key)
        if not prompt_data:
            continue
        tool = extract_final_tool_from_prompt(prompt_data)
        if tool:
            tools[model] = tool
    return tools


def discover_candidates(models: List[str]) -> Dict[str, Dict[str, str]]:
    candidates_by_key: Dict[str, Dict[str, str]] = {}
    for model in models:
        model_dir = model_dir_for(model)
        if not model_dir:
            continue
        rel_prefix = rel_model_dir(model)
        for image_path in sorted(model_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            key = image_path.stem
            rel_image = f"{rel_prefix}/{image_path.name}"
            candidates_by_key.setdefault(key, {})[model] = rel_image
    return candidates_by_key


def export_originals(keys: set[str]) -> Dict[str, str]:
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    key_to_path: Dict[str, str] = {}

    parquet_files = sorted(PARQUET_DIR.glob("train-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No parquet files found under datasets/GEditBench-v2/data")

    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file, columns=["key", "source_image"])
        for row in table.to_pylist():
            key = row["key"]
            if key not in keys:
                continue
            if key in key_to_path:
                continue
            source_image = row.get("source_image") or {}
            payload = source_image.get("bytes")
            original_path = source_image.get("path") or ""
            ext = Path(original_path).suffix.lower() or ".png"
            if not payload:
                continue
            output_name = f"{key}{ext}"
            output_path = ORIGINAL_DIR / output_name
            output_path.write_bytes(payload)
            key_to_path[key] = f"data/originals/{output_name}"
        if len(key_to_path) == len(keys):
            break

    return key_to_path


def build_manifest() -> dict:
    models = discover_models()
    metadata_records, metadata_files_used = load_metadata_records()
    candidates_by_key = discover_candidates(models)
    target_keys = set(candidates_by_key)
    key_to_original = export_originals(target_keys)
    translation_cache = load_translation_cache()

    final_records = []
    for key in sorted(candidates_by_key):
        original = key_to_original.get(key)
        if not original:
            continue
        record = metadata_records.get(key, {"prompt": "", "task": ""})

        normalized_candidates = {}
        for model in models:
            rel_image = candidates_by_key[key].get(model)
            if not rel_image:
                continue
            abs_path = GALLERY_DIR / rel_image
            if not abs_path.exists():
                continue
            normalized_candidates[model] = f"../datasets/GEditBench-v2-CandidatesGallery/{rel_image}"

        prompt_en = record["prompt"]
        agent_final_tools = build_agent_final_tools(key, list(normalized_candidates))
        entry = {
            "key": key,
            "prompt": prompt_en,
            "prompt_zh": translation_cache.get(prompt_en, ""),
            "task": record["task"],
            "original": original,
            "candidates": normalized_candidates,
        }
        if agent_final_tools:
            entry["agent_final_tools"] = agent_final_tools
        final_records.append(entry)

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "metadata_file": latest_metadata_file().name,
        "metadata_files": [p.name for p in metadata_files_used],
        "models": models,
        "records": final_records,
    }


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    output_path = DATA_DIR / "manifest.json"
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Manifest written: {output_path}")
    print(f"Records: {len(manifest['records'])}")
    print(f"Models: {len(manifest['models'])}")


if __name__ == "__main__":
    main()
