#!/usr/bin/env python3
"""Translate manifest English prompts to Chinese using local gemma4 vLLM.

Outputs (under ``web/data/``):
  - ``translation_cache.json``: ``{english_prompt: chinese_prompt}`` (dedup map).
  - ``prompts_zh.jsonl``: one record per manifest entry with key/task/en/zh.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent
WEB_DATA = ROOT / "web" / "data"
MANIFEST_PATH = WEB_DATA / "manifest.json"
CACHE_PATH = WEB_DATA / "translation_cache.json"
PER_RECORD_PATH = WEB_DATA / "prompts_zh.jsonl"

DEFAULT_ENDPOINT = "http://127.0.0.1:25931/v1/chat/completions"
DEFAULT_MODEL = "gemma4"

SYSTEM_PROMPT = (
    "你是一名专业的中英翻译，专门翻译图像编辑/生成指令。请将用户给出的英文指令翻译成"
    "自然、地道、保留原意的简体中文。严格遵守以下规则：\n"
    "1. 只输出中文译文本身，不要任何解释、引号包裹、前后缀或额外说明。\n"
    "2. 保留所有在原文中以英文引号、单引号或反引号包裹的字符串字面值，"
    "原样保留它们的英文内容与引号，不要翻译这些字符串。\n"
    "3. 数字、单位、坐标、颜色十六进制（如 #FF0000）、文件名、变量名等保持原样。\n"
    "4. 表达专业的图像编辑动词（如 inpaint、relight、bokeh 等）使用业内常见中文译法，"
    "若无统一译法可保留英文。\n"
    "5. 不要添加原文没有的信息，也不要遗漏原文中的细节。"
)


_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def load_cache() -> Dict[str, str]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in data.items() if isinstance(v, str)}


def save_cache(cache: Dict[str, str]) -> None:
    tmp = CACHE_PATH.with_suffix(CACHE_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(CACHE_PATH)


def call_chat(
    endpoint: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 2048,
    timeout: int = 180,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": max_tokens,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["choices"][0]["message"]["content"]


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_translation(text: str) -> str:
    text = _THINK_RE.sub("", text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = text.strip().strip("\"' \n\t")
    return text


def translate_one(
    prompt: str,
    *,
    endpoint: str,
    model: str,
    max_retries: int = 3,
) -> str:
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = call_chat(endpoint, model, prompt)
            zh = clean_translation(raw)
            if zh:
                return zh
            last_err = RuntimeError("empty translation")
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
        time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"translate failed after {max_retries} retries: {last_err}")


def collect_prompts() -> tuple[List[dict], List[str]]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    records = manifest["records"]
    unique = sorted({r["prompt"] for r in records if r.get("prompt")})
    return records, unique


def write_per_record(records: List[dict], cache: Dict[str, str]) -> None:
    tmp = PER_RECORD_PATH.with_suffix(PER_RECORD_PATH.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            en = r.get("prompt", "")
            row = {
                "key": r["key"],
                "task": r.get("task", ""),
                "prompt_en": en,
                "prompt_zh": cache.get(en, ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(PER_RECORD_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Persist cache to disk every N completed translations.",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Only translate first N (debug)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-translate even if cache already has a value.",
    )
    args = parser.parse_args()

    records, unique_prompts = collect_prompts()
    log(f"records={len(records)} unique_prompts={len(unique_prompts)}")

    cache = load_cache()
    log(f"loaded cache entries: {len(cache)} (filled: {sum(1 for v in cache.values() if v)})")

    todo: List[str] = []
    for p in unique_prompts:
        if not args.force and cache.get(p):
            continue
        todo.append(p)
    if args.limit:
        todo = todo[: args.limit]
    log(f"to translate: {len(todo)}")

    if not todo:
        write_per_record(records, cache)
        log(f"nothing to do; rewrote {PER_RECORD_PATH}")
        return 0

    completed = 0
    failures: List[tuple[str, str]] = []
    start = time.time()

    def worker(p: str) -> tuple[str, str | None, str | None]:
        try:
            return p, translate_one(p, endpoint=args.endpoint, model=args.model), None
        except Exception as exc:
            return p, None, str(exc)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, p) for p in todo]
        for fut in as_completed(futures):
            en, zh, err = fut.result()
            completed += 1
            if zh is None:
                failures.append((en, err or "unknown"))
                log(f"[{completed}/{len(todo)}] FAIL: {en[:60]!r} -> {err}")
            else:
                cache[en] = zh
            if completed % args.save_every == 0 or completed == len(todo):
                save_cache(cache)
                elapsed = time.time() - start
                rate = completed / elapsed if elapsed else 0.0
                log(
                    f"[{completed}/{len(todo)}] saved cache, "
                    f"elapsed={elapsed:.1f}s rate={rate:.2f}/s "
                    f"fail={len(failures)}"
                )

    save_cache(cache)
    write_per_record(records, cache)
    log(f"wrote cache: {CACHE_PATH}")
    log(f"wrote per-record: {PER_RECORD_PATH}")
    if failures:
        log(f"FAILED {len(failures)} prompts (first 5):")
        for en, err in failures[:5]:
            log(f"  - {en[:80]!r}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
