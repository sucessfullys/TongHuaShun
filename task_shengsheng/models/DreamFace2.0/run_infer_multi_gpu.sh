#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change these when needed.
GPUS="${GPUS:-0,1,2,3,4}"
INPUT_JSONL="${INPUT_JSONL:-${PROJECT_DIR}/prompt_jobs/dreamface2_highrisk_hand_foot_prompts_20260803.jsonl}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
RUN_NAME="${RUN_NAME:-highrisk_hand_foot}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_DIR}/output/${RUN_NAME}_${RUN_ID}}"

HEIGHT="${HEIGHT:-1152}"
WIDTH="${WIDTH:-896}"
STEPS="${STEPS:-4}"
CFG="${CFG:-1.0}"
SEEDS_PER_PROMPT="${SEEDS_PER_PROMPT:-3}"
SEED_MIN="${SEED_MIN:-1}"
SEED_MAX="${SEED_MAX:-2147483647}"
TORCH_DTYPE="${TORCH_DTYPE:-bf16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

require_file() {
    local name="$1"
    local path="$2"
    if [[ ! -f "${path}" ]]; then
        echo "[ERROR] ${name} does not exist: ${path}" >&2
        exit 1
    fi
}

require_dir() {
    local name="$1"
    local path="$2"
    if [[ ! -d "${path}" ]]; then
        echo "[ERROR] ${name} does not exist: ${path}" >&2
        exit 1
    fi
}

require_file INPUT_JSONL "${INPUT_JSONL}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "[ERROR] PYTHON_BIN is not executable or not found: ${PYTHON_BIN}" >&2
    exit 1
fi
require_dir MODEL_PATH "${MODEL_PATH}"

IFS=',' read -r -a GPU_ARRAY <<< "${GPUS}"
NUM_SHARDS="${#GPU_ARRAY[@]}"
if [[ "${NUM_SHARDS}" -lt 1 ]]; then
    echo "[ERROR] GPUS is empty" >&2
    exit 1
fi

mkdir -p "${OUT_ROOT}/shards" "${OUT_ROOT}/logs" "${OUT_ROOT}/tmp"

echo "============================================"
echo "  DreamFace2.0 multi-GPU inference"
echo "  GPUs      : ${GPUS}"
echo "  Shards    : ${NUM_SHARDS}"
echo "  Input     : ${INPUT_JSONL}"
echo "  Model     : ${MODEL_PATH}"
echo "  Output    : ${OUT_ROOT}"
echo "  Size      : ${WIDTH}x${HEIGHT}"
echo "  Steps/CFG : ${STEPS}/${CFG}"
echo "  Seeds     : ${SEEDS_PER_PROMPT} random seed(s) per prompt, range [${SEED_MIN}, ${SEED_MAX}]"
echo "============================================"

INPUT_JSONL="${INPUT_JSONL}" OUT_ROOT="${OUT_ROOT}" NUM_SHARDS="${NUM_SHARDS}" SEEDS_PER_PROMPT="${SEEDS_PER_PROMPT}" SEED_MIN="${SEED_MIN}" SEED_MAX="${SEED_MAX}" "${PYTHON_BIN}" - <<'PY'
import json
import os
import random
from pathlib import Path

input_jsonl = Path(os.environ["INPUT_JSONL"])
out_root = Path(os.environ["OUT_ROOT"])
num_shards = int(os.environ["NUM_SHARDS"])
seeds_per_prompt = int(os.environ["SEEDS_PER_PROMPT"])
seed_min = int(os.environ["SEED_MIN"])
seed_max = int(os.environ["SEED_MAX"])
shard_dir = out_root / "shards"
shard_dir.mkdir(parents=True, exist_ok=True)

if seeds_per_prompt < 1:
    raise SystemExit(f"SEEDS_PER_PROMPT must be >= 1, got {seeds_per_prompt}")
if seed_min < 0 or seed_max < seed_min:
    raise SystemExit(f"Invalid seed range: [{seed_min}, {seed_max}]")

rng = random.SystemRandom()
expanded = []
used_seeds_by_id = {}
for line_no, line in enumerate(input_jsonl.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    base_id = str(row.get("id") or f"row_{line_no:06d}")
    used = used_seeds_by_id.setdefault(base_id, set())
    for seed_index in range(1, seeds_per_prompt + 1):
        while True:
            seed = rng.randint(seed_min, seed_max)
            if seed not in used:
                used.add(seed)
                break
        item = dict(row)
        item["original_id"] = base_id
        item["seed_index"] = seed_index
        item["seed"] = seed
        item["id"] = f"{base_id}_seed{seed}"
        expanded.append(json.dumps(item, ensure_ascii=False, separators=(",", ":")))

writers = [(shard_dir / f"shard_{i}.jsonl").open("w", encoding="utf-8") for i in range(num_shards)]
try:
    for idx, line in enumerate(expanded):
        writers[idx % num_shards].write(line + "\n")
finally:
    for f in writers:
        f.close()

print(f"[split] prompts={sum(len(v) for v in used_seeds_by_id.values()) // seeds_per_prompt} seeds_per_prompt={seeds_per_prompt} expanded_total={len(expanded)} shards={num_shards}")
for i in range(num_shards):
    p = shard_dir / f"shard_{i}.jsonl"
    n = sum(1 for _ in p.open(encoding="utf-8"))
    print(f"[split] shard_{i}: {n}")
PY

pids=()
for shard_idx in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[${shard_idx}]}"
    shard_jsonl="${OUT_ROOT}/shards/shard_${shard_idx}.jsonl"
    shard_out="${OUT_ROOT}/gpu${gpu}"
    log_file="${OUT_ROOT}/logs/shard_${shard_idx}_gpu${gpu}.log"
    tmp_dir="${OUT_ROOT}/tmp/gpu${gpu}"
    mkdir -p "${tmp_dir}"

    echo "[launch] shard=${shard_idx}/${NUM_SHARDS} gpu=${gpu} log=${log_file}"
    TMPDIR="${tmp_dir}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" \
        "${PROJECT_DIR}/infer_dreamface2_batch.py" \
        --model-path "${MODEL_PATH}" \
        --input-jsonl "${shard_jsonl}" \
        --output-dir "${shard_out}" \
        --height "${HEIGHT}" \
        --width "${WIDTH}" \
        --steps "${STEPS}" \
        --cfg "${CFG}" \
        --seed -1 \
        --torch-dtype "${TORCH_DTYPE}" \
        ${EXTRA_ARGS} \
        > "${log_file}" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done

if [[ "${failed}" -ne 0 ]]; then
    echo "[ERROR] at least one shard failed. Check logs: ${OUT_ROOT}/logs" >&2
    exit 1
fi

MERGED="${OUT_ROOT}/results_merged.jsonl"
: > "${MERGED}"
for gpu in "${GPU_ARRAY[@]}"; do
    if [[ -f "${OUT_ROOT}/gpu${gpu}/results.jsonl" ]]; then
        cat "${OUT_ROOT}/gpu${gpu}/results.jsonl" >> "${MERGED}"
    fi
done

echo "[DONE] Output: ${OUT_ROOT}"
echo "[DONE] Merged results: ${MERGED}"
