# MCP Tools — Design Reference

## Overview

This directory contains the design documentation and JSON Schema definitions for
the eight Model Context Protocol (MCP) tool candidates planned for the
System-Prompt-Retrieval-Agent. No server implementation is included here; see
"Why Defer the Server" below.

All schemas live in `mcp_tools/schemas/` and follow JSON Schema Draft 2020-12.
Each tool has an `.input.json` and `.output.json` file.

---

## Why Defer the Server

MVP ships Python-first. The Python modules under
`src/system_prompt_retrieval_agent/` establish the definitive contracts for
remote dispatch, memory management, evaluation, and scoring. MCP server wrappers
are planned post-S10 (after all Python contracts have stabilized and been
integration-tested). Implementing the server before the Python contracts are
stable would require maintaining two contract surfaces simultaneously, which
creates drift risk with no benefit. Once the Python layer is stable, wrapping it
in an MCP server is a thin I/O shim.

---

## Tool Catalog

### 1. `remote_prompt_pair_pipeline`

**Summary:** Run the full three-stage remote pipeline (Gemma -> Flux -> Qwen)
for a single prompt pair on the remote GPU server.

**Python owner:** `system_prompt_retrieval_agent.remote.client`

**Input schema:** `schemas/remote_prompt_pair_pipeline.input.json`
Mirrors `RemoteStageRequest` from `schemas.py`.

**Output schema:** `schemas/remote_prompt_pair_pipeline.output.json`
Mirrors `RemoteStageResponse` from `schemas.py`.

**Error codes:**

| Code | Meaning |
|---|---|
| `REMOTE_UNREACHABLE` | SSH tunnel or HTTP connection failed. |
| `PIPELINE_PARTIAL` | Some samples failed; `allow_partial` was False. |
| `BUDGET_EXCEEDED` | Run was aborted due to cost guard. |
| `SCHEMA_MISMATCH` | Server returned an unexpected response shape. |

**Example call:**

```json
{
  "tool": "remote_prompt_pair_pipeline",
  "input": {
    "run_id": "run_2026_01",
    "round_id": 3,
    "prompt_pair_id": "pp_0042",
    "system_prompt_id": "sp_0011",
    "system_prompt_text": "A fashionable outfit with clean lines...",
    "negative_prompt_id": "np_0005",
    "negative_prompt": "blurry, distorted, watermark",
    "dataset_root": "/mnt/image-edit/datasets/xywang/dataset",
    "output_root": "/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/outputs/run_2026_01/round_03",
    "limit": 0,
    "allow_partial": false
  }
}
```

---

### 2. `remote_gemma_stage`

**Summary:** Dispatch only the Gemma text-conditioning stage via
`/stage/gemma` on the remote server.

**Python owner:** `system_prompt_retrieval_agent.remote.client`

**Input schema:** `schemas/remote_gemma_stage.input.json`
Mirrors `RemoteStageRequest`. Same shape as `remote_prompt_pair_pipeline.input.json`.

**Output schema:** `schemas/remote_gemma_stage.output.json`
Mirrors `RemoteStageResponse` with `stage` always `"gemma"`.

**Error codes:**

| Code | Meaning |
|---|---|
| `REMOTE_UNREACHABLE` | SSH tunnel or HTTP connection failed. |
| `STAGE_FAILED` | Gemma worker reported a fatal error. |
| `VRAM_OOM` | GPU ran out of memory during inference. |

**Example call:**

```json
{
  "tool": "remote_gemma_stage",
  "input": {
    "run_id": "run_2026_01",
    "round_id": 3,
    "prompt_pair_id": "pp_0042",
    "system_prompt_id": "sp_0011",
    "system_prompt_text": "A fashionable outfit with clean lines...",
    "dataset_root": "/mnt/image-edit/datasets/xywang/dataset",
    "output_root": "/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/outputs/run_2026_01/round_03"
  }
}
```

---

### 3. `remote_flux_stage`

**Summary:** Dispatch only the Flux image-generation stage via `/stage/flux`
on the remote server.

**Python owner:** `system_prompt_retrieval_agent.remote.client`

**Input schema:** `schemas/remote_flux_stage.input.json`
Mirrors `RemoteStageRequest`. Same shape as `remote_gemma_stage.input.json`.

**Output schema:** `schemas/remote_flux_stage.output.json`
Mirrors `RemoteStageResponse` with `stage` always `"flux"`.

**Error codes:**

| Code | Meaning |
|---|---|
| `REMOTE_UNREACHABLE` | SSH tunnel or HTTP connection failed. |
| `STAGE_FAILED` | Flux worker reported a fatal error. |
| `VRAM_OOM` | GPU ran out of memory during inference. |

**Example call:**

```json
{
  "tool": "remote_flux_stage",
  "input": {
    "run_id": "run_2026_01",
    "round_id": 3,
    "prompt_pair_id": "pp_0042",
    "system_prompt_id": "sp_0011",
    "system_prompt_text": "A fashionable outfit with clean lines...",
    "dataset_root": "/mnt/image-edit/datasets/xywang/dataset",
    "output_root": "/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/outputs/run_2026_01/round_03"
  }
}
```

---

### 4. `remote_qwen_stage`

**Summary:** Dispatch only the Qwen VLM scoring stage via `/stage/qwen`
on the remote server.

**Python owner:** `system_prompt_retrieval_agent.remote.client`

**Input schema:** `schemas/remote_qwen_stage.input.json`
Mirrors `RemoteStageRequest`. Same shape as `remote_gemma_stage.input.json`.

**Output schema:** `schemas/remote_qwen_stage.output.json`
Mirrors `RemoteStageResponse` with `stage` always `"qwen"`.

**Error codes:**

| Code | Meaning |
|---|---|
| `REMOTE_UNREACHABLE` | SSH tunnel or HTTP connection failed. |
| `STAGE_FAILED` | Qwen worker reported a fatal error. |
| `VRAM_OOM` | GPU ran out of memory during inference. |

**Example call:**

```json
{
  "tool": "remote_qwen_stage",
  "input": {
    "run_id": "run_2026_01",
    "round_id": 3,
    "prompt_pair_id": "pp_0042",
    "system_prompt_id": "sp_0011",
    "system_prompt_text": "A fashionable outfit with clean lines...",
    "dataset_root": "/mnt/image-edit/datasets/xywang/dataset",
    "output_root": "/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/outputs/run_2026_01/round_03"
  }
}
```

---

### 5. `artifact_manager`

**Summary:** Copy generated artifacts from the remote server to the local
machine via rsync and verify that required files arrived.

**Python owner:** `system_prompt_retrieval_agent.remote.copyback`

**Input schema:** `schemas/artifact_manager.input.json`

Fields: `remote_path`, `local_path`, `ssh_alias`, `required_file?`

**Output schema:** `schemas/artifact_manager.output.json`

Fields: `ok`, `verified_files`, `error?`, `bytes_transferred?`

**Error codes:**

| Code | Meaning |
|---|---|
| `TRANSFER_FAILED` | rsync returned a non-zero exit code. |
| `VERIFY_FAILED` | `required_file` was not found under `local_path`. |
| `SSH_UNREACHABLE` | Could not connect via `ssh_alias`. |

**Example call:**

```json
{
  "tool": "artifact_manager",
  "input": {
    "remote_path": "/mnt/image-edit/datasets/xywang/code/System-Prompt-Retrieval-Agent/outputs/run_2026_01/round_03/pp_0042",
    "local_path": "/Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/outputs/run_2026_01/round_03/pp_0042",
    "ssh_alias": "3h100",
    "required_file": "manifest.json"
  }
}
```

---

### 6. `memory_manager`

**Summary:** Invoke one of four memory operations: load context for generation,
append a pair to long memory, persist a pair to round memory, or prune stale
long-memory entries.

**Python owner:** `system_prompt_retrieval_agent.memory.manager`

**Input schema:** `schemas/memory_manager.input.json`

Fields: `op` (enum), `args` (object; shape depends on `op`)

Supported `op` values:

| Op | Description |
|---|---|
| `load_for_generation` | Retrieve `PromptPairHistoryContext` for the current round. |
| `append_long_memory` | Append a pair to the long-term memory store. |
| `write_pair` | Persist a `PromptPair` to the round memory store. |
| `prune_long_memory` | Remove low-scoring entries, keeping top-k. |

**Output schema:** `schemas/memory_manager.output.json`

Fields: `result` (any; shape depends on `op`), `schema_version`, `error?`

**Error codes:**

| Code | Meaning |
|---|---|
| `UNKNOWN_OP` | `op` value not recognized. |
| `MISSING_ARG` | A required arg was absent for the given `op`. |
| `STORAGE_ERROR` | Disk I/O failure reading or writing memory files. |

**Example call:**

```json
{
  "tool": "memory_manager",
  "input": {
    "op": "load_for_generation",
    "args": {
      "run_id": "run_2026_01",
      "round_id": 3,
      "top_k": 5
    }
  }
}
```

---

### 7. `evaluation_runner`

**Summary:** Run local API evaluation and Qwen output parsing for a batch of
generated image samples; return per-sample scores and total USD cost.

**Python owner:** `system_prompt_retrieval_agent.evaluation.local_api_eval`,
`system_prompt_retrieval_agent.evaluation.qwen_parser`

**Input schema:** `schemas/evaluation_runner.input.json`

Fields: `samples` (array of sample objects), `budget_usd?`

Each sample: `sample_id`, `image_path`, `reference_image_path?`, `category?`,
`prompt_pair_id?`, `qwen_output_path?`

**Output schema:** `schemas/evaluation_runner.output.json`

Fields: `results` (array), `usd_spent`, `budget_exceeded`

**Error codes:**

| Code | Meaning |
|---|---|
| `BUDGET_EXCEEDED` | `budget_usd` hit during the batch; partial results returned. |
| `API_ERROR` | OpenAI API call failed for one or more samples. |
| `INVALID_IMAGE` | Image at `image_path` could not be read or decoded. |
| `RATE_LIMIT` | OpenAI rate limit hit (> 3 req/s guard triggered). |

**Example call:**

```json
{
  "tool": "evaluation_runner",
  "input": {
    "samples": [
      {
        "sample_id": "sample_0001",
        "image_path": "/Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/outputs/run_2026_01/round_03/pp_0042/sample_0001.png",
        "reference_image_path": "/Volumes/970SSD/Code/Git/System-Prompt-Retrieval-Agent/dataset/sample_0001_ref.png",
        "category": "garment_transfer",
        "prompt_pair_id": "pp_0042"
      }
    ],
    "budget_usd": 2.0
  }
}
```

---

### 8. `score_aggregator`

**Summary:** Aggregate per-dimension sub-scores into an overall weighted score
and optional per-category breakdown using the helpers in `scoring/aggregate.py`.

**Python owner:** `system_prompt_retrieval_agent.scoring.aggregate`

**Input schema:** `schemas/score_aggregator.input.json`

Fields: `sub_scores` (mirrors `PromptPairSubScores`), `weights` (object),
`category_scores?` (mirrors `dict[str, CategoryScoreContext]`)

**Output schema:** `schemas/score_aggregator.output.json`

Mirrors `PromptPairScoreContext` from `schemas.py`.
Fields: `overall_score`, `total_score`, `sub_scores`, `category_scores?`,
`missing_score_reason?`

**Error codes:**

| Code | Meaning |
|---|---|
| `NO_VALID_SCORES` | All sub-score values were null; cannot aggregate. |
| `WEIGHT_MISMATCH` | `weights` keys do not match any `sub_scores` keys. |

**Example call:**

```json
{
  "tool": "score_aggregator",
  "input": {
    "sub_scores": {
      "qwen_pass_rate": 0.82,
      "edit_correctness": 0.75,
      "garment_transfer_correctness": 0.88,
      "preservation": 0.91,
      "artifact_penalty": 0.05
    },
    "weights": {
      "qwen_pass_rate": 0.3,
      "edit_correctness": 0.2,
      "garment_transfer_correctness": 0.25,
      "preservation": 0.15,
      "artifact_penalty": 0.1
    }
  }
}
```

---

## Schema File Index

```
mcp_tools/schemas/
  remote_prompt_pair_pipeline.input.json
  remote_prompt_pair_pipeline.output.json
  remote_gemma_stage.input.json
  remote_gemma_stage.output.json
  remote_flux_stage.input.json
  remote_flux_stage.output.json
  remote_qwen_stage.input.json
  remote_qwen_stage.output.json
  artifact_manager.input.json
  artifact_manager.output.json
  memory_manager.input.json
  memory_manager.output.json
  evaluation_runner.input.json
  evaluation_runner.output.json
  score_aggregator.input.json
  score_aggregator.output.json
```

Validate all schemas with:

```
.venv/bin/python scripts/verify_mcp_schemas.py
```

---

## Pydantic Model Mapping

| Schema file | Pydantic model |
|---|---|
| `remote_prompt_pair_pipeline.input.json` | `RemoteStageRequest` |
| `remote_prompt_pair_pipeline.output.json` | `RemoteStageResponse` |
| `remote_gemma_stage.input.json` | `RemoteStageRequest` |
| `remote_gemma_stage.output.json` | `RemoteStageResponse` |
| `remote_flux_stage.input.json` | `RemoteStageRequest` |
| `remote_flux_stage.output.json` | `RemoteStageResponse` |
| `remote_qwen_stage.input.json` | `RemoteStageRequest` |
| `remote_qwen_stage.output.json` | `RemoteStageResponse` |
| `score_aggregator.input.json` | `PromptPairSubScores`, `CategoryScoreContext` |
| `score_aggregator.output.json` | `PromptPairScoreContext` |
| `evaluation_runner.output.json` | `PromptPairSubScores` (per result entry) |

`artifact_manager` and `memory_manager` do not wrap a single pydantic model but
are defined compositionally from the existing module contracts.
