# Skill: running-remote-stage

## Name
running-remote-stage

## Description
Execute one complete remote inference stage using the V0.2.1 **stage-major**
dispatch model (Gemma-for-all-pairs → FLUX-for-surviving-pairs →
Qwen-for-surviving-pairs).  The legacy `/stage/all` endpoint is NOT used by
V0.2.1 orchestration code.

The end-to-end flow:

1. **SSH tunnel** — call `open_tunnel(alias, local_port, remote_port)` to
   forward the remote controller's port to localhost.  The tunnel performs a
   health-check poll and raises `TunnelError` if the controller is unreachable.

2. **Build stage request** — construct the appropriate request object:
   - `GemmaStageRequest` — carries `prompt_pairs[]`, `user_prompts[]`,
     `user_prompt_corpus_hash`, and optional `sample_ids[]` /
     `sample_manifest_path` + `sample_manifest_path_purpose`.  Only Gemma
     requests may carry non-null `system_prompt_text` on their pairs.
   - `FluxStageRequest` — carries `prompt_pairs[]` (surviving pairs only),
     `user_prompts[]`, `user_prompt_corpus_hash`, and a `sample_manifest_path`
     pointing to the Gemma survivor JSONL with
     `sample_manifest_path_purpose="prior_stage_survivor_cells"`.
     **Must not** set `system_prompt_text` on any pair — raises `ValueError`.
   - `QwenStageRequest` — same pattern as FLUX with
     `sample_manifest_path_purpose="prior_stage_survivor_cells"` from the
     FLUX survivor JSONL.  **Must not** set `system_prompt_text`.

3. **Dispatch stage** — use `StageDispatcher` or `RemoteControllerClient.post_stage_v021`:
   ```python
   from system_prompt_retrieval_agent.remote.stage_dispatcher import StageDispatcher
   dispatcher = StageDispatcher(http_client, base_url="http://127.0.0.1:17700")
   manifest = await dispatcher.dispatch_gemma(req)
   ```
   The dispatcher:
   - Validates no `system_prompt_text` on FLUX/Qwen pairs.
   - POSTs the batched payload to the per-stage endpoint.
   - Parses the response as `StageManifest`.
   - Checks `user_prompt_corpus_hash` echo; raises `CorpusDriftDetected` on
     mismatch.

4. **Partition surviving / failed pairs** — call `partition_stage_pairs`:
   ```python
   from system_prompt_retrieval_agent.remote.partition import partition_stage_pairs
   surviving, failed = partition_stage_pairs(
       stage_manifest=manifest,
       allow_partial=False,
       enabled_user_prompt_ids=user_prompt_ids,
       expected_samples_for_pair={pair_id: sample_ids},
   )
   ```
   Use `surviving` to build the `prompt_pairs[]` list for the next stage.

5. **Cell-keyed barrier** — call `barrier_cell_keyed` before copy-back.  This
   check is **non-negotiable** for surviving pairs:
   ```python
   from system_prompt_retrieval_agent.remote.barrier import barrier_cell_keyed
   surviving, failed = barrier_cell_keyed(
       stage_manifest=manifest,
       allow_partial=False,
       enabled_user_prompt_ids=user_prompt_ids,
       sample_universe_for_pair={pair_id: sample_ids},
       manifest_purpose=None,  # or "resume_missing_cells" / "prior_stage_survivor_cells"
   )
   ```
   Raises `BarrierViolation` only on surviving-pair violations; non-empty
   `failed_pairs` does **not** raise.

6. **Lifecycle assertion** — call `assert_lifecycle_ok` after the barrier:
   ```python
   from system_prompt_retrieval_agent.remote.lifecycle import assert_lifecycle_ok
   assert_lifecycle_ok(manifest, expected_mode="cold")
   ```

7. **Build survivor manifest** — call `build_survivor_manifest_with_samples`
   to produce the `sample_manifest_path` JSONL for the next stage:
   ```python
   from system_prompt_retrieval_agent.remote.resume import build_survivor_manifest_with_samples
   survivor_path = build_survivor_manifest_with_samples(
       prior_stage_manifest=manifest,
       sample_ids_for_pair={pair_id: sample_ids},
       local_manifests_root=output_root / "manifests",
   )
   ```

8. **rsync copy-back** — call `rsync_copyback` to pull the stage output
   directory.  Verifies a sentinel file exists; retries once on miss before
   raising `CopybackError`.

## Inputs

| Parameter | Type | Description |
|---|---|---|
| `req` | `GemmaStageRequest \| FluxStageRequest \| QwenStageRequest` | Fully-populated V0.2.1 stage request. |
| `local_output_dir` | `pathlib.Path` | Local directory where rsync will deposit the stage outputs. |
| `stage` | `str` | One of `"gemma"`, `"flux"`, `"qwen"`. |
| `ssh_alias` | `str` | SSH host alias from `~/.ssh/config` (default `"3h100"`). |
| `local_port` | `int` | Local port for the tunnel (default `17700`). |
| `allow_partial` | `bool` | If `True`, barrier and validator accept non-zero errors. |
| `enabled_user_prompt_ids` | `list[str]` | Full enabled user-prompt ID list for the run. |
| `expected_samples_for_pair` | `dict[str, list[str]]` | Mapping of pair_id → sample_ids dispatched. |
| `manifest_purpose` | `Optional[str]` | `None`, `"resume_missing_cells"`, or `"prior_stage_survivor_cells"`. |

## Outputs

| Output | Type | Description |
|---|---|---|
| `manifest` | `StageManifest` | Parsed V0.2.1 stage manifest. |
| `surviving_pairs` | `list[str]` | Pair IDs that passed the barrier. |
| `failed_pairs` | `list[dict]` | Dicts `{prompt_pair_id, failure_reason}` for failed pairs. |
| `local_copyback_dir` | `pathlib.Path` | Local directory populated by rsync. |

## Constraints

- **Stage-major dispatch only.** Never call `/stage/all` from V0.2.1 code.
- **Cell-keyed barrier is non-negotiable.** Always call `barrier_cell_keyed`
  before copy-back.
- **Corpus hash must echo.** Any hash mismatch raises `CorpusDriftDetected`
  and aborts the round.
- **system_prompt_text** is allowed only on Gemma pairs. Passing it to FLUX
  or Qwen raises `ValueError` before the POST.
- **`sample_manifest_path` requires `sample_manifest_path_purpose`.**  Omitting
  the purpose raises `ValueError` locally (and HTTP 400 from the controller).
- **No real network or SSH in tests.** Use `httpx.ASGITransport(app=mock_app)`
  for HTTP and inject `subprocess_run`/`popen`/`http_get` for rsync/tunnel.
- **Rate limiting.** This skill does not call the OpenAI API directly, but any
  downstream evaluation must route through `rate_limiter.py` (≤ 3 req/s).
- **SSH alias only.** Always use the `3h100` alias; never hard-code `root@10.217.219.2`.
- **Dry-run before real rsync.** Always perform a dry-run review before the
  real rsync when running from the deploy skill.

## Error types

| Exception | Module | When |
|---|---|---|
| `TunnelError` | `remote.tunnel` | Health check fails within timeout. |
| `StageLockBusy` | `remote.client` | Lock retries exhausted. |
| `StageTimeoutError` | `remote.client` | Request timeout retries exhausted. |
| `StageServerError` | `remote.client` | 5xx retries exhausted. |
| `CorpusDriftDetected` | `remote.stage_dispatcher` | Response corpus hash != request hash. |
| `BarrierViolation` | `remote.barrier` | Surviving-pair violation or lifecycle mismatch. |
| `LifecycleAssertionError` | `remote.lifecycle` | Lifecycle state mismatch or VRAM/RAM below threshold. |
| `ManifestValidationError` | `remote.manifest_validator` | Deeper manifest violation. |
| `CopybackError` | `remote.copyback` | rsync failure or missing sentinel file. |

## Canonical imports

```python
from system_prompt_retrieval_agent.remote import (
    RemoteControllerClient,
    open_tunnel,
    assert_all_workers_done,
    validate_manifest,
    rsync_copyback,
    BarrierViolation,
    CopybackError,
    MANIFEST_VALIDATION_ERRORS,
)
from system_prompt_retrieval_agent.remote.stage_dispatcher import (
    StageDispatcher,
    CorpusDriftDetected,
)
from system_prompt_retrieval_agent.remote.partition import partition_stage_pairs
from system_prompt_retrieval_agent.remote.barrier import barrier_cell_keyed
from system_prompt_retrieval_agent.remote.lifecycle import (
    assert_lifecycle_ok,
    LifecycleAssertionError,
)
from system_prompt_retrieval_agent.remote.resume import (
    build_missing_manifest,
    build_survivor_manifest,
    build_survivor_manifest_with_samples,
)
from system_prompt_retrieval_agent.remote.manifest_validator import (
    validate_stage_manifest,
)
from system_prompt_retrieval_agent.schemas import (
    GemmaStageRequest,
    FluxStageRequest,
    QwenStageRequest,
    StageManifest,
    PromptPairRequest,
    GemmaUserPrompt,
)
```
