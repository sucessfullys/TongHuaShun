"""Tests for the Remote Client module (S04.09).

All tests run completely offline:
  - HTTP calls use ``httpx.ASGITransport`` backed by ``mock_app``.
  - rsync calls use an injected ``subprocess_run`` stub.
  - SSH tunnel calls use an injected ``popen`` stub.
  - No real ports, no real SSH, no real rsync.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from tests.mock_controller import mock_app, reset_counts, call_counts
from system_prompt_retrieval_agent.remote import (
    BarrierViolation,
    CopybackError,
    MANIFEST_VALIDATION_ERRORS,
    ManifestValidationError,
    RemoteControllerClient,
    assert_all_workers_done,
    rsync_copyback,
    validate_manifest,
)
from system_prompt_retrieval_agent.remote.tunnel import TunnelError, open_tunnel
from system_prompt_retrieval_agent.remote.stage_dispatcher import (
    CorpusDriftDetected,
    StageDispatcher,
)
from system_prompt_retrieval_agent.remote.manifest_validator import validate_stage_manifest
from system_prompt_retrieval_agent.schemas import (
    FluxStageRequest,
    GemmaStageRequest,
    GemmaUserPrompt,
    ManifestEntry,
    PerPairManifest,
    PerUserPromptManifest,
    PromptPairRequest,
    QwenStageRequest,
    RemoteManifest,
    RemoteStageRequest,
    StageManifest,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_URL = "http://test"


def _transport() -> httpx.ASGITransport:
    return httpx.ASGITransport(app=mock_app)


def _make_client(**kwargs) -> RemoteControllerClient:
    """Create a client backed by the mock ASGI app."""
    http_client = httpx.AsyncClient(transport=_transport(), base_url=BASE_URL)
    return RemoteControllerClient(BASE_URL, http_client=http_client, **kwargs)


def _make_req(run_id: str = "run-001", **kwargs) -> RemoteStageRequest:
    defaults: dict[str, Any] = dict(
        run_id=run_id,
        round_id=1,
        prompt_pair_id="pp-1",
        system_prompt_id="sp-1",
        system_prompt_text="A great photo",
        dataset_root="/data",
        output_root="/out",
    )
    defaults.update(kwargs)
    return RemoteStageRequest(**defaults)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_returns_valid_manifest():
    reset_counts()
    client = _make_client()
    req = _make_req(run_id="run-happy")
    resp = await client.run_prompt_pair_pipeline(req)

    assert resp.ok is True
    assert resp.manifest is not None
    m = resp.manifest
    assert m.ok == 2
    assert m.errors == 0
    assert m.total == 2
    # Barrier should pass without exception.
    assert_all_workers_done(m)


@pytest.mark.asyncio
async def test_health_endpoint():
    client = _make_client()
    data = await client.health()
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_status_endpoint():
    client = _make_client()
    data = await client.status()
    assert "current_stage" in data


# ---------------------------------------------------------------------------
# Lock contention retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_busy_retries_then_succeeds():
    """Lock-busy responses must trigger retries and eventually succeed."""
    reset_counts()

    # A counter that returns 400/locked for the first 2 attempts, then 200.
    attempt = {"n": 0}
    entries = [
        {"sample_id": "s1", "status": "ok", "output_path": "out/s1.jpg", "error": None},
        {"sample_id": "s2", "status": "ok", "output_path": "out/s2.jpg", "error": None},
    ]
    happy_body = {
        "ok": True,
        "stage": "all",
        "message": "",
        "run_id": "run-lock",
        "manifest": {
            "stage": "all",
            "run_id": "run-lock",
            "ok": 2,
            "errors": 0,
            "total": 2,
            "entries": entries,
            "workers": [{"id": 0, "status": "done"}, {"id": 1, "status": "done"}],
            "vram_free_gib": [72.0, 73.0],
        },
    }

    class _FakeTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):  # type: ignore[override]
            attempt["n"] += 1
            if attempt["n"] <= 2:
                return httpx.Response(400, json={"detail": "locked"})
            return httpx.Response(200, json=happy_body)

    http_client = httpx.AsyncClient(transport=_FakeTransport(), base_url=BASE_URL)
    client = RemoteControllerClient(
        BASE_URL,
        http_client=http_client,
        lock_contention_retry_max=3,
        lock_contention_wait_s=0.01,
    )
    req = _make_req(run_id="run-lock")
    resp = await client.run_prompt_pair_pipeline(req)
    assert resp.ok is True
    assert attempt["n"] == 3  # 2 locked + 1 success


@pytest.mark.asyncio
async def test_lock_busy_exhausted_raises():
    """After exhausting retries, StageLockBusy must be raised."""
    from system_prompt_retrieval_agent.remote.client import StageLockBusy

    class _AlwaysLocked(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):  # type: ignore[override]
            return httpx.Response(400, json={"detail": "locked"})

    http_client = httpx.AsyncClient(transport=_AlwaysLocked(), base_url=BASE_URL)
    client = RemoteControllerClient(
        BASE_URL,
        http_client=http_client,
        lock_contention_retry_max=2,
        lock_contention_wait_s=0.01,
    )
    with pytest.raises(StageLockBusy):
        await client.run_prompt_pair_pipeline(_make_req())


# ---------------------------------------------------------------------------
# Partial / strict mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_strict_partial_raises_validation_error():
    """When allow_partial=False and manifest has errors, validate_manifest raises."""
    client = _make_client()
    req = _make_req(run_id="PARTIAL")
    resp = await client.run_prompt_pair_pipeline(req)

    # Client returns the response regardless; caller validates.
    assert resp.manifest is not None
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(resp.manifest, allow_partial=False)
    assert exc_info.value.code == "strict_has_errors"


@pytest.mark.asyncio
async def test_allow_partial_passes_validation():
    """allow_partial=True must not raise even when errors > 0."""
    client = _make_client()
    req = _make_req(run_id="PARTIAL")
    resp = await client.run_prompt_pair_pipeline(req)
    assert resp.manifest is not None
    # Should not raise.
    validate_manifest(resp.manifest, allow_partial=True)


# ---------------------------------------------------------------------------
# Barrier: count mismatch
# ---------------------------------------------------------------------------


def test_barrier_count_mismatch():
    m = RemoteManifest(stage="all", run_id="r1", ok=1, errors=0, total=2, entries=[])
    with pytest.raises(BarrierViolation) as exc_info:
        assert_all_workers_done(m)
    assert exc_info.value.code == "count_mismatch"


# ---------------------------------------------------------------------------
# Barrier: VRAM leak
# ---------------------------------------------------------------------------


def test_barrier_vram_leak():
    m = RemoteManifest(
        stage="all",
        run_id="r1",
        ok=2,
        errors=0,
        total=2,
        entries=[],
        vram_free_gib=[50.0, 60.0, 55.0],
    )
    with pytest.raises(BarrierViolation) as exc_info:
        assert_all_workers_done(m, post_unload_vram_free_min_gib=70.0)
    assert exc_info.value.code == "vram_leak_detected"


def test_barrier_vram_ok_skips_check():
    m = RemoteManifest(
        stage="all",
        run_id="r1",
        ok=2,
        errors=0,
        total=2,
        entries=[],
        vram_free_gib=[80.0, 85.0],
    )
    assert_all_workers_done(m, post_unload_vram_free_min_gib=70.0)  # No exception.


@pytest.mark.asyncio
async def test_vram_leak_integration():
    """End-to-end: VRAM_LEAK run_id triggers barrier violation."""
    client = _make_client()
    resp = await client.run_prompt_pair_pipeline(_make_req(run_id="VRAM_LEAK"))
    assert resp.manifest is not None
    with pytest.raises(BarrierViolation) as exc_info:
        assert_all_workers_done(resp.manifest, post_unload_vram_free_min_gib=70.0)
    assert exc_info.value.code == "vram_leak_detected"


# ---------------------------------------------------------------------------
# Barrier: strict_has_errors via assert_all_workers_done
# ---------------------------------------------------------------------------


def test_barrier_strict_has_errors():
    m = RemoteManifest(
        stage="all", run_id="r1", ok=1, errors=1, total=2,
        entries=[
            ManifestEntry(sample_id="s1", status="ok"),
            ManifestEntry(sample_id="s2", status="error", error="boom"),
        ],
    )
    with pytest.raises(BarrierViolation) as exc_info:
        assert_all_workers_done(m, allow_partial=False)
    assert exc_info.value.code == "strict_has_errors"


def test_barrier_allow_partial_passes():
    m = RemoteManifest(
        stage="all", run_id="r1", ok=1, errors=1, total=2,
        entries=[
            ManifestEntry(sample_id="s1", status="ok"),
            ManifestEntry(sample_id="s2", status="error", error="boom"),
        ],
    )
    assert_all_workers_done(m, allow_partial=True)  # No exception.


# ---------------------------------------------------------------------------
# Manifest validator: duplicate sample_id
# ---------------------------------------------------------------------------


def test_validate_manifest_duplicate_sample_id():
    m = RemoteManifest(
        stage="all",
        run_id="r1",
        ok=2,
        errors=0,
        total=2,
        entries=[
            ManifestEntry(sample_id="dup", status="ok"),
            ManifestEntry(sample_id="dup", status="ok"),
        ],
    )
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(m)
    assert exc_info.value.code == "duplicate_sample_id"


@pytest.mark.asyncio
async def test_duplicate_sample_id_integration():
    """End-to-end: DUPLICATE_ID run_id triggers validation error."""
    client = _make_client()
    resp = await client.run_prompt_pair_pipeline(_make_req(run_id="DUPLICATE_ID"))
    assert resp.manifest is not None
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(resp.manifest)
    assert exc_info.value.code == "duplicate_sample_id"


# ---------------------------------------------------------------------------
# Manifest validator: missing output file
# ---------------------------------------------------------------------------


def test_validate_manifest_missing_output_file(tmp_path: Path):
    m = RemoteManifest(
        stage="all",
        run_id="r1",
        ok=1,
        errors=0,
        total=1,
        entries=[
            ManifestEntry(sample_id="s1", status="ok", output_path="out/s1.jpg"),
        ],
    )
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(
            m,
            local_output_root=tmp_path,
            verify_existing_files=True,
        )
    assert exc_info.value.code == "missing_output_file"


def test_validate_manifest_file_exists(tmp_path: Path):
    output_file = tmp_path / "out" / "s1.jpg"
    output_file.parent.mkdir(parents=True)
    output_file.touch()

    m = RemoteManifest(
        stage="all",
        run_id="r1",
        ok=1,
        errors=0,
        total=1,
        entries=[
            ManifestEntry(sample_id="s1", status="ok", output_path="out/s1.jpg"),
        ],
    )
    validate_manifest(m, local_output_root=tmp_path, verify_existing_files=True)


# ---------------------------------------------------------------------------
# Tunnel tests
# ---------------------------------------------------------------------------


def test_open_tunnel_healthy():
    """Tunnel contextmanager yields when health check returns 200."""
    fake_proc = MagicMock()
    fake_proc.terminate = MagicMock()
    fake_proc.wait = MagicMock(return_value=0)

    calls = {"popen": [], "http_get": []}

    def fake_popen(cmd, start_new_session=False):
        calls["popen"].append(cmd)
        return fake_proc

    def fake_http_get(url, timeout=None):
        calls["http_get"].append(url)
        resp = MagicMock()
        resp.status_code = 200
        return resp

    with open_tunnel(
        alias="test-host",
        local_port=19999,
        remote_port=17700,
        health_timeout_s=5.0,
        popen=fake_popen,
        sleep=lambda _: None,
        http_get=fake_http_get,
    ):
        pass  # Should not raise.

    assert len(calls["popen"]) == 1
    assert len(calls["http_get"]) >= 1
    fake_proc.terminate.assert_called_once()


def test_open_tunnel_always_503_raises():
    """Tunnel raises TunnelError when health check never returns 200."""
    fake_proc = MagicMock()
    fake_proc.terminate = MagicMock()
    fake_proc.wait = MagicMock(return_value=0)

    def fake_popen(cmd, start_new_session=False):
        return fake_proc

    def fake_http_get(url, timeout=None):
        resp = MagicMock()
        resp.status_code = 503
        return resp

    # Use a very short timeout so the test is fast.
    with pytest.raises(TunnelError):
        with open_tunnel(
            alias="test-host",
            local_port=19999,
            remote_port=17700,
            health_timeout_s=0.05,  # tiny timeout
            popen=fake_popen,
            sleep=lambda _: None,
            http_get=fake_http_get,
        ):
            pass


def test_open_tunnel_exception_in_http_get_retries():
    """When http_get raises, tunnel keeps polling until timeout."""
    fake_proc = MagicMock()
    fake_proc.terminate = MagicMock()
    fake_proc.wait = MagicMock(return_value=0)

    def fake_popen(cmd, start_new_session=False):
        return fake_proc

    def fake_http_get(url, timeout=None):
        raise ConnectionRefusedError("not ready")

    with pytest.raises(TunnelError):
        with open_tunnel(
            alias="test-host",
            local_port=19999,
            remote_port=17700,
            health_timeout_s=0.05,
            popen=fake_popen,
            sleep=lambda _: None,
            http_get=fake_http_get,
        ):
            pass


# ---------------------------------------------------------------------------
# rsync copyback tests
# ---------------------------------------------------------------------------


def test_rsync_copyback_success(tmp_path: Path):
    """rsync_copyback passes when rsync succeeds and required file exists."""
    required_rel = "stage3_qwen/stage_summary.json"
    required_path = tmp_path / required_rel
    required_path.parent.mkdir(parents=True, exist_ok=True)

    call_log: list[list[str]] = []

    def fake_run(cmd, capture_output=False):
        call_log.append(cmd)
        # Create the required file on first call to simulate successful rsync.
        required_path.touch()
        return SimpleNamespace(returncode=0, stderr=b"")

    rsync_copyback(
        remote_path="/mnt/remote/output",
        local_path=tmp_path,
        ssh_alias="test-alias",
        required_file=required_rel,
        subprocess_run=fake_run,
    )

    assert len(call_log) == 1
    assert "rsync" in call_log[0]


def test_rsync_copyback_retries_on_missing_file(tmp_path: Path):
    """rsync_copyback retries once when required file is absent after first rsync."""
    required_rel = "stage3_qwen/stage_summary.json"
    required_path = tmp_path / required_rel
    required_path.parent.mkdir(parents=True, exist_ok=True)

    call_log: list[list[str]] = []

    def fake_run(cmd, capture_output=False):
        call_log.append(cmd)
        if len(call_log) == 1:
            # First call: rsync succeeds but don't create the file.
            return SimpleNamespace(returncode=0, stderr=b"")
        # Second call: create the file.
        required_path.touch()
        return SimpleNamespace(returncode=0, stderr=b"")

    rsync_copyback(
        remote_path="/mnt/remote/output",
        local_path=tmp_path,
        ssh_alias="test-alias",
        required_file=required_rel,
        max_retries=1,
        subprocess_run=fake_run,
    )

    assert len(call_log) == 2  # retried once


def test_rsync_copyback_second_miss_raises(tmp_path: Path):
    """rsync_copyback raises CopybackError when file still absent after retry."""
    required_rel = "stage3_qwen/stage_summary.json"
    (tmp_path / "stage3_qwen").mkdir(parents=True, exist_ok=True)

    def fake_run(cmd, capture_output=False):
        # Never create the required file.
        return SimpleNamespace(returncode=0, stderr=b"")

    with pytest.raises(CopybackError):
        rsync_copyback(
            remote_path="/mnt/remote/output",
            local_path=tmp_path,
            ssh_alias="test-alias",
            required_file=required_rel,
            max_retries=1,
            subprocess_run=fake_run,
        )


def test_rsync_copyback_non_zero_exit_raises(tmp_path: Path):
    """rsync_copyback raises CopybackError immediately on non-zero exit code."""

    def fake_run(cmd, capture_output=False):
        return SimpleNamespace(returncode=1, stderr=b"permission denied")

    with pytest.raises(CopybackError):
        rsync_copyback(
            remote_path="/mnt/remote/output",
            local_path=tmp_path,
            ssh_alias="test-alias",
            subprocess_run=fake_run,
        )


# ---------------------------------------------------------------------------
# MANIFEST_VALIDATION_ERRORS list completeness
# ---------------------------------------------------------------------------


def test_manifest_validation_errors_list():
    expected = {
        "count_mismatch",
        "strict_has_errors",
        "missing_output_file",
        "duplicate_sample_id",
        "worker_not_done",
    }
    assert set(MANIFEST_VALIDATION_ERRORS) == expected


# ---------------------------------------------------------------------------
# Worker not done
# ---------------------------------------------------------------------------


def test_validate_manifest_worker_not_done():
    m = RemoteManifest(
        stage="all",
        run_id="r1",
        ok=1,
        errors=0,
        total=1,
        entries=[ManifestEntry(sample_id="s1", status="ok")],
        workers=[{"id": 0, "status": "running"}],  # non-terminal
    )
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_manifest(m, require_all_gpu_workers=True)
    assert exc_info.value.code == "worker_not_done"


def test_validate_manifest_worker_done_ok():
    for terminal in ("ok", "error", "done"):
        m = RemoteManifest(
            stage="all",
            run_id="r1",
            ok=1,
            errors=0,
            total=1,
            entries=[ManifestEntry(sample_id="s1", status="ok")],
            workers=[{"id": 0, "status": terminal}],
        )
        validate_manifest(m, require_all_gpu_workers=True)  # Should not raise.


# ---------------------------------------------------------------------------
# V0.2.1 dispatch helpers
# ---------------------------------------------------------------------------

_CORPUS_HASH = "a" * 64
_USER_PROMPTS = [
    GemmaUserPrompt(user_prompt_id="zh_001", language="zh", text="穿衣", enabled=True),
    GemmaUserPrompt(user_prompt_id="en_001", language="en", text="wear clothes", enabled=True),
]
_PAIR_A = PromptPairRequest(
    prompt_pair_id="pair_A",
    system_prompt_id="sp_A",
    system_prompt_text="A great prompt",
)
_PAIR_A_NO_TEXT = PromptPairRequest(
    prompt_pair_id="pair_A",
    system_prompt_id="sp_A",
)


def _make_gemma_req(**kwargs) -> GemmaStageRequest:
    defaults = dict(
        run_id="V021_HAPPY",
        round_id=1,
        prompt_pairs=[_PAIR_A],
        user_prompts=_USER_PROMPTS,
        user_prompt_corpus_hash=_CORPUS_HASH,
        dataset_root="/data",
        output_root="/out",
    )
    defaults.update(kwargs)
    return GemmaStageRequest(**defaults)


def _make_flux_req(**kwargs) -> FluxStageRequest:
    defaults = dict(
        run_id="V021_HAPPY",
        round_id=1,
        prompt_pairs=[_PAIR_A_NO_TEXT],
        user_prompts=_USER_PROMPTS,
        user_prompt_corpus_hash=_CORPUS_HASH,
        dataset_root="/data",
        output_root="/out",
    )
    defaults.update(kwargs)
    return FluxStageRequest(**defaults)


def _make_qwen_req(**kwargs) -> QwenStageRequest:
    defaults = dict(
        run_id="V021_HAPPY",
        round_id=1,
        prompt_pairs=[_PAIR_A_NO_TEXT],
        user_prompts=_USER_PROMPTS,
        user_prompt_corpus_hash=_CORPUS_HASH,
        dataset_root="/data",
        output_root="/out",
    )
    defaults.update(kwargs)
    return QwenStageRequest(**defaults)


def _make_dispatcher() -> StageDispatcher:
    http_client = httpx.AsyncClient(transport=_transport(), base_url=BASE_URL)
    return StageDispatcher(http_client, base_url=BASE_URL)


# ---------------------------------------------------------------------------
# V0.2.1 StageDispatcher happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_gemma_happy_path():
    """dispatch_gemma returns a StageManifest with surviving_pairs populated."""
    dispatcher = _make_dispatcher()
    req = _make_gemma_req()
    manifest = await dispatcher.dispatch_gemma(req)

    assert isinstance(manifest, StageManifest)
    assert manifest.stage == "gemma"
    assert len(manifest.surviving_pairs) >= 1 or len(manifest.pairs) >= 1


@pytest.mark.asyncio
async def test_dispatcher_flux_happy_path():
    """dispatch_flux returns StageManifest; no system_prompt_text on pairs."""
    dispatcher = _make_dispatcher()
    req = _make_flux_req()
    manifest = await dispatcher.dispatch_flux(req)

    assert isinstance(manifest, StageManifest)


@pytest.mark.asyncio
async def test_dispatcher_qwen_happy_path():
    """dispatch_qwen returns StageManifest."""
    dispatcher = _make_dispatcher()
    req = _make_qwen_req()
    manifest = await dispatcher.dispatch_qwen(req)

    assert isinstance(manifest, StageManifest)


# ---------------------------------------------------------------------------
# V0.2.1 FLUX / Qwen must not carry system_prompt_text
# ---------------------------------------------------------------------------


def test_flux_request_rejects_system_prompt_text():
    """FluxStageRequest raises ValueError on construction with system_prompt_text."""
    bad_pair = PromptPairRequest(
        prompt_pair_id="pair_A",
        system_prompt_id="sp_A",
        system_prompt_text="should be rejected",
    )
    with pytest.raises(ValueError):
        FluxStageRequest(
            run_id="r1",
            prompt_pairs=[bad_pair],
            user_prompts=_USER_PROMPTS,
            user_prompt_corpus_hash=_CORPUS_HASH,
            dataset_root="/data",
            output_root="/out",
        )


def test_qwen_request_rejects_system_prompt_text():
    """QwenStageRequest raises ValueError on construction with system_prompt_text."""
    bad_pair = PromptPairRequest(
        prompt_pair_id="pair_A",
        system_prompt_id="sp_A",
        system_prompt_text="should be rejected",
    )
    with pytest.raises(ValueError):
        QwenStageRequest(
            run_id="r1",
            prompt_pairs=[bad_pair],
            user_prompts=_USER_PROMPTS,
            user_prompt_corpus_hash=_CORPUS_HASH,
            dataset_root="/data",
            output_root="/out",
        )


# ---------------------------------------------------------------------------
# V0.2.1 corpus hash mismatch raises CorpusDriftDetected
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_corpus_hash_mismatch_raises():
    """When controller returns wrong corpus hash, CorpusDriftDetected is raised."""
    dispatcher = _make_dispatcher()
    req = _make_gemma_req(run_id="V021_CORPUS_MISMATCH")
    with pytest.raises(CorpusDriftDetected):
        await dispatcher.dispatch_gemma(req)


@pytest.mark.asyncio
async def test_client_post_stage_v021_corpus_hash_mismatch_raises():
    """post_stage_v021 also raises CorpusDriftDetected on hash mismatch."""
    client = _make_client()
    req = _make_gemma_req(run_id="V021_CORPUS_MISMATCH")
    with pytest.raises(CorpusDriftDetected):
        await client.post_stage_v021(req)


# ---------------------------------------------------------------------------
# V0.2.1 one-pair-failed scenario
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatcher_one_pair_failed_no_raise():
    """One pair in failed_pairs does not raise; surviving_pairs has N-1 pairs."""
    dispatcher = _make_dispatcher()
    req = _make_gemma_req(
        run_id="V021_ONE_PAIR_FAILED",
        prompt_pairs=[_PAIR_A_NO_TEXT, PromptPairRequest(prompt_pair_id="pair_B")],
    )
    manifest = await dispatcher.dispatch_gemma(req)

    assert "pair_B" in [fp.get("prompt_pair_id") for fp in manifest.failed_pairs]
    assert "pair_A" in manifest.surviving_pairs


# ---------------------------------------------------------------------------
# V0.2.1 sample_manifest_path without purpose → HTTP 400
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sample_manifest_path_without_purpose_raises_client_side():
    """Pydantic raises ValueError when sample_manifest_path set without purpose."""
    with pytest.raises(ValueError, match="missing_manifest_purpose"):
        GemmaStageRequest(
            run_id="r1",
            prompt_pairs=[_PAIR_A],
            user_prompts=_USER_PROMPTS,
            user_prompt_corpus_hash=_CORPUS_HASH,
            dataset_root="/data",
            output_root="/out",
            sample_manifest_path="/some/path.jsonl",
            # sample_manifest_path_purpose deliberately omitted
        )


@pytest.mark.asyncio
async def test_sample_manifest_path_without_purpose_server_400():
    """Mock controller returns HTTP 400 for V021_NO_PURPOSE run_id."""
    dispatcher = _make_dispatcher()
    req = _make_gemma_req(run_id="V021_NO_PURPOSE")
    with pytest.raises(Exception):  # httpx raises on non-2xx
        await dispatcher.dispatch_gemma(req)


# ---------------------------------------------------------------------------
# V0.2.1 post_stage_v021 happy path via RemoteControllerClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_post_stage_v021_gemma_happy():
    """post_stage_v021 with GemmaStageRequest returns a StageManifest."""
    client = _make_client()
    req = _make_gemma_req()
    manifest = await client.post_stage_v021(req)

    assert isinstance(manifest, StageManifest)


@pytest.mark.asyncio
async def test_client_post_stage_v021_flux_happy():
    """post_stage_v021 with FluxStageRequest returns a StageManifest."""
    client = _make_client()
    req = _make_flux_req()
    manifest = await client.post_stage_v021(req)

    assert isinstance(manifest, StageManifest)


def test_client_post_stage_v021_flux_rejects_system_prompt_text_at_construction():
    """FluxStageRequest constructor raises ValueError when any pair has system_prompt_text.

    The pydantic validator fires at construction time, preventing bad requests
    from being created in the first place.  This is the primary defense — the
    dispatcher's redundant check is a belt-and-suspenders guard for requests
    created via model_construct or other bypass paths.
    """
    bad_pair = PromptPairRequest(
        prompt_pair_id="pair_A",
        system_prompt_id="sp_A",
        system_prompt_text="should be rejected at schema level",
    )
    with pytest.raises(ValueError):
        FluxStageRequest(
            run_id="r1",
            prompt_pairs=[bad_pair],
            user_prompts=_USER_PROMPTS,
            user_prompt_corpus_hash=_CORPUS_HASH,
            dataset_root="/data",
            output_root="/out",
        )


# ---------------------------------------------------------------------------
# V0.2.1 validate_stage_manifest
# ---------------------------------------------------------------------------


def test_validate_stage_manifest_happy():
    """validate_stage_manifest passes on a well-formed manifest."""
    m = StageManifest(
        stage="gemma",
        run_id="r1",
        pairs={
            "pair_A": PerPairManifest(
                prompt_pair_id="pair_A",
                ok=2,
                errors=0,
                total=2,
                per_user_prompt={
                    "zh_001": PerUserPromptManifest(ok=1, errors=0, total=1),
                    "en_001": PerUserPromptManifest(ok=1, errors=0, total=1),
                },
            ),
        },
        surviving_pairs=["pair_A"],
        failed_pairs=[],
        user_prompt_corpus_hash="a" * 64,
        lifecycle_state_after="disk_unloaded",
    )
    validate_stage_manifest(m, allow_partial=False)  # No exception.


def test_validate_stage_manifest_invalid_corpus_hash():
    """validate_stage_manifest raises on malformed corpus hash."""
    from system_prompt_retrieval_agent.remote.manifest_validator import ManifestValidationError
    m = StageManifest(
        stage="gemma",
        run_id="r1",
        pairs={
            "pair_A": PerPairManifest(
                prompt_pair_id="pair_A",
                ok=1,
                errors=0,
                total=1,
                per_user_prompt={"zh_001": PerUserPromptManifest(ok=1, errors=0, total=1)},
            ),
        },
        surviving_pairs=["pair_A"],
        failed_pairs=[],
        user_prompt_corpus_hash="not_hex",
    )
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_stage_manifest(m)
    assert exc_info.value.code == "invalid_corpus_hash_format"


def test_validate_stage_manifest_pair_in_both_surviving_and_failed():
    """validate_stage_manifest raises when pair is in both surviving and failed."""
    from system_prompt_retrieval_agent.remote.manifest_validator import ManifestValidationError
    m = StageManifest(
        stage="gemma",
        run_id="r1",
        pairs={
            "pair_A": PerPairManifest(
                prompt_pair_id="pair_A",
                ok=1,
                errors=0,
                total=1,
                per_user_prompt={"zh_001": PerUserPromptManifest(ok=1, errors=0, total=1)},
            ),
        },
        surviving_pairs=["pair_A"],
        failed_pairs=[{"prompt_pair_id": "pair_A", "failure_reason": "worker_crash"}],
        user_prompt_corpus_hash="a" * 64,
    )
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_stage_manifest(m)
    assert exc_info.value.code == "pair_in_both_surviving_and_failed"
