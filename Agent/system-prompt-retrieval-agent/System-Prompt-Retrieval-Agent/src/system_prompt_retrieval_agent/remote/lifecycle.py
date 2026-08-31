"""Lifecycle state helpers for V0.2.1 stage manifests.

Plan reference: §6.4 rule #5, §6.6, S04.10.
"""
from __future__ import annotations

from system_prompt_retrieval_agent.schemas import StageManifest


class LifecycleAssertionError(RuntimeError):
    """Raised when the post-stage lifecycle state does not match the expected mode.

    Attributes
    ----------
    code:
        Short identifier for the violation type.
    detail:
        Human-readable description.
    """

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"[{code}] {detail}" if detail else code)
        self.code = code
        self.detail = detail


def assert_lifecycle_ok(
    manifest: StageManifest,
    expected_mode: str,
    *,
    post_unload_vram_free_min_gib: float = 70.0,
    post_stage_host_ram_free_gib: float = 64.0,
) -> None:
    """Assert the post-stage lifecycle state in *manifest* matches *expected_mode*.

    Parameters
    ----------
    manifest:
        The stage manifest returned by the remote controller.
    expected_mode:
        ``"cold"`` or ``"warm"``.  In cold mode the manifest must report
        ``lifecycle_state_after ∈ {"disk_unloaded", "cold"}`` and per-GPU
        free VRAM must exceed *post_unload_vram_free_min_gib*.  In warm
        mode ``lifecycle_state_after ∈ {"gpu_unloaded_cpu_retained",
        "disk_unloaded"}`` is accepted and host-RAM headroom is checked
        instead.
    post_unload_vram_free_min_gib:
        Minimum acceptable per-GPU free VRAM (GiB) in cold mode.
    post_stage_host_ram_free_gib:
        Minimum acceptable host-RAM headroom (GiB) in warm mode.

    Raises
    ------
    LifecycleAssertionError
        - ``"lifecycle_state_missing"`` — ``manifest.lifecycle_state_after``
          is ``None`` and cannot be checked.
        - ``"lifecycle_mode_mismatch"`` — state does not match the expected
          mode.
        - ``"vram_leak_detected"`` — cold mode; any GPU reports free VRAM
          below the threshold.
        - ``"host_ram_insufficient"`` — warm mode; host-RAM headroom is
          below the threshold.
    """
    state = manifest.lifecycle_state_after

    if state is None:
        raise LifecycleAssertionError(
            "lifecycle_state_missing",
            "manifest.lifecycle_state_after is None; cannot verify lifecycle",
        )

    if expected_mode == "cold":
        if state not in ("disk_unloaded", "cold"):
            raise LifecycleAssertionError(
                "lifecycle_mode_mismatch",
                f"expected disk_unloaded (cold mode) but got '{state}'",
            )
        # VRAM check.
        if manifest.vram_free_gib is not None:
            for gib in manifest.vram_free_gib:
                if gib < post_unload_vram_free_min_gib:
                    raise LifecycleAssertionError(
                        "vram_leak_detected",
                        f"GPU reports only {gib:.1f} GiB free "
                        f"(threshold {post_unload_vram_free_min_gib:.1f} GiB)",
                    )

    elif expected_mode == "warm":
        if state not in ("gpu_unloaded_cpu_retained", "disk_unloaded"):
            raise LifecycleAssertionError(
                "lifecycle_mode_mismatch",
                f"expected gpu_unloaded_cpu_retained or disk_unloaded (warm mode) "
                f"but got '{state}'",
            )
        # Host RAM headroom check (only relevant when CPU-retained).
        if state == "gpu_unloaded_cpu_retained":
            if manifest.host_ram_free_gib is not None:
                if manifest.host_ram_free_gib < post_stage_host_ram_free_gib:
                    raise LifecycleAssertionError(
                        "host_ram_insufficient",
                        f"host RAM free {manifest.host_ram_free_gib:.1f} GiB "
                        f"< required {post_stage_host_ram_free_gib:.1f} GiB",
                    )

    else:
        raise LifecycleAssertionError(
            "lifecycle_mode_mismatch",
            f"unknown expected_mode '{expected_mode}'; must be 'cold' or 'warm'",
        )
