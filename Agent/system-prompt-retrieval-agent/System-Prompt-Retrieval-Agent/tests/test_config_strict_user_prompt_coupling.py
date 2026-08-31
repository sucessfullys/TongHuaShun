"""S09.06a — Config-coupling assertion for strict-mode user-prompt rule.

Plan §6.4: under workflow.allow_partial_samples=False, the
gemma_user_prompts.min_surviving_user_prompts_per_pair knob is
silently overridden to enabled-count. The agent_loop helper raises
StrictModeUserPromptCouplingWarning so callers can log + acknowledge.
"""
from __future__ import annotations

import pytest

from system_prompt_retrieval_agent.agent_loop_helpers import (
    StrictModeUserPromptCouplingWarning,
    assert_config_strict_user_prompt_coupling,
)
from system_prompt_retrieval_agent.config import (
    GemmaUserPromptEntry,
    GemmaUserPromptsConfig,
    ScoringConfig,
    WorkflowConfig,
)


class _FakeCfg:
    def __init__(self, *, allow_partial: bool, library_size: int, knob: int):
        self.workflow = WorkflowConfig(allow_partial_samples=allow_partial)
        self.gemma_user_prompts = GemmaUserPromptsConfig(
            library=[
                GemmaUserPromptEntry(
                    user_prompt_id=f"zh_{i:03d}",
                    language="zh",
                    text="x",
                    enabled=True,
                )
                for i in range(library_size)
            ],
            min_surviving_user_prompts_per_pair=knob,
        )
        self.scoring = ScoringConfig()


def test_strict_mode_with_knob_below_enabled_raises_warning():
    """Strict mode + knob < enabled-count → StrictModeUserPromptCouplingWarning."""
    cfg = _FakeCfg(allow_partial=False, library_size=4, knob=1)
    with pytest.raises(StrictModeUserPromptCouplingWarning):
        assert_config_strict_user_prompt_coupling(cfg)


def test_strict_mode_with_knob_equal_enabled_no_raise():
    cfg = _FakeCfg(allow_partial=False, library_size=4, knob=4)
    # Equal: no silent weakening; should not raise.
    assert_config_strict_user_prompt_coupling(cfg)


def test_allow_partial_with_knob_below_enabled_no_raise():
    """When allow_partial=True the knob is honored; no coupling violation."""
    cfg = _FakeCfg(allow_partial=True, library_size=4, knob=1)
    assert_config_strict_user_prompt_coupling(cfg)


def test_strict_mode_with_no_enabled_user_prompts_no_raise():
    cfg = _FakeCfg(allow_partial=False, library_size=0, knob=1)
    assert_config_strict_user_prompt_coupling(cfg)
