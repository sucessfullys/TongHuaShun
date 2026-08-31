"""prompt_generation — Prompt-Pair Generation subsystem.

Public API:
    generate_prompt_pairs  — Main entry point; returns (list[PromptPair], fallback_used).
    build_history_context  — Delegate to memory_manager.load_for_generation.
    PromptPairGenerator    — Low-level OpenAI-backed generator class.
"""
from system_prompt_retrieval_agent.prompt_generation.context_builder import build_history_context
from system_prompt_retrieval_agent.prompt_generation.generator import (
    PromptPairGenerator,
    generate_prompt_pairs,
)

__all__ = [
    "generate_prompt_pairs",
    "build_history_context",
    "PromptPairGenerator",
]
