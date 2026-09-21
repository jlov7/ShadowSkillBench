"""Frozen prompt for provider-native tool turns."""

from __future__ import annotations

from hashlib import sha256

NATIVE_TOOL_TURN_PROMPT_PROFILE = "SSB-OLLAMA-NATIVE-TOOLS-TURN-PROMPT1"
NATIVE_TOOL_TURN_PROMPT = (
    "# SSB-OLLAMA-NATIVE-TOOLS-TURN-PROMPT1\n\n"
    "Choose exactly one available native tool for this turn. Use prior tool results and the "
    "current observation. Do not repeat a successful read unless necessary. When the objective "
    "is satisfied, call finish_task exactly once. Stay within the visible remaining turn, tool, "
    "and token budgets."
)
NATIVE_TOOL_TURN_PROMPT_HASH = (
    "sha256:" + sha256(NATIVE_TOOL_TURN_PROMPT.encode("utf-8")).hexdigest()
)

__all__ = [
    "NATIVE_TOOL_TURN_PROMPT",
    "NATIVE_TOOL_TURN_PROMPT_HASH",
    "NATIVE_TOOL_TURN_PROMPT_PROFILE",
]
