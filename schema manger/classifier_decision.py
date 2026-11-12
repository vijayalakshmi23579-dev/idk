"""
classifier_decision.py — safe-tool allowlist for auto-mode classifier skip.

Extracted from src/utils/permissions/classifierDecision.ts. Tools on this
list are always allowed in auto mode without hitting the classifier API.
The rationale: read-only tools cannot cause harm, so classifying them is
wasted latency + cost.

Used by permissions.py's _apply_auto_mode_classifier() to skip the LLM call.
"""
from __future__ import annotations


# Read-only tools + tools whose side effects are already permission-gated.
AUTO_MODE_ALLOWLIST: frozenset[str] = frozenset({
    "Read",
    "Glob",
    "Grep",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "TaskList",
    "TaskGet",
    "TaskOutput",
    "MemorySearch",
    "MemoryExpand",
})


def is_auto_mode_allowlisted_tool(tool_name: str) -> bool:
    """True → skip the classifier, allow immediately."""
    return tool_name in AUTO_MODE_ALLOWLIST