"""
query_transitions.py — Continue and Terminal discriminated unions.

The source transitions.ts is a STUB (interface with empty bodies):
    export interface Continue {}
    export interface Terminal {}

These shapes are INFERRED from every `continue` and `return` in query.ts
(1729 lines). Cross-checked against the state machine's documented behavior.

Every `continue` in queryLoop() carries `state.transition = {...}` with one
of 7 reasons. Every `return` carries one of 10 Terminal reasons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ── Continue — why the loop continued ───────────────────────────────────
# Source: every `continue` in query.ts sets state.transition to one of these.

ContinueReason = Literal[
    "next_turn",                     # Normal tool execution completed
    "collapse_drain_retry",          # 413, context collapse staged → drain
    "reactive_compact_retry",        # 413, no collapse → summarize
    "max_output_tokens_escalate",    # 8k cap hit → retry at 64k
    "max_output_tokens_recovery",    # 64k also hit → nudge (up to 3×)
    "token_budget_continuation",     # budget not done → nudge
    "stop_hook_blocking",            # stop hook returned blocking error
]


@dataclass
class Continue:
    reason: ContinueReason
    # Only populated for collapse_drain_retry
    committed: int = 0
    # Only populated for max_output_tokens_recovery (1, 2, 3)
    attempt: int = 0


# ── Terminal — why the loop ended ───────────────────────────────────────

TerminalReason = Literal[
    "completed",              # Model produced a text-only response
    "aborted_streaming",      # User pressed Ctrl+C during stream
    "aborted_tools",          # User pressed Ctrl+C during tools
    "blocking_limit",         # Hard token limit (auto-compact disabled)
    "prompt_too_long",        # 413 error, no recovery
    "image_error",            # Image too large, no recovery
    "model_error",            # API error, no recovery
    "max_turns",              # Turn count exceeded maxTurns
    "hook_stopped",           # PostToolUse hook prevented continuation
    "stop_hook_prevented",    # Stop hook prevented continuation
]


@dataclass
class Terminal:
    reason: TerminalReason
    # Only for model_error
    error: Exception | None = None
    # Only for max_turns
    turn_count: int | None = None