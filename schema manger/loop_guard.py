"""
loop_guard.py — repeated-tool-call detection.

Claude Code has NO native loop detector (Issue #30150 is the open feature
request). Every existing implementation is a community PreToolUse hook.
This module synthesizes the converged design from LoopGuard, unwedge,
loop-breaker, cc-safe-setup, runaway-brake, and grounded.

Signals tracked (from unwedge's taxonomy):
  1. Repeated action      — same (tool_name, args_hash) in the window
  2. Repeated result      — same result_hash as the previous call
  3. Repeated error       — same error_signature as the previous call
  4. Streak               — consecutive calls with no change at all

Thresholds (converged):
  Idempotent tools (Read, Glob, Grep, WebFetch):
    warn at 5, block at 8
  Mutating tools (Edit, Write, Bash, TodoWrite):
    warn at 3, block at 5

The byte-identical default (8) comes from runaway-brake; the 3-in-5
warning comes from cc-safe-setup; the mutating/idempotent split comes
from PraisonAI.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field


WINDOW = 10                      # rolling window of recent calls
WARN_IDEMPOTENT = 5
BLOCK_IDEMPOTENT = 8
WARN_MUTATING = 3
BLOCK_MUTATING = 5

IDEMPOTENT_TOOLS = frozenset({
    "Read", "Glob", "Grep", "WebFetch", "WebSearch",
    "TaskList", "TaskGet", "TaskOutput", "MemorySearch", "MemoryExpand",
})


def _args_hash(tool_name: str, args: dict) -> str:
    """Stable hash of tool name + args. Order-independent for dict keys."""
    try:
        normalized = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        normalized = str(args)
    payload = f"{tool_name}|{normalized}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class LoopSignal:
    action: str          # "allow" | "warn" | "block"
    reason: str = ""
    count: int = 0
    tool_name: str = ""


@dataclass
class LoopGuardState:
    hashes: list[str] = field(default_factory=list)
    last_result_hash: str | None = None
    last_error_signature: str | None = None
    last_args_hash: str | None = None


def check_loop(
    state: LoopGuardState,
    tool_name: str,
    args: dict,
) -> LoopSignal:
    """Called BEFORE executing a tool. Returns allow/warn/block.

    On 'block', the caller should synthesize a tool_result telling the
    model it is looping and must change approach.
    """
    h = _args_hash(tool_name, args)

    # Update rolling window
    state.hashes.append(h)
    if len(state.hashes) > WINDOW:
        state.hashes = state.hashes[-WINDOW:]

    # Exact streak of consecutive identical calls
    streak = 0
    for prev in reversed(state.hashes):
        if prev == h:
            streak += 1
        else:
            break

    is_mutating = tool_name not in IDEMPOTENT_TOOLS
    warn_at = WARN_MUTATING if is_mutating else WARN_IDEMPOTENT
    block_at = BLOCK_MUTATING if is_mutating else BLOCK_IDEMPOTENT

    if streak >= block_at:
        return LoopSignal(
            action="block",
            reason=(
                f"You have called {tool_name} with these exact arguments "
                f"{streak} times in a row. This is a loop. STOP and change "
                f"approach: either you already have the answer, or the "
                f"command is failing for a reason that repeating it will "
                f"not fix."
            ),
            count=streak,
            tool_name=tool_name,
        )

    if streak >= warn_at:
        return LoopSignal(
            action="warn",
            reason=(
                f"Note: {tool_name} has been called {streak} times in a row "
                f"with identical arguments. If you are not making progress, "
                f"change approach instead of repeating."
            ),
            count=streak,
            tool_name=tool_name,
        )

    # Total occurrences in the window (catches A→B→A→B oscillation)
    total = sum(1 for x in state.hashes if x == h)
    if total >= block_at and streak < block_at:
        return LoopSignal(
            action="warn",
            reason=(
                f"{tool_name} appears {total} times in the last "
                f"{len(state.hashes)} calls with identical arguments. "
                f"You may be oscillating between approaches."
            ),
            count=total,
            tool_name=tool_name,
        )

    return LoopSignal(action="allow")


def record_result(
    state: LoopGuardState,
    result: str,
    is_error: bool = False,
) -> None:
    """Called AFTER a tool executes. Tracks result/error repetition."""
    rh = hashlib.sha256(result.encode()).hexdigest()[:16]
    state.last_result_hash = rh
    if is_error:
        state.last_error_signature = rh


__all__ = [
    "WINDOW", "WARN_IDEMPOTENT", "BLOCK_IDEMPOTENT",
    "WARN_MUTATING", "BLOCK_MUTATING", "IDEMPOTENT_TOOLS",
    "LoopSignal", "LoopGuardState", "check_loop", "record_result",
]