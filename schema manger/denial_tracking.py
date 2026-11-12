"""
denial_tracking.py — consecutive and total denial tracking for auto mode.

Extracted from src/utils/permissions/denialTracking.ts.

In auto mode, the classifier blocks actions it deems unsafe. But if the
classifier is overly cautious, it could block everything — that's bad UX.
So we track denials and force fallback to human prompting when:

  - 3 CONSECUTIVE denials (classifier is probably wrong)
  - 20 TOTAL denials (session-wide caution threshold)

Once either limit is hit, subsequent 'ask' results fall back to prompting
the human directly, bypassing the classifier.

State lives on AppState.denial_tracking for the main thread, and on
ToolUseContext.local_denial_tracking for async subagents (whose setAppState
is a no-op, so the global counter would never accumulate).
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Limits ────────────────────────────────────────────────────────────────

DENIAL_LIMITS = {
    "max_consecutive": 3,
    "max_total": 20,
}


# ── State ─────────────────────────────────────────────────────────────────

@dataclass
class DenialTrackingState:
    """Counters for classifier denials. Mutable for in-place subagent updates."""
    consecutive_denials: int = 0
    total_denials: int = 0


# ── Factories ─────────────────────────────────────────────────────────────

def create_denial_tracking_state() -> DenialTrackingState:
    """Fresh state with zero denials."""
    return DenialTrackingState(consecutive_denials=0, total_denials=0)


# ── Mutators (immutable-style; caller commits) ───────────────────────────

def record_denial(state: DenialTrackingState) -> DenialTrackingState:
    """Increment both counters. Returns a new state."""
    return DenialTrackingState(
        consecutive_denials=state.consecutive_denials + 1,
        total_denials=state.total_denials + 1,
    )


def record_success(state: DenialTrackingState) -> DenialTrackingState:
    """Reset consecutive. Keep total. Returns SAME INSTANCE if consecutive is 0.

    The identity check lets callers skip a store update when there's nothing
    to change (matches the Object.is check in the TS store).
    """
    if state.consecutive_denials == 0:
        return state
    return DenialTrackingState(
        consecutive_denials=0,
        total_denials=state.total_denials,
    )


def reset_denials(state: DenialTrackingState) -> DenialTrackingState:
    """Reset BOTH counters. Called when total limit hits and we fall back."""
    return DenialTrackingState(consecutive_denials=0, total_denials=0)


# ── Queries ───────────────────────────────────────────────────────────────

def should_fallback_to_prompting(state: DenialTrackingState) -> bool:
    """True when either limit is exceeded — caller should ask the human."""
    return (
        state.consecutive_denials >= DENIAL_LIMITS["max_consecutive"]
        or state.total_denials >= DENIAL_LIMITS["max_total"]
    )


def hit_total_limit(state: DenialTrackingState) -> bool:
    """True when the session-wide total limit is hit (triggers reset)."""
    return state.total_denials >= DENIAL_LIMITS["max_total"]


# ── Subagent in-place mutation ────────────────────────────────────────────

def apply_in_place(target: DenialTrackingState, source: DenialTrackingState) -> None:
    """
    Mutate `target` to match `source`. For subagents whose set_app_state
    is a no-op, this is how the local counter gets updated.

    Mirrors Object.assign(context.localDenialTracking, newState) in the TS
    code. Do NOT use for main-thread state — replace AppState.denial_tracking
    with a new instance there.
    """
    target.consecutive_denials = source.consecutive_denials
    target.total_denials = source.total_denials