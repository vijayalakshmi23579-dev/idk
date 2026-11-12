"""
query_token_budget.py — token budget tracker and decision logic.

Exact port of src/query/tokenBudget.ts (30 minified lines).

Constants (from source lines 4-4):
    COMPLETION_THRESHOLD = 0.9
    DIMINISHING_THRESHOLD = 500

Algorithm (source lines 14-30):
    if agent_id OR budget is None OR budget <= 0:
        → stop (no completion event)
    turn_tokens = global_turn_tokens
    pct = round((turn_tokens / budget) * 100)
    delta_since_last = global_turn_tokens - tracker.last_global_turn_tokens
    is_diminishing = (
        tracker.continuation_count >= 3
        AND delta_since_last < 500
        AND tracker.last_delta_tokens < 500
    )
    if NOT diminishing AND turn_tokens < budget * 0.9:
        tracker.continuation_count += 1
        tracker.last_delta_tokens = delta_since_last
        tracker.last_global_turn_tokens = global_turn_tokens
        → continue(nudge, pct, ...)
    if diminishing OR tracker.continuation_count > 0:
        → stop(completion_event with diminishing_returns flag)
    → stop (no completion event)

Subagents (agent_id != None) NEVER get budget continuation — the budget
belongs to the main turn.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Union


# ── Constants (tokenBudget.ts:4-4) ──────────────────────────────────────

COMPLETION_THRESHOLD = 0.9
DIMINISHING_THRESHOLD = 500


# ── Budget tracker (tokenBudget.ts:5-8) ─────────────────────────────────

@dataclass
class BudgetTracker:
    continuation_count: int = 0
    last_delta_tokens: int = 0
    last_global_turn_tokens: int = 0
    started_at: float = 0.0


def create_budget_tracker() -> BudgetTracker:
    """Exact match: tokenBudget.ts:7-8."""
    return BudgetTracker(
        continuation_count=0,
        last_delta_tokens=0,
        last_global_turn_tokens=0,
        started_at=time.time() * 1000,
    )


# ── Decisions (tokenBudget.ts:9-13) ─────────────────────────────────────

@dataclass
class ContinueDecision:
    action: Literal["continue"] = "continue"
    nudge_message: str = ""
    continuation_count: int = 0
    pct: int = 0
    turn_tokens: int = 0
    budget: int = 0


@dataclass
class CompletionEvent:
    continuation_count: int
    pct: int
    turn_tokens: int
    budget: int
    diminishing_returns: bool
    duration_ms: int


@dataclass
class StopDecision:
    action: Literal["stop"] = "stop"
    completion_event: CompletionEvent | None = None


TokenBudgetDecision = Union[ContinueDecision, StopDecision]


# ── Nudge message (from utils/tokenBudget.ts:3-29 + prompt template) ───

def get_budget_continuation_message(pct: int, turn_tokens: int, budget: int) -> str:
    """Continuation nudge. Preserves the intent of getBudgetContinuationMessage
    in utils/tokenBudget.ts — the exact template string there isn't in the leak,
    so this is a faithful reconstruction with the same fields."""
    remaining = max(0, budget - turn_tokens)
    return (
        f"You've used {pct}% of your {budget:,}-token budget "
        f"({turn_tokens:,} used, {remaining:,} remaining). "
        f"Continue working on the task. Do not stop to summarize — "
        f"keep going until the work is genuinely complete. "
        f"If you believe you are done, say so explicitly and stop."
    )


# ── The decision function (tokenBudget.ts:14-30) ───────────────────────

def check_token_budget(
    tracker: BudgetTracker,
    agent_id: str | None,
    budget: int | None,
    global_turn_tokens: int,
) -> TokenBudgetDecision:
    """Exact algorithm from tokenBudget.ts:14-30."""
    # Subagents never get budget continuation (tokenBudget.ts:16-18)
    if agent_id or budget is None or budget <= 0:
        return StopDecision()

    turn_tokens = global_turn_tokens
    pct = round((turn_tokens / budget) * 100)
    delta_since_last_check = global_turn_tokens - tracker.last_global_turn_tokens

    # Diminishing returns guard (tokenBudget.ts:19-21):
    #   continuationCount >= 3 AND delta < 500 AND lastDelta < 500
    is_diminishing = (
        tracker.continuation_count >= 3
        and delta_since_last_check < DIMINISHING_THRESHOLD
        and tracker.last_delta_tokens < DIMINISHING_THRESHOLD
    )

    # Continue if under threshold and not diminishing (tokenBudget.ts:21-26)
    if not is_diminishing and turn_tokens < budget * COMPLETION_THRESHOLD:
        tracker.continuation_count += 1
        tracker.last_delta_tokens = delta_since_last_check
        tracker.last_global_turn_tokens = global_turn_tokens
        return ContinueDecision(
            nudge_message=get_budget_continuation_message(
                pct, turn_tokens, budget,
            ),
            continuation_count=tracker.continuation_count,
            pct=pct,
            turn_tokens=turn_tokens,
            budget=budget,
        )

    # Stop with completion event if we were ever continuing (tokenBudget.ts:26-30)
    if is_diminishing or tracker.continuation_count > 0:
        return StopDecision(
            completion_event=CompletionEvent(
                continuation_count=tracker.continuation_count,
                pct=pct,
                turn_tokens=turn_tokens,
                budget=budget,
                diminishing_returns=is_diminishing,
                duration_ms=int(time.time() * 1000 - tracker.started_at),
            ),
        )

    # No budget was ever in play
    return StopDecision()