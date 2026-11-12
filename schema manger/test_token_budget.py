"""test_token_budget.py — every branch of check_token_budget."""
import time

from query_token_budget import (
    BudgetTracker, create_budget_tracker, check_token_budget,
    ContinueDecision, StopDecision, CompletionEvent,
    COMPLETION_THRESHOLD, DIMINISHING_THRESHOLD,
)


def main():
    # ── Constants match source ──
    assert COMPLETION_THRESHOLD == 0.9
    assert DIMINISHING_THRESHOLD == 500
    print("✅ Constants OK")

    # ── Fresh tracker ──
    t = create_budget_tracker()
    assert t.continuation_count == 0
    assert t.last_delta_tokens == 0
    assert t.last_global_turn_tokens == 0
    assert t.started_at > 0
    print("✅ Fresh tracker OK")

    # ── Subagent: never continues ──
    t = create_budget_tracker()
    d = check_token_budget(t, "agent-123", budget=100_000, global_turn_tokens=10_000)
    assert isinstance(d, StopDecision)
    assert d.completion_event is None
    assert t.continuation_count == 0
    print("✅ Subagent never continues OK")

    # ── No budget ──
    t = create_budget_tracker()
    d = check_token_budget(t, None, budget=None, global_turn_tokens=10_000)
    assert isinstance(d, StopDecision) and d.completion_event is None
    print("✅ None budget → stop OK")

    # ── Zero/negative budget ──
    t = create_budget_tracker()
    for b in (0, -100):
        d = check_token_budget(t, None, budget=b, global_turn_tokens=10_000)
        assert isinstance(d, StopDecision) and d.completion_event is None
    print("✅ Zero/negative budget → stop OK")

    # ── Under threshold → continue ──
    t = create_budget_tracker()
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=50_000)
    assert isinstance(d, ContinueDecision)
    assert d.pct == 50
    assert d.turn_tokens == 50_000
    assert d.budget == 100_000
    assert d.continuation_count == 1
    assert "50%" in d.nudge_message
    # Tracker was mutated
    assert t.continuation_count == 1
    assert t.last_global_turn_tokens == 50_000
    assert t.last_delta_tokens == 50_000
    print("✅ Under threshold → continue OK")

    # ── Continuation increments + tracks delta ──
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=60_000)
    assert isinstance(d, ContinueDecision)
    assert d.continuation_count == 2
    assert t.last_delta_tokens == 10_000   # 60_000 - 50_000
    assert t.last_global_turn_tokens == 60_000
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=70_000)
    assert isinstance(d, ContinueDecision)
    assert d.continuation_count == 3
    assert t.last_delta_tokens == 10_000
    print("✅ Continuation + delta tracking OK")

    # ── Over threshold → stop with completion event ──
    t = create_budget_tracker()
    t.continuation_count = 2
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=95_000)
    assert isinstance(d, StopDecision)
    assert d.completion_event is not None
    assert d.completion_event.pct == 95
    assert d.completion_event.continuation_count == 2
    assert d.completion_event.diminishing_returns is False
    print("✅ Over threshold → stop with event OK")

    # ── Diminishing returns: continuationCount >= 3 AND delta < 500 × 2 ──
    t = create_budget_tracker()
    t.continuation_count = 3
    t.last_delta_tokens = 100      # < 500 ✓
    t.last_global_turn_tokens = 50_000
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=50_300)  # delta=300 < 500
    assert isinstance(d, StopDecision)
    assert d.completion_event is not None
    assert d.completion_event.diminishing_returns is True
    assert d.completion_event.continuation_count == 3
    print("✅ Diminishing returns → stop OK")

    # ── Not diminishing if delta >= 500 ──
    t = create_budget_tracker()
    t.continuation_count = 3
    t.last_delta_tokens = 100
    t.last_global_turn_tokens = 50_000
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=50_600)  # delta=600 >= 500
    assert isinstance(d, ContinueDecision)  # not diminishing, under threshold
    assert d.continuation_count == 4
    print("✅ Delta >= 500 avoids diminishing OK")

    # ── Not diminishing if last_delta >= 500 ──
    t = create_budget_tracker()
    t.continuation_count = 3
    t.last_delta_tokens = 600      # >= 500 ✗
    t.last_global_turn_tokens = 50_000
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=50_100)
    assert isinstance(d, ContinueDecision)
    print("✅ lastDelta >= 500 avoids diminishing OK")

    # ── Stop with no event when continuation_count == 0 and over threshold ──
    t = create_budget_tracker()
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=95_000)
    assert isinstance(d, StopDecision)
    assert d.completion_event is None  # never continued → no event
    print("✅ No continuation + over threshold → stop with no event OK")

    # ── Realistic: 3 continuations then diminishing ──
    t = create_budget_tracker()
    # continue at 40k, 45k, 50k (deltas 40k, 5k, 5k — all >= 500, so not diminishing)
    for tokens in (40_000, 45_000, 50_000):
        d = check_token_budget(t, None, budget=100_000, global_turn_tokens=tokens)
        assert isinstance(d, ContinueDecision)
    assert t.continuation_count == 3
    # now delta = 200 < 500, lastDelta = 5_000 >= 500 → NOT diminishing yet
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=50_200)
    assert isinstance(d, ContinueDecision)
    assert t.continuation_count == 4
    assert t.last_delta_tokens == 200   # now lastDelta < 500
    # next delta = 100 < 500, lastDelta = 200 < 500, count = 4 >= 3 → diminishing
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=50_300)
    assert isinstance(d, StopDecision)
    assert d.completion_event.diminishing_returns is True
    print("✅ Realistic 5-turn diminishing sequence OK")

    # ── Duration is tracked ──
    t = create_budget_tracker()
    time.sleep(0.01)
    t.continuation_count = 1
    d = check_token_budget(t, None, budget=100_000, global_turn_tokens=95_000)
    assert isinstance(d, StopDecision)
    assert d.completion_event.duration_ms >= 10
    print("✅ Duration tracked OK")

    print("\n🎉 All query_token_budget.py tests passed")


main()