"""test_denial_tracking.py"""
from denial_tracking import (
    DenialTrackingState, DENIAL_LIMITS,
    create_denial_tracking_state, record_denial, record_success,
    reset_denials, should_fallback_to_prompting, hit_total_limit,
    apply_in_place,
)

# ── Fresh state ──
s = create_denial_tracking_state()
assert s.consecutive_denials == 0
assert s.total_denials == 0
print("✅ Fresh state OK")

# ── record_denial increments both ──
s = create_denial_tracking_state()
s = record_denial(s)
assert s.consecutive_denials == 1
assert s.total_denials == 1
s = record_denial(s)
assert s.consecutive_denials == 2
assert s.total_denials == 2
print("✅ record_denial OK")

# ── record_success resets consecutive, keeps total ──
s = create_denial_tracking_state()
s = record_denial(s)
s = record_denial(s)
assert s.consecutive_denials == 2
assert s.total_denials == 2
s = record_success(s)
assert s.consecutive_denials == 0
assert s.total_denials == 2  # total preserved
print("✅ record_success OK")

# ── record_success returns SAME instance when consecutive already 0 ──
s = create_denial_tracking_state()
s2 = record_success(s)
assert s2 is s  # identity check
print("✅ record_success identity check OK")

# ── reset_denials clears both ──
s = DenialTrackingState(consecutive_denials=5, total_denials=25)
s = reset_denials(s)
assert s.consecutive_denials == 0
assert s.total_denials == 0
print("✅ reset_denials OK")

# ── Consecutive limit fires at 3 ──
s = create_denial_tracking_state()
assert not should_fallback_to_prompting(s)
s = record_denial(s)
assert not should_fallback_to_prompting(s)  # 1
s = record_denial(s)
assert not should_fallback_to_prompting(s)  # 2
s = record_denial(s)
assert should_fallback_to_prompting(s)      # 3 → fires
print("✅ Consecutive limit (3) OK")

# ── Total limit fires at 20 (consecutive kept at 0) ──
s = create_denial_tracking_state()
for _ in range(19):
    s = record_denial(s)
    s = record_success(s)  # keep consecutive low
assert s.consecutive_denials == 0
assert s.total_denials == 19
assert not should_fallback_to_prompting(s)
s = record_denial(s)  # total = 20
assert should_fallback_to_prompting(s)
print("✅ Total limit (20) OK")

# ── hit_total_limit only fires at 20 ──
s = DenialTrackingState(consecutive_denials=3, total_denials=15)
assert not hit_total_limit(s)
s = DenialTrackingState(consecutive_denials=0, total_denials=20)
assert hit_total_limit(s)
print("✅ hit_total_limit OK")

# ── apply_in_place mutates target ──
target = DenialTrackingState(consecutive_denials=0, total_denials=0)
source = DenialTrackingState(consecutive_denials=2, total_denials=5)
apply_in_place(target, source)
assert target.consecutive_denials == 2
assert target.total_denials == 5
assert target is not source
print("✅ apply_in_place OK")

# ── Realistic scenario: 3 denials → fallback → human approves → reset ──
s = create_denial_tracking_state()
s = record_denial(s)
s = record_denial(s)
s = record_denial(s)
assert should_fallback_to_prompting(s)  # 3 consecutive → fallback
s = record_success(s)  # human approves
assert s.consecutive_denials == 0
assert s.total_denials == 3
assert not should_fallback_to_prompting(s)  # back below threshold
print("✅ Realistic scenario OK")

print("\n🎉 All denial_tracking.py tests passed")