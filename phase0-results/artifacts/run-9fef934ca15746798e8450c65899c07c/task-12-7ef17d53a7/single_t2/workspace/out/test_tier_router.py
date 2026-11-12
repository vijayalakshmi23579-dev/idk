"""Boundary tests for tier_for_complexity.

Tests cover the valid ranges and edge cases, plus that invalid values raise ValueError.
"""

from tier_router import tier_for_complexity


def test_tier_1_is_T1():
    assert tier_for_complexity(1) == "T1"


def test_tier_2_is_T1():
    assert tier_for_complexity(2) == "T1"


def test_tier_3_is_T2():
    assert tier_for_complexity(3) == "T2"


def test_tier_4_is_T2():
    assert tier_for_complexity(4) == "T2"


def test_tier_5_is_T3():
    assert tier_for_complexity(5) == "T3"


def test_invalid_zero_raises():
    try:
        tier_for_complexity(0)
        assert False, "Expected ValueError for value 0"
    except ValueError:
        pass


def test_invalid_six_raises():
    try:
        tier_for_complexity(6)
        assert False, "Expected ValueError for value 6"
    except ValueError:
        pass


def test_invalid_negative_raises():
    try:
        tier_for_complexity(-1)
        assert False, "Expected ValueError for value -1"
    except ValueError:
        pass


def test_invalid_float_raises():
    try:
        tier_for_complexity(2.5)
        assert False, "Expected ValueError for non‑integer value 2.5"
    except ValueError:
        pass