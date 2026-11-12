import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from split_gate import decide_mode


def test_sequential_returns_single():
    assert decide_mode(steps=20, sequential=True, branches=5,
                       read_heavy=True, disjoint_writes=True) == "single"
    assert decide_mode(steps=100, sequential=True, branches=0,
                       read_heavy=False, disjoint_writes=False) == "single"


def test_steps_leq_8_returns_single():
    assert decide_mode(steps=8, sequential=False, branches=5,
                       read_heavy=True, disjoint_writes=True) == "single"
    assert decide_mode(steps=1, sequential=False, branches=10,
                       read_heavy=True, disjoint_writes=True) == "single"
    assert decide_mode(steps=5, sequential=False, branches=0,
                       read_heavy=False, disjoint_writes=False) == "single"


def test_wide_branches_and_read_heavy():
    assert decide_mode(steps=20, sequential=False, branches=3,
                       read_heavy=True, disjoint_writes=False) == "wide"
    assert decide_mode(steps=50, sequential=False, branches=4,
                       read_heavy=True, disjoint_writes=False) == "wide"


def test_wide_branches_and_disjoint_writes():
    assert decide_mode(steps=20, sequential=False, branches=3,
                       read_heavy=False, disjoint_writes=True) == "wide"
    assert decide_mode(steps=15, sequential=False, branches=10,
                       read_heavy=False, disjoint_writes=True) == "wide"


def test_fewer_than_3_branches_returns_single():
    assert decide_mode(steps=20, sequential=False, branches=2,
                       read_heavy=True, disjoint_writes=True) == "single"
    assert decide_mode(steps=20, sequential=False, branches=1,
                       read_heavy=True, disjoint_writes=False) == "single"
    assert decide_mode(steps=20, sequential=False, branches=0,
                       read_heavy=False, disjoint_writes=True) == "single"


def test_branches_without_read_heavy_or_disjoint_writes_returns_single():
    assert decide_mode(steps=20, sequential=False, branches=3,
                       read_heavy=False, disjoint_writes=False) == "single"
    assert decide_mode(steps=30, sequential=False, branches=5,
                       read_heavy=False, disjoint_writes=False) == "single"


if __name__ == "__main__":
    test_sequential_returns_single()
    test_steps_leq_8_returns_single()
    test_wide_branches_and_read_heavy()
    test_wide_branches_and_disjoint_writes()
    test_fewer_than_3_branches_returns_single()
    test_branches_without_read_heavy_or_disjoint_writes_returns_single()
    print("All tests passed!")
