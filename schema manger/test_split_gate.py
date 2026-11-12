"""Offline tests for deterministic Pyramid split decisions."""
from split_gate import SplitEstimate, decide_mode


def estimate(**overrides):
    values = {
        "estimated_steps": 12,
        "sequential_dependency": False,
        "independent_branches": 3,
        "read_heavy": True,
        "disjoint_writes": False,
        "uncertain_steps": False,
    }
    values.update(overrides)
    return SplitEstimate(**values)


def main():
    assert decide_mode(None).mode == "single"
    assert decide_mode(estimate(sequential_dependency=True)).mode == "single"
    assert decide_mode(estimate(estimated_steps=8)).mode == "single"
    assert decide_mode(estimate()).mode == "wide"
    assert decide_mode(estimate(independent_branches=2, read_heavy=False)).mode == "single"
    assert decide_mode(estimate(
        independent_branches=2, read_heavy=False, uncertain_steps=True,
    )).mode == "single_then_escalate"
    print("Deterministic split gate tests passed")


if __name__ == "__main__":
    main()