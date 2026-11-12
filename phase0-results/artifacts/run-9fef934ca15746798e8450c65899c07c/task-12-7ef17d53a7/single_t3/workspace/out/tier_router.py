"""Tier router mapping complexity scores to support tiers."""


def tier_for_complexity(value):
    """Map a complexity value to a tier.

    Args:
        value: An integer complexity score.

    Returns:
        A tier string: "T1", "T2", or "T3".

    Raises:
        ValueError: If value is not 1-5.
        TypeError: If value is not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("complexity must be an integer")
    if value < 1 or value > 5:
        raise ValueError("complexity must be between 1 and 5")
    if value <= 2:
        return "T1"
    if value <= 4:
        return "T2"
    return "T3"
