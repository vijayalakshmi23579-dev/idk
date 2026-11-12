def tier_for_complexity(value):
    """Map a complexity score to a tier label.

    Args:
        value: An integer complexity score.

    Returns:
        A tier string: "T1", "T2", or "T3".

    Raises:
        ValueError: If value is not 1, 2, 3, 4, or 5.
    """
    if value in (1, 2):
        return "T1"
    if value in (3, 4):
        return "T2"
    if value == 5:
        return "T3"
    raise ValueError(f"Invalid complexity value: {value!r}. Expected 1-5.")