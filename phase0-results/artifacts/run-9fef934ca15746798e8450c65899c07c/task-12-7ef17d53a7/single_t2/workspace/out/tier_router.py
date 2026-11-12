def tier_for_complexity(value):
    """Map a complexity value to a tier.

    - 1 or 2 -> T1
    - 3 or 4 -> T2
    - 5 -> T3
    - any other value raises ValueError.
    """
    if value in (1, 2):
        return "T1"
    elif value in (3, 4):
        return "T2"
    elif value == 5:
        return "T3"
    else:
        raise ValueError(f"Complexity value {value} out of range (1-5)")