def decide_mode(steps, sequential, branches, read_heavy, disjoint_writes):
    """Return 'single' or 'wide' based on the given parameters."""
    if sequential:
        return 'single'
    if steps <= 8:
        return 'single'
    if branches >= 3 and (read_heavy or disjoint_writes):
        return 'wide'
    return 'single'