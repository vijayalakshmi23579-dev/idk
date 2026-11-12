def decide_mode(steps, sequential, branches, read_heavy, disjoint_writes):
    """Decide execution mode based on split-gate conditions.

    Returns "single" when execution is sequential or the step count is at most
    eight. Returns "wide" when there are at least three independent branches
    plus either read-heavy work or disjoint writes. Otherwise returns "single".
    """
    if sequential or steps <= 8:
        return "single"
    if branches >= 3 and (read_heavy or disjoint_writes):
        return "wide"
    return "single"
