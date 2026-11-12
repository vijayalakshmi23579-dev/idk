def decide_mode(steps, sequential, branches, read_heavy, disjoint_writes):
    """Decide execution mode based on pipeline characteristics.

    Returns 'single' for sequential execution or small step counts (<= 8).
    Returns 'wide' when there are at least 3 branches AND either read-heavy
    workload or disjoint writes. Otherwise returns 'single'.
    """
    if sequential or steps <= 8:
        return "single"

    if branches >= 3 and (read_heavy or disjoint_writes):
        return "wide"

    return "single"
