def normalize_name(value):
    """Trim whitespace, collapse internal whitespace, and casefold."""
    # Remove leading/trailing whitespace
    value = value.strip()
    # Collapse internal whitespace sequences to a single space
    import re
    value = re.sub(r'\s+', ' ', value)
    # Casefold
    value = value.casefold()
    return value