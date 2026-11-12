"""Name normalization utilities."""


def normalize_name(value):
    """Trim whitespace, collapse internal whitespace, and casefold."""
    return " ".join(str(value).strip().split()).casefold()
