"""Path containment checks with safe cross-drive handling."""

import os


def is_within_root(root, candidate):
    """Return True if *candidate* resolves to a path inside *root*.

    Both paths are resolved (symlinks normalized, ``..`` components removed)
    before comparison.  A trailing separator on *root* is handled so that
    ``/app/data`` and ``/app/data/`` behave identically.  Cross-drive
    comparisons (e.g. ``C:`` vs ``D:`` on Windows) always return ``False``
    because ``os.path.commonpath`` raises ``ValueError`` for paths on
    different drives.

    Parameters
    ----------
    root : str | os.PathLike
        The directory that is treated as the boundary.
    candidate : str | os.PathLike
        The path to test for containment within *root*.

    Returns
    -------
    bool
        ``True`` when *candidate* is equal to or nested inside *root*,
        otherwise ``False``.
    """
    root_resolved = os.path.realpath(root)
    candidate_resolved = os.path.realpath(candidate)

    # Normalise the root so a trailing separator does not break the prefix
    # comparison (os.path.commonpath treats "foo" and "foo/" differently).
    root_norm = os.path.normpath(root_resolved)
    if root_norm != os.sep:
        root_norm = root_norm.rstrip(os.sep)

    try:
        common = os.path.commonpath([root_norm, candidate_resolved])
    except ValueError:
        # Raised on mixed absolute/relative paths or different drives.
        return False

    return common == root_norm
