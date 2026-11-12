"""Path‑within‑root checks with safe cross‑drive handling."""

import os


def is_within_root(root: str, candidate: str) -> bool:
    """Return True if *candidate* is inside *root* directory tree.

    The function resolves both paths to their real (symlink‑free) absolute
    forms, then checks that the candidate stays under the root.  It also
    protects against cross‑drive attacks on Windows‑style paths by ensuring
    both paths reside on the same driveletter (if any).

    Parameters
    ----------
    root : str
        The prospective root directory (may be relative or absolute).
    candidate : str
        The path to test (may be relative or absolute).

    Returns
    -------
    bool
        ``True`` when *candidate* is the same as *root* or lies somewhere
        beneath it; otherwise ``False``.
    """
    # Resolve to absolute, symlink‑free paths
    root_resolved = os.path.realpath(root)
    cand_resolved = os.path.realpath(candidate)

    # Split drive letters (Windows) – empty string on POSIX
    root_drive, _ = os.path.splitdrive(root_resolved)
    cand_drive, _ = os.path.splitdrive(cand_resolved)

    # Cross‑drive check – different drives can never be inside each other
    if root_drive.lower() != cand_drive.lower():
        return False

    # On POSIX root_drive is '' and the comparison below always passes;
    # on Windows it guarantees same‑drive containment.
    # Use os.sep to avoid matching e.g. root="/foo" and candidate="/foobar"
    # without a separator.
    prefix = root_resolved + os.sep
    return cand_resolved.startswith(prefix) or cand_resolved == root_resolved