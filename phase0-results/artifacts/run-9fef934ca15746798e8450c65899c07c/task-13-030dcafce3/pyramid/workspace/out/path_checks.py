"""Filesystem path containment checks.

Provides :func:`is_within_root` for safely determining whether a candidate
path lies within (or is equal to) a given root directory.  The implementation
resolves both paths to absolute canonical paths, handles cross-drive cases on
Windows, and rejects ``..`` traversal or symlink escapes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def is_within_root(root: str | os.PathLike[str], candidate: str | os.PathLike[str]) -> bool:
    """Return True if ``candidate`` is inside ``root`` (or equal to it).

    Both paths are resolved to absolute canonical paths.  ``..`` components
    are normalised by ``Path.resolve()``, so a candidate that escapes the root
    via ``..`` will not be considered within the root.  On Windows, if the
    two paths live on different drives, the function returns False
    immediately (cross-drive safety).

    Args:
        root: The directory that acts as the containment boundary.
        candidate: The path to test against the boundary.

    Returns:
        True when the resolved candidate path is within or equal to the
        resolved root path; False otherwise.
    """
    try:
        root_path = Path(root).resolve(strict=False)
        candidate_path = Path(candidate).resolve(strict=False)
    except (OSError, ValueError):
        return False

    # Cross-drive handling: on Windows, paths on different volumes can never
    # be within each other.
    if sys.platform == "win32":
        root_drive = root_path.drive
        candidate_drive = candidate_path.drive
        if root_drive.lower() != candidate_drive.lower():
            return False

    # ``is_relative_to`` (Python 3.9+) is the cleanest containment test.
    if hasattr(candidate_path, "is_relative_to"):
        return candidate_path.is_relative_to(root_path)

    # Fallback for older interpreters: compare via common path.
    try:
        return os.path.commonpath([root_path, candidate_path]) == root_path
    except ValueError:
        # commonpath raises ValueError when paths are on different drives
        # on non-Windows platforms or when mixing absolute/relative.
        return False


if __name__ == "__main__":
    # Basic self-checks.
    tmp = Path(__file__).resolve().parent
    assert is_within_root(tmp, tmp) is True, "root should be within itself"
    assert is_within_root(tmp, tmp / "subdir" / "file.txt") is True, "nested child should be within root"
    assert is_within_root(tmp, tmp.parent) is False, "parent should be outside root"
    assert is_within_root(tmp, tmp / ".." / "other") is False, ".. traversal should escape root"
    print("path_checks self-checks passed")
