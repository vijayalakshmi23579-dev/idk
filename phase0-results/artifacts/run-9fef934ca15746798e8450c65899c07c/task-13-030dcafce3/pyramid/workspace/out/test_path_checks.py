"""Unit tests for :func:`is_within_root`.

Run with:  pytest test_path_checks.py
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

# Make path_checks importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from path_checks import is_within_root  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tmp_root(tmp_path: Path) -> Path:
    """Return a deterministic root directory under tmp_path."""
    root = tmp_path / "root"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Inside cases
# ---------------------------------------------------------------------------

def test_candidate_equal_to_root(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    assert is_within_root(root, root) is True


def test_root_with_trailing_separator(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    assert is_within_root(str(root) + os.sep, root) is True


def test_direct_child_is_within_root(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    child = root / "child.txt"
    child.touch()
    assert is_within_root(root, child) is True


def test_nested_child_is_within_root(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    nested = root / "a" / "b" / "c.txt"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.touch()
    assert is_within_root(root, nested) is True


def test_string_paths_are_handled(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    assert is_within_root(str(root), str(root / "file.txt")) is True


# ---------------------------------------------------------------------------
# Outside cases
# ---------------------------------------------------------------------------

def test_sibling_directory_is_outside(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    sibling = tmp_path / "sibling"
    sibling.mkdir(parents=True, exist_ok=True)
    assert is_within_root(root, sibling) is False


def test_dotdot_traversal_escapes_root(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    escaped = root / ".." / "outside.txt"
    assert is_within_root(root, escaped) is False


def test_absolute_path_outside_root(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    outside = tmp_path / "outside.txt"
    outside.touch()
    assert is_within_root(root, outside) is False


def test_parent_of_root_is_outside(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    assert is_within_root(root, root.parent) is False


def test_nonexistent_candidate_inside_root(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    nonexistent = root / "does_not_exist" / "deep" / "file.txt"
    assert is_within_root(root, nonexistent) is True


def test_nonexistent_candidate_outside_root(tmp_path: Path) -> None:
    root = _tmp_root(tmp_path)
    nonexistent = tmp_path / "outside" / "deep" / "file.txt"
    assert is_within_root(root, nonexistent) is False


def test_symlink_pointing_outside_root(tmp_path: Path) -> None:
    """A symlink inside root that points outside must be rejected."""
    if not hasattr(os, "symlink"):
        return
    root = _tmp_root(tmp_path)
    target_outside = tmp_path / "outside_target.txt"
    target_outside.touch()
    link_inside = root / "link_outside.txt"
    try:
        os.symlink(target_outside, link_inside)
    except OSError:
        return
    assert is_within_root(root, link_inside) is False


def test_symlink_pointing_inside_root(tmp_path: Path) -> None:
    """A symlink inside root that points inside root should be accepted."""
    if not hasattr(os, "symlink"):
        return
    root = _tmp_root(tmp_path)
    target_inside = root / "real_inside.txt"
    target_inside.touch()
    link_inside = root / "link_inside.txt"
    try:
        os.symlink(target_inside, link_inside)
    except OSError:
        return
    assert is_within_root(root, link_inside) is True


@unittest.skipUnless(sys.platform == "win32", "cross-drive test requires Windows")
def test_cross_drive_is_outside(tmp_path: Path) -> None:
    """A candidate on a different Windows drive must be rejected."""
    root = _tmp_root(tmp_path)
    other_drive = Path("D:") / "some" / "path"
    assert is_within_root(root, other_drive) is False


# ---------------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        test_candidate_equal_to_root(td_path)
        test_root_with_trailing_separator(td_path)
        test_direct_child_is_within_root(td_path)
        test_nested_child_is_within_root(td_path)
        test_string_paths_are_handled(td_path)
        test_sibling_directory_is_outside(td_path)
        test_dotdot_traversal_escapes_root(td_path)
        test_absolute_path_outside_root(td_path)
        test_parent_of_root_is_outside(td_path)
        test_nonexistent_candidate_inside_root(td_path)
        test_nonexistent_candidate_outside_root(td_path)
        test_symlink_pointing_outside_root(td_path)
        test_symlink_pointing_inside_root(td_path)
    print("test_path_checks standalone checks passed")
