"""Tests for :mod:`path_checks.is_within_root`.

Uses the workspace fixtures (architecture.md, events.json, puzzles.md,
sales.csv, users.json) as test data that lives inside the project tree.
"""

import pathlib
import sys
import os

# Ensure the package under test is importable
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from path_checks import is_within_root

# Resolve the project root – the directory that contains the fixture files
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Paths that should be considered inside the root
INSIDE_CASES = [
    # Direct children of PROJECT_ROOT
    str(PROJECT_ROOT / "architecture.md"),
    str(PROJECT_ROOT / "events.json"),
    str(PROJECT_ROOT / "puzzles.md"),
    str(PROJECT_ROOT / "sales.csv"),
    str(PROJECT_ROOT / "users.json"),

    # Nested directories (if they exist)
    str(PROJECT_ROOT / "out"),
    str(PROJECT_ROOT / "out" / "path_checks.py"),
]

# Paths that should be outside the root
OUTSIDE_CASES = [
    # Parent of PROJECT_ROOT
    str(PROJECT_ROOT.parent / "some_other_dir"),

    # Sibling of PROJECT_ROOT
    str(PROJECT_ROOT.parent / "another_project"),

    # Paths that escape via .. outside the project root
    str((PROJECT_ROOT / "architecture.md").parent.parent / "escape"),

    # Absolute paths that point elsewhere on the filesystem
    "/tmp/nonexistent_dir",

    # Paths that resolve to a different drive letter (Windows‑style simulation)
    # On POSIX these will simply be outside because of the drive check;
    # we add them for completeness.
    "C:\\Windows\\System32\\config",
]

all_pass = True

print("--- inside cases ---")
for cand in INSIDE_CASES:
    result = is_within_root(str(PROJECT_ROOT), cand)
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"  {status}: is_within_root(root='{PROJECT_ROOT}', candidate='{cand}')")

print("\n--- outside cases ---")
for cand in OUTSIDE_CASES:
    result = is_within_root(str(PROJECT_ROOT), cand)
    status = "PASS" if not result else "FAIL"
    if result:
        all_pass = False
    print(f"  {status}: is_within_root(root='{PROJECT_ROOT}', candidate='{cand}')")

print("\n" + ("All tests passed!" if all_pass else "Some tests FAILED!"))
sys.exit(0 if all_pass else 1)