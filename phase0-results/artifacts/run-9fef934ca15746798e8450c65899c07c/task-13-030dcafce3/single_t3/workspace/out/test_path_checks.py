"""Tests for is_within_root covering inside and outside cases."""

import os
import tempfile
import unittest

from path_checks import is_within_root


class TestIsWithinRoot(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.root = os.path.join(self.tmpdir, "root")
        os.makedirs(self.root)

    def tearDown(self):
        # Best-effort cleanup of the temporary directory.
        for root, dirs, files in os.walk(self.tmpdir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(self.tmpdir)

    # ---- inside cases ---------------------------------------------------

    def test_root_itself(self):
        self.assertTrue(is_within_root(self.root, self.root))

    def test_nested_file(self):
        nested = os.path.join(self.root, "sub", "file.txt")
        self.assertTrue(is_within_root(self.root, nested))

    def test_deeply_nested(self):
        deep = os.path.join(self.root, "a", "b", "c", "d")
        self.assertTrue(is_within_root(self.root, deep))

    def test_root_with_trailing_separator(self):
        nested = os.path.join(self.root, "file.txt")
        self.assertTrue(is_within_root(self.root + os.sep, nested))

    def test_relative_candidate_within_root(self):
        nested = os.path.join(self.root, "file.txt")
        # Candidate resolved relative to the current working directory;
        # make it absolute by joining with tmpdir then relpath from cwd.
        self.assertTrue(is_within_root(self.root, nested))

    def test_symlink_inside_root_resolves_within(self):
        target = os.path.join(self.root, "real.txt")
        open(target, "w").close()
        link = os.path.join(self.tmpdir, "link_to_root_file")
        os.symlink(target, link)
        self.assertTrue(is_within_root(self.root, link))

    # ---- outside cases --------------------------------------------------

    def test_sibling_directory(self):
        sibling = os.path.join(self.tmpdir, "sibling")
        os.makedirs(sibling)
        self.assertFalse(is_within_root(self.root, sibling))

    def test_parent_directory(self):
        self.assertFalse(is_within_root(self.root, self.tmpdir))

    def test_dotdot_escape(self):
        escape = os.path.join(self.root, "sub", "..", "..", "escape.txt")
        self.assertFalse(is_within_root(self.root, escape))

    def test_completely_unrelated_path(self):
        unrelated = os.path.join(self.tmpdir, "unrelated", "file.txt")
        self.assertFalse(is_within_root(self.root, unrelated))

    def test_cross_drive_windows_style(self):
        # On non-Windows these are just distinct roots; on Windows they would
        # be different drives and must not be considered within each other.
        self.assertFalse(is_within_root("C:\\root", "D:\\root\\file.txt"))

    def test_nonexistent_candidate(self):
        # A nonexistent path still resolves lexically and must not leak out.
        escape = os.path.join(self.root, "..", "outside.txt")
        self.assertFalse(is_within_root(self.root, escape))


if __name__ == "__main__":
    unittest.main()
