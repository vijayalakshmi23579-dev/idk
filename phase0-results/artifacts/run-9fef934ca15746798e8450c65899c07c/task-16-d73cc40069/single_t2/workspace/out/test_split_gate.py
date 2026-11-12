import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from split_gate import decide_mode
import unittest

class TestDecideMode(unittest.TestCase):
    def test_sequential_returns_single(self):
        # sequential=True always returns 'single'
        mode = decide_mode(steps=10, sequential=True, branches=1, read_heavy=False, disjoint_writes=False)
        self.assertEqual(mode, 'single')

    def test_steps_leq8_returns_single(self):
        # steps <= 8 returns 'single' even if not sequential
        mode = decide_mode(steps=5, sequential=False, branches=3, read_heavy=True, disjoint_writes=False)
        self.assertEqual(mode, 'single')

    def test_branches_and_read_heavy_returns_wide(self):
        # at least 3 branches plus read_heavy => 'wide'
        mode = decide_mode(steps=10, sequential=False, branches=3, read_heavy=True, disjoint_writes=False)
        self.assertEqual(mode, 'wide')

    def test_branches_and_disjoint_writes_returns_wide(self):
        # at least 3 branches plus disjoint_writes => 'wide'
        mode = decide_mode(steps=10, sequential=False, branches=4, read_heavy=False, disjoint_writes=True)
        self.assertEqual(mode, 'wide')

    def test_default_falls_back_to_single(self):
        # otherwise falls back to 'single'
        mode = decide_mode(steps=20, sequential=False, branches=5, read_heavy=False, disjoint_writes=False)
        self.assertEqual(mode, 'single')

if __name__ == '__main__':
    unittest.main()