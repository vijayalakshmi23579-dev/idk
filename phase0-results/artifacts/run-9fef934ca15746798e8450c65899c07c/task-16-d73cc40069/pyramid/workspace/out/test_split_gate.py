import unittest

from split_gate import decide_mode


class TestDecideMode(unittest.TestCase):
    def test_sequential_returns_single(self):
        self.assertEqual(decide_mode(steps=12, sequential=True, branches=5, read_heavy=True, disjoint_writes=True), "single")

    def test_steps_at_most_eight_returns_single(self):
        self.assertEqual(decide_mode(steps=8, sequential=False, branches=4, read_heavy=True, disjoint_writes=True), "single")
        self.assertEqual(decide_mode(steps=1, sequential=False, branches=5, read_heavy=True, disjoint_writes=True), "single")

    def test_wide_with_read_heavy(self):
        self.assertEqual(decide_mode(steps=10, sequential=False, branches=3, read_heavy=True, disjoint_writes=False), "wide")
        self.assertEqual(decide_mode(steps=20, sequential=False, branches=5, read_heavy=True, disjoint_writes=False), "wide")

    def test_wide_with_disjoint_writes(self):
        self.assertEqual(decide_mode(steps=10, sequential=False, branches=4, read_heavy=False, disjoint_writes=True), "wide")

    def test_fewer_than_three_branches_returns_single(self):
        self.assertEqual(decide_mode(steps=12, sequential=False, branches=2, read_heavy=True, disjoint_writes=True), "single")

    def test_no_read_heavy_and_no_disjoint_writes_returns_single(self):
        self.assertEqual(decide_mode(steps=12, sequential=False, branches=4, read_heavy=False, disjoint_writes=False), "single")

    def test_both_read_heavy_and_disjoint_writes_returns_wide(self):
        self.assertEqual(decide_mode(steps=15, sequential=False, branches=3, read_heavy=True, disjoint_writes=True), "wide")


if __name__ == "__main__":
    unittest.main()
