"""Boundary tests for tier_for_complexity."""

import unittest

from tier_router import tier_for_complexity


class TestTierRouter(unittest.TestCase):
    """Tests covering valid mappings and rejection boundaries."""

    def test_t1_lower_bound(self):
        self.assertEqual(tier_for_complexity(1), "T1")

    def test_t1_upper_bound(self):
        self.assertEqual(tier_for_complexity(2), "T1")

    def test_t2_lower_bound(self):
        self.assertEqual(tier_for_complexity(3), "T2")

    def test_t2_upper_bound(self):
        self.assertEqual(tier_for_complexity(4), "T2")

    def test_t3(self):
        self.assertEqual(tier_for_complexity(5), "T3")

    def test_zero_rejected(self):
        with self.assertRaises(ValueError):
            tier_for_complexity(0)

    def test_six_rejected(self):
        with self.assertRaises(ValueError):
            tier_for_complexity(6)

    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            tier_for_complexity(-1)

    def test_float_rejected(self):
        with self.assertRaises(TypeError):
            tier_for_complexity(2.5)

    def test_string_rejected(self):
        with self.assertRaises(TypeError):
            tier_for_complexity("3")

    def test_bool_rejected(self):
        with self.assertRaises(TypeError):
            tier_for_complexity(True)

    def test_none_rejected(self):
        with self.assertRaises(TypeError):
            tier_for_complexity(None)


if __name__ == "__main__":
    unittest.main()
