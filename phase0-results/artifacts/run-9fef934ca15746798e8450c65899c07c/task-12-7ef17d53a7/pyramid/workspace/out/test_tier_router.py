import unittest
from tier_router import tier_for_complexity


class TestTierRouter(unittest.TestCase):
    def test_t1_boundary_low(self):
        self.assertEqual(tier_for_complexity(1), "T1")

    def test_t1_boundary_high(self):
        self.assertEqual(tier_for_complexity(2), "T1")

    def test_t2_boundary_low(self):
        self.assertEqual(tier_for_complexity(3), "T2")

    def test_t2_boundary_high(self):
        self.assertEqual(tier_for_complexity(4), "T2")

    def test_t3(self):
        self.assertEqual(tier_for_complexity(5), "T3")

    def test_reject_zero(self):
        with self.assertRaises(ValueError):
            tier_for_complexity(0)

    def test_reject_above_max(self):
        with self.assertRaises(ValueError):
            tier_for_complexity(6)

    def test_reject_negative(self):
        with self.assertRaises(ValueError):
            tier_for_complexity(-1)

    def test_reject_non_integer(self):
        with self.assertRaises(ValueError):
            tier_for_complexity("3")

    def test_reject_float(self):
        with self.assertRaises(ValueError):
            tier_for_complexity(2.5)


if __name__ == "__main__":
    unittest.main()