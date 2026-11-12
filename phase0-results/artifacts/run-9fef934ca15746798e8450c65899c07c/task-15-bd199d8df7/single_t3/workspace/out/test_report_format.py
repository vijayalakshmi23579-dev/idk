import json
import unittest

from report_format import format_summary


class TestFormatSummary(unittest.TestCase):
    def test_key_order_is_stable(self):
        metrics = {
            "zulu": 1,
            "alpha": 2,
            "mike": 3,
        }
        first = format_summary(metrics)
        second = format_summary(metrics)
        self.assertEqual(first, second)

        parsed = json.loads(first)
        self.assertEqual(list(parsed.keys()), ["alpha", "mike", "zulu"])

    def test_nested_keys_are_sorted(self):
        metrics = {
            "b": {"d": 4, "a": 1},
            "a": {"c": 3, "b": 2},
        }
        result = format_summary(metrics)
        parsed = json.loads(result)
        self.assertEqual(list(parsed.keys()), ["a", "b"])
        self.assertEqual(list(parsed["a"].keys()), ["b", "c"])
        self.assertEqual(list(parsed["b"].keys()), ["a", "d"])


if __name__ == "__main__":
    unittest.main()
