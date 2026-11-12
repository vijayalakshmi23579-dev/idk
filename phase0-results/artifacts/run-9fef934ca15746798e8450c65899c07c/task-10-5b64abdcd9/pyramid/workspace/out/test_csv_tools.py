import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from csv_tools import parse_rows


class TestCsvTools(unittest.TestCase):
    def test_two_row_input(self):
        text = (
            "date,region,sku,units,revenue\n"
            "2026-01-01,North,A,2,20.0\n"
            "2026-01-01,South,B,1,15.0\n"
        )
        rows = parse_rows(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["units"], 2)
        self.assertEqual(rows[0]["revenue"], 20.0)
        self.assertEqual(rows[1]["units"], 1)
        self.assertEqual(rows[1]["revenue"], 15.0)


if __name__ == "__main__":
    unittest.main()
