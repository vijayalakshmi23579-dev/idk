import unittest
from csv_tools import parse_rows


class TestParseRows(unittest.TestCase):
    def test_two_row_input(self):
        text = "date,region,sku,units,revenue\n2026-01-01,North,A,2,20.0\n2026-01-01,South,B,1,15.0"
        rows = parse_rows(text)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"date": "2026-01-01", "region": "North", "sku": "A", "units": 2, "revenue": 20.0})
        self.assertEqual(rows[1], {"date": "2026-01-01", "region": "South", "sku": "B", "units": 1, "revenue": 15.0})
        self.assertIsInstance(rows[0]["units"], int)
        self.assertIsInstance(rows[0]["revenue"], float)


if __name__ == "__main__":
    unittest.main()
