import pytest
from csv_tools import parse_rows


def test_parse_rows_two_rows():
    text = "date,region,sku,units,revenue\n2026-01-01,North,A,2,20.0\n2026-01-01,South,B,1,15.0"
    rows = parse_rows(text)
    assert len(rows) == 2
    assert rows[0]["units"] == 2
    assert rows[0]["revenue"] == 20.0
    assert rows[1]["units"] == 1
    assert rows[1]["revenue"] == 15.0