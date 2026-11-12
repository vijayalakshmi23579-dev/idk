from metrics import summarize_regions


def test_two_north_rows_sum_to_5_units():
    rows = [
        {"date": "2026-01-01", "region": "North", "sku": "A", "units": 2, "revenue": 20.0},
        {"date": "2026-01-02", "region": "North", "sku": "B", "units": 3, "revenue": 45.0},
    ]
    result = summarize_regions(rows)
    assert result["North"]["units"] == 5
