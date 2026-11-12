from metrics import summarize_regions


def test_north_rows_sum_to_five_units():
    """Two North rows should sum to 5 units total."""
    rows = [
        {'region': 'North', 'units': 2, 'revenue': 20.0},
        {'region': 'North', 'units': 3, 'revenue': 45.0},
    ]
    summary = summarize_regions(rows)
    assert summary['North']['units'] == 5