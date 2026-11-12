import sys
sys.path.insert(0, '/tmp/phase0-case-8p7bbiju')
from metrics import summarize_regions

def test_north_two_rows_sum_to_5_units():
    rows = [
        {'date': '2026-01-01', 'region': 'North', 'sku': 'A', 'units': 2, 'revenue': 20.0},
        {'date': '2026-01-02', 'region': 'North', 'sku': 'B', 'units': 3, 'revenue': 45.0},
    ]
    result = summarize_regions(rows)
    north = result['North']
    assert north['units'] == 5, f"Expected 5 units, got {north['units']}"
    assert north['revenue'] == 65.0

if __name__ == '__main__':
    test_north_two_rows_sum_to_5_units()
    print('Test passed')