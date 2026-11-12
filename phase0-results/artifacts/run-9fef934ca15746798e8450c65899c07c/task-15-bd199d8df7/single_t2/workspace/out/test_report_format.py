import json

from report_format import format_summary


def test_key_order_is_stable():
    """Assert that format_summary returns JSON with a consistent key order."""
    metrics = {"accuracy": 0.95, "loss": 0.23, "epochs": 10}
    # Call twice and ensure the ordering is the same (sorted keys)
    result1 = format_summary(metrics)
    result2 = format_summary(metrics)
    # Both should produce identical sorted JSON
    assert result1 == result2
    # Verify that the keys appear in sorted order by parsing back
    parsed = json.loads(result1)
    # The string representation should have keys in alphabetical order
    expected = json.dumps(metrics, sort_keys=True)
    assert result1 == expected