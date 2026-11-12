import json

from report_format import format_summary


def test_key_order_is_stable():
    metrics = {
        "zebra": 1,
        "apple": 2,
        "mango": 3,
    }
    first = format_summary(metrics)
    second = format_summary(metrics)
    assert first == second
    assert first == json.dumps(metrics, sort_keys=True)
