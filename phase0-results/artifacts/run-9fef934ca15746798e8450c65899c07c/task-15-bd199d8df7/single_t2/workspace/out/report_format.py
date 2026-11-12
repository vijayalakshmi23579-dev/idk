import json


def format_summary(metrics):
    """Return a stable, sorted JSON representation of metrics."""
    return json.dumps(metrics, sort_keys=True)