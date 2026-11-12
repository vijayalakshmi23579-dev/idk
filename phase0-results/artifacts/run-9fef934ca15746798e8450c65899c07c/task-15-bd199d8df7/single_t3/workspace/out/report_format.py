import json


def format_summary(metrics):
    """Return stable, sorted JSON text from metrics."""
    return json.dumps(metrics, sort_keys=True, indent=2)
