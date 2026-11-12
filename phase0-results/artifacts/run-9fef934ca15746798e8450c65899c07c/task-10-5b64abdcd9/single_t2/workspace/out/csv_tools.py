import csv
from io import StringIO


def parse_rows(text: str):
    """Parse CSV text and convert units to int and revenue to float.

    Returns a list of dicts, one per row (excluding the header).
    """
    rows = []
    reader = csv.DictReader(StringIO(text))
    for row in reader:
        row["units"] = int(row["units"])
        row["revenue"] = float(row["revenue"])
        rows.append(row)
    return rows