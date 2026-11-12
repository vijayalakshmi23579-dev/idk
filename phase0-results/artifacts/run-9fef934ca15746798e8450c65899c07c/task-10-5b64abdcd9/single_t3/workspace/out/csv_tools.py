import csv
from io import StringIO


def parse_rows(text):
    """Parse CSV text into a list of dicts, converting units to int and revenue to float."""
    reader = csv.DictReader(StringIO(text))
    rows = []
    for row in reader:
        rows.append({
            "date": row["date"],
            "region": row["region"],
            "sku": row["sku"],
            "units": int(row["units"]),
            "revenue": float(row["revenue"]),
        })
    return rows
