import csv
import io


def parse_rows(text):
    """Parse CSV text, converting units to int and revenue to float."""
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for row in reader:
        row["units"] = int(row["units"])
        row["revenue"] = float(row["revenue"])
        rows.append(row)
    return rows
