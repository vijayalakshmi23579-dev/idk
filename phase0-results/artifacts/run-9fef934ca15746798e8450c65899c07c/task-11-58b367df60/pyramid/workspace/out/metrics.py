def summarize_regions(rows):
    """Return units and revenue totals keyed by region.

    Each row is expected to be a mapping with ``region``, ``units``, and
    ``revenue`` keys. Returns a dict mapping region name to a dict with
    ``units`` and ``revenue`` totals.
    """
    totals = {}
    for row in rows:
        region = row["region"]
        units = int(row["units"])
        revenue = float(row["revenue"])
        if region not in totals:
            totals[region] = {"units": 0, "revenue": 0.0}
        totals[region]["units"] += units
        totals[region]["revenue"] += revenue
    return totals
