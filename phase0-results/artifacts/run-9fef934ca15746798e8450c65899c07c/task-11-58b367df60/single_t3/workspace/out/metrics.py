def summarize_regions(rows):
    """Return units and revenue totals keyed by region.

    Args:
        rows: An iterable of dict-like rows, each containing at least
              the keys 'region', 'units', and 'revenue'.

    Returns:
        A dict mapping region name to a dict with 'units' and 'revenue'
        totals for that region.
    """
    result = {}
    for row in rows:
        region = row['region']
        if region not in result:
            result[region] = {'units': 0, 'revenue': 0.0}
        result[region]['units'] += row['units']
        result[region]['revenue'] += row['revenue']
    return result