def summarize_regions(rows):
    """Return dict mapping region to {'units': int, 'revenue': float} totals."""
    totals = {}
    for row in rows:
        region = row['region']
        units = row['units']
        revenue = row['revenue']
        if region not in totals:
            totals[region] = {'units': 0, 'revenue': 0.0}
        totals[region]['units'] += units
        totals[region]['revenue'] += revenue
    return totals