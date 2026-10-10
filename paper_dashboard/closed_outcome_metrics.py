"""Shared reporting-only metrics; never changes account accounting."""
import math

def closed_trade_drawdown(rows: list[dict]) -> dict:
    """A closed-outcome curve starts at zero before the first outcome.

    This is not intratrade/account drawdown, which comes from persisted equity.
    """
    if not rows:
        return {'available': False, 'reason': 'No closed outcomes in period'}
    running = peak = maximum = 0.0
    for row in rows:
        try:
            value = float(row['net_pnl'])
        except (KeyError, TypeError, ValueError):
            return {'available': False, 'reason': 'Missing net outcome'}
        if not math.isfinite(value):
            return {'available': False, 'reason': 'Nonfinite net outcome'}
        running += value
        peak = max(peak, running)
        maximum = min(maximum, running - peak)
    return {'available': True, 'current_usd': running - peak, 'maximum_usd': maximum,
            'basis': 'Chronological included closed net outcomes with initial zero baseline; excludes open-trade equity'}
