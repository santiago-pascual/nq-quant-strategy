"""Presentation-only normalization for heterogeneous diagnostic fields."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone


def market_display_state(alerts, *, now=None):
    """Only a fresh calendar evaluation can label a session open or closed."""
    market = (alerts or {}).get('market_state') or {}
    now = now or datetime.now(timezone.utc)
    try:
        stamp = datetime.fromisoformat(str(market['evaluated_at_utc']).replace('Z','+00:00'))
        if stamp.tzinfo is None or not 0 <= (now-stamp).total_seconds() <= 120:
            return 'MARKET_UNKNOWN'
    except (KeyError, ValueError, TypeError):
        return 'MARKET_UNKNOWN'
    state = market.get('state')
    return state if state in {'MARKET_OPEN','MARKET_CLOSED','MARKET_BREAK','MARKET_UNKNOWN'} else 'MARKET_UNKNOWN'


def account_max_drawdown(data):
    """Account high-water drawdown must not be replaced by a closed-trade sample."""
    value = (((data or {}).get('status') or {}).get('portfolio') or {}).get('max_drawdown')
    if isinstance(value, (int,float)) and not isinstance(value,bool) and math.isfinite(value):
        return value
    return None


def alert_table_rows(rows):
    result = []
    for source in rows:
        row = dict(source)
        for field in ('expected', 'observed', 'threshold'):
            value = row.get(field)
            row[field] = ('Unavailable' if value is None else value if isinstance(value, str)
                          else json.dumps(value, sort_keys=True, default=str))
        result.append(row)
    return result
