from paper_dashboard.display_values import alert_table_rows
from paper_dashboard.display_values import market_display_state
from paper_dashboard.display_values import account_max_drawdown
from datetime import datetime, timezone, timedelta


def test_market_state_requires_fresh_reviewed_evidence():
    now = datetime(2026,10,10,tzinfo=timezone.utc)
    assert market_display_state({},now=now) == 'MARKET_UNKNOWN'
    market = {'market_state':{'state':'MARKET_CLOSED','evaluated_at_utc':now.isoformat()}}
    assert market_display_state(market,now=now) == 'MARKET_CLOSED'
    assert market_display_state(market,now=now+timedelta(seconds=121)) == 'MARKET_UNKNOWN'


def test_account_drawdown_never_uses_incompatible_trade_sample():
    data = {'portfolio':{'max_drawdown_usd':0},'status':{'portfolio':{'max_drawdown':-409.11}}}
    assert account_max_drawdown(data) == -409.11
    assert account_max_drawdown({'portfolio':{'max_drawdown_usd':0}}) is None


def test_mixed_alert_fields_are_strings_without_mutating_persisted_records():
    source = [{'expected': True, 'observed': False, 'threshold': None},
              {'expected': 'RUNNING', 'observed': {'state': 'ERROR'}, 'threshold': 10}]
    rows = alert_table_rows(source)
    assert rows[0]['expected'] == 'true'
    assert rows[0]['threshold'] == 'Unavailable'
    assert rows[1]['expected'] == 'RUNNING'
    assert rows[1]['observed'] == '{"state": "ERROR"}'
    assert all(isinstance(row[key], str) for row in rows for key in ('expected', 'observed', 'threshold'))
    assert source[0]['expected'] is True
    assert source[1]['observed'] == {'state': 'ERROR'}
