from paper_dashboard.forward_diagnostics import forward_comparison


def research():
    return [{'strategy': 'ORB', 'r_multiple': .5} for _ in range(30)]


def paper(count, result):
    return [{'trade_id': str(i), 'strategy': 'ORB', 'realized_r': result, 'net_pnl': result * 100,
             'exit_timestamp_utc': f'2026-10-09T18:{i:02d}:00Z'} for i in range(count)]


def test_one_losing_trade_cannot_be_alpha_decay():
    result = forward_comparison(research(), paper(1, -1))
    assert result['groups']['PORTFOLIO']['state'] == 'INSUFFICIENT_DATA'
    assert result['statistically_concerning_enabled'] is False


def test_low_mean_is_descriptive_watch_without_significance_claim():
    result = forward_comparison(research(), paper(25, -1))
    assert result['groups']['ORB']['state'] == 'WATCH'
    assert result['groups']['MRL1']['state'] == 'INSUFFICIENT_DATA'
    assert result == forward_comparison(research(), reversed(paper(25, -1)))


def test_costs_do_not_create_usd_to_r_conversion():
    rows = paper(25, .5)
    for row in rows:
        row['net_pnl'] = 10000
    result = forward_comparison(research(), rows)
    assert result['groups']['PORTFOLIO']['forward_metrics']['expectancy_r'] == .5
    assert result['groups']['PORTFOLIO']['state'] == 'WITHIN_EXPECTATIONS'


def test_duplicate_or_invalid_records_withhold_comparison():
    rows = paper(1, .5)
    assert forward_comparison(research(), rows * 2)['available'] is False
    rows[0]['realized_r'] = None
    assert forward_comparison(research(), rows)['state'] == 'UNAVAILABLE'
