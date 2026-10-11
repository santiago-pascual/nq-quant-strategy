import json
from pathlib import Path

import pandas as pd
import pytest

from src.paper.historical_extension import START, END, isolated_output, validate_frame, ROOT
from paper_dashboard.historical_extension import load_extension, source_timeline


def frame():
    return pd.DataFrame({'timestamp': pd.to_datetime(['2019-05-05T22:03Z','2026-06-21T22:00Z','2026-10-07T23:59Z','2026-10-08T13:02Z'], utc=True),
        'open': [100]*4,'high': [101]*4,'low': [99]*4,'close': [100]*4,'volume': [1]*4,
        'symbol': ['MNQ.v.0']*4,'instrument_id': [1]*4})


def test_observed_rows_are_not_a_minute_grid_certificate():
    evidence = validate_frame(frame())
    assert evidence['observed_rows'] == 2
    assert evidence['minute_grid_completeness_claimed'] is False
    assert evidence['first_observed_utc'].startswith('2026-06-21')


@pytest.mark.parametrize('failure', ['duplicate','order','ohlc','symbol','mapping','endpoint'])
def test_extension_input_integrity(failure):
    raw = frame()
    if failure == 'duplicate': raw.loc[2,'timestamp'] = raw.loc[1,'timestamp']
    if failure == 'order': raw = raw.iloc[::-1]
    if failure == 'ohlc': raw.loc[2,'high'] = 90
    if failure == 'symbol': raw.loc[2,'symbol'] = 'spread'
    if failure == 'mapping': raw.loc[2,'instrument_id'] = 0
    if failure == 'endpoint': raw = raw.iloc[:2]
    with pytest.raises(ValueError): validate_frame(raw)


def test_extension_never_targets_production_or_frozen_results():
    assert isolated_output(ROOT/'results/paper/historical_extension_fixture').name == 'historical_extension_fixture'
    for name in ['delayed_mnqz6_paper_accepted_20261008_1303_r3','full_research_replay_final_v3','oos_test']:
        with pytest.raises(ValueError): isolated_output(ROOT/'results/paper'/name)


def test_missing_or_partial_extension_has_no_returns(tmp_path):
    result = load_extension(tmp_path)
    assert result['available'] is False and result['trades'] == []
    directory = tmp_path/'results/paper/historical_extension_20260620_20261008'
    directory.mkdir(parents=True)
    (directory/'extension_analysis.json').write_text('{partial')
    assert load_extension(tmp_path)['available'] is False


def test_timeline_separates_units_and_preserves_gap(tmp_path):
    timeline = source_timeline(tmp_path, '2026-10-08T13:03:00Z')
    assert timeline[1]['end'].startswith('2026-06-19')
    assert timeline[2]['start'] == START.isoformat()
    assert timeline[2]['end'] == END.isoformat()
    assert timeline[3]['start'] == '2026-10-08T13:03:00Z'
    assert 'Research R' == timeline[1]['units']
    assert load_extension(tmp_path)['trades'] == []
    assert timeline[0]['start'] is None and timeline[0]['end'] is None


def completed_extension(tmp_path):
    from src.paper.historical_extension import DEFAULT_OUTPUT, sha256
    directory = tmp_path/'results/paper'/DEFAULT_OUTPUT.name
    directory.mkdir(parents=True)
    evidence = {}
    for name in ('status.json', 'events.jsonl', 'historical_extension.json'):
        (directory/name).write_text('{}', encoding='utf-8')
        evidence[name] = sha256(directory/name)
    value = {'schema_version': 1, 'classification': 'HISTORICAL_CAUSAL_SIMULATION',
             'completed': True, 'start_utc_inclusive': START.isoformat(),
             'end_utc_exclusive': END.isoformat(), 'artifacts_sha256': evidence,
             'trades': [{'trade_id': 'extension-only', 'strategy': 'ORB',
                         'entry_timestamp_utc': '2026-07-01T14:00:00Z',
                         'exit_timestamp_utc': '2026-07-01T14:30:00Z',
                         'realized_r': .5, 'net_pnl': 10.0}]}
    (directory/'extension_analysis.json').write_text(json.dumps(value))
    return directory, value


def test_completed_extension_requires_unchanged_source_provenance(tmp_path):
    directory, _ = completed_extension(tmp_path)
    assert load_extension(tmp_path)['available'] is True
    (directory/'events.jsonl').write_text('changed')
    result = load_extension(tmp_path)
    assert result['available'] is False and result['trades'] == []


@pytest.mark.parametrize('failure', ['future_exit', 'prior_entry', 'duplicate', 'missing_entry'])
def test_completed_extension_rejects_cross_segment_or_duplicate_trades(tmp_path, failure):
    directory, value = completed_extension(tmp_path)
    if failure == 'future_exit': value['trades'][0]['exit_timestamp_utc'] = END.isoformat()
    if failure == 'prior_entry': value['trades'][0]['entry_timestamp_utc'] = '2026-06-19T14:00:00Z'
    if failure == 'duplicate': value['trades'].append(dict(value['trades'][0]))
    if failure == 'missing_entry': value['trades'][0]['entry_timestamp_utc'] = None
    (directory/'extension_analysis.json').write_text(json.dumps(value))
    assert load_extension(tmp_path)['available'] is False


def test_read_only_api_reports_extension_scope_without_forward_account(tmp_path):
    from threading import Thread
    from urllib.request import Request, urlopen
    from src.paper.monitoring_api import create_monitoring_server
    (tmp_path/'historical_extension.json').write_text(json.dumps({
        'classification': 'HISTORICAL_CAUSAL_SIMULATION',
        'start_utc_inclusive': START.isoformat(), 'end_utc_exclusive': END.isoformat()}))
    (tmp_path/'status.json').write_text(json.dumps({'mode': 'PAPER', 'system': {
        'state': 'STOPPED', 'last_bar': '2026-10-07T23:59:00Z'}}))
    token = 'isolated-extension-contract-token'
    server = create_monitoring_server(tmp_path, token=token, port=0)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(f'http://127.0.0.1:{server.server_port}/v1/dashboard',
                          headers={'Authorization': f'Bearer {token}'})
        with urlopen(request, timeout=5) as response:
            value = json.load(response)['data']
        assert value['run']['kind'] == 'HISTORICAL_SIMULATION_EXTENSION'
        assert value['run']['status'] == 'COMPLETED'
        assert value['run']['end_utc_exclusive'] == END.isoformat()
        assert value['account'] is None and value['equity_curve'] == []
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=2)
