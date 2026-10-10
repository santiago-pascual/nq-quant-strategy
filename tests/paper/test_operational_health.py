from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import pytest

from src.paper.operational_health import evaluate, inspect, journal_tail, save_report

NOW = datetime(2026, 10, 9, 19, tzinfo=timezone.utc)


def status(**feed):
    return {'system': {'state': 'RUNNING', 'last_bar': '2026-10-09T18:45:00Z',
                       'bars_processed': 10, 'feed_health': {
                           'provider_connected': True, 'connection_state': 'CONNECTED',
                           'latest_available_bar': '2026-10-09T18:45:00Z',
                           'mode': 'CAUGHT_UP', 'backlog_bars': 0, **feed}}}


@pytest.mark.parametrize('market', ['MARKET_CLOSED', 'MARKET_BREAK', 'MARKET_UNKNOWN'])
def test_closure_and_unknown_are_never_a_feed_stall(market):
    report = evaluate(status(connection_state='DISCONNECTED'), {'state': market}, now=NOW,
                      status_age_seconds=900)
    assert report['state'] == market
    assert report['real_session_validation'] == 'PENDING'
    assert report['evidence']['connection_state'] == 'DISCONNECTED'


def test_only_measured_commit_progress_is_session_progress():
    initial = evaluate(status(), {'state': 'MARKET_OPEN'}, now=NOW)
    assert initial['state'] == 'WAITING_FOR_DELAYED_DATA'
    assert initial['real_session_validation'] == 'PENDING'
    previous = {'evidence': {'last_committed_bar': '2026-10-09T18:44:00Z',
                             'provider_frontier': '2026-10-09T18:44:00Z'}}
    report = evaluate(status(), {'state': 'MARKET_OPEN'}, now=NOW, previous=previous)
    assert report['state'] == 'CONNECTED'
    assert report['real_session_validation'] == 'OBSERVED_COMMIT_PROGRESSION'


@pytest.mark.parametrize('changes,expected', [({'mode': 'RECOVERING'}, 'RECOVERING'),
                                             ({'connection_state': 'RECONNECTING'}, 'RECONNECTING')])
def test_recovery_and_disconnect_are_explicit(changes, expected):
    assert evaluate(status(**changes), {'state': 'MARKET_OPEN'}, now=NOW)['state'] == expected


def test_cursor_regression_overrides_weekend_and_connected_socket():
    report = evaluate(status(), {'state': 'MARKET_CLOSED'}, now=NOW,
                      previous={'evidence': {'last_committed_bar': '2026-10-09T18:46:00Z'}})
    assert report['state'] == 'ERROR'
    assert 'committed_cursor_regressed' in report['integrity_findings']


def test_missing_status_is_unavailable():
    assert evaluate(None, {'state': 'MARKET_CLOSED'}, now=NOW)['state'] == 'UNAVAILABLE'


def test_tail_partial_write_and_duplicate_detection_without_production_mutation():
    scratch = Path(__file__).resolve().parents[2] / '.test_scratch'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        run = Path(temp) / 'run'; run.mkdir()
        (run / 'status.json').write_text(json.dumps(status()), encoding='utf-8')
        row = {'event_id': 'same', 'event_type': 'market_data',
               'payload': {'timestamp': '2026-10-09T18:45:00Z'}}
        raw = (json.dumps(row)+'\n')*2 + '{"unfinished":'
        events = run / 'events.jsonl'; events.write_text(raw, encoding='utf-8')
        sample = journal_tail(events)
        assert sample['partial_append'] is True
        assert sample['invalid_complete_records'] == 0
        report = inspect(run, None, now=NOW)
        assert report['state'] == 'ERROR'
        assert report['event_evidence']['duplicate_event_ids'] == 1
        assert events.read_text() == raw
        with pytest.raises(ValueError, match='outside'):
            save_report(run / 'report.json', report, run)
        save_report(Path(temp) / 'report.json', report, run)
        assert json.loads((Path(temp) / 'report.json').read_text()) == report


def test_bounded_tail_does_not_claim_full_journal_validation():
    scratch = Path(__file__).resolve().parents[2] / '.test_scratch'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        events = Path(temp) / 'events.jsonl'
        events.write_text((' {"event_id":"a"}\n')*100)
        sample = journal_tail(events, maximum_bytes=100)
        assert sample['complete_journal'] is False
        assert sample['bytes_read'] <= 100


def test_unknown_gap_is_review_required_not_healthy_or_proven_loss(tmp_path):
    (tmp_path/'status.json').write_text(json.dumps(status()))
    rows = [{'event_id':str(i),'event_type':'market_data','timestamp':stamp,
             'payload':{'timestamp':stamp}} for i,stamp in enumerate(
                 ('2026-10-09T14:00:00Z','2026-10-09T14:03:00Z'))]
    (tmp_path/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report=inspect(tmp_path,None,now=NOW)
    assert report['state']=='REQUIRES_REVIEW'
    assert report['processing_validation']['gaps'][0]['state']=='CALENDAR_UNKNOWN'
    assert not report['processing_validation']['source_completeness_certified']


def test_operational_api_is_authenticated_and_preserves_missing_state():
    from threading import Thread
    from urllib.request import Request, build_opener, ProxyHandler
    from urllib.error import HTTPError
    from src.paper.monitoring_api import create_monitoring_server
    scratch = Path(__file__).resolve().parents[2] / '.test_scratch'
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as temp:
        run = Path(temp)
        token = 'isolated-monitor-token-long-enough'
        server = create_monitoring_server(run, token=token, port=0)
        thread = Thread(target=server.serve_forever, daemon=True); thread.start()
        opener = build_opener(ProxyHandler({}))
        url = f'http://127.0.0.1:{server.server_port}/v1/operational-health'
        try:
            with pytest.raises(HTTPError) as error:
                opener.open(url)
            assert error.value.code == 401
            response = opener.open(Request(url, headers={'Authorization': f'Bearer {token}'}))
            body = json.load(response)['data']
            assert body['state'] == 'UNAVAILABLE'
            assert body['account'] is None
            with pytest.raises(HTTPError) as error:
                opener.open(Request(url, method='POST', headers={'Authorization': f'Bearer {token}'}))
            assert error.value.code == 405
            assert list(run.iterdir()) == []
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)
