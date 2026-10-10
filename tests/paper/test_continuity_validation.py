from datetime import datetime, timedelta, timezone
from pathlib import Path
from src.paper.continuity_validation import validate_bar_events
from src.paper.cme_calendar import CMETradingCalendar, CMECalendarSnapshot


def calendar():
    path = Path(__file__).resolve().parents[2] / 'src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json'
    return CMETradingCalendar(CMECalendarSnapshot.from_json(path))


def events(stamp, ordinal):
    def row(kind, payload):
        return {'event_id': f'{ordinal}-{kind}-{len(rows)}', 'event_type': kind,
                'timestamp': stamp, 'payload': payload}
    rows = []
    rows.append(row('market_data', {'timestamp': stamp}))
    for strategy in ('MRL1','MRS2','S2R','ORB'):
        rows.append(row('strategy_decision', {'strategy_name': strategy}))
    for stream in ('MR','S2R'):
        rows.append(row('hmm_state', {'stream': stream,'posterior':[.1,.2,.7],'raw_state':2,'model_hash':'fitted'}))
    rows.append(row('checkpoint_saved', {'last_processed_bar': stamp}))
    return rows


def test_recorded_bar_chain_and_open_minute_uncertainty():
    rows = events('2026-10-09T14:00:00Z',0) + events('2026-10-09T14:01:00Z',1)
    assert validate_bar_events(rows, calendar(), complete_tail=True)['state'] == 'VALID_OBSERVED_PROCESSING'
    rows += events('2026-10-09T14:03:00Z',2)
    result = validate_bar_events(rows, calendar(), complete_tail=True)
    assert result['state'] == 'UNRESOLVED'
    assert result['gaps'][0]['absent_expected_minutes'] == 1
    assert not result['source_completeness_certified']


def test_weekend_verified_closure_and_unknown_calendar():
    rows = events('2026-10-09T20:59:00Z',0) + events('2026-10-11T22:00:00Z',1)
    result = validate_bar_events(rows, calendar(), complete_tail=True)
    assert result['gaps'][0]['state'] == 'VERIFIED_CLOSURE'
    assert validate_bar_events(rows, None)['state'] == 'UNRESOLVED'


def test_duplicate_and_missing_strategy_hmm_checkpoint_are_errors():
    rows = events('2026-10-09T14:00:00Z',0)
    rows = [r for r in rows if r['event_type'] != 'checkpoint_saved' and r['payload'].get('strategy_name') != 'ORB']
    rows[0]['event_id'] = rows[1]['event_id']
    next(r for r in rows if r['event_type']=='hmm_state')['payload']['posterior'] = [1,1,1]
    issues = validate_bar_events(rows, calendar(), complete_tail=True)['issues']
    assert {'duplicate_event_id','strategy_processing_incomplete_or_duplicate','hmm_posterior_invalid','durable_checkpoint_event_unavailable'} <= set(issues)


def test_partial_edge_group_does_not_fabricate_failure():
    rows = events('2026-10-09T14:00:00Z',0)[:1]
    assert not validate_bar_events(rows, calendar())['issues']
    assert validate_bar_events(rows, calendar())['state'] == 'PARTIAL_SAMPLE'
    assert validate_bar_events([], calendar())['state'] == 'UNAVAILABLE'
    assert validate_bar_events(rows, calendar(), complete_tail=True)['issues']


def test_missing_hmm_identity_is_unavailable_not_a_fabricated_model_error():
    rows = events('2026-10-09T14:00:00Z',0)
    next(r for r in rows if r['event_type']=='hmm_state')['payload']['model_hash'] = None
    result = validate_bar_events(rows, calendar(), complete_tail=True)
    assert not result['issues']
    assert result['unavailable_checks'] == ['per_bar_hmm_model_identity']


def test_boolean_is_not_valid_hmm_state_or_probability():
    rows = events('2026-10-09T14:00:00Z',0)
    state = next(r for r in rows if r['event_type']=='hmm_state')['payload']
    state.update(raw_state=True, posterior=[True,0,0])
    result=validate_bar_events(rows,calendar(),complete_tail=True)
    assert {'hmm_state_invalid','hmm_posterior_invalid'} <= set(result['issues'])
