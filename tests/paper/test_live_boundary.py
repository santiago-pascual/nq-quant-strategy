import pytest

from src.paper.live_boundary import compare_position_snapshots, live_readiness, require_live_submission_authorization


def test_live_submission_is_unconditionally_disabled(monkeypatch):
    monkeypatch.setenv('LIVE_TRADING_ENABLED', 'true')
    with pytest.raises(PermissionError):
        require_live_submission_authorization()
    assert live_readiness()['live_execution_enabled'] is False


def test_missing_broker_snapshot_is_not_healthy_or_flat():
    assert compare_position_snapshots([], None)['state'] == 'UNAVAILABLE'


def test_signed_position_mismatch_does_not_execute_any_action():
    result = compare_position_snapshots([{'contract_id': 815824267, 'signed_quantity': 1}],
                                        [{'contract_id': 815824267, 'signed_quantity': -1}])
    assert result['state'] == 'MISMATCH'
    assert result['actions_taken'] == []
    assert result['execution_enabled'] is False


def test_duplicate_broker_rows_and_missing_identity_are_unavailable():
    position = {'contract_id': 815824267, 'signed_quantity': 1}
    assert compare_position_snapshots([position], [position, position])['state'] == 'UNAVAILABLE'
    assert compare_position_snapshots([], [{'signed_quantity': 1}])['state'] == 'UNAVAILABLE'
    assert compare_position_snapshots([position], [position])['state'] == 'MATCH'


def test_independent_emergency_plan_never_enables_broker_execution():
    from src.paper.live_boundary import protective_response
    for evidence in (True,False,None):
        result = protective_response(positions_reconciled=evidence,orders_reconciled=True,
                                     independent_limits_healthy=True)
        assert not result['new_submission_allowed'] and result['actions_taken']==[]
    assert protective_response(positions_reconciled=None,orders_reconciled=True,
                independent_limits_healthy=True)['state']=='BLOCKED_UNVERIFIED'


def test_future_intent_key_is_idempotent_without_submission():
    from src.paper.live_boundary import future_order_intent_key
    intent = dict(account_id='isolated-fixture',strategy='ORB',contract_id=1,
                  market_timestamp_utc='2026-10-09T14:00:00Z',action='ENTER',quantity=1)
    first=future_order_intent_key(intent)
    assert first==future_order_intent_key({**intent,'audit_timestamp':'ignored'})
    assert first!=future_order_intent_key({**intent,'quantity':2})
