"""Future read-only broker reconciliation; live execution is unconditionally off.

No IBKR transport is imported. Connecting a real account, cancellation and
liquidation require a separately authorized implementation and validation.
"""
from __future__ import annotations

from typing import Mapping, Sequence, Any
import hashlib
import json


LIVE_EXECUTION_ENABLED = False


def protective_response(*, positions_reconciled: bool | None,
                        orders_reconciled: bool | None, independent_limits_healthy: bool | None,
                        operator_stop_requested: bool = False) -> dict:
    """Design-only independent emergency response, never broker commands.

    A future implementation must stop new submissions before bounded cancellation
    and broker reconciliation. Liquidation is a separate authorized policy, not
    an automatic action inferred here. Unknown evidence is blocking.
    """
    evidence = (positions_reconciled, orders_reconciled, independent_limits_healthy)
    if operator_stop_requested or any(item is False for item in evidence):
        state = 'EMERGENCY_STOP_RECOMMENDED'
    elif any(item is not True for item in evidence):
        state = 'BLOCKED_UNVERIFIED'
    else:
        state = 'DESIGN_CHECKS_PASSED_EXECUTION_DISABLED'
    return {'state':state,'live_execution_enabled':False,'actions_taken':[],
            'new_submission_allowed':False,'manual_intervention_required':state!='DESIGN_CHECKS_PASSED_EXECUTION_DISABLED',
            'future_procedure':['Stop new submissions','Persist incident and intent identifiers',
                                'Reconcile independent broker orders, fills and positions',
                                'Apply separately approved bounded cancellation policy',
                                'Require explicit recovery authorization; no silent restart']}


def future_order_intent_key(intent: Mapping[str, Any]) -> str:
    """Stable diagnostic intent identity; it cannot submit an order."""
    required = ('account_id','strategy','contract_id','market_timestamp_utc','action','quantity')
    if any(intent.get(key) is None for key in required):
        raise ValueError('complete intent identity required')
    quantity = intent['quantity']
    if not isinstance(quantity,int) or isinstance(quantity,bool) or quantity <= 0:
        raise ValueError('positive integral intent quantity required')
    if intent['action'] not in {'ENTER','EXIT'}:
        raise ValueError('unsupported intent action')
    canonical = {key:intent[key] for key in required}
    return hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def require_live_submission_authorization() -> None:
    raise PermissionError('Live execution is not implemented or authorized; broker submission is disabled')


def compare_position_snapshots(expected: Sequence[Mapping[str, Any]] | None,
                               observed: Sequence[Mapping[str, Any]] | None) -> dict:
    """Compare explicit signed contract quantities, never invent a flat account.

    This is a read-only diagnostic for a future independently supplied broker
    snapshot. A missing/ambiguous identity or duplicate row is unavailable.
    """
    if expected is None or observed is None:
        return {'state': 'UNAVAILABLE', 'reason': 'Both independent snapshots are required'}
    def index(rows):
        result = {}
        for row in rows:
            key = str(row.get('contract_id') or '')
            quantity = row.get('signed_quantity')
            if not key or key in result or not isinstance(quantity, int) or isinstance(quantity, bool):
                raise ValueError('ambiguous contract/quantity snapshot')
            result[key] = quantity
        return result
    try:
        left, right = index(expected), index(observed)
    except (AttributeError, TypeError, ValueError):
        return {'state': 'UNAVAILABLE', 'reason': 'Invalid or duplicate position identities'}
    differences = [{'contract_id': key, 'expected_quantity': left.get(key, 0),
                    'observed_quantity': right.get(key, 0)} for key in sorted(left.keys() | right.keys())
                   if left.get(key, 0) != right.get(key, 0)]
    return {'state': 'MISMATCH' if differences else 'MATCH', 'differences': differences,
            'execution_enabled': False, 'actions_taken': []}


def live_readiness() -> dict:
    return {'live_execution_enabled': False, 'broker_order_submission': 'DISABLED',
            'broker_position_reconciliation': 'INTERFACE_ONLY_NO_CONNECTED_LIVE_ACCOUNT',
            'broker_kill_switch': 'DESIGN_ONLY_NO_BROKER_COMMANDS',
            'required_before_live': ['Separate explicit authorization', 'Verified live account identity',
                                    'Independent broker position/order/fill reconciliation',
                                    'Broker execution idempotency and order-state-machine validation',
                                    'Independent hard risk controls and tested emergency procedures',
                                    'Observed execution telemetry and operational failure testing']}
