"""Compare matched recorded inputs and decisions without changing either run."""
from __future__ import annotations

from collections import defaultdict
from src.paper.operational_health import _stamp


FIELDS = {
    'market_data': ('open','high','low','close','volume','market_data_contract_id','market_period'),
    'strategy_decision': ('strategy_name','action','signal','reason'),
    'hmm_state': ('stream','raw_state','posterior','model_hash','model_version'),
    'risk_decision': ('strategy_name','approved','reason','quantity'),
    'order_submitted': ('strategy_name','side','quantity','order_type','price'),
    'fill': ('strategy_name','side','quantity','price','commission'),
}


def compare_event_samples(left: list[dict], right: list[dict], *,
                          left_config: dict, right_config: dict) -> dict:
    def index(rows):
        result = defaultdict(list)
        for row in rows:
            kind = row.get('event_type'); stamp = _stamp(row.get('timestamp'))
            if kind not in FIELDS or stamp is None: continue
            payload = row.get('payload') or {}
            key = (stamp.isoformat(), kind, payload.get('strategy_name') or payload.get('stream') or 'SYSTEM')
            result[key].append({field: payload.get(field) for field in FIELDS[kind]})
        return result
    left_index, right_index = index(left), index(right)
    market_keys = {k for k in left_index if k[1]=='market_data'}
    other_market = {k for k in right_index if k[1]=='market_data'}
    if not market_keys or not other_market:
        return {'state': 'UNAVAILABLE', 'reason': 'Both samples require recorded market inputs', 'trading_actions': []}
    for sample, keys in ((left_index, market_keys), (right_index, other_market)):
        if any(len(sample[key]) != 1 for key in keys):
            return {'state': 'UNAVAILABLE', 'reason': 'Duplicate recorded market inputs', 'trading_actions': []}
        if any(sample[key][0].get(field) is None for key in keys for field in ('open','high','low','close','volume')):
            return {'state': 'UNAVAILABLE', 'reason': 'Incomplete recorded OHLCV inputs', 'trading_actions': []}
    same_input = market_keys == other_market and all(left_index[k] == right_index[k] and len(left_index[k])==1 for k in market_keys)
    same_config = left_config == right_config
    differences = []
    for key in sorted(left_index.keys() | right_index.keys()):
        if left_index.get(key) == right_index.get(key): continue
        category = ('INPUT_OR_SESSION_DIFFERENCE' if key[1]=='market_data' else
                    'HMM_INFERENCE_OR_FIT_DIFFERENCE' if key[1]=='hmm_state' else
                    'CONFIGURATION_DIFFERENCE' if not same_config else
                    'UNEXPLAINED_MATCHED_INPUT_DIFFERENCE' if same_input else 'UNMATCHED_INPUTS')
        differences.append({'timestamp_utc': key[0], 'event_type': key[1], 'strategy_or_stream': key[2],
                            'category': category, 'left': left_index.get(key, []), 'right': right_index.get(key, [])})
    return {'state': 'MATCHED_SAMPLE_EQUIVALENT' if not differences and same_config else 'DIFFERENCES_RECORDED',
            'identical_recorded_inputs': same_input, 'identical_configuration': same_config,
            'matched_market_timestamps': len(market_keys & other_market), 'differences': differences,
            'unavailable_fields': sorted({f'{key[1]}.{field}' for sample in (left_index,right_index)
                for key, rows in sample.items() for row in rows for field, value in row.items() if value is None}),
            'scope': 'Provided recorded sample only; no full-run or research parity claimed',
            'unknown_event_types_excluded': sorted({r.get('event_type','UNKNOWN') for r in left+right if r.get('event_type') not in FIELDS}),
            'limitations': ['HMM differences need explicit inference/fit attribution',
                           'Missing execution fields cannot establish fill or fee equivalence',
                           'Different market periods cannot establish strategy-level parity'],
            'trading_actions': []}
