"""Observed-bar processing validation; absent trade bars are not fabricated."""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta
import math
from src.paper.operational_health import _stamp


STRATEGIES = {'MRL1', 'MRS2', 'S2R', 'ORB'}


def validate_bar_events(rows: list[dict], calendar, *, complete_tail: bool = False) -> dict:
    """Audit complete recorded bar groups; skip partially retained edge groups.

    No claim of source completeness follows from OHLCV sparsity. Missing open
    minutes are unresolved until independent coverage evidence classifies them.
    """
    groups = defaultdict(list); ids = set(); issues = []; last = None; gaps = []; unavailable = []; inspected = 0
    for row in rows:
        event_id = row.get('event_id')
        if event_id:
            if event_id in ids: issues.append('duplicate_event_id')
            ids.add(event_id)
        stamp = _stamp(row.get('timestamp'))
        if stamp: groups[stamp].append(row)
    markets = [(stamp, [r for r in group if r.get('event_type') == 'market_data'])
               for stamp, group in groups.items()]
    markets = [(stamp, values) for stamp, values in markets if values]
    for index, (stamp, values) in enumerate(markets):
        if len(values) != 1: issues.append('duplicate_market_bar')
        if last and stamp <= last: issues.append('nonmonotonic_market_bars')
        if last and stamp > last + timedelta(minutes=1):
            # Bound a diagnostic request. Long outages need the existing backfill
            # validator, not an unbounded monitoring scan.
            if (stamp-last).total_seconds() > 7*86400:
                gaps.append({'start': last.isoformat(), 'end': stamp.isoformat(), 'state': 'BOUNDED_SCAN_REQUIRED'})
            elif calendar is None:
                gaps.append({'start': last.isoformat(), 'end': stamp.isoformat(), 'state': 'CALENDAR_UNKNOWN'})
            else:
                try:
                    expected = calendar.expected_missing_minutes(last, stamp)
                    gaps.append({'start': last.isoformat(), 'end': stamp.isoformat(),
                                 'state': 'UNRESOLVED_OPEN_MINUTES' if expected else 'VERIFIED_CLOSURE',
                                 'absent_expected_minutes': len(expected)})
                except (ValueError, RuntimeError):
                    gaps.append({'start': last.isoformat(), 'end': stamp.isoformat(), 'state': 'CALENDAR_UNKNOWN'})
        last = stamp
        # Last group may still be being appended, first may have been truncated.
        if not complete_tail and index in {0, len(markets)-1}: continue
        inspected += 1
        group = groups[stamp]; payload = values[0].get('payload') or {}
        decisions = [(r.get('payload') or {}).get('strategy_name') for r in group if r.get('event_type') == 'strategy_decision']
        if set(decisions) != STRATEGIES or len(decisions) != 4: issues.append('strategy_processing_incomplete_or_duplicate')
        for stream in ('MR', 'S2R'):
            states = [r.get('payload') or {} for r in group if r.get('event_type') == 'hmm_state' and (r.get('payload') or {}).get('stream') == stream]
            if len(states) != 1: issues.append('hmm_processing_incomplete_or_duplicate'); continue
            state = states[0]; posterior = state.get('posterior')
            if isinstance(state.get('raw_state'),bool) or state.get('raw_state') not in (0,1,2):
                issues.append('hmm_state_invalid')
            if not state.get('model_hash'):
                unavailable.append('per_bar_hmm_model_identity')
            if (not isinstance(posterior,list) or len(posterior)!=3 or
                any(isinstance(p,bool) or not isinstance(p,(float,int)) or not math.isfinite(p) or p<0 for p in posterior) or
                abs(sum(posterior)-1)>1e-8): issues.append('hmm_posterior_invalid')
        saved = [r for r in group if r.get('event_type') == 'checkpoint_saved' and _stamp((r.get('payload') or {}).get('last_processed_bar')) == stamp]
        if not saved: issues.append('durable_checkpoint_event_unavailable')
        observed = _stamp(payload.get('market_data_first_observed_at_utc'))
        finalized = _stamp(payload.get('market_data_finalized_at_utc'))
        if observed and finalized and finalized < observed: issues.append('finalization_precedes_observation')
    return {'state': 'ERROR' if issues else 'UNRESOLVED' if any(g['state'] != 'VERIFIED_CLOSURE' for g in gaps) else
                'VALID_OBSERVED_PROCESSING' if inspected else 'PARTIAL_SAMPLE' if markets else 'UNAVAILABLE',
            'issues': sorted(set(issues)), 'observed_bar_groups': len(markets), 'gaps': gaps,
            'fully_inspected_bar_groups': inspected,
            'source_completeness_certified': False, 'partial_edge_groups_excluded': not complete_tail,
            'unavailable_checks': sorted(set(unavailable))}
