"""Descriptive forward comparisons in risk units; no execution decisions."""
from __future__ import annotations

from typing import Any, Mapping, Iterable
from paper_dashboard.alpha_monitor import completed_looks

from paper_dashboard.research_analytics import (STRATEGIES, moving_block_expectancy_band,
    reconstruct_paper_outcomes, rolling_metrics, summarize_r)


def forward_comparison(research: Iterable[Mapping[str, Any]], paper: Iterable[Mapping[str, Any]],
                       *, window: int = 25) -> dict[str, Any]:
    reference = list(research)
    reconstructed = reconstruct_paper_outcomes(paper)
    if not reconstructed['available'] or reconstructed.get('invalid_rows'):
        return {'available': False, 'state': 'UNAVAILABLE',
                'reason': reconstructed.get('reason') or 'Invalid Paper outcomes; comparison withheld'}
    results = {}
    for name in ('PORTFOLIO', *STRATEGIES):
        historical = [row for row in reference if name == 'PORTFOLIO' or row.get('strategy') == name]
        forward = [row for row in reconstructed['trades'] if name == 'PORTFOLIO' or row.get('strategy') == name]
        reference_values = [row['r_multiple'] for row in historical]
        observed = [row['realized_r'] for row in forward[-window:]]
        band = moving_block_expectancy_band(reference_values, window=window,
                                             block_length=max(2, min(10, window // 5)))
        metrics = summarize_r(observed)
        state = 'INSUFFICIENT_DATA'
        if len(observed) >= window and band.get('available'):
            state = 'WATCH' if metrics['expectancy_r'] < band['lower'] else 'WITHIN_EXPECTATIONS'
        results[name] = {'state': state, 'forward_sample_size': len(forward),
                         'evaluation_sample_size': len(observed), 'required_window': window,
                         'forward_metrics': metrics, 'research_reference': band,
                         'sequential_safeguards': completed_looks([row['realized_r'] for row in forward]),
                         'rolling': rolling_metrics(forward, value_key='realized_r',
                                                    time_key='exit_timestamp_utc', window=window)}
    return {'available': True, 'groups': results,
            'units': 'Research stored R versus Paper stored net R; execution costs differ',
            'statistically_concerning_enabled': False,
            'limitations': ['Descriptive bands are not calibrated sequential hypothesis tests.',
                           'Repeated windows and five groups are dependent; no unadjusted significance claim.',
                           'Research selection bias and regime nonstationarity remain.',
                           'Insufficient samples cannot diagnose alpha decay.',
                           'No trading action or strategy retraining is triggered.']}
