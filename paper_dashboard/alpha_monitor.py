"""Predeclared descriptive monitoring looks; no trading or significance claims."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import random
from paper_dashboard.research_analytics import summarize_r


@dataclass(frozen=True)
class MonitoringPlan:
    trades_per_look: int = 25
    maximum_looks: int = 10
    groups: int = 5
    familywise_alpha: float = .05
    block_length: int = 5
    seed: int = 20261010

    def identity(self):
        if self.trades_per_look < 2 or self.maximum_looks < 1 or self.groups < 1 or not 0 < self.familywise_alpha < 1 or not 1 <= self.block_length <= self.trades_per_look:
            raise ValueError('invalid monitoring plan')
        return sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def drawdown(values):
    equity = peak = maximum = 0.0
    for value in values:
        equity += value; peak = max(peak, equity); maximum = min(maximum, equity-peak)
    return maximum


def reference_distribution(values, *, plan: MonitoringPlan, draws: int = 2000):
    """Moving blocks preserve short-range order; approximate, conditional bands.

    Frozen strategy selection, nonstationarity and dependence beyond block size
    prevent distribution-free coverage guarantees. No p-values are asserted.
    """
    identity = plan.identity()
    data = [float(v) for v in values]
    if not all(math.isfinite(v) for v in data): raise ValueError('invalid reference')
    if len(data) < plan.trades_per_look or draws < 100:
        return {'available': False, 'state': 'INSUFFICIENT_DATA'}
    rng = random.Random(plan.seed); samples = []
    for _ in range(draws):
        sequence = []
        while len(sequence) < plan.trades_per_look:
            start = rng.randrange(len(data)-plan.block_length+1)
            sequence.extend(data[start:start+plan.block_length])
        sequence = sequence[:plan.trades_per_look]
        metrics = summarize_r(sequence); metrics['drawdown_r'] = drawdown(sequence)
        samples.append(metrics)
    # Multiplicity budget is disclosed, not misrepresented as proven calibration.
    nominal = plan.familywise_alpha/(plan.groups*plan.maximum_looks*4)
    bands = {}
    for key in ('expectancy_r','win_rate','profit_factor','drawdown_r'):
        ordered = sorted(s[key] for s in samples if s[key] is not None and math.isfinite(s[key]))
        if not ordered:
            bands[key] = {'available': False}; continue
        bands[key] = {'available': True, 'lower': ordered[int((len(ordered)-1)*nominal/2)],
                      'upper': ordered[int((len(ordered)-1)*(1-nominal/2))],
                      'finite_draws': len(ordered),
                      'tail_resolution_sufficient': len(ordered)*nominal/2 >= 10}
    return {'available': True, 'plan_id': identity, 'bands': bands,
            'nominal_per_metric_look_alpha': nominal, 'coverage_guaranteed': False,
            'inference': 'DESCRIPTIVE_ONLY_UNCALIBRATED_TAILS', 'block_length': plan.block_length}


def completed_looks(values, *, plan: MonitoringPlan = MonitoringPlan()):
    """Only disjoint completed looks; later trades cannot revise an earlier look."""
    identity = plan.identity(); data = [float(v) for v in values]
    if not all(math.isfinite(v) for v in data): raise ValueError('invalid forward outcome')
    count = min(len(data)//plan.trades_per_look, plan.maximum_looks)
    rows = []
    for look in range(count):
        sample = data[look*plan.trades_per_look:(look+1)*plan.trades_per_look]
        rows.append({'look': look+1, 'trades_start': look*plan.trades_per_look,
                     'trades_end_exclusive': (look+1)*plan.trades_per_look,
                     **summarize_r(sample), 'drawdown_r': drawdown(sample)})
    return {'plan_id': identity, 'state': 'INSUFFICIENT_DATA' if not rows else 'DESCRIPTIVE_LOOKS_AVAILABLE',
            'looks': rows, 'maximum_looks_reached': len(data)//plan.trades_per_look >= plan.maximum_looks,
            'trading_actions': [], 'formal_significance_enabled': False,
            'frequency_status': 'UNAVAILABLE_WITHOUT_COMPARABLE_SESSION_EXPOSURE',
            'safeguards': ['Fixed disjoint trade windows', 'Finite look budget across five groups',
                          'No optional-stopping significance claim', 'No automatic execution changes']}
