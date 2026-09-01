# Mean Reversion Research

## Objective

Investigate whether statistically exploitable mean-reversion behavior
exists in the NQ/MNQ 1-minute dataset.

## Research principle

No strategy parameters are assumed in advance.

The research process is:

1. Feature construction
2. Statistical analysis
3. Edge discovery
4. Strategy specification
5. Backtest
6. Failure analysis
7. Robustness testing
8. Walk-forward / OOS validation
9. Freeze
10. Modular reproduction

## Candidate phenomena

- VWAP deviation
- Z-score
- Ornstein-Uhlenbeck dynamics
- Price displacement
- Return autocorrelation
- Volatility regimes
- Intraday structure
- Session effects

## Important

Features are research variables first.
They are not automatically strategy signals.

No parameter should be frozen before the corresponding
research and validation stage.
