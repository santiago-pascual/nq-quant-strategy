# NQ Quant Strategy Project

Quantitative research repository for developing a modular, regime-aware strategy portfolio for Nasdaq futures, with initial emphasis on the regular trading hours (RTH) session and research centered on MNQ/NQ intraday data.

## Objective

The current objective is to build a research pipeline in which:

1. Market regime is identified from observable features.
2. Strategies are evaluated only in the regimes where they may be applicable.
3. Only strategies with positive out-of-sample evidence in a given regime are considered for activation there.
4. Strategy outputs are intended to feed a shared execution, risk, and account-validation framework.

This repository reflects the research architecture and current development stage of that process. It does not represent a completed production trading system.

## Current Status

- `S2` research has been extensively developed and is currently treated as frozen.
- Active `S2` research scripts remain under `src/research/`.
- Generated `S2` research outputs are organized under `src/research/results/s2/`.
- A preserved backup of the active `S2` research scripts exists under `src/research/backup_s2/`.
- The reusable strategy library has been started under `src/strategies/`.
- The current priority is completing the RTH strategy portfolio before expanding to additional sessions.

At this stage:

- `S2` research: frozen
- `S2` results organization: complete
- Strategy library: started
- RTH portfolio: not complete
- Final system validation: not started
- Final funded-account validation: not started
- Live deployment: not ready

## Research Architecture

The project separates research code from reusable strategy components.

- `src/research/` contains exploratory research, validation scripts, and generated research outputs.
- `src/strategies/` is intended to hold modular strategy implementations behind a common interface.
- `src/models/` contains reusable modeling components such as the volatility regime model.
- `tests/` contains automated tests for core data, feature, session, target, and regime behavior.

The current architecture is designed around a regime-aware portfolio concept rather than running every strategy continuously across all market conditions.

Conceptually:

`regime -> eligible strategies -> strategy selection -> execution -> risk management -> account model`

A strategy may be active only in the regimes where out-of-sample evidence supports trading, and inactive elsewhere.

## S2 Research Context

`S2` is a selective Nasdaq futures research line evaluated across volatility regimes. The repository structure and research outputs indicate that this work progressed through multiple stages including:

- exploratory robustness and feature-related analysis
- regime routing and selective execution
- final validation
- statistical validation
- funded-account simulation variants
- architecture comparison

The central research principle is that a strategy should not be judged only by aggregate results across all market conditions if the strategy is intended to operate only in a subset of regimes.

Accordingly, the research process distinguishes between:

- strategy discovery
- regime-specific evaluation
- walk-forward and out-of-sample validation
- strategy and regime selection
- portfolio construction
- final system-level validation

`S2` is currently treated as a completed research component rather than an actively optimized one, except in the case of a genuine implementation or data issue.

## Validation Philosophy

The repository reflects a progressively stricter validation approach:

1. Hypothesis formation
2. Strategy implementation
3. Unit and invariant testing
4. Historical backtesting
5. Regime analysis
6. Walk-forward validation
7. Out-of-sample evaluation
8. Strategy and regime selection
9. Portfolio and system construction
10. Final statistical validation
11. Funded-account simulation
12. Only later, potential live deployment

Statistical tools already present in the research workflow include or support:

- bootstrap analysis
- block bootstrap analysis
- Monte Carlo path simulation
- drawdown analysis
- losing-streak analysis
- monthly performance analysis
- walk-forward and out-of-sample evaluation

An important constraint in this project is that validation of an individual strategy is not automatically treated as validation of the final portfolio. Final statistical assessment is intended to be performed on the complete regime-aware system after portfolio construction.

## Implemented Core Components

The repository already includes several reusable core building blocks:

- dataset loading and validation in `src/data_loader.py` and `src/data_validator.py`
- session labeling in `src/session_engine.py`
- return, volatility, and target generation in `src/feature_engine.py` and `src/targets.py`
- a volatility regime model in `src/models/regime.py`
- an initial strategy package scaffold in `src/strategies/`

The current dataset loaded by the project is:

- `data/Dataset_NQ_1min_2022_2025.csv`

Tests currently present cover:

- data loading
- feature engineering
- session logic
- target construction
- regime-model behavior

## Strategy Library Direction

The strategy library is intended to hold independent, deterministic, modular strategy implementations that can be evaluated consistently.

Future strategy families may include:

- mean reversion
- volatility expansion
- momentum
- breakout
- trend following
- liquidity or sweep-based ideas
- other statistically testable short-term Nasdaq strategies

These should be understood as research directions, not as claims that the strategies are already implemented or validated in this repository.

Each strategy is intended to eventually:

- expose a clearly defined signal model
- behave deterministically
- integrate with shared execution conventions
- produce comparable trade-level outputs
- be tested independently
- be evaluated across regimes
- be evaluated out of sample before portfolio inclusion

## Market Scope

Current scope:

- Nasdaq futures research
- primary focus on MNQ/NQ intraday data
- initial emphasis on the RTH trading window

Potential future research directions:

- New York afternoon
- London session
- Asia session
- cross-session information
- broader 24-hour market structure

These are future directions only and should not be interpreted as current system coverage.

## Funded-Account Research

Funded-account simulation is treated as a separate layer from strategy discovery. The repository contains `S2` funded-account research scripts, but those should not be interpreted as final conclusions about the complete portfolio.

The broader intended account-level research includes topics such as:

- evaluation or challenge-stage constraints
- funded-account constraints
- position and risk sizing
- drawdown limits
- profit targets
- payout timing and rules
- withdrawal policy
- evaluation or subscription costs
- time to pass
- pass and failure probabilities
- expected payouts
- account survival

Complete funded-account validation is intended to follow completion of the broader RTH strategy portfolio.

## Design Principles

- No look-ahead bias
- No leakage between training and out-of-sample data
- Frozen parameters during final validation
- No accidental optimization during validation
- Separation of research code from reusable strategy code
- Modular implementations
- Reproducible experiments
- Explicit execution costs
- Regime-aware evaluation
- Preference for out-of-sample evidence over in-sample fit
- Validation of the complete portfolio, not only isolated components
- Preference for robustness over maximum backtest performance

## Repository Structure

```text
src/
├── strategies/
│   ├── __init__.py
│   └── base.py
├── models/
│   └── regime.py
└── research/
    ├── results/
    │   └── s2/
    ├── backup_s2/
    └── s2_*.py

data/
tests/
```

Briefly:

- `src/strategies/` contains reusable strategy components and interfaces.
- `src/research/` contains experiments, validation scripts, and research workflows.
- `src/research/results/` contains generated research outputs.
- `src/research/backup_s2/` contains preserved pre-refactor copies of active `S2` research scripts.
- `tests/` contains automated test coverage for core infrastructure.

## Roadmap

### Phase 1 - S2

- research `S2`
- identify where `S2` is effective
- freeze `S2`
- organize research outputs

Status: done / frozen

### Phase 2 - Strategy Library

- establish a common strategy interface
- migrate `S2` into the modular architecture without changing behavior
- implement additional candidate strategy families
- test each strategy independently

Status: in progress

### Phase 3 - Regime Coverage

- evaluate each candidate across volatility regimes
- identify uncovered or weak regime areas
- prioritize new strategy research for those gaps
- avoid forcing strategies into regimes where evidence is weak

### Phase 4 - RTH Portfolio

- combine validated strategies
- establish regime-aware strategy selection
- establish common execution and risk handling
- validate the complete RTH system

### Phase 5 - Final Validation

- out-of-sample validation of the complete system
- bootstrap and block bootstrap
- Monte Carlo analysis
- drawdown and robustness analysis
- funded-account simulation

### Phase 6 - Expansion

Only after the RTH system is robust:

- New York afternoon
- London
- Asia
- cross-session information
- broader market coverage

## Disclaimer

This repository is a quantitative research project. Historical or backtested results do not guarantee future performance.

## License

Copyright © 2026 Santiago Pascual. All Rights Reserved.

This repository is publicly available for viewing, educational, and
research purposes only.

Reproduction, redistribution, commercial use, or derivative works
require prior written permission from the author.

See [LICENSE](LICENSE) for the complete terms.

