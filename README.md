# NQ Quant Strategy Project

This repository is a quantitative research codebase for Nasdaq index futures, centered on the MNQ 1-minute market and a modular, regime-aware mean-reversion portfolio. It is not a completed production trading system.

## Current state

The current validated portfolio is a three-strategy mean-reversion system built around the canonical Databento MNQ market feed and the modular research pipeline:

- MRL1
- S2R
- MRS2

These are the current validated strategies for the repository, and they are documented below as the authoritative portfolio state. The old `S2`-only and legacy dataset references in earlier documentation are no longer the primary state of the project.

## Canonical market data and migration

The current canonical market dataset is the Databento MNQ 1-minute OHLVC feed loaded through:

- `src.databento_loader.load_databento_mnq()`

The raw data lives under:

- `data/raw/mnq/ohlcv_1m/`

The current canonical dataset coverage is:

- 2,577,661 rows
- start: 2019-05-05 18:03 -04:00
- end: 2026-08-26 19:59 -04:00

The repository has migrated from the older, narrower research dataset workflow to this full Databento-based pipeline. The file `data/Dataset_NQ_1min_2022_2025.csv` is a legacy dataset artifact and should not be treated as the current canonical validator for the modular pipeline. The current research code and validation scripts use `load_databento_mnq()` and `src.session_engine` rather than the older market engine workflow.

## Repository architecture

The project currently contains the following relevant areas:

- `src/` — core market, feature, session, risk, and strategy infrastructure
- `src/research/` — research scripts, validation workflows, and generated outputs
- `src/research/mean_reversion/` — current mean-reversion research lineage
- `src/research/results/s2_extended/` — authoritative S2R benchmark and robustness outputs
- `src/strategies/` — modular strategy library
- `src/models/` — model components and regime-related utilities
- `tests/` — project validation and core behavior tests
- `trade_visualizer.py` — current trade visualization tool for frozen strategy trade streams

## Current validated mean-reversion strategies

### MRL1

Direction: LONG

Signal conditions:

- HMM state 1
- VOL20–40
- Z <= -2.5

Execution:

- TP = 25 points
- SL = 37.5 points
- Horizon = 8 bars
- RR = 0.6667

Validated 08AA result:

- 483 trades
- win rate = 60.2484%
- net = +20.34R
- expectancy = +0.042112R
- profit factor = 1.196042
- max drawdown = -8.393333R

### S2R

S2R has multiple historical streams in the repository, but only one is authoritative for the current frozen benchmark.

The authoritative current stream is:

- `src/research/results/s2_extended/s2r_modular_authoritative_reproduction.csv`

This is the 537-trade frozen reproduction and should be treated as the canonical S2R validation stream. The 5231-row `s2r_modular_full_databento_trades.csv` file is a broad export stream and is not the validated benchmark. It must not be presented as the authoritative S2R result.

Current S2R rules:

- HMM state 2
- VOL40–60
- SHORT
- Quality >= 0.75
- SL = 25 points
- TP = 43.75 points
- Horizon = 20 bars
- Reward/risk = 1.75R
- MAE/recovery condition:
  - 0.70R MAE threshold
  - +0.20R recovery
  - deadline = 6 bars

Validated authoritative result for the frozen trade stream:

- 537 trades
- total = +23.2572R
- win rate = 45.4376%
- profit factor = 1.095879
- expectancy = +0.043309R
- max drawdown ≈ -16.7936R using the contribution-based calculation

### MRS2

Direction: SHORT

Signal conditions:

- HMM state 2
- VOL80–100
- Z >= +2.0

Execution:

- TP = 27.5 points
- SL = 25 points
- Horizon = 30 bars
- RR = 1.10

Validated 08AA result:

- 1,052 trades
- win rate = 51.616%
- net = +83.51R
- expectancy = +0.079382R
- profit factor = 1.170366
- max drawdown = -14.45R

## 08AA modular reproduction and validation audit

The current modular reproduction pipeline is:

- `src/research/mean_reversion/research/08aa_modular_reproduction.py`

This script replays the frozen mean-reversion candidates through the modular backtest and lifecycle architecture. It is the authoritative reproduction check for the MRL1 and MRS2 research pipeline and is the code path that verifies the research context before strategy execution is accepted.

Current audit facts from the modular pipeline:

- Research 07 events: 825,717
- HMM rows: 825,717
- Missing HMM: 0
- Missing z-score: 0
- Market rows: 2,577,661
- Valid realized_vol_30: 2,577,631
- RTH rows: 825,746
- Event timestamp == RTH timestamp: 825,717 / 825,717
- Event close == market close: 825,717 / 825,717
- Complete events: 825,717

Volatility bucket counts:

- VOL0–20: 163,633
- VOL20–40: 168,992
- VOL40–60: 170,842
- VOL60–80: 171,449
- VOL80–100: 150,801

Candidate events:

- MRS2: 2,253
- MRL1: 840

Modular trade reproduction totals:

- MRS2: 1,052
- MRL1: 483

Audit status: PASS

This audit establishes that the market, event context, HMM assignment, z-score data, and RTH alignment are all internally consistent before strategy-level trades are treated as valid. It is validation of the research pipeline, not a claim that every regime bucket is equally mature or that new strategy work is complete.

## Current three-strategy portfolio

The portfolio analysis script is:

- `src/research/mean_reversion/research/12_mr_3_strategy_visual_report.py`

This script aggregates the frozen trade streams for the three validated strategies and confirms the current portfolio count and ordering.

Current portfolio composition:

- MRL1 = 483 trades
- S2R = 537 trades
- MRS2 = 1,052 trades
- total = 2,072 trades

Current portfolio metrics:

- total R = +127.1072R
- expectancy = +0.061345R
- median trade = +0.0733R
- win rate = 52.03%
- profit factor = 1.1520
- max drawdown = -17.2469R
- average win = +0.8939R
- average loss = -0.8450R
- payoff = 1.0579
- longest win streak = 12
- longest loss streak = 9
- annualized Sharpe = 1.5692
- Sortino = 2.6524

Duplicate check:

- duplicate (entry_timestamp, strategy) = 0

Daily correlations:

- MRL1 / S2R = 0.016
- MRL1 / MRS2 = 0.010
- S2R / MRS2 = 0.125

Yearly returns:

- 2019: +1.19R
- 2020: -1.5167R
- 2021: +9.6277R
- 2022: +47.5944R
- 2023: +1.8648R
- 2024: +16.1801R
- 2025: +28.0595R
- 2026: +24.1073R

Important: 2019 and 2026 are partial years and should not be described as full-year performance.

The visual-report output directory expected by the script is:

- `src/research/results/portfolio_3_strategy/`

This directory does not currently appear in the checked-in repository snapshot, so the script is present but its generated visual products are not currently populated in this checkout.

## Regime map

The current mean-reversion regime coverage is:

- VOL0–20: uncovered
- VOL20–40: MRL1
- VOL40–60: S2R
- VOL60–80: uncovered
- VOL80–100: MRS2

Research philosophy:

- do not add a strategy merely to fill a regime bucket
- a new candidate may replace, complement, or extend an existing strategy only when the evidence supports it
- there has been limited research into VOL0–20 and VOL60–80, and those buckets are therefore currently treated as uncovered rather than as failed strategy territories

## Trade visualizer

The current trade visualizer is:

- `trade_visualizer.py`

It is designed to load and inspect frozen trade streams against the canonical Databento market data. It can:

- load one or more trade CSV files
- load the canonical Databento MNQ market data via `load_databento_mnq()`
- render local price action around an entry and exit
- overlay entry, exit, stop-loss, and take-profit levels when available
- filter trades by strategy, side, result, and volatility regime
- move through prior/next/random trades in the selected CSV

Important implementation detail: the visualizer defaults to the validated S2R trade stream:

- `src/research/results/s2_extended/s2r_modular_authoritative_reproduction.csv`

This is the 537-trade stream. The broader 5231-row `s2r_modular_full_databento_trades.csv` export is intentionally not treated as the authoritative frozen trade list and should not be documented as the validated S2R stream.

## Funded simulation work

The repository contains funded simulation scripts, but they are historical research artifacts rather than final full-portfolio validation.

Relevant scripts:

- `src/research/mean_reversion/research/09_mr_funded_simulation.py`
- `src/research/mean_reversion/research/10_mr_portfolio_funded_simulation.py`

Important distinction:

- the older funded simulation work excluded S2R and therefore should not be presented as the final/current full three-strategy portfolio result
- the current validated three-strategy portfolio is the historical trade-stream analysis above, not a funded-account simulation result

The repo contains historical account-based research around the MR strategies, but the final code snapshot does not contain a new funded simulation that re-asserts the full 3-strategy portfolio as the latest account-level validation result.

## Git and project status

Current repository status as checked in this workspace:

- branch: `santiago-pascual-quant-strategy-fix`
- git tag: `v1.1-mr-validation-complete`
- checkpoint commit: `4b54c1b`
- working tree: clean

The project is intentionally not creating another major branch at this stage. The current rule in the project workflow is to branch only when core strategy research is complete, the portfolio and model stack are validated, and paper trading or the next major operational phase begins.

## Session Momentum research status

Session Momentum is a separate research stream and is not part of the validated production portfolio.

There is currently no active `src/research/mean_reversion/research/13_session_momentum_analysis.py` implementation in this repository snapshot. The repo should therefore treat Session Momentum as exploratory, not validated, unless a later branch adds a committed implementation and supporting results.

Current concept from the research notes:

- NASDAQ / NY session open
- M5 timeframe
- EMA12
- EMA120
- stop = 8 × ATR
- direction determined from the first NY opening candle relative to EMA12
- one trade per session
- no fixed profit target
- once trade reaches +0.5R, trailing activates using EMA120

Important entry definition:

- opening candle is 09:30–09:35 America/New_York
- entry occurs at 09:35 America/New_York
- entry price is the close of the 09:30–09:35 opening candle
- not 09:40 and not the next candle open

ATR audit status:

- ATR grid: [5, 7, 10, 12, 14, 16, 20, 24, 30, 40]
- ATR multiplier: 8.0
- EMA signal: 12
- EMA trail: 120
- trail trigger: +0.5R

Opening-candle audit (data-integrity / implementation check, not profitability evidence):

- 1-minute rows: 2,577,661
- complete RTH M5 bars: 144,866
- sessions: 1,887
- expected signal candle: 09:30–09:35 America/New_York
- expected entry time: 09:35
- expected entry price: close of opening candle
- sessions with 09:30 bar: 1,885
- sessions without 09:30 bar: 2
- 09:30 bars found: 1,885
- bad opening-minute labels: 0

Session Momentum should remain separated from the validated portfolio. It is a research avenue for strategy-result analysis, regime/context analysis, and robustness checks, but not a currently validated production component.

## Research philosophy

The project is built around the following rules:

- validate market and event integrity before strategy conclusions
- separate exploratory work from frozen validation output
- keep strategy implementation and research analysis modular
- prefer verified, frozen trade streams over broad event exports
- do not imply portfolio validation from a single strategy result
- keep regime-aware strategy selection evidence-driven
- avoid hyper-optimization when the underlying regime signal is still diagnostic rather than proven

The current repo state is a validated mean-reversion portfolio on the canonical Databento MNQ dataset, with S2R, MRL1, and MRS2 as the active frozen strategy references, and the rest of the more recent research directions kept clearly separated as exploratory or work-in-progress.

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

