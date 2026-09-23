# NQ Quant Strategy

Quantitative research infrastructure for Nasdaq futures (NQ/MNQ), built around canonical 1-minute Databento data, modular strategy implementations, reproducible backtests, reconciliation audits, portfolio analysis, robustness testing, and execution/account simulations.

The repository is at a **pre-paper-trading** stage. Its historical results and account simulations are research evidence only; they are not live validation or a guarantee of future performance.

## Overview

The project combines four frozen strategy components into one research portfolio:

- **MRL1** — mean reversion, long
- **S2R** — regime-filtered short strategy with adverse-excursion recovery logic
- **MRS2** — mean reversion, short
- **ORB** — 30-minute opening-range breakout

The codebase separates strategy logic from research orchestration and result generation. The validation workflow emphasizes deterministic data loading, baseline-to-modular reconciliation, a common out-of-sample (OOS) period, execution-friction stress, statistical path analysis, and chronological funded-account simulation.

## Final System

The current frozen system is **MRL1 + S2R + MRS2 + ORB**. The four components are confirmed by the current portfolio scripts, modular strategy code, and the `v1.3-full-system-validation` milestone.

### MRL1

| Field | Frozen definition |
|---|---|
| Context | HMM state 1; volatility percentile 20–40 |
| Direction | Long |
| Entry | Z-score at or below -2.5 |
| Exit | 25-point target, 37.5-point stop, or 8-bar horizon |
| Reward/risk | 25 / 37.5 = 0.6667R |

### S2R

| Field | Frozen definition |
|---|---|
| Context | HMM state 2; volatility percentile 40–60 |
| Direction | Short |
| Entry | Quality score at least 0.75 |
| Exit | 25-point stop, 1.75R target, or 20-bar horizon |
| Lifecycle | 0.70R adverse-excursion threshold, +0.20R recovery, six-bar recovery deadline |

### MRS2

| Field | Frozen definition |
|---|---|
| Context | HMM state 2; volatility percentile 80–100 |
| Direction | Short |
| Entry | Z-score at or above +2.0 |
| Exit | 27.5-point target, 25-point stop, or 30-bar horizon |
| Reward/risk | 27.5 / 25 = 1.10R |

### ORB

| Field | Frozen definition |
|---|---|
| Context | New York regular trading hours |
| Opening range | 09:30–10:00 America/New_York |
| Direction | Long above the range high; short below the range low |
| Entry | Breakout touch between 10:00 and 11:00 |
| Exit | 2R target, opposite-side stop, or RTH close |
| Lifecycle | One trade per session; stop takes priority when stop and target occur in the same bar |

The ORB modular reproduction checks the frozen standalone OOS benchmark of **1,442 trades**, **+156.026568R**, **0.1082015R expectancy**, **48.404993% win rate**, **1.255941 profit factor**, and **-14.175719R maximum drawdown**. These are ORB reconciliation values, not live results and not the four-strategy portfolio scorecard.

## Portfolio Architecture

The portfolio is assembled from the frozen trade streams and sorted chronologically by entry timestamp, with strategy attribution retained for every trade. The portfolio analysis performs:

- full-sample stream-count checks: MRL1 **483**, S2R **537**, MRS2 **1,052**, ORB **1,747**; total **3,819**
- common-OOS partitioning for the official window
- duplicate checks on `(entry_timestamp, strategy)`
- strategy-level and combined accounting in R
- daily, monthly, and yearly breakdowns
- daily strategy correlations, same-day interaction, entry overlap, concurrency, and contribution analysis

The common OOS trade-count audits in the funded simulation are MRL1 **430**, S2R **520**, MRS2 **863**, and ORB **1,442**, for **3,255** trades. The analysis preserves chronological ordering; it does not shuffle individual trades.

The repository does not contain the generated four-strategy `portfolio_metrics.csv` or common-OOS portfolio output in the checked-in tree. Consequently, this README does not reproduce unverified portfolio expectancy, profit factor, drawdown, Sharpe, Sortino, or total-R values.

## Validation Methodology

The validation pipeline is chronological:

1. **Individual strategy validation** — freezes strategy definitions and evaluates their historical trade streams.
2. **Modular/baseline reconciliation** — compares modular implementations against the established research baselines. Mean reversion uses the modular 08AA reproduction; ORB uses baseline and modular reconciliation scripts.
3. **Common-OOS portfolio validation** — merges the four frozen streams, applies the exact shared OOS window, checks counts and ordering, and produces portfolio-level accounting.
4. **Robustness analysis** — tests whether the frozen portfolio's conclusions depend on one path, one time window, one strategy, or a narrow parameter/execution assumption.
5. **Transaction-cost and slippage stress** — applies the configured MNQ cost model and adverse deterministic/random execution assumptions to the common OOS stream.
6. **Monte Carlo and bootstrap analysis** — evaluates trade permutation, IID trade, IID daily, moving-block daily, strategy-preserving daily, execution, and missed-trade paths.
7. **Funded-account simulation** — replays the chronological common-OOS portfolio under Combine and XFA account policies without independently shuffling trades.

These stages test reproducibility and historical robustness. They do not establish live profitability or production readiness.

## Out-of-Sample Results

The official common OOS window is:

**2020-06-23 through 2026-06-19**, inclusive, in the New York session calendar.

The committed analysis code verifies the following common-OOS counts:

| Strategy | OOS trades |
|---|---:|
| MRL1 | 430 |
| S2R | 520 |
| MRS2 | 863 |
| ORB | 1,442 |
| **Total** | **3,255** |

The portfolio analysis script writes the exact combined metrics to `src\research\results\portfolio\portfolio_metrics.csv` when run. That generated file is not committed in the current repository snapshot, so no portfolio-level expectancy, profit factor, win rate, maximum drawdown, Sharpe, Sortino, trading-day count, or total-R claim is made here.

## Robustness Testing

The full-system robustness engine is `src\research\portfolio\22_mr_orb_portfolio_robustness.py`. It operates on the official common OOS and implements:

| Test | Failure mode examined |
|---|---|
| Deterministic cost/slippage grid | Sensitivity to repeatable execution friction |
| Random adverse slippage | Variation in fills rather than one fixed slippage path |
| Trade-sequence permutation | Dependence of drawdown on the observed trade order |
| IID trade bootstrap | Uncertainty under independent trade resampling |
| IID daily bootstrap | Uncertainty while retaining daily aggregation |
| Moving-block daily bootstrap | Dependence on clustered daily outcomes |
| Strategy-preserving daily bootstrap | Whether strategy composition matters beyond aggregate daily returns |
| Missed-trade stress | Degradation from execution failures or unavailable fills |
| Worst-tail degradation | Sensitivity to additional damage in the worst historical trades |
| Year/month/week stability | Concentration in particular calendar periods |
| Drawdown episodes and duration | Depth and persistence of loss periods |
| Strategy contribution/correlation | Dependence on one component or correlated components |
| Entry overlap/concurrency | Simultaneous exposure and operational load |

The engine is configured for **50,000** simulations, moving blocks of **5, 10, and 20 days**, random adverse slippage up to **10 ticks per side**, and missed-trade stress of **1%, 2%, 5%, and 10%**. These are test configurations, not claims about actual live execution.

Generated scorecards and simulation tables are written under `src\research\results\portfolio_robustness\`. They are generated artifacts and are not present in the checked-in repository snapshot; numerical robustness conclusions are therefore intentionally not reproduced here.

## Transaction Costs and Slippage

The full-system scripts model the following MNQ assumptions:

- MNQ tick size: **0.25 points**
- MNQ tick value: **$0.50**
- MNQ point value: **$2.00**
- configured Topstep MNQ round-turn cost: **$1.22**

The deterministic portfolio stress grid tests 0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 16, and 20 ticks per side. Random execution stress samples adverse slippage independently on each side up to the configured scenario maximum. These are explicit research assumptions, not guaranteed broker, exchange, or funded-account fills.

## Funded Account Validation

`src\research\portfolio\22_full_system_funded_simulation.py` performs account-policy simulation separately from strategy validation:

- exact common-OOS portfolio sequence only
- chronological trade order preserved
- real historical trading-day boundaries preserved
- no individual trade shuffling
- **50,000** replay paths
- vectorized and scalar engines checked by a deterministic parity audit over **25** paths, capped at **250** trades for the audit

### Combine model

- starting balance: **$50,000**
- profit target: **+$3,000**
- maximum loss: **-$2,000**
- maximum simulated path length: **500 trades**
- risk policies include 0.25%, 0.50%, 0.75%, 1.00%, and 0.50%-to-1.00% after a $1,000 threshold

### XFA model

- starting balance: **$50,000**
- fixed loss floor: **$48,000**
- minimum winning days: **5**
- minimum winning-day profit: **$150**
- maximum simulated path length: **2,000 trades**
- payout intervals: 20, 21, and 22 days
- payout amounts: $500 through $2,000 across the configured grid

The code writes Combine and XFA result tables under `src\research\results\portfolio\funded\`. Those generated outputs are not committed in the current repository snapshot, so no pass rates, survival rates, payouts, or account-level performance figures are asserted here. This is a simulation of account constraints, not evidence of future funded-account performance.

## Reproducibility

The canonical market path is:

- loader: `src\databento_loader.py`, via `load_databento_mnq()`
- raw data: `data\raw\mnq\ohlcv_1m\`
- legacy dataset: `data\Dataset_NQ_1min_2022_2025.csv`; it is not the canonical modular validation source

Representative research commands from the committed scripts:

```powershell
python .\src\research\orb\20_orb_modular_reproduction.py
python .\src\research\portfolio\21_mr_orb_portfolio_analysis.py
python .\src\research\portfolio\22_mr_orb_portfolio_robustness.py
python .\src\research\portfolio\22_full_system_funded_simulation.py
```

The scripts perform their own input and count audits and write CSV/PNG reports under `src\research\results\`. The authoritative S2R stream is `src\research\results\s2_extended\s2r_modular_authoritative_reproduction.csv`; the broader `s2r_modular_full_databento_trades.csv` export is not the frozen benchmark.

## Project Structure

```text
data\
  raw\mnq\ohlcv_1m\       Canonical Databento MNQ 1-minute files
src\
  databento_loader.py     Canonical data loader
  session_engine.py       Session and RTH infrastructure
  strategies\
    mean_reversion\       MRL1 and MRS2 modular strategies
    s2r\                  S2R modular strategy and recovery logic
    orb\                  ORB modular strategy and lifecycle
  research\
    mean_reversion\       Mean-reversion validation and robustness
    orb\                  ORB baseline, reconciliation, robustness, funding
    portfolio\            Four-strategy portfolio and account validation
    results\              Frozen inputs and generated research outputs
tests\                    Strategy and lifecycle tests
trade_visualizer.py       Trade inspection against canonical market data
```

The repository also contains older S2, backup, directional, barrier, and exploratory research scripts. They remain research history and are not additional components of the frozen four-strategy system.

## Validation Status

- [x] MRL1 and MRS2 mean-reversion validation and modular reproduction
- [x] S2R modular reproduction and recovery-lifecycle audit
- [x] ORB baseline, execution audit, and baseline-to-modular reconciliation
- [x] Four-strategy portfolio integration and common-OOS count audits
- [x] Portfolio robustness test implementations
- [x] Transaction-cost and slippage stress-test implementations
- [x] Monte Carlo, bootstrap, block-bootstrap, tail, missed-trade, and concentration analyses implemented
- [x] Chronological Combine/XFA funded-account simulation implementation
- [x] Scalar/vector parity audit implementation
- [ ] Paper-trading validation
- [ ] Live execution validation
- [ ] Production risk/execution infrastructure

The latest repository milestone is `v1.3-full-system-validation`. The validation code is complete enough to support a reproducible pre-paper-trading research system, but the repository does not establish live or paper-trading performance.

## Limitations and Next Stage

The results are dependent on historical Databento bars, bar-level execution assumptions, conservative intrabar ambiguity rules, modeled costs, and the selected OOS window. Generated full-system metric and account-result files are not committed in this snapshot, and the research code cannot establish fill quality, liquidity, market impact, or future regime behavior.

The next stage is to freeze the reproducible research inputs, verify the complete outputs in a controlled environment, build the execution and risk infrastructure, and conduct paper trading. No claim is made that the historical system will work live.

## Disclaimer

Historical and backtested results are not guarantees of future performance. Simulated funded-account results are not live results. Execution quality, slippage, liquidity, transaction costs, market impact, and regime changes can materially affect live outcomes.

## License

Copyright © 2026 Santiago Pascual. All Rights Reserved.

This repository is publicly available for viewing, educational, and research purposes only. Reproduction, redistribution, commercial use, or derivative works require prior written permission from the author.

See [LICENSE](LICENSE) for the complete terms.
