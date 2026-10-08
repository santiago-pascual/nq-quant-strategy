# MNQ Quant Strategy Research and Paper Trading

Research and Paper execution infrastructure for four MNQ futures strategies. The repository includes frozen historical research, a deterministic Research/Paper replay, a causal HMM implementation under provisional validation, and a local Paper engine with analytics, checkpoints, shadow replay, and a protected read-only monitoring API.

**This project is paper-only. It does not route real-money orders.** Historical backtests and simulated Paper executions are not live results and do not predict future performance.

## Project status at a glance

| Area | Status | What the evidence supports |
|---|---|---|
| Historical Research OOS | Validated historical benchmark | Frozen 2020-06-23 through 2026-06-19 portfolio: 3,255 trades and the metrics below. |
| Research/Paper replay | Validated | 2,899 exact Paper executions; the other 356 Research trades were classified by existing account/risk gates; no mismatches or extras. |
| Causal HMM | Implemented; provisional | Forward filtering and causal refit paths exist. The long pseudo-live run was interrupted and not completed, so this is not production validation. |
| Realtime Paper engine | Implemented for replay/local operation | SQLite/WAL analytics, checkpoints, shadow replay and monitoring are implemented and have focused test coverage. |
| Live market data | Not available | No connected live MNQ feed/provider or credentials are included. Current supported input is historical replay data. |
| CME calendar | Loader/schema and validation exist; reviewed snapshot missing | Do not run calendar-dependent operation across uncovered dates. No 2026-10-07 through 2026-12-31 MNQ snapshot is installed. |
| Linux deployment | Example only | A systemd replay template is provided. It has not been installed or tested on Linux. |

## Strategies

- **MRL1** — long mean reversion. Its strategy contract consumes raw HMM state 1.
- **MRS2** — short mean reversion. Its strategy contract consumes raw HMM state 2.
- **S2R** — short regime strategy with recovery analysis. It consumes raw HMM state 2; recovery enrichment is analytical and does not replace the baseline executable lifecycle.
- **ORB** — New York opening-range breakout with one entry per session, stop/target handling, and session-close management.

Research strategy parameters and the frozen Research Replay are separate from the newer causal Paper HMM path. The causal provider preserves the raw fitted component IDs; it does not apply semantic state remapping.

## Historical out-of-sample benchmark

The frozen Research portfolio covers **2020-06-23 through 2026-06-19**. It contains **3,255 trades** and reports:

| Metric | Frozen Research OOS |
|---|---:|
| Total return | **+289.6619R** |
| Expectancy | **+0.08899R/trade** |
| Profit factor | **1.216** |
| Win rate | **50.66%** |
| Maximum drawdown | **−18.0935R** |

Strategy counts are MRL1 430, MRS2 863, S2R 520, and ORB 1,442. These are historical Research results under the repository's bar-level execution and cost assumptions. They are not realtime Paper or live results.

## Frozen Research/Paper Replay validation

The completed full OOS Research Replay used frozen Research HMM states and reference trade artifacts. The run summary records `causal_inference_used: false`: this validates Paper execution against the frozen Research methodology; it does **not** validate causal HMM inference.

| Strategy | Frozen Research trades | Exact Paper executions | Legitimate account/risk gate | Mismatches | Paper extras |
|---|---:|---:|---:|---:|---:|
| MRL1 | 430 | 428 | 2 | 0 | 0 |
| MRS2 | 863 | 806 | 57 | 0 | 0 |
| S2R | 520 | 519 | 1 | 0 | 0 |
| ORB | 1,442 | 1,146 | 296 | 0 | 0 |
| **Total** | **3,255** | **2,899** | **356** | **0** | **0** |

The non-executions were classified as daily-loss, maximum-daily-trades, or validated ORB per-contract risk-cap rejections. Every executed Paper trade matched the Research comparison fields; no unexplained differences or extra Paper trades remained. Shortened-session ORB closes and S2R final-20-bar entry eligibility were included in the final replay validation. The run artifacts are generated under `results/` and are intentionally not versioned.

## Causal HMM: implementation and validation status

`src/models/causal_hmm.py` implements causal forward filtering and model/refit/checkpoint support. The current strategy contract uses raw fitted IDs (MRL1 state 1; MRS2 and S2R state 2), with no semantic alignment between fits. The configured cadence is expanding MR training with four-calendar-month refits and rolling two-year S2R training with three-calendar-month refits. These are distinct from the frozen historical Research Replay path.

The causal HMM was provisionally accepted for further Paper integration after focused checks. A complete 2024-08-27 through 2026-08-26 pseudo-live validation did **not** complete; the run was interrupted due to runtime constraints. Therefore, no completed pseudo-live performance or full-period restart-equivalence claim is made here. Differences from frozen Research states/trades are expected because Research used its historical decoding/training methodology.

## Paper engine and analytics

The Paper subsystem is designed for simulated execution only. Its current components include:

- sequential Paper engine, risk/conflict/execution and simulated broker/fill paths;
- replay market-data source (no live MNQ provider is connected);
- SQLite analytics store using WAL, event and trade diagnostics;
- checkpoints and recovery/verification commands;
- shadow replay and daily parity reporting;
- CME calendar snapshot loader and fail-closed coverage validation;
- local single-writer lock for a shared output directory;
- bearer-token-protected, loopback-only, read-only monitoring API.

The monitoring API uses a versioned JSON envelope (`schema_version: "1.0"`) and read-only routes for health, account, strategies, positions, trades, HMM/refits, feed, checkpoint, parity, events, candidates, daily reports, and risk. It has no order-entry endpoints. Do not expose it publicly without an authenticated private access path and TLS termination. The local writer lock is not cross-host fencing; do not run simultaneous Windows and cloud instances against the same account or data stream.

### Market data and CME calendar

The current adapter supports deterministic replay, not a live feed. Live Paper remains blocked on a market-data provider, credentials, reconnect/backfill handling, and operational sequence validation.

The calendar loader accepts explicit reviewed snapshots. The official [CME trading-hours page](https://www.cmegroup.com/trading-hours.html) provides product/date selection and warns that schedules may change. No reviewed MNQ snapshot currently covers 2026-10-07 through 2026-12-31. Product-specific hours still need verification for October 12, November 11 and November 25–28, and December 24–26 and 31, 2026, before those exceptions can be marked covered. The updater/validator does not manufacture hours; uncovered sessions fail closed.

### Deployment

- **Windows:** local deterministic Paper replay can be launched with `scripts/start_paper_replay.ps1` after installing the pinned environment. Use a dedicated output directory and retain its checkpoints and SQLite database.
- **Linux:** `deploy/systemd/mnq-paper-replay.service.example` is a replay-service template only. Linux installation and runtime have not been tested in this project environment. Review paths, service user, environment file, storage and resource limits before use.
- **Cloud:** no cloud resource has been configured. Published free-tier quotas do not establish that the workload is operationally suitable. Full history bootstrap, complete S2R refit cost, Linux runtime, and multi-host fencing remain unverified.

## Reproducibility and tests

Use Python 3.13 on the tested Windows x86-64 environment with `requirements-realtime-paper-py313.lock`. The lock pins versions but is not a platform-independent hash lock. Linux x86-64/ARM64 wheels were inspected, but installation/runtime was not tested; Windows ARM64 is also untested.

Install in a project virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-realtime-paper-py313.lock
```

Focused validation commands:

```powershell
pytest -q tests\models\test_causal_hmm.py
pytest -q tests\paper\test_realtime_paper.py tests\paper\test_analytics_store.py
pytest -q tests\paper\test_monitoring_api.py tests\paper\test_cme_calendar.py tests\paper\test_single_writer.py tests\paper\test_systemd_notify.py
```

On 2026-10-07, the focused causal-HMM, Paper/realtime, calendar, monitoring, risk and S2R reconstruction validation set completed with **117 passed**. This does not include a full OOS replay or completed pseudo-live run.

Local replay CLI (PAPER mode only; choose timestamps inside locally available data and supply a reviewed calendar snapshot and explicit cost policy):

```powershell
python -m src.paper.run_realtime_paper --command run --mode PAPER `
  --replay-start "<UTC_START_IN_LOCAL_DATA>" --replay-end "<UTC_END_IN_LOCAL_DATA>" `
  --calendar-snapshot "<REVIEWED_MNQ_CALENDAR_JSON>" `
  --cost-config ".\src\paper\config\topstepx_mnq_fees_2026-07.json" `
  --output-dir results/paper/local_replay
```

The supported CLI commands also include validation, stop/status, event inspection, checkpoint verification, shadow/daily reports, database backup, analytics and the read-only API. For argument details, run `python -m src.paper.run_realtime_paper --help`. Use only a calendar-covered interval and the appropriate explicit cost configuration. The deterministic historical Research Replay is separate and can be invoked with `python -m src.paper.run_research_replay --start 2020-06-23 --end 2026-06-19 --output-dir results/paper/research_replay`; it is long-running and is not required for the focused tests above.

## Current limitations and roadmap

### Before autonomous realtime Paper

1. Connect and validate a real MNQ market-data provider, including credentials, reconnects, backfill and duplicate/out-of-order handling.
2. Obtain and review authoritative product-specific CME session exceptions; install and validate the covered calendar snapshot.
3. Validate full causal history bootstrap and scheduled refit cost; complete pseudo-live/restart validation under an approved runtime budget.
4. Test the pinned dependencies and end-to-end replay on the chosen Linux target, if using Linux.
5. Add cross-host/account-level fencing and operational controls before moving between machines.

### Before a mobile client

1. Freeze and version the monitoring API schema.
2. Choose a private authenticated transport, TLS boundary, access policy, and retention/backup policy.
3. Build the client against the read-only API; no mobile order-entry functionality is part of this project scope.

No step above authorizes real-money order routing. This repository currently supports PAPER mode only.

## License and risk notice

See [LICENSE](LICENSE). Historical backtests and simulated fills depend on the available data, bar-level assumptions, modeled transaction costs and execution rules. Actual fills, slippage, liquidity, market impact, system failures and future regimes may differ materially. This is quantitative research and paper-trading software, not investment advice or a guarantee of future results.
