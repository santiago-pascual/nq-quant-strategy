# Delayed IBKR Paper startup gate

`src.paper.delayed_paper_cli` is the explicit operator interface for delayed
IBKR Paper. It labels the service `DELAYED_IBKR_PAPER`, routes no orders, pins
MNQZ6 / conId 815824267 / expiry 20261218, and uses the existing per-bar
checkpoint/acknowledgment service. It does not start a bootstrap.

## Offline readiness check

From the repository root with the project virtual environment active:

```powershell
python -m src.paper.delayed_paper_cli readiness `
  --activation-utc 2026-10-09T00:00:00Z `
  --calendar src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json `
  --calendar-review src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.review.json `
  --bootstrap <activation-ready-causal-bootstrap.json> `
  --bootstrap-identity <trusted-bootstrap-identity.json> `
  --recovery-validation <strategy-recovery-validation.json> `
  --cost-config src/paper/config/topstepx_mnq_fees_2026-07.json `
  --output-dir results/paper/delayed_mnqz6_20261009 `
  --tws-host 127.0.0.1 --tws-port 7496
```

Readiness is offline: it validates files/configuration but does not contact TWS
or start Paper. The current recovery validation record passes, but a compatible
activation-ready causal seed is absent and the historical coverage certificate
is not fully certified. This session snapshot only covers October 8–31, 2026.

## Read-only delayed streaming probe

Before treating TWS as a continuous delayed feed, run the separate bounded
streaming probe (it does not use historical bars and never calls an order API):

```powershell
\.venv\Scripts\python.exe -m src.paper.ibkr_streaming_probe `
  --host 127.0.0.1 --port 7497 --client-id 192 `
  --duration-seconds 25 --timeout-seconds 8
```

The probe reports success only after both a delayed-mode callback (`3`) and at
least one finite positive streaming price tick; size-only callbacks do not
count. A successful historical-bar request or a delayed-mode callback alone
is insufficient. The 2026-10-09 probe connected to the IBKR Paper account and
resolved MNQZ6 / conId 815824267 / expiry 20261218. A 25-second streaming
request reported market-data type 3 but produced no price ticks. The separate
snapshot diagnostic reported error 10167 (market data not subscribed; delayed
fallback announced). The historical diagnostic returned 1,190 valid ordered
bars from 2026-10-08 22:00 through 2026-10-09 17:49 UTC; this confirms
historical access only. The quote-tick stream remains unverified. The Paper bar
adapter does not require individual `reqMktData` ticks; its historical-polling
path is assessed below. Strategy-enabled startup remains blocked by the missing
activation bootstrap and uncertified historical coverage.

## Historical 1-minute polling path

The Paper adapter acquires completed one-minute bars using the IBKR historical
TRADES API; it does not depend on `reqMktData` quote ticks. The API path can
therefore be evaluated independently from quote-stream availability. To run a
bounded historical-poll diagnostic while the reviewed calendar covers the
requested timestamps:

```powershell
\.venv\Scripts\python.exe -m src.paper.ibkr_historical_poll `
  --host 127.0.0.1 --port 7497 --client-id 197 `
  --con-id 815824267 --local-symbol MNQZ6 --polls 3 `
  --interval-seconds 30 --duration-seconds 86400 --timeout 20 `
  --output-dir results/diagnostics/ibkr_historical_poll_20261009_session_test `
  --calendar-snapshot src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json
```

The cursor-aware acquisition/finalization path can be exercised separately
from Paper with:

```powershell
\.venv\Scripts\python.exe -m src.paper.ibkr_delayed_feed_smoke `
  --host 127.0.0.1 --port 7497 --client-id 198 `
  --con-id 815824267 --local-symbol MNQZ6 --expiry 20261218 `
  --cycles 3 --interval-seconds 30 `
  --calendar-snapshot src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json `
  --output-dir results/diagnostics/ibkr_delayed_feed_smoke_20261009_polling
```

The October 9 open-session poll returned 1,431 unique valid bars across three
overlapping responses. One new minute appeared on poll two; the interval was
30 seconds, request starts were at least 35 seconds apart, and there were no
invalid or out-of-order rows. The sole 21:00–22:00 UTC gap was the reviewed
maintenance break. A recent bar first appeared about 9m57s after its bar end;
an overlapping response later revised its close, low and volume. The
finalization gate reset the stability count and withheld that version until it
met the 10-minute age and two-observation/30-second rules. The poll analyzer
keeps its conservative `historical_poll_not_validated` verdict when it sees
any revision; the separate finalization smoke demonstrates that this observed
pre-delivery revision was safely withheld rather than silently accepted.

The separate acquisition smoke used at most two requests per cycle and five
requests in about 63 seconds. It appended 121 observations, tracked 31 unique
bar timestamps, and finalized 20 bars. A second process using the same
persisted journals finalized two new bars. The resulting delivery journal has
22 unique, strictly increasing contract/timestamp keys, with zero duplicate
deliveries, zero late revisions and no provider errors. These bounded tests
support historical polling for further isolated Paper development. They do not
prove an uninterrupted service, a futures roll schedule, or strategy-enabled
readiness. Neither smoke was connected to the Paper engine. A quote-tick stream
is not required by the 1-minute OHLCV contract; historical polling is the
intended delayed bar transport.

The poll analyzer accepts the same reviewed calendar snapshot and classifies
observed timestamp gaps as scheduled closures, unresolved expected-session
gaps, or uncertified-calendar gaps. This classifies session timing only: an
absent OHLCV minute during an expected session is not proof of lost data. The
saved October 9 poll JSON predates this diagnostic improvement and retains its
original unclassified gap label. Tests cover the reviewed maintenance break,
an open-session gap, and dates outside calendar coverage.

The supported Windows Paper runtime includes the pinned official `ibapi`
distribution (`ibapi==9.81.1.post1`) in
`requirements-realtime-paper-py313.lock`. Install the complete pinned runtime
into the project environment with
`\.venv\Scripts\python.exe -m pip install -r requirements-realtime-paper-py313.lock`.
The separate `requirements-ibkr-diagnostic.txt` remains available for an
isolated connectivity diagnostic environment; the delayed Paper runner must
use the main `.venv` so IBKR, NumPy, HMM and Paper imports share one runtime.

When the readiness command exits 0 and all prerequisites have been independently
reviewed, start explicitly:

```powershell
python -m src.paper.delayed_paper_cli start `
  --activation-utc 2026-10-09T00:00:00Z `
  --calendar src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json `
  --calendar-review src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.review.json `
  --bootstrap <activation-ready-causal-bootstrap.json> `
  --bootstrap-identity <trusted-bootstrap-identity.json> `
  --recovery-validation <strategy-recovery-validation.json> `
  --cost-config src/paper/config/topstepx_mnq_fees_2026-07.json `
  --output-dir results/paper/delayed_mnqz6_20261009 `
  --tws-host 127.0.0.1 --tws-port 7496 --confirm-delayed-paper
```

Resume this run with the same configuration and `--resume`. Request a graceful
stop with:

```powershell
python -m src.paper.delayed_paper_cli stop --output-dir results/paper/delayed_mnqz6_20261009
```

Inspect it with:

```powershell
python -m src.paper.delayed_paper_cli status --output-dir results/paper/delayed_mnqz6_20261009
```

`start` checks the complete readiness gate before connecting TWS or writing
Paper state. TWS access is historical read-only; this module does not import or
call broker order placement methods. Use a different output directory for a
new account; never point this interface at historical replay output.

## Historical calendar gap certificate

Generate the machine-readable, conservative inventory with:

```powershell
python scripts/build_mnq_calendar_coverage_certificate.py `
  --output results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08.json
```

It hashes the local compressed inputs and lists every observed timestamp gap.
Only explicit reviewed snapshot dates are classified. An absent OHLCV minute
inside an open session remains ambiguous between no trades and capture loss;
the report never promotes that to proven missing data. Uncovered dates remain
uncertified and no bars are synthesized.

The latest versioned inventory is
`results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_v2_recurring_rules.json`
(schema 2; details in the sibling `_gaps.csv`). It covers 2,619,604 rows from
2019-05-05 22:03 UTC through 2026-10-08 13:02 UTC. It separates the CME
Equity Index recurring schedule from date-specific reviewed snapshots:
Sunday-Friday 17:00-16:15 CT with a 15:15-15:30 CT pause and 16:15-17:00 CT
maintenance before the June 27, 2021 rule change; from that change onward,
Sunday-Friday 17:00-16:00 CT with 16:00-17:00 CT maintenance and no pause.
The rules are sourced to CME's 2012 hours notice and 2021 pause-elimination
notice. Date-specific holidays and early closes are still accepted only from
reviewed snapshots.

The v2 scan finds 5,937 gap spans: 3,435 consist only of recurring or reviewed
scheduled closures; 2,502 remain uncertain. Across all spans it classifies
1,238,690 absent timestamps as recurring/reviewed closures and 47,526 as
unverified open-session, holiday-exception, or no-trade-versus-capture cases.
These are missing timestamp counts, not proven lost trade bars. In particular,
OHLCV alone cannot distinguish a no-trade minute from lost capture. The scan
found no duplicate timestamps, non-monotonic transitions, or invalid OHLC
rows. The September 7, 2026 interval from 16:59 UTC to 22:00 UTC is now
classified as 300 scheduled early-close/holiday minutes, not missing market
data.

The recurring rules improve gap classification, but do not certify holidays,
early closes, or the cause of absent open-session timestamps across the full
2019-origin interval. A later activation date can shorten strategy evaluation,
but cannot truncate MR's expanding training origin (2019-05-05 22:03 UTC) or
S2R's rolling two-year training window while preserving the frozen model
specification. Do not label the 2019-origin bootstrap ready until the
unverified date-specific sessions and source-data continuity are resolved or
explicitly accepted by a reviewed coverage policy.

## Paper-only residual history-risk acceptance

The strict v5 certificate remains `NOT_FULLY_CERTIFIED`. The user explicitly
accepted its residual Databento degraded-source risk for internal simulated
Paper only. The separate acceptance record is
`results/paper/coverage_acceptance/paper_research_quality_accepted_2019_2026_v5.json`.
It pins the strict certificate, gap inventory, raw partition hashes, Databento
condition/manifest hashes, 15 degraded dates, and the eight unresolved spans
(1,051 minutes specifically classified as degraded-source uncertainty).
Already-classified maintenance/weekend minutes within those gap spans are
recorded separately. The acceptance never relabels the strict certificate,
never synthesizes bars, and cannot authorize live execution. Any source hash,
manifest, gap inventory, or accepted-scope change invalidates it.

Validate it without changing strict validation output:

```powershell
\.venv\Scripts\python.exe -m src.paper.coverage_acceptance_cli validate `
  --certificate results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_sparse_semantics_v5.json `
  --output results/paper/coverage_acceptance/paper_research_quality_accepted_2019_2026_v5.json
```

The resumable bootstrap uses an explicit flag and a separate checkpoint/cache
directory. Never run `resume` while the original process is alive:

```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:LOKY_MAX_CPU_COUNT='4'
\.venv\Scripts\python.exe -m src.paper.causal_bootstrap_cli status `
  --checkpoint results/paper/bootstrap_paper_accepted_20261008_1303/checkpoint.json
\.venv\Scripts\python.exe -m src.paper.causal_bootstrap_cli resume `
  --certificate results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_sparse_semantics_v5.json `
  --coverage-acceptance results/paper/coverage_acceptance/paper_research_quality_accepted_2019_2026_v5.json `
  --accept-paper-research-quality --confirm-full-bootstrap `
  --origin-utc 2019-05-05T22:03:00Z --activation-utc 2026-10-08T13:03:00Z `
  --chunk-rows 10000 `
  --checkpoint results/paper/bootstrap_paper_accepted_20261008_1303/checkpoint.json `
  --identity results/paper/bootstrap_paper_accepted_20261008_1303/identity.json `
  --cache-dir results/paper/bootstrap_paper_accepted_20261008_1303/feature_cache `
  --progress results/paper/bootstrap_paper_accepted_20261008_1303/progress.json
```
