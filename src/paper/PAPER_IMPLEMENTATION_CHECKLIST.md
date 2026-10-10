# MNQ Delayed Paper implementation checklist

Updated: 2026-10-09 UTC

This checklist tracks infrastructure work only. It does not authorize an
activation, unattended Paper process, broker order, or full historical run.

## Completed in this continuation

- [x] Inspect local Databento batch request, manifest, file hashes, import
  manifests, and condition metadata without network access.
- [x] Add `scripts/report_databento_coverage_evidence.py` to verify local batch
  provenance and summarize date-level condition status against uncertain gaps.
- [x] Add focused tests proving degraded condition overlap remains diagnostic
  and cannot promote coverage to certified.
- [x] Attempt read-only TWS Paper check on `127.0.0.1:7497`; socket timed out,
  no API handshake, contract query, quote, historical request, or order occurred.
- [x] Fix bounded benchmark result assembly and Windows peak-memory counter
  handle-width configuration; add a Windows counter regression test.
- [x] Run focused bootstrap, artifact, coverage, delayed-readiness, and
  Databento evidence tests: 30 passed.
- [x] Complete bounded real-data bootstrap/resume benchmark through the first
  two-year S2R fit. Result:
  `results/diagnostics/causal_bootstrap_may2019_first_s2r_fit_20261009_retry.json`.
  It consumed 697,999 actual rows (2019-05-05 22:03Z through 2021-05-06
  23:59Z), made 7 expanding MR fits and 1 rolling two-year S2R fit; a split at
  row 350,000 resumed exactly (nonsemantic timing fields excluded). The
  serialized split context was 38,291,364 bytes. Total runtime was 1,768.6s;
  cache hit; observed process working set peaked at about 1.58 GB by periodic
  sampling. Native peak-memory capture was unavailable in that run; the
  Windows counter implementation was corrected and tested afterward.
- [x] Fix the readiness-validation artifact lookup: readiness now defaults to
  the repository recovery record and verifies current runtime and test-source
  fingerprints. The old record was present but not supplied to the CLI, which
  caused the “missing” blocker.
- [x] Add `scripts/record_delayed_paper_recovery_validation.py`. It writes
  evidence only after the hard-crash ORB/HMM/refit/one-hour-catch-up test, the
  pending-order checkpoint/restore test, and the uninterrupted/recovered
  service test all pass.
- [x] Extend Paper checkpoint serialization/restoration to retain pending
  in-memory order authorizations, strategy-order links, broker submissions,
  and simulated fill queues. Added consistency checks and preserved str-backed
  enum semantics.
- [x] Add an actual ORB-generated pending-entry test: checkpoint before fill,
  restore, settle the same simulated fill, and compare final execution state.
- [x] Regenerate the strategy-enabled recovery record after its required tests
  passed. Offline readiness now passes recovery validation and reports only
  the missing activation-ready bootstrap artifact.
- [x] Extend the Databento evidence report to separate scheduled closure spans,
  uncertain spans, degraded-date overlap, certificate minute classes, and
  demonstrated-missing-required-observations from source completeness claims.
- [x] Run offline IBKR historical-poll, acquisition/finalization, observation
  ledger, journal, Paper service, checkpoint/recovery, and delayed CLI tests.
  No repeat TWS socket attempt was made after the prior timeout.
- [x] Rerun causal-bootstrap `validate` after checkpoint changes: source
  hashes/data integrity pass, but the certificate remains not fully certified
  and activation is blocked.

## Current blockers

- Databento manifests and checksums establish request identity and package
  integrity, not complete delivery of every trade-derived OHLCV minute.
- Databento date-level `available`/`degraded` condition is not a
  per-symbol/per-minute completeness attestation. The evidence report records
  18 degraded dates in observed coverage; 1,289 timestamps inside unresolved
  gap spans coincide with five degraded UTC dates.
- No required trade observation has been demonstrated missing; this is not a
  completeness proof. There are 47,526 uncertain absent open/exception/no-trade
  minutes; Databento trade-derived OHLCV cannot distinguish a no-trade minute
  from capture loss from local files alone.
- Historical date-specific CME holiday/early-close evidence remains missing
  for broad intervals; the coverage certificate remains
  `NOT_FULLY_CERTIFIED`. No absent bars are synthesized or certified.
- An activation-ready causal bootstrap artifact is not present.
- Delayed-Paper readiness still lacks the activation-ready causal bootstrap
  artifact. The recovery validation record is present and fingerprint-valid.
- TWS must be running in Paper mode with its API socket enabled at port 7497
  before the read-only connection diagnostic can run successfully.
- Bootstrap `validate` reports the source files match the certificate and the
  causal computation is possible on observed rows, but activation remains
  blocked by the certificate's uncertain coverage.

## 2026-10-09 coverage-resolution continuation

- [x] Confirmed the Databento client package is installed (`0.85.0`) and the
  local batch/certificate evidence is readable. The initial shell lookup did
  not find `DATABENTO_API_KEY`; a later explicitly authorized User-environment
  lookup succeeded without displaying the key, and bounded authenticated
  queries were made under the USD 1.00 task cap. See the request ledger below.
- [x] Reviewed current official Databento schema/API documentation. For
  `ohlcv-1m`, bars are trade-derived and no record is emitted for an interval
  with no trade. `trades` records expose venue sequence numbers and flags.
  Dataset condition is date-level: `available` means no known issue;
  `degraded` means data may be missing or have correctness issues. These
  signals do not attest completeness for the MNQ continuous symbol.
- [x] Initial local reclassification at
  `results/diagnostics/databento_coverage_evidence_2019_2026_20261009_reclassified.json`:
  2,619,604 rows; 3,435 recurring/verified closure spans; 2,502 uncertain
  spans (47,526 absent minutes); 18 degraded metadata dates across the source
  coverage; five degraded
  dates overlap six uncertain spans (1,289 absent minutes). The local batch
  manifests and hashes prove package integrity, not source completeness. This
  minute-grid count was later refined by the schema-aware certificate below.
- [x] Kept activation fail-closed. No coverage rule was relaxed: ordinary
  sparse OHLCV is not called a proven missing bar, but source-capture loss and
  unverified historical holiday/session exceptions remain unresolved.
- [x] Ran bounded targeted trades queries only after cost estimation; the
  evidence and unresolved 504 requests are recorded below.
- [ ] Obtain product-specific CME historical exception schedules for the
  unresolved dates or an authoritative CME/data-provider coverage statement;
  do not infer exceptions from generic holiday or clearing schedules.
- [ ] Only after those evidence gaps are resolved, rerun bootstrap validation,
  then start/resume the full checkpointed bootstrap and validate the v2 seed.

At that stage no code, source data, checkpoints, or research artifacts were
changed. No bootstrap or delayed Paper service was started.

## Commands / evidence

- Coverage evidence report:
  `python scripts/report_databento_coverage_evidence.py --output results/diagnostics/databento_coverage_evidence_2019_2026_20261009.json`
- Offline service readiness:
  `python -m src.paper.delayed_paper_cli readiness --output-dir results/paper/delayed_readiness_20261009 --tws-host 127.0.0.1 --tws-port 7497 --tws-client-id 93`
- Readiness status is `ready: false`; the only blocker is the absent bootstrap
  artifact. No service was started.
- Source/condition evidence:
  `results/diagnostics/databento_coverage_evidence_2019_2026_20261009.json`.
- Reclassified report:
  `results/diagnostics/databento_coverage_evidence_2019_2026_20261009_reclassified.json`.
- Recovery record:
  `results/diagnostics/delayed_paper_strategy_recovery_validation_20261009.json`.
- Bootstrap validation result:
  `results/diagnostics/causal_bootstrap_validate_20261009_after_checkpoint.json`.
- Official CME 2026 holiday page lists the Labor Day holiday span as Sept 6–8;
  the exact MNQ product-filtered schedule is not exposed in the captured
  product-neutral/browser-readable rows. It does not justify certifying
  historical gaps beyond reviewed calendar coverage.
- Final combined focused offline suite after record regeneration:
  49 passed (98.96s; 776 upstream hmmlearn/NumPy deprecation warnings).
- The first post-change run exposed the expected stale record and one earlier
  Windows checkpoint-rename permission error; the isolated rename retry passed,
  the record was regenerated from passing tests, and the final combined suite
  passed without failures.
- Test commands included `tests/paper/test_databento_coverage_evidence.py` and
  `tests/paper/test_causal_bootstrap_cli.py`; both passed after the Windows
  peak-memory counter correction.
- No full bootstrap, Paper run, broker order, commit, or push was performed.
## 2026-10-09 bounded Databento evidence continuation

- [x] Confirmed `DATABENTO_API_KEY` is available in the execution environment without displaying it; authenticated requests were made only after per-request cost estimation.
- [x] Captured three additional bounded 2025-11-28 MNQ.v.0 trades samples into `results/diagnostics/databento_gap_trades_20261009/` with SHA-256 hashes and response timestamps/IDs/flags/sequences.
- [x] Updated `results/diagnostics/mnq_coverage_investigation_20261009.json` with all submitted request estimates, evidence, and remaining uncertainty. Estimated exposure including the prior definition quote is USD 0.022566423257; actual posted charges were not reported by API responses. Two 2020 requests returned HTTP 504, so actual charges are unknown.
- [x] Reconfirmed TWS Paper endpoint 127.0.0.1:7497 previously returned the paper/demo account class, MNQZ6 conId 815824267, delayed market-data type 3, and ordered historical 1-minute TRADES bars; streaming quotes were not observed.
- [ ] Coverage remains uncertified: 2020 degraded open-session intervals and 2025 outage/close windows are not proven complete by trade records; exact date-specific CME Globex evidence is still missing for the 2025 exceptions.
- [ ] Activation artifact is absent, so no full bootstrap, new replay, or delayed Paper start is permitted by readiness gates.

No strategy, HMM, risk, replay, or service source code was changed. No broker orders, full bootstrap, new replay, commit, or push occurred.

## 2026-10-09 sparse OHLCV coverage semantics

- [x] Updated the certificate builder to verify the local Databento `condition.json` against its manifest hash and carry the dataset/date-scoped provenance into a new certificate.
- [x] Distinguished absent trade-derived OHLCV minute-grid rows on `available` dates from known recurring exchange closures and from degraded/unknown source intervals. The sparse label does not assert that no trade occurred; no synthetic bars are created.
- [x] Kept degraded/unknown source intervals activation-blocking. The revised certificate remains `NOT_FULLY_CERTIFIED`.
- [x] Rebuilt a new certificate, leaving prior certificates untouched: `results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_sparse_semantics_v3.json`.
- [x] Full local data: 2,619,604 rows; 46,791 absent minutes now correctly categorized as not-required sparse trade-derived OHLCV rows; 735 absent minutes in six degraded-source spans remain unresolved; 17 Databento-degraded dates overlap observed history; zero demonstrated missing required observations.
- [x] Validator confirms raw files and certificate match, but `activation_ready=false` due unresolved degraded-source evidence.
- [ ] Do not start full bootstrap or delayed Paper until degraded-source intervals are addressed through authoritative per-instrument evidence or another approved safe policy.

## 2026-10-09 maintenance-boundary correction

- [x] Fixed the coverage classifier so daily maintenance ends at 17:00 America/Chicago; Friday's session close transitions into the weekly closure rather than being mislabeled as daily maintenance through midnight.
- [x] Added timezone/season regression cases for pre-2021 and current hours, daily reopen, Friday weekly close, and vectorized gap classification.
- [x] Rebuilt the certificate as v4 without overwriting v2/v3. The corrected inventory has 5,937 spans: 1,842 verified scheduled-closure spans, 3,477 sparse-only trade-derived OHLCV spans, 610 mixed sparse/closure spans, and 8 source-quality-unresolved spans.
- [x] V4 classifies 54,452 absent minutes as not-required minute-grid rows and retains 1,051 absent minutes across eight degraded-source spans as unresolved. A corrected maintenance boundary exposed an additional 120-minute open interval on 2020-06-30 and one additional 1-minute degraded interval on 2025-09-17.
- [x] Bootstrap validator reports valid input integrity and causal computation on observed rows, but activation remains blocked by the 17 vendor-degraded dates and eight unresolved spans. No coverage gate was relaxed.
- [x] Focused calendar, coverage-evidence, delayed CLI and bootstrap CLI tests: 23 passed.
- [ ] Exact CME Nov 28, 2025 MNQ outage/reopening and early-close evidence, plus per-instrument completeness evidence for degraded-source intervals, remains necessary before activation.

No bootstrap, replay, delayed Paper service, broker order, commit, or push was started.

## 2026-10-09 final implementation continuation

- [x] Searched local datasets and caches for independent replacements for the
  eight degraded-source spans. The additional `data/Dataset_NQ_1min_2022_2025.csv`
  is NQ (not MNQ), has no Databento instrument mapping/provenance in its CSV
  header, and cannot certify MNQ bars or trade-channel completeness. Feature
  cache files are derived from the canonical MNQ batch and are not independent
  evidence. No source observations were copied into the canonical dataset.
- [x] Added a bounded read-only streaming probe:
  `src/paper/ibkr_streaming_probe.py`. It pins MNQZ6/conId 815824267/expiry
  20261218, checks the managed-account class without recording account IDs,
  requests delayed mode, and reports success only after a finite positive
  price tick (size-only callbacks do not count). It exposes no broker order API.
- [x] Main `.venv` TWS Paper probe on 127.0.0.1:7497 connected; one managed
  account had the `DU` Paper prefix; pinned contract resolved; callback reported
  market-data type 3 on the first direct probe. The repeatable 25-second
  streaming probe reported market-data type 3 but no price ticks. A separate
  snapshot request reported IBKR error 10167 (unsubscribed; delayed fallback
  announced). Delayed streaming is NOT confirmed. A separate historical request returned 1,190
  valid ordered 1-minute bars from 2026-10-08 22:00 UTC through 2026-10-09
  17:49 UTC; this confirms historical access only.
- [x] Offline delayed-Paper readiness recheck remains `ready=false`; the
  activation-ready causal bootstrap artifact is missing. The existing recovery
  validation record and October 8–31 calendar review passed their file/runtime
  checks.
- [x] Focused calendar, coverage-evidence, delayed CLI, bootstrap CLI, and
  streaming-probe tests: 28 passed.
- [x] Saved the repeatable streaming probe output at
  `results/diagnostics/ibkr_delayed_stream_probe_20261009.json` (SHA-256
  `0504d905fd0c24d8c553de6bf36f446e160b12b6dc1d5ec89788010753c32e7c`).
- [ ] Resolve source-degraded historical observations and produce a genuinely
  activation-ready bootstrap without changing the frozen schedule.
- [ ] Validate finalized-bar cadence and reconnect/backfill on a longer bounded
  polling run before Paper integration; a quote-tick stream is not required by
  the 1-minute OHLCV input contract.

No bootstrap, historical Paper replay, delayed Paper service, broker order,
Databento request, commit, or push was performed in this continuation.

## 2026-10-09 historical-polling integration check

- [x] Verified the date/session at 2026-10-09 18:09 UTC: the pinned TWS
  contract reports the Friday Globex session open; its 24-hour historical
  window contains the reviewed 21:00–22:00 UTC maintenance break.
- [x] Three read-only `reqHistoricalData` polls, 30 seconds apart, returned
  1,431 unique valid ordered bars, including one newly available minute on
  poll two. No malformed rows or out-of-order timestamps were found. The only
  gap was the reviewed maintenance break. A recent bar first appeared 597.6
  seconds after bar end, then changed in a later response before finalization.
  The poll analyzer keeps the conservative `historical_poll_not_validated`
  verdict when it sees any revision.
- [x] Finalization withheld the revised bar until it had the required age and
  repeated identical observations; no late revision was delivered. The
  cursor-aware acquisition smoke finalized 20 bars from 121 observations
  (31 unique bar starts), using at most two requests per 30-second cycle.
- [x] Restarted the isolated acquisition process with the same observation,
  delivery and cursor journals. It finalized two new bars; total finalized
  ledger count advanced from 20 to 22. All 22 contract/timestamp keys are
  unique and strictly increasing; no provider errors or late revisions.
- [x] Acquisition/finalization/recovery suites: 19 passed; full strategy
  recovery module: 15 passed. The combined run with `test_realtime_paper.py`
  encountered Windows ACL failures for pytest temp fixtures/session cleanup;
  the affected four-module run did not yield a valid suite summary. Running
  the recovery and adapter suites separately with repository-root basetemp
  succeeded.
- [x] Actual polling remains isolated from Paper execution. The existing
  delayed runner uses the historical TRADES polling source; no strategies or
  broker order APIs were connected during these tests.
- [x] Poll-gap diagnostics now use the reviewed calendar to distinguish
  scheduled closures, unresolved expected-session gaps, and uncertified dates;
  they do not equate sparse OHLCV with proven data loss. Polling/finalization
  and gap-classification tests: 28 passed.
- [ ] Obtain activation-ready bootstrap and clear historical source/calendar
  coverage gates before enabling strategy execution.
- [ ] Continue verifying real delayed-bar availability, finalization cadence,
  and disconnect/outage recovery during a longer but bounded test.

## 2026-10-09 final coverage certificate v5

- [x] V5 records source-condition date status, per-degraded-date observed row counts, and only exempts zero-row UTC dates when all 1,440 minutes are independently classified as scheduled closed.
- [x] Two zero-row degraded Saturdays (2026-01-31, 2026-03-21) are verified wholly inside the recurring weekly closure and excluded from MNQ training-impacting degraded dates; 15 degraded dates contain observed MNQ bars and remain flagged.
- [x] Final validator: `VALID` source/certificate integrity, 2,619,604 rows, 54,452 sparse OHLCV absent minutes not required by the minute grid, 8 unresolved spans / 1,051 minutes on degraded source dates, zero demonstrated missing required observations. Activation remains false.
- [ ] The eight degraded-source spans and 15 source-degraded dates containing observed MNQ rows still lack per-instrument completeness evidence; full bootstrap and delayed Paper remain blocked.
