# June–October historical causal simulation

This is a separate, fresh simulated account, not forward Paper and not a change
to the 3,255-trade frozen Research benchmark.

## Verified inputs and present gates

The source-hash-checked canonical loader found 107,280 observed bars in
`[2026-06-20T00:00Z, 2026-10-08T00:00Z)`. Actual first/last observations are
June 21 22:00 UTC / October 7 23:59 UTC. June 20 is a weekend; it is not filled
with synthetic bars. Whole source: 2,619,604 bars, May 5, 2019 through October 8,
2026 13:02 UTC. Source-quality acceptance remains explicitly Paper-only.

No completed extension is currently published. Two gates prevent execution:

1. Reviewed supported-schema calendar for June 20–October 7. An October-only
   calendar cannot certify earlier sessions. Existing historical evidence may
   be reused only within its actual reviewed scope.
2. Activation-ready seed for June 20. The completed October artifact contains
   future observations and must never initialize this interval.

## Preparation and bootstrap

From the repository root, use the deterministic thread settings documented in
the README. Preparation reads canonical bars and manifests, does not fit models,
and writes only a separate diagnostic report:

```powershell
.\.venv\Scripts\python.exe -m src.paper.historical_extension prepare `
  --certificate results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_sparse_semantics_v5.json `
  --report results/diagnostics/historical_extension_preparation_v1.json
```

Exit 2 means prerequisites are missing; it does not imply prices are missing.
Once the calendar gate is satisfied, create an activation-specific seed through
the existing bootstrap (long work; keep the original October seed untouched):

```powershell
.\.venv\Scripts\python.exe -m src.paper.causal_bootstrap_cli start `
  --activation-utc 2026-06-20T00:00:00Z `
  --certificate results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_sparse_semantics_v5.json `
  --coverage-acceptance results/paper/coverage_acceptance/paper_research_quality_accepted_2019_2026_v5.json `
  --accept-paper-research-quality --confirm-full-bootstrap `
  --checkpoint results/paper/historical_extension_seed_20260620/checkpoint.json `
  --identity results/paper/historical_extension_seed_20260620/identity.json `
  --progress results/paper/historical_extension_seed_20260620/progress.json `
  --cache-dir results/paper/bootstrap_paper_accepted_20261008_1303/feature_cache
```

The cache validates prefix/source/config compatibility; a nonmatching cache is
not relabeled as a hit. Resume uses the same command with `resume` instead of
`start`. No bootstrap was launched during preparation.

## Replay, resume and publication

After both gates pass, supply the actual reviewed calendar paths:

```powershell
.\.venv\Scripts\python.exe -m src.paper.historical_extension run `
  --certificate results/diagnostics/mnq_calendar_coverage_2019-05-05_2026-10-08_sparse_semantics_v5.json `
  --calendar $ReviewedExtensionCalendar --calendar-review $ReviewedExtensionReview `
  --seed results/paper/historical_extension_seed_20260620/checkpoint.json `
  --seed-identity results/paper/historical_extension_seed_20260620/identity.json
```

`resume` retains identical arguments and restores the isolated checkpoint. The
pipeline uses all original strategy constructors, fresh account/risk state,
causal context, current versioned commission/exchange/regulatory fees, and zero
artificial slippage. No broker is connected. Source prefix, runtime, calendar,
costs and bounds must agree. Dedicated writer ownership and per-bar checkpoints
prevent competing execution; the retry store preserves authoritative envelopes.

Only a fully exhausted replay with matching processed count and last checkpoint
can publish `extension_analysis.json`. The dashboard verifies its source hashes,
trade identities and date bounds. Partial runs and changed evidence remain
unavailable. Research and Paper are unchanged; extension USD/R use an independent
baseline. Differences from full-sequence Research states are expected and must
not be optimized away. Completion proves infrastructure execution, not alpha or
Research parity. Stops/targets, fees and risk exclusions retain the engine rules.

The current July 2026 fee configuration is an explicitly selected scenario
cost profile. It is not evidence of the exact contemporaneous June fees.
Do not label the extension a reconstruction of those unverified historical fees.
