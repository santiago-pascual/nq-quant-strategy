# Local Paper monitoring dashboard

The dashboard is a read-only Streamlit and Plotly view over the existing authenticated GET-only Paper Monitoring API. The API remains loopback-only. The optional phone-access configuration binds only the Streamlit UI to this PC's Tailscale IPv4; it does not bind to Wi-Fi/Ethernet or a wildcard address. The UI has no order-submission or engine-configuration controls. Historical replay runs are labeled as historical replay; delayed IBKR Paper runs are labeled as delayed simulated fills.

## Windows start

From the repository root in PowerShell:

```powershell
& .\scripts\install_paper_dashboard.ps1
& .\scripts\start_paper_dashboard.ps1 -DashboardAddress 100.114.250.67 -DashboardPort 8501
```

The launcher defaults to `results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3`, binds only the supplied Tailscale address, opens the browser, and runs Streamlit in the background. It fails if that address is not assigned or if port 8501 is occupied; it never widens the listener or silently changes the URL. To choose a run explicitly:

```powershell
& .\scripts\start_paper_dashboard.ps1 -OutputDir results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3 -DashboardAddress 100.114.250.67 -DashboardPort 8501
```

It writes timestamped dashboard stdout/stderr and launcher errors to `results/paper/.dashboard_runtime/logs`. To make the dashboard start at the current Windows user's next logon (after reboot), register a limited-privilege task:

```powershell
& .\scripts\register_paper_dashboard_startup.ps1
```

The per-user logon entry starts the dashboard only; it does not control Paper Engine. Install it with:

```powershell
& .\scripts\register_paper_dashboard_startup.ps1
```

Remove it with `& .\scripts\register_paper_dashboard_startup.ps1 -Remove`. If the dashboard needs to be stopped, use the guarded stop script:

```powershell
& .\scripts\stop_paper_dashboard.ps1
```

This does not stop or alter the Paper engine. The Streamlit UI listens only on `100.114.250.67:8501`; its monitoring API uses an ephemeral loopback port and is not exposed to the tailnet or public networks. The selected run's status, database and persisted charts are read from that run directory only. The dashboard does not merge replay trades with a delayed Paper account. Desktop access uses the same Tailscale URL; Streamlit cannot bind one process simultaneously to loopback and a single interface IP without adding a proxy or second process.

## Pages and refresh

Command Center, Performance, Strategies, Trade Explorer, Positions & Orders, Risk Monitor and System Observatory refresh the selected view every 30 seconds. While a run initializes, missing values are marked unavailable. Delayed-feed committed bar, provider frontier, backlog and persisted feed errors are displayed separately from the engine state.

## Data limits

Account equity and drawdown use persisted account snapshots. Trade statistics use stored closed-trade rows and show the sample basis/truncation. The cumulative P&L chart is a closed-trade series, not account equity. Daily results group by exit date in America/New_York. HMM timelines display raw recorded states without semantic remapping. Risk limits, failures, data gaps, and progress percentages remain unavailable unless persisted. Trade price paths appear only when the selected run stored market bars for the selected trade.

All pages are read-only. No browser or API operation can submit broker orders. Never expose the monitoring API directly to a public network.

## Private phone access and firewall

From an elevated PowerShell, install/verify the narrow Windows Firewall rule:

```powershell
& .\scripts\configure_paper_dashboard_firewall.ps1
```

It disables Windows' broad `Tailscale-In` inbound allow rules (backing up their prior state), then permits only TCP 8501 from phone `100.104.47.115` to this PC's Tailscale IPv4 `100.114.250.67`, on the Tailscale adapter, for the dashboard Python executable. A rollback script restores the previous `Tailscale-In` enabled states. Then start the dashboard using the command above and open `http://100.114.250.67:8501` on the phone while Tailscale is connected. No router port-forward, Tailscale Funnel, public listener, TWS port, or IBKR API access is configured. If either device's tailnet IP changes, update the launcher and firewall rule accordingly. The dashboard's read-only monitoring API remains loopback-only.

## Telegram notifications

Notifications run as a separate read-only sidecar that tails this run's
append-only `events.jsonl`; it never imports the engine or writes trading
state. On first start it begins at the existing end of the file, so recovered
historical activity is not replayed as fresh alerts. Events marked
`RECOVERED_PAPER` are suppressed for trade-open/close alerts by default. The
delivery journal and cursor live under `results/paper/.notifications/<run>`.
Backoff and deduplication are persisted. Delivery is at-least-once around an
ambiguous network failure because Telegram does not provide a client
idempotency key.

Credentials are entered locally through hidden console prompts and encrypted
with Windows DPAPI for the current Windows user. The file is stored outside
the repository at `%LOCALAPPDATA%\MNQPaperDashboard\telegram.credentials.json`;
the plaintext token is never printed or written to a project file. Do not paste
the token into chat, source code, shell history or logs. Configure credentials
and send a clearly labeled connectivity test:

```powershell
& .\.venv\Scripts\python.exe -m src.paper.notification_cli configure
& .\.venv\Scripts\python.exe -m src.paper.notification_cli test `
  --run-dir results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3
```

Start/inspect/stop the event notifier and its independent Paper-process
watchdog without starting, stopping or controlling the Paper Engine:

```powershell
& .\scripts\start_paper_notifications.ps1
& .\.venv\Scripts\python.exe -m src.paper.notification_cli status `
  --run-dir results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3
& .\scripts\stop_paper_notifications.ps1
```

When Windows CIM process inspection is restricted, pass the Paper Python PID
explicitly with `-EnginePid <pid>` to the launcher; this override is suitable
for the current session only because process IDs change after restart. With
normal same-user Windows process visibility, the separate watchdog discovers
the Engine by its run directory and runner command. It never treats a missing
status file alone as proof that the Engine terminated.

To start both read-only sidecars automatically at the current user's logon
(including after reboot), register a limited-privilege scheduled task:

```powershell
& .\scripts\register_paper_notifications_startup.ps1
```

Remove it with `& .\scripts\register_paper_notifications_startup.ps1 -Remove`.
The notifier sends trade open/close, risk, feed and critical/recovery alerts;
the separate watchdog detects an unexpected Paper process exit and recovery.
First startup begins at the current end of the existing event file. Subsequent
catch-up trades tagged `RECOVERED_PAPER` are suppressed by default; explicitly
critical recovery errors remain visible. Delivery is deduplicated by persisted
event ID and rate-limited across both sidecars. Telegram has no client-side
idempotency key, so a process crash after Telegram accepts a message but before
the local delivery journal is fsynced can still result in one duplicate.

A daily summary is sent only after a non-recovered bar is observed on the
current New York date and persisted daily P&L is available. Calendar/contract
reminders are sent only from an optional reviewed JSON schedule passed with
`-ReminderSchedule`; each event must include `reviewed: true`, an effective UTC
timestamp and a source URL. No verified future contract-roll schedule is
currently installed, so no roll date is inferred from expiry. Logs and delivery
journals are under `results/paper/.notifications/<run>`; credentials are not.

Reviewed schedule example:

```json
{
  "schema_version": 1,
  "events": [
    {
      "id": "mnq-roll-review-2026-12",
      "kind": "contract_roll",
      "label": "Operator-reviewed MNQ contract transition",
      "effective_at_utc": "2026-12-01T00:00:00Z",
      "reviewed": true,
      "source_url": "https://www.cmegroup.com/"
    }
  ]
}
```

Only put a date in this file after verifying it from the relevant official
source; the example values above are schema placeholders and must not be used
as an actual MNQ roll schedule.

## Persistent Paper risk and execution alerts

`src.paper.monitoring_alerts` is an independent, read-only sidecar. It reads
the selected run's atomic `status.json`, append-only `events.jsonl`, and a
short-timeout SQLite connection opened with `mode=ro`. It writes its own
versioned state, alert transition JSONL, and Telegram delivery journal under
`results/paper/.notifications/<run>/`; it never writes Paper state or calls an
IBKR order API. First installation starts its event cursor at the current end
of the Paper journal to avoid sending historical catch-up as new alerts.

The monitor resolves `MARKET_OPEN`, `MARKET_CLOSED`, `MARKET_BREAK`, or
`MARKET_UNKNOWN` from the local MNQ snapshot and its matching product review
record. It requires the snapshot to cover the evaluated date; uncovered or
invalid dates stay `MARKET_UNKNOWN`. The evaluator is read-only and does not
advance feed cursors. A weekend/maintenance period suppresses stall alerts,
while a reported provider disconnect remains separately visible with the
calendar state in its details.

Provider and Paper processing stalls are evaluated separately. The accepted
delayed mode uses a 600-second provider delay, a 600-second finalization age,
and 30-second confirmation spacing. Delay and finalization age overlap, so the
expected bar frontier is based on `max(600, 600) + 30 = 630` seconds behind the
clock, not their sum. Default warning/critical tolerances beyond that expected
frontier are 300/900 seconds; processing-frontier lag thresholds are 300/900
seconds. Both stall conditions must persist for 60 seconds before notification.
These are operational monitoring tolerances, not statistically fitted limits
or trading rules: warning allows five minutes beyond the configured frontier;
critical allows fifteen minutes. The backlog warning is 60 bars (one hour of
one-minute bars). The persisted run contains 15,794 observation-to-finalization
measurements (31.9-second minimum, 132.7-second median, 774.4-second p90,
11,705.9-second maximum), but every associated event is marked
`RECOVERED_PAPER`; those catch-up measurements are not treated as a live-feed
latency distribution.

Daily loss reads the persisted Paper `max_daily_loss` risk policy and daily
P&L. The current configuration has a $500 hard daily loss gate. Drawdown is
measured from persisted Paper equity snapshots, but a separate account
drawdown rule is not configured; it remains unavailable for threshold alerts
instead of reusing the daily-loss limit or historical research drawdown.
Current persisted fills are recovered catch-up fills with zero modeled
slippage and no observed spread, so they do not establish a meaningful
slippage threshold. Execution/acquisition latency likewise lacks a
non-recovered timing distribution. Those thresholds intentionally remain
unset. Unknown inputs are shown as unavailable, never interpreted as healthy.
Conditions may transition to `RECOVERED`; event incidents remain active until
acknowledged or resolved with the sidecar CLI. The dashboard and API remain
GET-only.

Monitoring thresholds can be set before launching the sidecar:

```powershell
$env:MNQ_ALERT_BACKLOG_BARS = "60"
$env:MNQ_IBKR_DELAY_SECONDS = "600"
$env:MNQ_BAR_FINALIZATION_MIN_AGE_SECONDS = "600"
$env:MNQ_BAR_CONFIRMATION_SPACING_SECONDS = "30"
$env:MNQ_ALERT_PROVIDER_STALL_WARNING_SECONDS = "300"
$env:MNQ_ALERT_PROVIDER_STALL_CRITICAL_SECONDS = "900"
$env:MNQ_ALERT_PROCESSING_STALL_WARNING_SECONDS = "300"
$env:MNQ_ALERT_PROCESSING_STALL_CRITICAL_SECONDS = "900"
$env:MNQ_ALERT_STALL_DEBOUNCE_SECONDS = "60"
# Leave these unset until a non-recovered timing/fill sample supports them:
# MNQ_ALERT_SLIPPAGE_WARNING_TICKS / MNQ_ALERT_SLIPPAGE_CRITICAL_TICKS
# MNQ_ALERT_EXECUTION_LATENCY_WARNING_MS
# MNQ_ALERT_ACQUISITION_LATENCY_WARNING_SECONDS
# MNQ_ALERT_DRAWDOWN_WARNING_USD / MNQ_ALERT_DRAWDOWN_CRITICAL_USD
```

The notifier writes an atomic heartbeat every poll. The independent watchdog
checks this heartbeat even if Windows process enumeration is unavailable. New
delayed-Paper processes can be registered with
`scripts/register_paper_process_identity.ps1`; it writes
`paper_process_identity.json` with the run ID, PID, executable and OS
process-creation identity. The watchdog verifies that exact process directly,
avoiding global CIM enumeration and PID-reuse false positives. A stale lease
from a crashed process is treated as absent; corrupt or inaccessible identity
data remains unavailable, never healthy. Register only after the persisted
status is RUNNING:

```powershell
& .\scripts\register_paper_process_identity.ps1 -RunDir "results\paper\delayed_mnqz6_paper_accepted_20261008_1303_r3" -EnginePid <PAPER_PID>
```
watchdog checks the Paper `status.json` heartbeat while the Paper process is
visible, waiting through a persistence interval before reporting a stale
writer. The watchdog emits a recovery event when a stale heartbeat resumes. The
Paper alert monitor remains responsible for persisted acquisition errors and
backlog, avoiding a duplicate watchdog notification while the notifier is
healthy. If the notifier stops, the watchdog reports its stale heartbeat using
its own delivery process. TWS maintenance is not inferred from error 1100 or
from a weekend; an operator maintenance note may be supplied with
`MNQ_TWS_MAINTENANCE_NOTE` and is displayed as unverified operator context.

Broker-fill, broker order lifecycle, and broker-position reconciliation remain
disabled. Strategy anomalies are emitted only for explicit persisted
invariant/input/out-of-session diagnostics; a losing or unusual-but-valid
trade is not an anomaly. The future live broker monitor remains an interface
only and is not instantiated or enabled.

Start/stop the two notification-sidecar processes (not Paper) and inspect their
state with:

```powershell
& .\scripts\start_paper_notifications.ps1 -RunDir "results\paper\delayed_mnqz6_paper_accepted_20261008_1303_r3"
& .\scripts\stop_paper_notifications.ps1 -RunName "delayed_mnqz6_paper_accepted_20261008_1303_r3"
& .\.venv\Scripts\python.exe -m src.paper.notification_cli status --run-dir "results\paper\delayed_mnqz6_paper_accepted_20261008_1303_r3"
```

If the independent watchdog is already running and only the updated notifier
needs to be started, pass `-SkipWatchdog`; this leaves that watchdog process and
its state untouched:

```powershell
& .\scripts\start_paper_notifications.ps1 -RunDir "results\paper\delayed_mnqz6_paper_accepted_20261008_1303_r3" -SkipWatchdog
```

Send the clearly labeled connectivity test only when desired:

```powershell
& .\.venv\Scripts\python.exe -m src.paper.notification_cli test --run-dir "results\paper\delayed_mnqz6_paper_accepted_20261008_1303_r3"
```

Telegram delivery errors are journaled as sanitized categories. Transient
network/server failures use bounded exponential retry; HTTP 429 honors
Telegram's retry-after value. Permanent authentication, chat/access or message
format rejections pause the same pending event without advancing its cursor.
After correcting the configuration, explicitly requeue that same event ID:

```powershell
& .\.venv\Scripts\python.exe -m src.paper.notification_cli retry-pending --channel event --run-dir "results\paper\delayed_mnqz6_paper_accepted_20261008_1303_r3"
```

Run one local evaluation without starting a background service or sending a
message:

```powershell
& .\.venv\Scripts\python.exe -m src.paper.notification_cli monitor-once --run-dir "results\paper\delayed_mnqz6_paper_accepted_20261008_1303_r3"
```

To acknowledge an alert without changing Paper state:

```powershell
& .\.venv\Scripts\python.exe -m src.paper.notification_cli acknowledge --run-dir "results\paper\delayed_mnqz6_paper_accepted_20261008_1303_r3" --alert-id <alert-id>
```

Add `--resolve` only after the incident is resolved. Telegram retries use the
existing persistent delivery journal, exponential backoff and cross-process
rate limiter. Telegram cannot provide exactly-once delivery after an
ambiguous network failure; one duplicate remains possible after Telegram
accepts a message but before local fsync.

## Research-to-Paper Analytics

The Streamlit **Analytics** workspace reads the frozen independent Research OOS portfolio at `src/research/results/portfolio/independent_reproduction/` and the selected run's closed outcomes from the authenticated, loopback-only `/v1/analytics` endpoint. Research output CSV hashes and component-source hashes are checked against the validated artifact identity before rendering. The API uses a bounded read-only SQLite query (up to 5,000 closed Paper outcomes); chart refresh does not modify Paper state.

The comparable chart uses Research `r_multiple` and Paper `realized_r` (net Paper P&L divided by its persisted initial risk). Paper is rebased to the Research endpoint for display only. The interval between Research OOS end (2026-06-19) and the actual activation timestamp in the selected delayed-Paper manifest is blank; no returns are interpolated. Native Research R and Paper USD account equity are also shown separately. If the selected run does not have a valid delayed-Paper manifest, the dashboard omits the transition marker and Paper baseline instead of assuming an activation date.

Rolling metrics are trade ordered and require the selected window to be populated. Alpha reference ranges use a deterministic moving-block bootstrap of the chronological Research R series; they are descriptive, preserve short-range serial dependence, and are not a hypothesis test. A small Paper sample is explicitly marked insufficient. Execution cost and latency charts only show persisted fields; broker slippage and broker execution latency are unavailable for simulated fills. Historical Research regime attribution is unavailable because the validated trade ledger has no HMM-state field. Paper raw HMM IDs may be reported without semantic remapping.

Launch the local dashboard with the existing Windows script:

```powershell
& .\scripts\start_paper_dashboard.ps1
```

Open the configured local/Tailscale dashboard URL and select **Analytics**.
