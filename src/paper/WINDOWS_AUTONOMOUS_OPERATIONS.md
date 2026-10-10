# Windows delayed Paper supervision

`mnq_paper_system.ps1` provides per-user startup tasks and read-only recovery
preflight for the existing delayed IBKR Paper run. It never places IBKR orders.
The Paper Engine auto-resume task is intentionally absent and
`engine_auto_resume_enabled` is hard-coded false. Do not enable unattended
Paper restart until the enabled recovery orchestration is validated end to end
and the restart policy is explicitly authorized.

## Install and operate

Run these from the repository root in an interactive PowerShell session after
the user has signed in. TWS requires an interactive desktop session and may
require manual password/2FA/API confirmation; this automation does not bypass
those controls.

```powershell
# Install per-user startup tasks. Supply the actual TWS executable if desired.
.\scripts\mnq_paper_system.ps1 -Mode Install -TwsExecutable 'C:\Path\To\Trader Workstation\tws.exe'

# Inspect checkpoint, delivery cursor, writer lock, TWS API/contract, dashboard,
# notifier and scheduled-task status. The TWS check is one bounded read-only API
# handshake and contractDetails lookup, not a market-data request.
.\scripts\mnq_paper_system.ps1 -Mode Check

# Gracefully request Engine stop if it is active; stop notifier/watchdog and
# dashboard sidecars. TWS and Tailscale remain running.
.\scripts\mnq_paper_system.ps1 -Mode Stop

# Restart only the read-only notifier/watchdog and dashboard sidecars.
.\scripts\mnq_paper_system.ps1 -Mode RestartSidecars

# Remove the per-user scheduled tasks; this leaves running processes and data
# untouched.
.\scripts\mnq_paper_system.ps1 -Mode Uninstall
```

No administrator rights are normally needed for per-user Task Scheduler tasks.
The installer registers tasks for optional TWS launch at login, the monitoring
loop, Telegram notifier plus independent process watchdog, and the read-only
Streamlit dashboard. TWS launch is skipped unless its executable path is
provided. Tailscale must already be installed and authenticated; the dashboard
bind remains the configured Tailscale IPv4 address. TWS starts first, then the
monitor, notifier, and dashboard after short logon delays. A TWS GUI launch is
not proof that authentication or the API is ready.

The monitor checks TWS through one read-only handshake and exact MNQZ6
contract lookup at a paced interval. It does not kill a TWS process. If TWS is
absent and an executable is configured, it makes at most three launch attempts
before requiring operator action. If TWS is present but the API is unavailable,
it reports manual login/2FA/API-settings/maintenance as possible causes and
does not launch another copy. TWS failure/recovery transitions are persisted in
the existing notification sidecar journal for deduplicated Telegram delivery.

## Paper Engine recovery policy

The supervisor preflight is read-only. It verifies the Paper-only run manifest
and MNQZ6 identity, checkpoint checksum/schema, runtime fingerprint, committed
timestamp, authoritative delivery reconciliation (including legitimate same-bar saves), writer lock availability,
process identity lease, reviewed calendar identity, persisted account/execution
state, and read-only IBKR handshake/contract identity. Any unavailable or
mismatched evidence blocks resume. It does not rewrite checkpoints, reconcile
journals, or start Paper. Same-bar checkpoint rewrites are evaluated by the authoritative recovery
validator, rather than requiring file-byte equality with an older acknowledgment.
Execution-critical runtime changes remain blocked; reporting compatibility uses
the existing explicitly versioned and validated rules.

The monitor may relaunch TWS only. It never restarts the Paper Engine. A future
Engine recovery task requires separate explicit authorization and a passing
preflight; it must use `delayed_paper_cli start --resume` with the same validated
bootstrap, calendar, recovery-validation, cost and run arguments. Never use the
historical replay launcher for delayed continuous Paper.

## Reliability boundaries

- Startup runs at user logon, not before login. Authentication, Windows Hello,
  IBKR 2FA, and interactive TWS consent remain manual when requested.
- Windows Update remains enabled. Configure active hours/restart notices in
  Windows Settings; do not disable security updates.
- For a desktop that must stay awake, use Windows Power settings or an
  administrator-approved AC power policy. Do not disable sleep/hibernation
  globally without deciding the power/thermal tradeoff. Reboot recovery still
  depends on power restoration, user login, TWS authentication, and the gates.
- Logs are under `results/paper/.automation/logs`, dashboard logs under
  `results/paper/.dashboard_runtime/logs`, and notification state under
  `results/paper/.notifications/<run-id>`. They contain operational diagnostics
  only; credentials remain in the existing user-scoped encrypted store.
- Task Scheduler restarts the monitor/notifier/dashboard after a process crash.
  Telegram queueing remains persistent if delivery is unavailable.

## Current installation and limitation

Per-user logon tasks are installed: `MNQ Paper TWS Login`,
`MNQ Paper Supervisor Monitor`, `MNQ Paper Read-only Dashboard`, and the
pre-existing `MNQ Paper Telegram Notifications` task was reused without being
overwritten. The supervisor monitor is currently running. Its latest persisted
read-only checks report the approved TWS contract, notifier, watchdog, and
dashboard healthy. No Paper Engine startup/restart task is installed, and the
config explicitly records `engine_auto_resume_enabled=false`.

## Verified operation — October 10, 2026

The user-authorized manual resume is active: Engine PID 39536, committed bar
2026-10-09 20:59 UTC, 1,857 processed bars, balance/equity $49,760.78, no open
positions. The reviewed calendar identifies the weekend closure. No new market
bars should be inferred from a fresh API connection alone.

`recovery_orchestrator.py` adds an isolated disabled-by-default recovery
controller with strict preflight, an OS lock, durable launch reservations,
cooldown and restart budget. Its concrete launcher uses the existing
`delayed_paper_cli start --resume`; it is not installed as an Engine task.
Process creation is `LAUNCHED_UNVERIFIED`, not successful feed recovery.
Post-launch identity registration and observed durable-progress verification
remain required before unattended Engine restart can be enabled. The current
Engine must not be restarted to test this mechanism.

## Read-only reports and health

```powershell
.\scripts\write_paper_quant_reports.ps1
.\scripts\write_paper_quant_reports.ps1 -Send
.\scripts\register_paper_quant_reports.ps1
.\scripts\register_paper_quant_reports.ps1 -Remove
.\.venv\Scripts\python.exe -m src.paper.operational_health --run-dir results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3 --calendar src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json --output results/diagnostics/current_paper_health.json
```

The installed `MNQ Paper Quant Reports` task runs at 06:15 PC-local time after
user login. Daily periods and Monday's prior-week report use New York calendar
boundaries. Reports never claim certified period completeness; current account
snapshots are distinguished from period-end equity. Delivery uses a separate
persistent journal, stable run/period IDs, retries and shared Telegram pacing.
Calendar review reminders are tied to the existing approved snapshot identity.
No future calendar coverage or contract roll is approved by a reminder.

Transient Windows status replacement warnings have been observed. Readers are
short-lived, but concurrent atomic replacement is not guaranteed on this
Windows installation. Status remains fresh; retained warnings must not be
silently treated as successful persistence. No Engine writer was changed.

See `AUTONOMOUS_ROADMAP_PROGRESS.md` for completed tests and pending acceptance
work. Task installation is not evidence that future scheduled execution passed.

## Continued acceptance and maintenance

An isolated scheduled-task execution of the read-only reporting script completed
with LastTaskResult=0. It did not send Telegram messages. Actual reboot/login
acceptance is still pending; the test task was removed.

```powershell
# Read-only; output is deliberately outside the production run.
.\.venv\Scripts\python.exe -m src.paper.maintenance_health --run-dir results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3 --output results/diagnostics/current_maintenance_health.json
.\scripts\start_paper_dashboard.ps1
# Only for stale dashboard PID metadata, after exact listener identity checks:
.\scripts\stop_paper_dashboard.ps1 -AdoptVerifiedListeners
```

Dashboard URLs: `http://100.114.250.67:8501` (private Tailscale) and
`http://127.0.0.1:8501` (desktop). No wildcard/public binding was introduced.
Stopping the dashboard does not stop Paper. Do not use it as an Engine recovery
command. Dashboard sources and theme may change without changing trading state.

Recovery post-launch verification distinguishes the Windows venv launcher from
the Python worker using exact creation/executable identity and direct kernel
parentage. It requires writer ownership and fresh status; open sessions require
observed durable progression. Closed-session waiting is a separate state.
The controller remains disabled and is not an installed Engine restart task.

RetrySafeCheckpointStore has passed isolated deterministic/real ORB crash tests,
but is not deployed into the active Engine. Production status replacement
warnings are retained and remain an infrastructure limitation. Installing that
writer requires compatible runtime validation and explicit restart permission.
