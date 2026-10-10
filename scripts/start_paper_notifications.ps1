param(
    [string]$RunDir = "results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3",
    [string]$ReminderSchedule,
    [int]$EnginePid = 0,
    [switch]$SkipWatchdog
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunPath = if ([System.IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $Root $RunDir }
$RunPath = (Resolve-Path -LiteralPath $RunPath).Path
if (-not (Test-Path -LiteralPath $Python)) { throw "Project venv is missing: $Python" }
$CredentialStore = Join-Path $env:LOCALAPPDATA "MNQPaperDashboard\telegram.credentials.json"
if (-not (Test-Path -LiteralPath $CredentialStore -PathType Leaf)) {
    throw "Telegram credentials are not configured. Run: .\.venv\Scripts\python.exe -m src.paper.notification_cli configure"
}
$Runtime = Join-Path $Root ("results\paper\.notifications\" + (Split-Path $RunPath -Leaf))
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
$PidFile = Join-Path $Runtime "notifier.pid"
$WatchdogPidFile = Join-Path $Runtime "watchdog.pid"
 $pidFilesToCheck = if ($SkipWatchdog) { @($PidFile) } else { @($PidFile, $WatchdogPidFile) }
foreach ($candidatePidFile in $pidFilesToCheck) {
    if (Test-Path -LiteralPath $candidatePidFile) {
        $oldPid = 0
        [void][int]::TryParse((Get-Content -LiteralPath $candidatePidFile -Raw), [ref]$oldPid)
        if ($oldPid -gt 0) {
            $oldProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue
            if ($oldProcess -and $oldProcess.CommandLine -match "src\.paper\.notification_cli" -and
                $oldProcess.CommandLine -match [regex]::Escape((Split-Path $RunPath -Leaf))) {
                throw "Notification sidecar PID $oldPid already exists; inspect with notification_cli status first."
            }
        }
    }
}
$LogDir = Join-Path $Runtime "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Stdout = Join-Path $LogDir "notifier_$Stamp.stdout.log"
$Stderr = Join-Path $LogDir "notifier_$Stamp.stderr.log"
$logs = Get-ChildItem -LiteralPath $LogDir -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($logs.Count -gt 14) { $logs | Select-Object -Skip 14 | Remove-Item -Force }
$StopFile = Join-Path $Runtime "stop.request"
$WatchdogStopFile = Join-Path $Runtime "watchdog.stop.request"
Remove-Item -LiteralPath $StopFile -Force -ErrorAction SilentlyContinue
if (-not $SkipWatchdog) { Remove-Item -LiteralPath $WatchdogStopFile -Force -ErrorAction SilentlyContinue }
$Arguments = "-m src.paper.notification_cli run --run-dir `"$RunPath`""
if ($ReminderSchedule) {
    $SchedulePath = if ([IO.Path]::IsPathRooted($ReminderSchedule)) { $ReminderSchedule } else { Join-Path $Root $ReminderSchedule }
    if (-not (Test-Path -LiteralPath $SchedulePath -PathType Leaf)) { throw "Reviewed reminder schedule not found: $SchedulePath" }
    $Arguments += " --reminder-schedule `"$SchedulePath`""
}
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $Root `
    -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii
$WatchProcess = $null
try {
    if (-not $SkipWatchdog) {
        $WatchStdout = Join-Path $LogDir "watchdog_$Stamp.stdout.log"
        $WatchStderr = Join-Path $LogDir "watchdog_$Stamp.stderr.log"
        $WatchArguments = "-m src.paper.notification_cli watchdog --run-dir `"$RunPath`""
        if ($EnginePid -gt 0) { $WatchArguments += " --engine-pid $EnginePid" }
        $WatchProcess = Start-Process -FilePath $Python -ArgumentList $WatchArguments -WorkingDirectory $Root `
            -WindowStyle Hidden -RedirectStandardOutput $WatchStdout -RedirectStandardError $WatchStderr -PassThru
        Set-Content -LiteralPath $WatchdogPidFile -Value $WatchProcess.Id -Encoding ascii
    }
}
catch {
    Set-Content -LiteralPath $StopFile -Value "startup rollback" -Encoding ascii
    if ($WatchProcess -and -not $WatchProcess.HasExited) {
        Set-Content -LiteralPath $WatchdogStopFile -Value "startup rollback" -Encoding ascii
    }
    throw
}
Start-Sleep -Seconds 2
if ($Process.HasExited -or ($WatchProcess -and $WatchProcess.HasExited)) {
    $tail = if (Test-Path -LiteralPath $Stderr) { Get-Content -LiteralPath $Stderr -Tail 20 } else { @() }
    if (-not $Process.HasExited) { Set-Content -LiteralPath $StopFile -Value "startup rollback" -Encoding ascii }
    if ($WatchProcess -and -not $WatchProcess.HasExited) { Set-Content -LiteralPath $WatchdogStopFile -Value "startup rollback" -Encoding ascii }
    throw "Notification sidecar exited during startup. Review $Stderr`n$($tail -join "`n")"
}
$NotifierPid = $Process.Id
$HeartbeatPath = Join-Path $Runtime "notifier_heartbeat.json"
if (Test-Path -LiteralPath $HeartbeatPath -PathType Leaf) {
    try {
        $heartbeat = Get-Content -LiteralPath $HeartbeatPath -Raw | ConvertFrom-Json
        $heartbeatPid = 0
        if ($heartbeat.component -eq "paper_notification_service" -and
            $heartbeat.run_id -eq (Split-Path $RunPath -Leaf) -and
            [int]::TryParse([string]$heartbeat.pid, [ref]$heartbeatPid) -and $heartbeatPid -gt 0) {
            $heartbeatProcess = Get-Process -Id $heartbeatPid -ErrorAction SilentlyContinue
            if ($heartbeatProcess -and $heartbeatProcess.ProcessName -match "^pythonw?$") {
                # The service's own atomic heartbeat is authoritative if the
                # Windows launcher PID differs from the running interpreter.
                $NotifierPid = $heartbeatPid
                Set-Content -LiteralPath $PidFile -Value $NotifierPid -Encoding ascii
            }
        }
    } catch {
        # Keep the launcher PID when a heartbeat is unavailable or malformed;
        # the process lock remains the final duplicate-start guard.
    }
}
$watchdogText = if ($WatchProcess) { "; independent Paper watchdog PID=$($WatchProcess.Id)" } elseif ($SkipWatchdog) { "; existing Paper watchdog left untouched" } else { "; watchdog not started" }
Write-Host "Read-only notifier PID=$NotifierPid$watchdogText; run=$([IO.Path]::GetFileName($RunPath)); logs=$LogDir"
