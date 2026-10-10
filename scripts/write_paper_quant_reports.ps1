[CmdletBinding()]
param(
    [string]$RunDir = 'results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3',
    [switch]$Send
)
$ErrorActionPreference = 'Stop'
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$EasternNow = [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTimeOffset]::UtcNow, 'Eastern Standard Time')
$ReportDay = $EasternNow.Date.AddDays(-1).ToString('yyyy-MM-dd')
$ReportArgs = @('-m', 'src.paper.quant_report', '--run-dir', $RunDir, '--date', $ReportDay)
if ($Send) { $ReportArgs += '--send' }
$LogDir = Join-Path $ProjectRoot 'results/paper/.automation/logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$ReportLog = Join-Path $LogDir ("quant-report-" + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
Push-Location $ProjectRoot
try {
    & $Python @ReportArgs --period daily *>> $ReportLog
    if ($LASTEXITCODE -ne 0) { throw 'Daily quantitative report failed; inspect stderr.' }
    if ($EasternNow.DayOfWeek -eq [DayOfWeek]::Monday) {
        & $Python @ReportArgs --period weekly *>> $ReportLog
        if ($LASTEXITCODE -ne 0) { throw 'Weekly quantitative report failed; inspect stderr.' }
        # Checkpoint checks read large immutable snapshots; run weekly, outside
        # the Engine, not on every dashboard request or provider poll.
        & $Python -m src.paper.maintenance_health --run-dir $RunDir `
          --output ("results/diagnostics/quant_reports/" + (Split-Path $RunDir -Leaf) + '/maintenance_' + $ReportDay + '.json') *>> $ReportLog
    }
    & $Python -m src.paper.operational_health --run-dir $RunDir `
      --calendar 'src/paper/config/cme_mnq_calendar_2026-10-08_2026-10-31.json' `
      --output ("results/diagnostics/quant_reports/" + (Split-Path $RunDir -Leaf) + '/daily_health_' + $ReportDay + '.json') *>> $ReportLog
    # An unhealthy/unknown monitoring result is evidence to retain, not a reason
    # to reset the Engine or hide the successfully persisted performance report.
    Write-Host "Read-only reports completed. Logs: $ReportLog"
} finally { Pop-Location }
