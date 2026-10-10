param(
    [string]$RunName = "delayed_mnqz6_paper_accepted_20261008_1303_r3"
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runtime = Join-Path $Root ("results\paper\.notifications\" + $RunName)
if (-not (Test-Path -LiteralPath $Runtime -PathType Container)) {
    throw "Notification runtime state not found: $Runtime"
}
Set-Content -LiteralPath (Join-Path $Runtime "stop.request") -Value "operator requested graceful notifier stop" -Encoding ascii
Set-Content -LiteralPath (Join-Path $Runtime "watchdog.stop.request") -Value "operator requested graceful watchdog stop" -Encoding ascii
Write-Host "Graceful notifier and watchdog stops requested. Paper Engine is unaffected."
