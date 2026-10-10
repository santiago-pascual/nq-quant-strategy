param(
    [string]$OutputDir = "results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3",
    [string]$DashboardAddress = "100.114.250.67",
    [int]$DashboardPort = 8501,
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Launcher = Join-Path $RepoRoot "scripts\start_paper_dashboard.ps1"
$RunKey = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Run"
$ValueName = "MNQPaperDashboardTailscale"
if ($Remove) {
    Remove-ItemProperty -Path $RunKey -Name $ValueName -ErrorAction SilentlyContinue
    Write-Host "Removed this user's MNQ dashboard logon startup entry."
    exit 0
}
$PowerShell = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
if (-not $PowerShell) { $PowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source }
$command = "`"$PowerShell`" -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Launcher`" -OutputDir `"$OutputDir`" -DashboardAddress `"$DashboardAddress`" -DashboardPort $DashboardPort -WaitForAddressSeconds 180 -NoBrowser"
New-Item -Path $RunKey -Force | Out-Null
Set-ItemProperty -Path $RunKey -Name $ValueName -Value $command -Type String
Write-Host "Registered per-user dashboard startup at logon (no administrator rights required)."
Write-Host "The launcher waits up to 180 seconds for Tailscale and writes errors to results\paper\.dashboard_runtime\logs\launcher.log."
Write-Host "Remove with: .\scripts\register_paper_dashboard_startup.ps1 -Remove"
