$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this rollback from an elevated PowerShell window."
}
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $RepoRoot "results\paper\.dashboard_runtime\logs"
$BackupPath = Join-Path $LogDir "tailscale-firewall-backup.json"
Get-NetFirewallRule -DisplayName "MNQ Paper Dashboard Tailscale Phone" -ErrorAction SilentlyContinue | Remove-NetFirewallRule
if (-not (Test-Path -LiteralPath $BackupPath)) { throw "Firewall backup not found at $BackupPath; no Tailscale-In rules were restored." }
foreach ($entry in (Get-Content -LiteralPath $BackupPath -Raw | ConvertFrom-Json)) {
    Get-NetFirewallRule -Name $entry.Name -ErrorAction SilentlyContinue | Set-NetFirewallRule -Enabled ([bool]$entry.Enabled)
}
Write-Host "Removed dashboard-specific rule and restored prior Tailscale-In enabled states."
