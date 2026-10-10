param(
    [string]$LocalTailscaleAddress = "100.114.250.67",
    [string]$PhoneTailscaleAddress = "100.104.47.115",
    [int]$DashboardPort = 8501
)

$ErrorActionPreference = "Stop"
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an elevated PowerShell window. It narrows Tailscale inbound access to the phone and dashboard port."
}
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Program = (Resolve-Path (Join-Path $RepoRoot ".venv-dashboard\Scripts\python.exe")).Path
$LogDir = Join-Path $RepoRoot "results\paper\.dashboard_runtime\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$BackupPath = Join-Path $LogDir "tailscale-firewall-backup.json"
$tailscaleAddress = Get-NetIPAddress -AddressFamily IPv4 -IPAddress $LocalTailscaleAddress -ErrorAction Stop
if ($tailscaleAddress.InterfaceAlias -ne "Tailscale") {
    throw "$LocalTailscaleAddress is assigned to '$($tailscaleAddress.InterfaceAlias)', not the expected Tailscale adapter. No rule was changed."
}
$ruleName = "MNQ Paper Dashboard Tailscale Phone"
$tailscaleDefaults = @(Get-NetFirewallRule -DisplayName "Tailscale-In" -ErrorAction SilentlyContinue)
if (-not (Test-Path -LiteralPath $BackupPath)) {
    $backup = @($tailscaleDefaults | ForEach-Object { [pscustomobject]@{ Name=$_.Name; Enabled=[bool]$_.Enabled } })
    $backup | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $BackupPath -Encoding utf8
}
try {
    # The installer-provided Tailscale-In rule allows every port from every
    # tailnet peer. Disable it and retain only this explicit phone/dashboard rule.
    $tailscaleDefaults | Disable-NetFirewallRule
    Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    New-NetFirewallRule -DisplayName $ruleName `
        -Description "Allow only the specified Tailscale phone to read the local MNQ Paper dashboard." `
        -Direction Inbound -Action Allow -Enabled True -Profile Any `
        -Protocol TCP -LocalPort $DashboardPort `
        -LocalAddress $LocalTailscaleAddress -RemoteAddress $PhoneTailscaleAddress `
        -InterfaceAlias Tailscale -Program $Program | Out-Null
    $rule = Get-NetFirewallRule -DisplayName $ruleName
    $addresses = Get-NetFirewallAddressFilter -AssociatedNetFirewallRule $rule
    $ports = Get-NetFirewallPortFilter -AssociatedNetFirewallRule $rule
    $application = Get-NetFirewallApplicationFilter -AssociatedNetFirewallRule $rule
    if ($rule.Action -ne "Allow" -or $ports.LocalPort -notcontains "$DashboardPort" -or
        $addresses.LocalAddress -notcontains $LocalTailscaleAddress -or
        $addresses.RemoteAddress -notcontains $PhoneTailscaleAddress -or
        $application.Program -ne $Program -or
        (Get-NetFirewallRule -DisplayName "Tailscale-In" -ErrorAction SilentlyContinue | Where-Object Enabled)) {
        throw "Firewall rule verification failed."
    }
} catch {
    Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule
    if (Test-Path -LiteralPath $BackupPath) {
        foreach ($entry in (Get-Content -LiteralPath $BackupPath -Raw | ConvertFrom-Json)) {
            Get-NetFirewallRule -Name $entry.Name -ErrorAction SilentlyContinue | Set-NetFirewallRule -Enabled ([bool]$entry.Enabled)
        }
    }
    throw
}
Write-Host "Verified narrow access: $PhoneTailscaleAddress -> $LocalTailscaleAddress`:$DashboardPort (TCP, Tailscale adapter, dashboard Python only)."
Write-Host "Disabled broad installer Tailscale-In allow rules; previous states saved to $BackupPath."
Write-Host "Rollback: .\scripts\restore_paper_dashboard_firewall.ps1 (elevated PowerShell)."
