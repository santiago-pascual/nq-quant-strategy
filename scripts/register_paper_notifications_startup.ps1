param(
    [string]$RunDir = "results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$TaskName = "MNQ Paper Telegram Notifications"
if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed the per-user Telegram notification startup task."
    exit 0
}
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Launcher = Join-Path $Root "scripts\start_paper_notifications.ps1"
$PowerShell = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
if (-not $PowerShell) { $PowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source }
$RunPath = if ([IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $Root $RunDir }
$RunPath = [IO.Path]::GetFullPath($RunPath)
$Arguments = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Launcher`" -RunDir `"$RunPath`""
$Action = New-ScheduledTaskAction -Execute $PowerShell -Argument $Arguments -WorkingDirectory $Root
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 2) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
Write-Host "Registered per-user logon startup for the read-only Telegram sidecar."
Write-Host "It does not start, stop or control the Paper Engine. Credentials remain DPAPI-encrypted in this user profile."
Write-Host "Remove with: .\scripts\register_paper_notifications_startup.ps1 -Remove"
