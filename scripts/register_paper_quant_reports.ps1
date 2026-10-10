[CmdletBinding()]
param(
    [string]$RunDir = 'results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3',
    [switch]$Remove
)
$ErrorActionPreference = 'Stop'
$TaskName = 'MNQ Paper Quant Reports'
if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host 'Removed quantitative reporting task. Running Paper and its state are untouched.'
    exit 0
}
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PowerShellPath = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
if (-not $PowerShellPath) { $PowerShellPath = (Get-Command powershell.exe -ErrorAction Stop).Source }
$Launcher = Join-Path $ProjectRoot 'scripts/write_paper_quant_reports.ps1'
$RunPath = if ([IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $ProjectRoot $RunDir }
$RunPath = (Resolve-Path -LiteralPath $RunPath).Path
$Arguments = '-NoProfile -WindowStyle Hidden -ExecutionPolicy RemoteSigned -File "' + $Launcher + '" -RunDir "' + $RunPath + '" -Send'
$Action = New-ScheduledTaskAction -Execute $PowerShellPath -Argument $Arguments -WorkingDirectory $ProjectRoot
# 06:15 PC local time, after New York midnight for the current Windows setup.
# The reporting script computes New York date boundaries using its timezone.
$Trigger = New-ScheduledTaskTrigger -Daily -At '06:15'
$Settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 3 `
  -RestartInterval (New-TimeSpan -Minutes 10) -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
$Principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Force | Out-Null
Write-Host 'Installed read-only daily quantitative reporting at 06:15 PC local time; weekly reports on Monday. User login is required. No Engine restart is enabled.'
