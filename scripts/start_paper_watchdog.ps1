param([string]$RunDir = 'results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3')
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$RunPath = if ([IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $Root $RunDir }
$RunPath = (Resolve-Path -LiteralPath $RunPath).Path
$Runtime = Join-Path $Root ("results\paper\.notifications\" + (Split-Path $RunPath -Leaf))
New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
$PidFile = Join-Path $Runtime 'watchdog.pid'
if (Test-Path -LiteralPath $PidFile) {
    $prior = 0
    [void][int]::TryParse((Get-Content -LiteralPath $PidFile -Raw).Trim(),[ref]$prior)
    if ($prior -gt 0 -and (Get-Process -Id $prior -ErrorAction SilentlyContinue)) { throw "Watchdog PID $prior is still active; refusing duplicate startup." }
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Project .venv Python is missing.' }
Remove-Item -LiteralPath (Join-Path $Runtime 'watchdog.stop.request') -Force -ErrorAction SilentlyContinue
$LogDir = Join-Path $Runtime 'logs'; New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$process = Start-Process -FilePath $Python -ArgumentList @('-m','src.paper.notification_cli','watchdog','--run-dir',"`"$RunPath`"") `
    -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $LogDir "watchdog_$stamp.stdout.log") `
    -RedirectStandardError (Join-Path $LogDir "watchdog_$stamp.stderr.log") -PassThru
Set-Content -LiteralPath $PidFile -Value $process.Id -Encoding ascii
Start-Sleep -Seconds 2
if ($process.HasExited) { throw "Watchdog exited during startup; inspect $LogDir" }
Write-Host "Independent read-only Paper watchdog started. PID=$($process.Id); run=$(Split-Path $RunPath -Leaf)"
