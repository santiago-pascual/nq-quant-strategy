param([switch]$AdoptVerifiedListeners)
$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $RepoRoot "results\paper\.dashboard_runtime\logs"
$stopped = @()
if ($AdoptVerifiedListeners) {
    # Explicit repair of dashboard-only PID metadata. Never infer ownership
    # from port alone; both worker and parent must match this project's app.
    $ExpectedPython = Join-Path $RepoRoot '.venv-dashboard\Scripts\python.exe'
    foreach ($connection in (Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue)) {
        if ($connection.LocalAddress -notin @('127.0.0.1', '100.114.250.67')) { continue }
        $worker = Get-CimInstance Win32_Process -Filter "ProcessId = $($connection.OwningProcess)" -ErrorAction SilentlyContinue
        if (-not $worker -or $worker.Name -ne 'python.exe' -or
            $worker.CommandLine -notmatch [regex]::Escape($ExpectedPython) -or
            $worker.CommandLine -notmatch '-m\s+streamlit\s+run' -or
            $worker.CommandLine -notmatch 'paper_dashboard[\\/]app\.py' -or
            $worker.CommandLine -notmatch ('--server.address\s+' + [regex]::Escape($connection.LocalAddress) + '\s')) { continue }
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($worker.ParentProcessId)" -ErrorAction SilentlyContinue
        if (-not $parent -or $parent.Name -ne 'python.exe' -or $parent.CommandLine -ne $worker.CommandLine) { continue }
        $listener = Get-Process -Id $worker.ProcessId -ErrorAction Stop
        $launcher = Get-Process -Id $parent.ProcessId -ErrorAction Stop
        $record = @{ address=$connection.LocalAddress; port=8501; listener_pid=$listener.Id; launcher_pid=$launcher.Id;
            listener_start_utc_ticks=$listener.StartTime.ToUniversalTime().Ticks;
            launcher_start_utc_ticks=$launcher.StartTime.ToUniversalTime().Ticks }
        $record | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $LogDir "streamlit-$($connection.LocalAddress)-8501.json")
    }
}
foreach ($file in (Get-ChildItem -LiteralPath $LogDir -Filter "streamlit-*.json" -File -ErrorAction SilentlyContinue)) {
    $state = Get-Content -LiteralPath $file.FullName -Raw | ConvertFrom-Json
    $address = [string]$state.address
    $port = [int]$state.port
    $listenerPid = [int]$state.listener_pid
    $launcherPid = [int]$state.launcher_pid
    if (-not (netstat -ano -p tcp | Where-Object { $_ -match "^\s*TCP\s+$([regex]::Escape($address)):$port\s+\S+\s+LISTENING\s+$listenerPid\s*$" })) { continue }
    $listener = Get-Process -Id $listenerPid -ErrorAction SilentlyContinue
    $launcher = Get-Process -Id $launcherPid -ErrorAction SilentlyContinue
    if (-not $listener -or -not $launcher -or
        $listener.StartTime.ToUniversalTime().Ticks -ne [long]$state.listener_start_utc_ticks -or
        $launcher.StartTime.ToUniversalTime().Ticks -ne [long]$state.launcher_start_utc_ticks) { continue }
    Stop-Process -Id $launcherPid -Force -ErrorAction SilentlyContinue
    if ($listenerPid -ne $launcherPid) { Stop-Process -Id $listenerPid -Force -ErrorAction SilentlyContinue }
    Remove-Item -LiteralPath $file.FullName -Force -ErrorAction SilentlyContinue
    $stopped += "$address`:$port (PID $launcherPid tree)"
}
Write-Host "Stopped recorded dashboard process trees: $($stopped -join '; ')"
Write-Host "Paper Engine processes were not targeted."
