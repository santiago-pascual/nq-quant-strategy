param(
    [string]$OutputDir = "results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3",
    [string]$DashboardAddress = "100.114.250.67",
    [int]$DashboardPort = 8501,
    [int]$WaitForAddressSeconds = 90,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogDir = Join-Path $RepoRoot "results\paper\.dashboard_runtime\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LaunchLog = Join-Path $LogDir "launcher.log"
$Process = $null
$Ready = $false
trap {
    $entry = "{0} ERROR {1}" -f (Get-Date -Format o), $_.Exception.ToString()
    Add-Content -LiteralPath $LaunchLog -Value $entry -Encoding utf8
    if ($Process -and -not $Ready) { & taskkill.exe /PID $Process.Id /T /F 2>$null | Out-Null }
    throw
}
$Python = Join-Path $RepoRoot ".venv-dashboard\Scripts\python.exe"
$App = Join-Path $RepoRoot "paper_dashboard\app.py"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Dashboard environment missing. Run scripts\install_paper_dashboard.ps1 first."
}
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $RepoRoot $OutputDir
}
if (-not (Test-Path -LiteralPath $OutputDir -PathType Container)) {
    throw "Paper output directory not found: $OutputDir"
}
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

# Bind only to the requested Tailscale address. Never fall back to 0.0.0.0 or
# another interface: that could expose the dashboard on Wi-Fi/Ethernet.
$BindAddress = [System.Net.IPAddress]::Parse($DashboardAddress)
function Test-AddressAssigned {
    foreach ($adapter in [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces()) {
        foreach ($unicast in $adapter.GetIPProperties().UnicastAddresses) {
            if ($unicast.Address.Equals($BindAddress)) { return $true }
        }
    }
    return $false
}
function Test-DashboardHealth([string]$Uri) {
    $handler = [System.Net.Http.HttpClientHandler]::new()
    $handler.UseProxy = $false
    $client = [System.Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(2)
    try {
        $response = $client.GetAsync($Uri).GetAwaiter().GetResult()
        $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult()
        return ($response.IsSuccessStatusCode -and $body -match "ok")
    } finally {
        $client.Dispose()
        $handler.Dispose()
    }
}
$addressDeadline = (Get-Date).AddSeconds([Math]::Max(0, $WaitForAddressSeconds))
while (-not (Test-AddressAssigned) -and (Get-Date) -lt $addressDeadline) { Start-Sleep -Seconds 2 }
if (-not (Test-AddressAssigned)) {
    throw "Dashboard address $DashboardAddress is not assigned to this PC. Confirm Tailscale is connected, then retry."
}
$probe = [System.Net.Sockets.TcpListener]::new($BindAddress, $DashboardPort)
try { $probe.Start(); $probe.Stop() } catch {
    try { $probe.Stop() } catch { }
    throw "Cannot bind $DashboardAddress`:$DashboardPort. Inspect the listener before restarting; this launcher will not select a different port."
}

$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$Stdout = Join-Path $LogDir "streamlit-$RunStamp.stdout.log"
$Stderr = Join-Path $LogDir "streamlit-$RunStamp.stderr.log"
$PidSuffix = "$DashboardAddress-$DashboardPort"
$PidFile = Join-Path $LogDir "streamlit-$PidSuffix.pid"
$LauncherPidFile = Join-Path $LogDir "streamlit-launcher-$PidSuffix.pid"
$IdentityFile = Join-Path $LogDir "streamlit-$PidSuffix.json"
$env:PAPER_MONITOR_OUTPUT_DIR = $OutputDir
$env:PAPER_MONITOR_DEFAULT_RUN = $OutputDir
$env:PAPER_MONITORING_API_PORT = "0"
$env:PAPER_DASHBOARD_LOG_DIR = $LogDir

$Arguments = "-m streamlit run `"$App`" --server.address $DashboardAddress --server.port $DashboardPort --server.headless true --browser.gatherUsageStats false"
$Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru
Set-Content -LiteralPath $PidFile -Value $Process.Id -Encoding ascii
$Url = "http://$DashboardAddress`:$DashboardPort"

for ($i = 0; $i -lt 40; $i++) {
    Start-Sleep -Milliseconds 500
    if ($Process.HasExited) { break }
    try { if (Test-DashboardHealth "$Url/_stcore/health") { $Ready = $true; break } } catch { }
}
if (-not $Ready) {
    $recent = @()
    foreach ($log in @($Stdout, $Stderr)) {
        if (Test-Path -LiteralPath $log) { $recent += Get-Content -LiteralPath $log -Tail 40 }
    }
    throw "Dashboard did not become healthy. PID=$($Process.Id). Logs: $Stdout and $Stderr`n$($recent -join "`n")"
}
$ListenerPid = $null
foreach ($line in (netstat -ano -p tcp)) {
    if ($line -match "^\s*TCP\s+$([regex]::Escape($DashboardAddress)):$DashboardPort\s+\S+\s+LISTENING\s+(\d+)\s*$") {
        $ListenerPid = [int]$Matches[1]
    }
}
if (-not $ListenerPid) {
    Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
    throw "No listener was found on $DashboardAddress`:$DashboardPort after a successful health check. Refusing ambiguous startup."
}
Set-Content -LiteralPath $PidFile -Value $ListenerPid -Encoding ascii
Set-Content -LiteralPath $LauncherPidFile -Value $Process.Id -Encoding ascii
$listenerProcess = Get-Process -Id $ListenerPid -ErrorAction Stop
$launcherProcess = Get-Process -Id $Process.Id -ErrorAction Stop
@{
    address = $DashboardAddress
    port = $DashboardPort
    listener_pid = $ListenerPid
    listener_start_utc_ticks = $listenerProcess.StartTime.ToUniversalTime().Ticks
    launcher_pid = $Process.Id
    launcher_start_utc_ticks = $launcherProcess.StartTime.ToUniversalTime().Ticks
    selected_run = $OutputDir
} | ConvertTo-Json | Set-Content -LiteralPath $IdentityFile -Encoding utf8
Add-Content -LiteralPath $LaunchLog -Value ("{0} HEALTHY url={1} listener_pid={2} run={3}" -f (Get-Date -Format o), $Url, $ListenerPid, $OutputDir) -Encoding utf8
if (-not $NoBrowser) { Start-Process $Url }
Write-Host "MNQ read-only Paper dashboard is healthy."
Write-Host "URL: $Url"
if ($DashboardAddress -eq "127.0.0.1") { Write-Host "Listener: $DashboardAddress`:$DashboardPort (loopback only)" } else { Write-Host "Listener: $DashboardAddress`:$DashboardPort (configured private interface only)" }
Write-Host "Selected run: $OutputDir"
Write-Host "Dashboard listener PID: $ListenerPid"
Write-Host "Dashboard launcher PID: $($Process.Id)"
Write-Host "Logs: $Stdout ; $Stderr"
Write-Host "Stop dashboard only: .\scripts\stop_paper_dashboard.ps1"
