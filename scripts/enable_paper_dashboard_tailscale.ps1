param(
    [int]$DashboardPort = 8501
)

$ErrorActionPreference = "Stop"
$tailscale = Get-Command tailscale -ErrorAction SilentlyContinue
if (-not $tailscale) {
    throw "Tailscale CLI is not installed. Install Tailscale on this Windows PC and your phone, sign both into the same private tailnet, then rerun this script."
}
$health = Invoke-WebRequest -Uri "http://127.0.0.1:$DashboardPort/_stcore/health" -TimeoutSec 3 -UseBasicParsing
if ($health.StatusCode -ne 200) { throw "The local dashboard is not healthy on 127.0.0.1:$DashboardPort." }

# Serve proxies the loopback-only Streamlit listener inside the authenticated
# tailnet. This does not enable Funnel or open a router/firewall port.
& $tailscale.Source serve --bg "http://127.0.0.1:$DashboardPort"
if ($LASTEXITCODE -ne 0) { throw "Tailscale Serve could not be enabled; no public exposure was attempted." }
& $tailscale.Source serve status
