$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$BasePython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$DashboardEnv = Join-Path $RepoRoot ".venv-dashboard"
$DashboardPython = Join-Path $DashboardEnv "Scripts\python.exe"
$Requirements = Join-Path $RepoRoot "requirements-paper-dashboard-py313.txt"

if (-not (Test-Path -LiteralPath $BasePython)) {
    throw "Project Python 3.13 environment not found: $BasePython"
}
if (-not (Test-Path -LiteralPath $DashboardPython)) {
    & $BasePython -m venv $DashboardEnv
}
& $DashboardPython -m pip install --disable-pip-version-check -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Dashboard dependency installation failed." }
& $DashboardPython -m pip freeze | Set-Content -Encoding utf8 (Join-Path $RepoRoot "requirements-paper-dashboard-windows.lock.txt")
Write-Host "Dashboard environment ready at $DashboardEnv"
