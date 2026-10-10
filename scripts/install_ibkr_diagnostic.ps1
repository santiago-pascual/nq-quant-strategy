param(
    [string]$Python = ""
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Venv = Join-Path $Root ".venv-ibkr-diagnostic"
if (-not $Python) {
    $ProjectPython = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $ProjectPython) { $Python = $ProjectPython }
    else { $Python = "py -3.13" }
}
if (-not (Test-Path (Join-Path $Venv "Scripts\python.exe"))) {
    if ($Python -eq "py -3.13") { & py -3.13 -m venv $Venv }
    else { & $Python -m venv $Venv }
    if ($LASTEXITCODE -ne 0) { throw "Could not create isolated IBKR diagnostic environment." }
}
$VenvPython = Join-Path $Venv "Scripts\python.exe"
& $VenvPython -m pip install --disable-pip-version-check -r (Join-Path $Root "requirements-ibkr-diagnostic.txt")
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned IBKR API client." }
Write-Host "Installed isolated IBKR diagnostic at $VenvPython"
