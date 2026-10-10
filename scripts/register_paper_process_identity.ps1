param(
    [Parameter(Mandatory = $true)][string]$RunDir,
    [Parameter(Mandatory = $true)][int]$EnginePid
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RunPath = if ([IO.Path]::IsPathRooted($RunDir)) { $RunDir } else { Join-Path $Root $RunDir }
$RunPath = (Resolve-Path -LiteralPath $RunPath).Path
$StatusPath = Join-Path $RunPath "status.json"
if (-not (Test-Path -LiteralPath $StatusPath -PathType Leaf)) { throw "Paper status.json is required." }
$Status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
$State = [string]$Status.system.state
if ($State -notin @("RUNNING", "RECOVERING", "DEGRADED", "PAUSED_REFIT")) {
    throw "Refusing to register a process while persisted Paper state is $State."
}
$Process = Get-Process -Id $EnginePid -ErrorAction Stop
if ($Process.ProcessName -notmatch '^pythonw?$') { throw "The supplied PID is not a Python process." }
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) { throw "Project virtual environment is missing." }
Push-Location $Root
try {
    & $Python -m src.paper.process_identity_cli --run-dir $RunPath --pid $EnginePid
    if ($LASTEXITCODE -ne 0) { throw "The Python process identity check failed." }
} finally {
    Pop-Location
}
