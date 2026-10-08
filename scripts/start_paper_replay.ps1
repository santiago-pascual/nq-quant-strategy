param(
    [Parameter(Mandatory=$true)][string]$Start,
    [Parameter(Mandatory=$true)][string]$End,
    [string]$OutputDir = "results/paper/realtime-replay",
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$Python = [System.IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $Python)) { throw "Project venv Python not found: $Python" }
if (-not $env:PAPER_COST_CONFIG) {
    $env:PAPER_COST_CONFIG = "src/paper/config/topstepx_mnq_fees_2026-07.json"
}
$PythonArgs = @("-m", "src.paper.run_realtime_paper", "--command", "run", "--mode", "PAPER",
                "--replay-start", $Start, "--replay-end", $End,
                "--output-dir", $OutputDir, "--cost-config", $env:PAPER_COST_CONFIG)
if ($Resume) { $PythonArgs += "--resume" }
Push-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
try {
    & $Python @PythonArgs
    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $ExitCode
