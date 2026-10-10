param(
    [Parameter(Mandatory=$true)][string]$Start,
    [string]$End,
    [string]$OutputDir = "results/paper/realtime-replay",
    [switch]$Resume,
    [switch]$AutoResume,
    [switch]$Oos,
    [string]$DataEndExclusive,
    [string]$ReplayEndExclusive
)
$ErrorActionPreference = "Stop"
if ($Resume -and $AutoResume) { throw "Use either -Resume or -AutoResume, not both" }
if ($Oos -and (-not $DataEndExclusive -or -not $ReplayEndExclusive)) {
    throw "OOS replay requires -DataEndExclusive and -ReplayEndExclusive"
}
if (-not $Oos -and -not $End) { throw "Historical replay requires -End" }
if (-not $Oos -and ($DataEndExclusive -or $ReplayEndExclusive)) {
    throw "Use -DataEndExclusive and -ReplayEndExclusive only with -Oos"
}
$RepositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
if (-not [System.IO.Path]::IsPathRooted($OutputDir)) {
    $OutputDir = Join-Path $RepositoryRoot $OutputDir
}
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$Python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$Python = [System.IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $Python)) { throw "Project venv Python not found: $Python" }
if (-not $env:PAPER_COST_CONFIG) {
    $env:PAPER_COST_CONFIG = "src/paper/config/topstepx_mnq_fees_2026-07.json"
}
$PythonArgs = @("-m", "src.paper.run_realtime_paper", "--command", "run", "--mode", "PAPER",
                "--replay-start", $Start,
                "--output-dir", $OutputDir, "--cost-config", $env:PAPER_COST_CONFIG)
if ($Oos) {
    $PythonArgs += @("--oos-run", "--data-end-exclusive", $DataEndExclusive,
                     "--replay-end-exclusive", $ReplayEndExclusive)
}
else {
    $PythonArgs += @("--replay-end", $End)
}
$Checkpoint = Join-Path $OutputDir "paper_checkpoint.json"
$Events = Join-Path $OutputDir "events.jsonl"
$Database = Join-Path $OutputDir "paper_analytics.sqlite3"
if ($Resume -or ($AutoResume -and (Test-Path -LiteralPath $Checkpoint))) {
    $PythonArgs += "--resume"
}
elseif ($AutoResume -and ((Test-Path -LiteralPath $Events) -or (Test-Path -LiteralPath $Database))) {
    throw "Paper output has state but no checkpoint; refusing automatic restart"
}
Push-Location $RepositoryRoot
try {
    & $Python @PythonArgs
    $ExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $ExitCode
