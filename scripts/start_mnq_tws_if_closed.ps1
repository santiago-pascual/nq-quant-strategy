param([Parameter(Mandatory=$true)][string]$Executable)
$ErrorActionPreference = 'Stop'
$exePath = (Resolve-Path -LiteralPath $Executable).Path
if (-not (Test-Path -LiteralPath $exePath -PathType Leaf)) { throw 'Configured TWS executable does not exist.' }
$name = [IO.Path]::GetFileNameWithoutExtension($exePath)
$logDir = Join-Path (Split-Path $PSScriptRoot -Parent) 'results\paper\.automation\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir 'tws-supervisor.log'
if ((Test-Path -LiteralPath $log) -and (Get-Item -LiteralPath $log).Length -ge 2MB) {
    for ($index=4; $index -ge 1; $index--) {
        $source = "$log.$index"; $destination = "$log.$($index+1)"
        if (Test-Path -LiteralPath $source) { Move-Item -LiteralPath $source -Destination $destination -Force }
    }
    Move-Item -LiteralPath $log -Destination "$log.1" -Force
}
function Get-TwsPresence {
    if (@(Get-Process -Name $name -ErrorAction SilentlyContinue).Count -gt 0) { return 'PRESENT' }
    $ambiguous = $false
    foreach ($java in @(Get-Process -Name java,javaw -ErrorAction SilentlyContinue)) {
        try {
            $image = [string]$java.MainModule.FileName
            if ($image -match '(?i)(\\Jts\\|Trader Workstation|\\IBGateway\\)') { return 'PRESENT' }
        } catch { $ambiguous = $true }
    }
    if ($ambiguous) { return 'UNKNOWN' }
    return 'ABSENT'
}
$presence = Get-TwsPresence
if ($presence -ne 'ABSENT') {
    # Ambiguity is treated as presence to avoid launching a duplicate GUI or
    # terminating/altering a possibly healthy TWS process.
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) TWS presence=$presence; launch skipped."
    exit 0
}
try {
    Start-Process -FilePath $exePath -WorkingDirectory (Split-Path $exePath -Parent) | Out-Null
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) TWS launch requested; interactive authentication may be required."
} catch {
    Add-Content -LiteralPath $log -Value "$(Get-Date -Format o) TWS launch failed: $($_.Exception.GetType().Name)"
    exit 1
}
