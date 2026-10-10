[CmdletBinding()]
param(
    [ValidateSet('Install','Check','Monitor','Stop','RestartSidecars','Uninstall')]
    [string]$Mode = 'Check',
    [string]$RunDir = 'results/paper/delayed_mnqz6_paper_accepted_20261008_1303_r3',
    [string]$TwsExecutable,
    [string]$TwsHost = '127.0.0.1',
    [int]$TwsPort = 7497,
    [string]$DashboardAddress = '100.114.250.67',
    [int]$DashboardPort = 8501,
    [switch]$SkipTwsProbe
)
$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RunPath = if ([IO.Path]::IsPathRooted($RunDir)) { [IO.Path]::GetFullPath($RunDir) } else { [IO.Path]::GetFullPath((Join-Path $Root $RunDir)) }
$RunName = Split-Path $RunPath -Leaf
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$PowerShell = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
if (-not $PowerShell) { $PowerShell = (Get-Command powershell.exe -ErrorAction Stop).Source }
$Runtime = Join-Path $Root 'results\paper\.automation'
$LogDir = Join-Path $Runtime 'logs'
$ConfigPath = Join-Path $Runtime 'windows-config.json'
$TaskPrefix = 'MNQ Paper '
$script:CreatedTasks = @()
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Log([string]$Message) {
    $path = Join-Path $LogDir 'windows-supervisor.log'
    if ((Test-Path -LiteralPath $path) -and (Get-Item -LiteralPath $path).Length -ge 2MB) {
        for ($index=4; $index -ge 1; $index--) {
            $source = "$path.$index"; $destination = "$path.$($index+1)"
            if (Test-Path -LiteralPath $source) { Move-Item -LiteralPath $source -Destination $destination -Force }
        }
        Move-Item -LiteralPath $path -Destination "$path.1" -Force
    }
    Add-Content -LiteralPath $path -Value "$(Get-Date -Format o) $Message" -Encoding utf8
}
function Read-Config {
    if (Test-Path -LiteralPath $ConfigPath) { return Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json }
    return [pscustomobject]@{schema_version=1; run_dir=$RunPath; tws_executable=$null; tws_host=$TwsHost; tws_port=$TwsPort; engine_auto_resume_enabled=$false}
}
function Invoke-TwsProbe {
    $args = @('-m','src.paper.windows_supervisor','--tws-probe-only','--tws-host',$TwsHost,'--tws-port',[string]$TwsPort)
    $output = & $Python @args 2>&1
    $code = $LASTEXITCODE
    $parsed = $null
    try { $parsed = ($output -join "`n") | ConvertFrom-Json } catch { }
    return [pscustomobject]@{success=($code -eq 0 -and $parsed.verified); report=$parsed; raw=($output -join "`n")}
}
function Start-TwsIfClosed([string]$Executable) {
    if (-not $Executable -or -not (Test-Path -LiteralPath $Executable -PathType Leaf)) { Write-Log 'TWS executable not configured; manual login/start required.'; return $false }
    $name = [IO.Path]::GetFileNameWithoutExtension($Executable)
    if (@(Get-Process -Name $name -ErrorAction SilentlyContinue).Count -gt 0) { return $false }
    & (Join-Path $PSScriptRoot 'start_mnq_tws_if_closed.ps1') -Executable $Executable
    return $true
}
function Get-TwsPresence([string]$Executable) {
    if (-not $Executable) { return 'UNKNOWN' }
    $name = [IO.Path]::GetFileNameWithoutExtension($Executable)
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
function Test-PidFileAlive([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { return $false }
    $target = 0
    if (-not [int]::TryParse((Get-Content -LiteralPath $Path -Raw).Trim(),[ref]$target) -or $target -le 0) { return $false }
    return [bool](Get-Process -Id $target -ErrorAction SilentlyContinue)
}
function Start-SidecarLauncher([string]$Name,[string]$Script,[string[]]$Arguments) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $out = Join-Path $LogDir ("$Name-$stamp.stdout.log")
    $err = Join-Path $LogDir ("$Name-$stamp.stderr.log")
    $quoted = @($Arguments | ForEach-Object { '"' + ([string]$_ -replace '"','`"') + '"' })
    $argumentText = '-NoProfile -ExecutionPolicy RemoteSigned -WindowStyle Hidden -File "' + $Script + '" ' + ($quoted -join ' ')
    Start-Process -FilePath $PowerShell -ArgumentList $argumentText -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err | Out-Null
    Write-Log "$Name launcher scheduled; logs=$out;$err"
}
function Send-TwsCondition([bool]$Active,[string]$Detail,[string]$Observed) {
    $condition = if ($Active) { 'active' } else { 'recovered' }
    $safeDetail = $Detail -replace '[\r\n]+',' '
    $args = @('-m','src.paper.windows_supervisor','--run-dir',$RunPath,'--record-tws-condition',$condition,'--details',$safeDetail,'--observed',$Observed)
    & $Python @args 2>&1 | Out-Null
}
function Send-ComponentCondition([string]$Component,[bool]$Active,[string]$Detail,[string]$Observed) {
    $condition = if ($Active) { 'active' } else { 'recovered' }
    $severity = if ($Component -in @('notifier','watchdog')) { 'CRITICAL' } else { 'WARNING' }
    $args = @('-m','src.paper.windows_supervisor','--run-dir',$RunPath,'--record-condition',$condition,
              '--condition',"windows-$Component",'--title',"Paper $Component unavailable",'--severity',$severity,
              '--details',($Detail -replace '[\r\n]+',' '),'--observed',$Observed)
    & $Python @args 2>&1 | Out-Null
}
function Register-LogonTask([string]$Name,[string]$ScriptPath,[string[]]$Arguments,[int]$DelaySeconds=0) {
    $existing = Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
    if ($existing) {
        $oldArguments = [string]$existing.Actions[0].Arguments
        $expected = @($ScriptPath) + $Arguments
        if ($oldArguments.Contains($ScriptPath) -and $oldArguments.Contains($RunPath)) {
            Write-Log "Reused existing matching startup task '$Name'; it was not overwritten."
            return $false
        }
        throw "Scheduled task '$Name' already exists with a different action; inspect/uninstall it manually before replacing."
    }
    $argumentText = '-NoProfile -ExecutionPolicy RemoteSigned -WindowStyle Hidden -File "' + $ScriptPath + '"'
    foreach ($item in $Arguments) { $argumentText += ' "' + ($item -replace '"','`"') + '"' }
    $action = New-ScheduledTaskAction -Execute $PowerShell -Argument $argumentText -WorkingDirectory $Root
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    if ($DelaySeconds -gt 0) { $trigger.Delay = "PT${DelaySeconds}S" }
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 2) -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force | Out-Null
    $script:CreatedTasks += $Name
    return $true
}
function Request-SidecarStops {
    $notificationRuntime = Join-Path $Root ("results\paper\.notifications\" + $RunName)
    New-Item -ItemType Directory -Force -Path $notificationRuntime | Out-Null
    Set-Content -LiteralPath (Join-Path $notificationRuntime 'stop.request') -Value 'operator requested sidecar stop' -Encoding ascii
    Set-Content -LiteralPath (Join-Path $notificationRuntime 'watchdog.stop.request') -Value 'operator requested sidecar stop' -Encoding ascii
    $monitorStop = Join-Path $Runtime 'monitor.stop.request'
    Set-Content -LiteralPath $monitorStop -Value 'operator requested monitor stop' -Encoding ascii
}
function Wait-RecordedProcesses([string[]]$PidFiles,[int]$TimeoutSeconds=30) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    do {
        $alive = @()
        foreach ($path in $PidFiles) {
            if (-not (Test-Path -LiteralPath $path)) { continue }
            $target=0
            if ([int]::TryParse((Get-Content -LiteralPath $path -Raw).Trim(),[ref]$target) -and $target -gt 0 -and (Get-Process -Id $target -ErrorAction SilentlyContinue)) { $alive += $target }
        }
        if ($alive.Count -eq 0) { return $true }
        Start-Sleep -Seconds 1
    } while ((Get-Date) -lt $deadline)
    return $false
}

switch ($Mode) {
    'Install' {
        if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Project .venv Python is missing.' }
        if ($TwsExecutable) {
            $TwsExecutable = (Resolve-Path -LiteralPath $TwsExecutable).Path
            if (-not (Test-Path -LiteralPath $TwsExecutable -PathType Leaf)) { throw 'TWS executable path is invalid.' }
        }
        try {
            if ($TwsExecutable) { [void](Register-LogonTask ($TaskPrefix + 'TWS Login') (Join-Path $PSScriptRoot 'start_mnq_tws_if_closed.ps1') @('-Executable',$TwsExecutable)) }
            [void](Register-LogonTask ($TaskPrefix + 'Supervisor Monitor') $PSCommandPath @('-Mode','Monitor','-RunDir',$RunPath,'-TwsHost',$TwsHost,'-TwsPort',[string]$TwsPort) 20)
            [void](Register-LogonTask ($TaskPrefix + 'Telegram Notifications') (Join-Path $PSScriptRoot 'start_paper_notifications.ps1') @('-RunDir',$RunPath) 10)
            [void](Register-LogonTask ($TaskPrefix + 'Read-only Dashboard') (Join-Path $PSScriptRoot 'start_paper_dashboard.ps1') @('-OutputDir',$RunPath,'-DashboardAddress',$DashboardAddress,'-DashboardPort',[string]$DashboardPort,'-WaitForAddressSeconds','180','-NoBrowser') 30)
        } catch {
            foreach ($created in $script:CreatedTasks) { Unregister-ScheduledTask -TaskName $created -Confirm:$false -ErrorAction SilentlyContinue }
            throw
        }
        $config = [ordered]@{schema_version=1; run_dir=$RunPath; tws_executable=$TwsExecutable; tws_host=$TwsHost; tws_port=$TwsPort; dashboard_address=$DashboardAddress; dashboard_port=$DashboardPort; engine_auto_resume_enabled=$false; engine_resume_task_installed=$false; installed_tasks=@($script:CreatedTasks); installed_for_user="$env:USERDOMAIN\$env:USERNAME"; installed_at_utc=[DateTime]::UtcNow.ToString('o')}
        $configTemp=$ConfigPath+'.tmp'
        try { $config | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $configTemp -Encoding utf8; Move-Item -LiteralPath $configTemp -Destination $ConfigPath -Force }
        catch { foreach ($created in $script:CreatedTasks) { Unregister-ScheduledTask -TaskName $created -Confirm:$false -ErrorAction SilentlyContinue }; Remove-Item -LiteralPath $configTemp -Force -ErrorAction SilentlyContinue; throw }
        Write-Log "Installed per-user logon tasks. Engine auto-resume remains disabled; run=$RunName"
        Write-Host 'Installed per-user startup tasks for the optional TWS GUI, read-only notifier/watchdog, dashboard, and monitor.'
        Write-Host 'No Paper Engine task was installed. Engine automatic restart remains DISABLED.'
        Write-Host 'TWS may require interactive login/2FA after each reboot; no authentication is automated.'
        Write-Host 'No administrator permission is normally required for these per-user tasks.'
    }
    'Check' {
        if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw 'Project .venv Python is missing.' }
        $args = @('-m','src.paper.windows_supervisor','--run-dir',$RunPath)
        if (-not $SkipTwsProbe) { $args += @('--check-tws','--tws-host',$TwsHost,'--tws-port',[string]$TwsPort) }
        $assessment = & $Python @args 2>&1
        $assessmentCode = $LASTEXITCODE
        Write-Host 'Read-only recovery assessment:'; $assessment | ForEach-Object { Write-Host $_ }
        $listener = Get-NetTCPConnection -LocalPort $DashboardPort -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        $dash = $false
        if ($listener) { try { $dash = (Invoke-WebRequest -Uri "http://${DashboardAddress}:$DashboardPort/_stcore/health" -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200 } catch { } }
        $notifyRuntime = Join-Path $Root ("results\paper\.notifications\" + $RunName)
        $notifierBeat = Join-Path $notifyRuntime 'notifier_heartbeat.json'
        $watchdogBeat = Join-Path $notifyRuntime 'engine_watchdog_state.json'
        Write-Host ("Dashboard listener={0}; health={1}" -f [bool]$listener,$dash)
        $tailscalePresent = [bool]([System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() | ForEach-Object { $_.GetIPProperties().UnicastAddresses } | Where-Object { $_.Address.ToString() -eq $DashboardAddress })
        $supervisorStatePath = Join-Path $Runtime 'supervisor-state.json'
        $supervisorState = $null
        if (Test-Path $supervisorStatePath) { try { $supervisorState = Get-Content $supervisorStatePath -Raw | ConvertFrom-Json } catch { } }
        $taskNames = @(($TaskPrefix + 'TWS Login'),($TaskPrefix + 'Supervisor Monitor'),($TaskPrefix + 'Telegram Notifications'),($TaskPrefix + 'Read-only Dashboard'))
        $taskStates = foreach ($name in $taskNames) {
            $task = Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
            [pscustomobject]@{Task=$name;State=$(if($task){[string]$task.State}else{'NOT_INSTALLED'})}
        }
        Write-Host ("Tailscale address assigned={0}; supervisor TWS verified={1}; last handshake UTC={2}" -f $tailscalePresent,$supervisorState.tws_available,$supervisorState.last_verified_utc)
        Write-Host ("Notifier heartbeat present={0}; watchdog state present={1}" -f (Test-Path $notifierBeat),(Test-Path $watchdogBeat))
        Write-Host 'Scheduled tasks:'; $taskStates | Format-Table -AutoSize | Out-String | Write-Host
        if ($assessmentCode -ne 0) { exit $assessmentCode }
    }
    'Monitor' {
        $monitorLog = Join-Path $LogDir 'supervisor-monitor.log'
        $monitorPidFile = Join-Path $Runtime 'supervisor.pid'
        Set-Content -LiteralPath $monitorPidFile -Value $PID -Encoding ascii
        $stop = Join-Path $Runtime 'monitor.stop.request'
        Remove-Item -LiteralPath $stop -Force -ErrorAction SilentlyContinue
        $statePath = Join-Path $Runtime 'supervisor-state.json'
        $stateDefaults = [ordered]@{
            schema_version=1; tws_available=$null; failures=0; relaunches=0
            next_retry_utc=$null; last_verified_utc=$null
            notifier_available=$null; notifier_retry_utc=$null
            watchdog_available=$null; watchdog_retry_utc=$null
            dashboard_available=$null; dashboard_retry_utc=$null
        }
        if (Test-Path $statePath) {
            $state = Get-Content $statePath -Raw | ConvertFrom-Json
            foreach ($key in $stateDefaults.Keys) {
                if ($null -eq $state.PSObject.Properties[$key]) {
                    $state | Add-Member -NotePropertyName $key -NotePropertyValue $stateDefaults[$key]
                }
            }
        } else { $state = [pscustomobject]$stateDefaults }
        try { while (-not (Test-Path -LiteralPath $stop)) {
            $cfg = Read-Config
            $TwsHost = [string]$cfg.tws_host; $TwsPort = [int]$cfg.tws_port
            $probe = Invoke-TwsProbe
            if ($probe.success) {
                if ($state.tws_available -ne $true) { Send-TwsCondition $false 'TWS read-only handshake and approved contract verified.' 'verified' }
                $state.tws_available=$true; $state.failures=0; $state.relaunches=0; $state.next_retry_utc=$null; $state.last_verified_utc=[DateTime]::UtcNow.ToString('o')
                Write-Log 'TWS read-only handshake and MNQZ6 identity verified.'
            } else {
                $wasHealthy = ($state.tws_available -eq $true)
                $state.tws_available=$false; $state.failures=[int]$state.failures+1
                $twsPresence = Get-TwsPresence ([string]$cfg.tws_executable)
                $twsProcess = $twsPresence -ne 'ABSENT'
                if ($twsPresence -eq 'PRESENT') { $detail='TWS process is present but read-only API handshake/contract validation failed; check login, 2FA, API settings or maintenance. A healthy TWS process was not killed.' }
                elseif ($twsPresence -eq 'UNKNOWN') { $detail='TWS process presence could not be established safely; no duplicate was launched. Check TWS process visibility and API login.' }
                else { $detail='TWS API is unavailable and no TWS process was found; authentication may be required. No broker order was sent.' }
                if ($wasHealthy -or $state.failures -eq 1) { Send-TwsCondition $true $detail 'handshake_unavailable' }
                Write-Log "TWS handshake unavailable; process_present=$twsProcess; failure=$($state.failures)."
                # Relaunch only when the configured application is absent. Never kill an existing TWS.
                $now = [DateTime]::UtcNow
                if ($twsPresence -eq 'ABSENT' -and $cfg.tws_executable -and [int]$state.relaunches -lt 3 -and
                    (-not $state.next_retry_utc -or $now -ge [DateTime]::Parse([string]$state.next_retry_utc).ToUniversalTime())) {
                    Start-TwsIfClosed ([string]$cfg.tws_executable)
                    $state.relaunches=[int]$state.relaunches+1
                    $delay = [Math]::Min(900,60*[Math]::Pow(2,[Math]::Min([int]$state.relaunches-1,4)))
                    $state.next_retry_utc=$now.AddSeconds($delay).ToString('o')
                }
                if ([int]$state.relaunches -ge 3) { Write-Log 'TWS relaunch budget exhausted; manual authentication/start is required.' }
            }
            # Sidecars are restarted only when their recorded OS process is
            # absent. A stale heartbeat with a live PID is reported, never
            # force-killed. The launcher/lock remains the duplicate guard.
            $notificationRuntime = Join-Path $Root ("results\paper\.notifications\" + $RunName)
            if (-not (Test-PidFileAlive (Join-Path $notificationRuntime 'notifier.pid'))) {
                if ($state.notifier_available -ne $false) { Send-ComponentCondition 'notifier' $true 'Telegram notifier process is absent; delivery queue remains persisted.' 'process_absent' }
                $state.notifier_available=$false
                $readyAt = if ($state.notifier_retry_utc) { [DateTime]::Parse([string]$state.notifier_retry_utc).ToUniversalTime() } else { [DateTime]::MinValue }
                if ([DateTime]::UtcNow -ge $readyAt) {
                    Start-SidecarLauncher 'notifier-recovery' (Join-Path $PSScriptRoot 'start_paper_notifications.ps1') @('-RunDir',$RunPath)
                    $state.notifier_retry_utc=[DateTime]::UtcNow.AddSeconds(120).ToString('o')
                }
            } else {
                $beatPath = Join-Path $notificationRuntime 'notifier_heartbeat.json'
                $fresh = (Test-Path $beatPath) -and ((Get-Date) - (Get-Item $beatPath).LastWriteTime).TotalSeconds -lt 45
                if ($fresh) {
                    if ($state.notifier_available -ne $true) { Send-ComponentCondition 'notifier' $false 'Telegram notifier heartbeat is fresh.' 'healthy' }
                    $state.notifier_available=$true
                } else {
                    if ($state.notifier_available -ne $false) { Send-ComponentCondition 'notifier' $true 'Notifier process exists but its heartbeat is stale; no process was killed.' 'heartbeat_stale' }
                    $state.notifier_available=$false
                }
            }
            if (-not (Test-PidFileAlive (Join-Path $notificationRuntime 'watchdog.pid'))) {
                if ($state.watchdog_available -ne $false) { Send-ComponentCondition 'watchdog' $true 'Independent Paper watchdog process is absent.' 'process_absent' }
                $state.watchdog_available=$false
                $readyAt = if ($state.watchdog_retry_utc) { [DateTime]::Parse([string]$state.watchdog_retry_utc).ToUniversalTime() } else { [DateTime]::MinValue }
                if ([DateTime]::UtcNow -ge $readyAt) {
                    Start-SidecarLauncher 'watchdog-recovery' (Join-Path $PSScriptRoot 'start_paper_watchdog.ps1') @('-RunDir',$RunPath)
                    $state.watchdog_retry_utc=[DateTime]::UtcNow.AddSeconds(120).ToString('o')
                }
            } else {
                $watchStatePath = Join-Path $notificationRuntime 'engine_watchdog_state.json'
                $watchFresh = $false
                try { $watchState = Get-Content $watchStatePath -Raw | ConvertFrom-Json; $watchFresh = (([DateTime]::UtcNow - [DateTime]::Parse([string]$watchState.last_probe_at_utc).ToUniversalTime()).TotalSeconds -lt 45) } catch { }
                if ($watchFresh) {
                    if ($state.watchdog_available -ne $true) { Send-ComponentCondition 'watchdog' $false 'Independent watchdog probe heartbeat is fresh.' 'healthy' }
                    $state.watchdog_available=$true
                } else {
                    if ($state.watchdog_available -ne $false) { Send-ComponentCondition 'watchdog' $true 'Watchdog process exists but its process-probe heartbeat is stale.' 'heartbeat_stale' }
                    $state.watchdog_available=$false
                }
            }
            $healthUrl = "http://${DashboardAddress}:$DashboardPort/_stcore/health"
            $dashboardHealthy = $false
            try { $dashboardHealthy = (Invoke-WebRequest -Uri $healthUrl -TimeoutSec 3 -UseBasicParsing).StatusCode -eq 200 } catch { }
            if (-not $dashboardHealthy) {
                if ($state.dashboard_available -ne $false) { Send-ComponentCondition 'dashboard' $true 'Read-only dashboard health endpoint is unavailable.' 'health_check_failed' }
                $state.dashboard_available=$false
                $tailscalePresent = [System.Net.NetworkInformation.NetworkInterface]::GetAllNetworkInterfaces() | ForEach-Object { $_.GetIPProperties().UnicastAddresses } | Where-Object { $_.Address.ToString() -eq $DashboardAddress }
                $readyAt = if ($state.dashboard_retry_utc) { [DateTime]::Parse([string]$state.dashboard_retry_utc).ToUniversalTime() } else { [DateTime]::MinValue }
                if ($tailscalePresent -and [DateTime]::UtcNow -ge $readyAt) {
                    Start-SidecarLauncher 'dashboard-recovery' (Join-Path $PSScriptRoot 'start_paper_dashboard.ps1') @('-OutputDir',$RunPath,'-DashboardAddress',$DashboardAddress,'-DashboardPort',[string]$DashboardPort,'-NoBrowser')
                    $state.dashboard_retry_utc=[DateTime]::UtcNow.AddSeconds(120).ToString('o')
                } elseif (-not $tailscalePresent) { Write-Log 'Tailscale dashboard address is not assigned; dashboard remains stopped until it returns.' }
            } else {
                if ($state.dashboard_available -ne $true) { Send-ComponentCondition 'dashboard' $false 'Dashboard health endpoint responded successfully on the configured private address.' 'healthy' }
                $state.dashboard_available=$true
            }
            $stateTemp = $statePath + '.tmp'
            $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $stateTemp -Encoding utf8
            Move-Item -LiteralPath $stateTemp -Destination $statePath -Force
            Start-Sleep -Seconds 60
        }} finally { Remove-Item -LiteralPath $monitorPidFile -Force -ErrorAction SilentlyContinue }
        Remove-Item -LiteralPath $stop -Force -ErrorAction SilentlyContinue
        Write-Log 'Supervisor monitor stopped gracefully.'
    }
    'Stop' {
        # Delayed-Paper CLI only writes a graceful stop request for active states.
        try { & $Python -m src.paper.delayed_paper_cli stop --output-dir $RunPath --tws-host $TwsHost --tws-port $TwsPort; $stopCode=$LASTEXITCODE }
        catch { $stopCode=1; Write-Log 'Paper stop request command failed; sidecars will still be stopped without force-killing the Engine.' }
        if ($stopCode -ne 0) { Write-Log "Paper stop request returned exit code $stopCode; persisted Engine state will be inspected." }
        $engineDeadline=(Get-Date).AddSeconds(60)
        while ((Get-Date) -lt $engineDeadline) {
            try { $engineState=[string]((Get-Content (Join-Path $RunPath 'status.json') -Raw | ConvertFrom-Json).system.state) } catch { $engineState='UNKNOWN' }
            if ($engineState -in @('STOPPED','COMPLETED','ERROR','FAILED')) { break }
            Start-Sleep -Seconds 1
        }
        Request-SidecarStops
        $notificationRuntime = Join-Path $Root ("results\paper\.notifications\" + $RunName)
        $pidFiles = @((Join-Path $notificationRuntime 'notifier.pid'),(Join-Path $notificationRuntime 'watchdog.pid'),(Join-Path $Runtime 'supervisor.pid'))
        if (-not (Wait-RecordedProcesses $pidFiles 30)) { Write-Log 'Some sidecars did not stop before timeout; no process was force-killed.' }
        & (Join-Path $PSScriptRoot 'stop_paper_dashboard.ps1')
        Write-Host 'Graceful stop requested. TWS and Tailscale were left running; no process was force-killed.'
    }
    'RestartSidecars' {
        Request-SidecarStops
        $notificationRuntime = Join-Path $Root ("results\paper\.notifications\" + $RunName)
        $pidFiles = @((Join-Path $notificationRuntime 'notifier.pid'),(Join-Path $notificationRuntime 'watchdog.pid'),(Join-Path $Runtime 'supervisor.pid'))
        if (-not (Wait-RecordedProcesses $pidFiles 30)) { throw 'A sidecar is still active; refusing duplicate restart.' }
        & (Join-Path $PSScriptRoot 'start_paper_notifications.ps1') -RunDir $RunPath
        & (Join-Path $PSScriptRoot 'start_paper_dashboard.ps1') -OutputDir $RunPath -DashboardAddress $DashboardAddress -DashboardPort $DashboardPort -NoBrowser
        $monitorArgs=@('-NoProfile','-ExecutionPolicy','RemoteSigned','-WindowStyle','Hidden','-File',('"'+$PSCommandPath+'"'),'-Mode','Monitor','-RunDir',('"'+$RunPath+'"'),'-TwsHost',$TwsHost,'-TwsPort',[string]$TwsPort)
        Start-Process -FilePath $PowerShell -ArgumentList $monitorArgs -WorkingDirectory $Root -WindowStyle Hidden | Out-Null
        Write-Host 'Read-only sidecars restarted; Paper Engine and TWS were not touched.'
    }
    'Uninstall' {
        $installed = @()
        if (Test-Path -LiteralPath $ConfigPath) { try { $installed=@((Get-Content $ConfigPath -Raw | ConvertFrom-Json).installed_tasks) } catch { } }
        foreach ($name in $installed) { Unregister-ScheduledTask -TaskName ([string]$name) -Confirm:$false -ErrorAction SilentlyContinue }
        Write-Log 'Scheduled tasks removed; running processes and Paper data left untouched.'
        Write-Host 'Removed configured MNQ Paper scheduled tasks. This does not stop running processes or alter Paper data.'
    }
}
