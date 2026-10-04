<#
.SYNOPSIS
    Gracefully stops the 5G Suraksha-Net IMC Demo.

.DESCRIPTION
    Terminates the local AI pipeline process (scripts\run_pipeline.py).
    Optionally stops FFmpeg/MediaMTX services on the Raspberry Pi without shutting down the Pi.

.PARAMETER StopPiServices
    Switch to also stop camera streaming services on the Raspberry Pi.
.PARAMETER PiHost
    IP address or hostname of the Raspberry Pi (default: 10.254.18.48).
.PARAMETER PiUser
    SSH username for the Raspberry Pi (default: student).
#>

[CmdletBinding()]
param(
    [switch]$StopPiServices,
    [string]$PiHost = "10.254.18.48",
    [string]$PiUser = "student"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "       5G SURAKSHA-NET - STOP DEMO      " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Find and stop local python pipeline instances
$stoppedCount = 0
try {
    $processes = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "python" -and ($_.CommandLine -match "run_pipeline\.py" -or $_.CommandLine -match "probe_stream\.py")
    }

    foreach ($proc in $processes) {
        Write-Host "Stopping process PID $($proc.ProcessId): $($proc.CommandLine)" -ForegroundColor Gray
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        $stoppedCount++
    }
} catch {
    Write-Host "[WARNING] Could not query process list via CIM: $_" -ForegroundColor Yellow
}

if ($stoppedCount -gt 0) {
    Write-Host "[OK] Stopped $stoppedCount local Suraksha-Net process(es)." -ForegroundColor Green
} else {
    Write-Host "[INFO] No active local Suraksha-Net pipeline processes found." -ForegroundColor Gray
}

# Stop Pi services if requested
if ($StopPiServices) {
    Write-Host "Stopping camera services on Raspberry Pi ($PiHost)..." -ForegroundColor Gray
    $sshCmd = Get-Command "ssh.exe" -ErrorAction SilentlyContinue
    if ($sshCmd) {
        $sshTarget = "$($PiUser)@$($PiHost)"
        # Note: We only terminate the camera publisher and mediamtx. We NEVER reboot or shutdown the Pi.
        $stopCmd = "pkill -x ffmpeg; pkill -x mediamtx; echo STOPPED"
        $res = & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget $stopCmd 2>$null
        if ($LASTEXITCODE -eq 0 -and $res -match "STOPPED") {
            Write-Host "[OK] Stopped camera publisher and MediaMTX on Pi." -ForegroundColor Green
            Write-Host "[INFO] Raspberry Pi remains powered ON and ready for next run." -ForegroundColor Gray
        } else {
            Write-Host "[NOTICE] Could not stop Pi services automatically (SSH password or key required)." -ForegroundColor Yellow
            Write-Host "You can stop them manually on the Pi with: pkill -f 'ffmpeg.*drone'" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "STATUS: IMC DEMO STOPPED" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
