<#
.SYNOPSIS
    One-Click IMC Demo Launcher for 5G Suraksha-Net.

.DESCRIPTION
    Automates preflight checks, Raspberry Pi MediaMTX + camera publisher verification,
    RTSP stream health probing, and launches the complete Suraksha-Net AI pipeline
    in a dedicated visible window.

.PARAMETER PiHost
    IP address or hostname of the Raspberry Pi (default: 10.179.184.48).
.PARAMETER PiUser
    SSH username for the Raspberry Pi (default: student).
.PARAMETER RtspPort
    RTSP streaming port (default: 8554).
.PARAMETER RtspPath
    RTSP stream path name (default: drone).
.PARAMETER ProbeTimeout
    Timeout in seconds for RTSP stream verification (default: 15).
.PARAMETER NoDisplay
    Run pipeline without GUI window (headless mode).
.PARAMETER DryRun
    Perform all preflight checks and stream verification without launching the AI pipeline.
#>

[CmdletBinding()]
param(
    [string]$PiHost = "10.179.184.48",
    [string]$PiUser = "student",
    [int]$RtspPort = 8554,
    [string]$RtspPath = "drone",
    [int]$ProbeTimeout = 20,
    [switch]$NoDisplay,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# Resolve paths relative to script location
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ProbeScript = Join-Path $ProjectRoot "scripts\probe_stream.py"
$PipelineScript = Join-Path $ProjectRoot "scripts\run_pipeline.py"
$CleanPath = $RtspPath.TrimStart('/')
$RtspUrl = "rtsp://$($PiHost):$($RtspPort)/$($CleanPath)"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "       5G SURAKSHA-NET - IMC DEMO       " -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Helper to print step results
function Print-Step([int]$num, [string]$desc, [string]$status, [string]$color) {
    $pad = 32 - $desc.Length
    if ($pad -lt 1) { $pad = 1 }
    $spacing = " " * $pad
    Write-Host "[$num/5] $desc$spacing" -NoNewline
    Write-Host "[$status]" -ForegroundColor $color
}

# ----------------------------------------------------
# [1/5] Checking Raspberry Pi & Environment
# ----------------------------------------------------
Write-Host "Checking local environment and network..." -ForegroundColor Gray

if (-not (Test-Path $VenvPython)) {
    Write-Host "`n[ERROR] Virtual environment not found at: $VenvPython" -ForegroundColor Red
    Write-Host "Please set up the Python venv first: python -m venv .venv; .venv\Scripts\pip install -e ." -ForegroundColor Yellow
    exit 1
}

# Fast TCP connectivity check to Pi port 22 or ping
$piReachable = $false
try {
    $tcp = New-Object System.Net.Sockets.TcpClient
    $connect = $tcp.BeginConnect($PiHost, 22, $null, $null)
    $success = $connect.AsyncWaitHandle.WaitOne(2000, $false)
    if ($success -and $tcp.Connected) {
        $piReachable = $true
        $tcp.EndConnect($connect)
    }
    $tcp.Close()
} catch {
    $piReachable = $false
}

if (-not $piReachable) {
    # Fallback to ICMP ping
    $ping = Test-Connection -ComputerName $PiHost -Count 1 -Quiet -ErrorAction SilentlyContinue
    if ($ping) { $piReachable = $true }
}

if (-not $piReachable) {
    Print-Step 1 "Checking Raspberry Pi..." "FAIL" "Red"
    Write-Host "`n[ERROR] Raspberry Pi ($PiHost) is unreachable on the network." -ForegroundColor Red
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Verify the Pi is powered on." -ForegroundColor Yellow
    Write-Host "  2. Confirm both laptop and Pi are on the same subnet (e.g. 10.254.18.x)." -ForegroundColor Yellow
    Write-Host "  3. Verify your Wi-Fi/Ethernet connection: ping $PiHost" -ForegroundColor Yellow
    exit 1
}

Print-Step 1 "Checking Raspberry Pi..." "PASS" "Green"

# ----------------------------------------------------
# [2/5] Checking Logitech C270 & Pi Services
# ----------------------------------------------------
# Test if MediaMTX and camera publisher are already actively streaming
$probeResult = $null
try {
    $quickRaw = & $VenvPython $ProbeScript --url $RtspUrl --timeout $ProbeTimeout --frames 2 --json 2>$null
    if ($quickRaw) {
        $parsed = $quickRaw | ConvertFrom-Json
        if ($parsed.success) {
            $probeResult = $parsed
        }
    }
} catch {
    $probeResult = $null
}

if ($probeResult) {
    Print-Step 2 "Checking Logitech C270..." "PASS" "Green"
    Print-Step 3 "Starting MediaMTX..." "PASS (Running)" "Green"
    Print-Step 4 "Starting RTSP camera stream..." "PASS (Active)" "Green"
} else {
    # Check if SSH is available to start remote services
    $sshCmd = Get-Command "ssh.exe" -ErrorAction SilentlyContinue
    if (-not $sshCmd) {
        Write-Host "`n[ERROR] OpenSSH client (ssh.exe) not found on Windows." -ForegroundColor Red
        Write-Host "Please enable OpenSSH client in Windows Optional Features." -ForegroundColor Yellow
        exit 1
    }

    $sshTarget = "$($PiUser)@$($PiHost)"
    Write-Host "Connecting to Pi ($PiHost) to inspect camera and services..." -ForegroundColor Gray

    # Check /dev/video0 exists on Pi (BatchMode=yes prevents hanging on password prompts)
    & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget "test -e /dev/video0" 2>$null
    $devCheckExit = $LASTEXITCODE

    if ($devCheckExit -eq 255) {
        # Fallback to standard interactive SSH if batch mode fails
        Write-Host "`n[NOTICE] SSH key authentication unavailable. Trying interactive SSH for $sshTarget..." -ForegroundColor Yellow
        & ssh.exe -o ConnectTimeout=8 -o StrictHostKeyChecking=no $sshTarget "test -e /dev/video0" 2>$null
        $devCheckExit = $LASTEXITCODE
    }

    if ($devCheckExit -ne 0) {
        Print-Step 2 "Checking Logitech C270..." "FAIL" "Red"
        Write-Host "`n[ERROR] /dev/video0 not found on Raspberry Pi." -ForegroundColor Red
        Write-Host "Ensure the Logitech C270 USB webcam is securely plugged into the Pi." -ForegroundColor Yellow
        exit 1
    }
    Print-Step 2 "Checking Logitech C270..." "PASS" "Green"

    # [3/5] MediaMTX check and start
    & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget "pgrep -x mediamtx" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Starting MediaMTX on Pi..." -ForegroundColor Gray
        $startMtxCmd = "nohup /home/student/mediamtx </dev/null >/home/student/mediamtx.log 2>&1 &"
        & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget $startMtxCmd 2>$null
        Start-Sleep -Seconds 1

        # Post-start verification
        & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget "pgrep -x mediamtx" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Print-Step 3 "Starting MediaMTX..." "FAIL" "Red"
            Write-Host "`n[ERROR] MediaMTX failed to start on Raspberry Pi." -ForegroundColor Red
            $mtxLog = & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget "cat /home/student/mediamtx.log 2>/dev/null"
            if ($mtxLog) {
                Write-Host "MediaMTX log output:`n$mtxLog" -ForegroundColor Yellow
            }
            exit 1
        }
        Print-Step 3 "Starting MediaMTX..." "PASS (Started)" "Green"
    } else {
        Print-Step 3 "Starting MediaMTX..." "PASS (Running)" "Green"
    }

    # [4/5] Camera publisher check and start
    & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget "pgrep -x ffmpeg" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Starting camera publisher (FFmpeg MJPEG 640x360@15 -> H.264 RTSP)..." -ForegroundColor Gray
        $startFfmpegCmd = "nohup ffmpeg -f v4l2 -input_format mjpeg -video_size 640x360 -framerate 15 -i /dev/video0 -c:v libx264 -preset ultrafast -tune zerolatency -g 15 -pix_fmt yuv420p -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:$($RtspPort)/$($CleanPath) </dev/null >/home/student/ffmpeg_drone.log 2>&1 &"
        & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget $startFfmpegCmd 2>$null
        Start-Sleep -Seconds 2

        # Post-start verification: verify process exists AND log was created
        & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget "pgrep -x ffmpeg && test -f /home/student/ffmpeg_drone.log" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Print-Step 4 "Starting RTSP camera stream..." "FAIL" "Red"
            Write-Host "`n[ERROR] FFmpeg camera publisher failed to start on Raspberry Pi." -ForegroundColor Red
            $ffLog = & ssh.exe -o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no $sshTarget "cat /home/student/ffmpeg_drone.log 2>/dev/null"
            if ($ffLog) {
                Write-Host "FFmpeg log output:`n$ffLog" -ForegroundColor Yellow
            } else {
                Write-Host "FFmpeg log file (/home/student/ffmpeg_drone.log) was not created." -ForegroundColor Yellow
            }
            exit 1
        }
        Print-Step 4 "Starting RTSP camera stream..." "PASS (Started)" "Green"
    } else {
        Print-Step 4 "Starting RTSP camera stream..." "PASS (Running)" "Green"
    }

    # Phase 3 probe after remote launch
    Write-Host "Probing RTSP stream for video frames..." -ForegroundColor Gray
    $probeJsonRaw = & $VenvPython $ProbeScript --url $RtspUrl --timeout $ProbeTimeout --frames 3 --json
    try {
        $probeResult = $probeJsonRaw | ConvertFrom-Json
    } catch {
        Write-Host "`n[ERROR] Failed to parse probe output:`n$probeJsonRaw" -ForegroundColor Red
        exit 1
    }

    if (-not $probeResult.success) {
        Write-Host "`n[ERROR] RTSP stream validation failed: $RtspUrl" -ForegroundColor Red
        Write-Host "Details: $($probeResult.error)" -ForegroundColor Yellow
        Write-Host "`nTroubleshooting:" -ForegroundColor Yellow
        Write-Host "  1. Test stream directly: .venv\Scripts\python.exe scripts\probe_stream.py --url $RtspUrl" -ForegroundColor Yellow
        Write-Host "  2. Check Pi MediaMTX log: ssh $PiUser@$PiHost 'cat /home/student/mediamtx.log'" -ForegroundColor Yellow
        Write-Host "  3. Check Pi FFmpeg log:   ssh $PiUser@$PiHost 'cat /home/student/ffmpeg_drone.log'" -ForegroundColor Yellow
        exit 1
    }
}

# ----------------------------------------------------
# [5/5] Start Suraksha-Net AI Pipeline
# ----------------------------------------------------
if ($DryRun) {
    Print-Step 5 "Starting Suraksha-Net AI..." "SKIPPED (DryRun)" "Cyan"
} else {
    $displayFlag = ""
    if (-not $NoDisplay) {
        $displayFlag = "--display"
    }

    $cmdLine = "Set-Location '$ProjectRoot'; & '$VenvPython' scripts\run_pipeline.py --source rtsp --path '$RtspUrl' $displayFlag"
    Start-Process powershell.exe -ArgumentList "-NoExit", "-Command", $cmdLine
    Print-Step 5 "Starting Suraksha-Net AI..." "PASS" "Green"
}

Write-Host ""
Write-Host "Camera: Logitech C270" -ForegroundColor White
Write-Host "Pi: $PiHost" -ForegroundColor White
Write-Host "Stream: $RtspUrl" -ForegroundColor White
Write-Host "Resolution: $($probeResult.width)x$($probeResult.height)" -ForegroundColor White
Write-Host "Target camera FPS: $($probeResult.source_fps)" -ForegroundColor White
Write-Host ""
Write-Host "STATUS: IMC DEMO READY" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
