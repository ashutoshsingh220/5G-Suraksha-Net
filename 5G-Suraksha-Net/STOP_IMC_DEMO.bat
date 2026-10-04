@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   Stopping 5G Suraksha-Net IMC Demo...
echo ========================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\stop_imc_demo.ps1" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Stop command encountered an issue [Exit Code %ERRORLEVEL%].
    pause
    exit /b %ERRORLEVEL%
)

exit /b 0
