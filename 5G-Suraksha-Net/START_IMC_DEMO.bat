@echo off
setlocal
cd /d "%~dp0"

echo ========================================================
echo   Launching 5G Suraksha-Net IMC Demo...
echo ========================================================
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_imc_demo.ps1" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Demo startup encountered an issue [Exit Code %ERRORLEVEL%].
    echo Please review the diagnostic messages above.
    pause
    exit /b %ERRORLEVEL%
)

exit /b 0
