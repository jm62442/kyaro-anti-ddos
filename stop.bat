@echo off
title Kyaro Anti-DDoS Stop
color 0a

echo ====================================================
echo              Kyaro Anti-DDoS Stop Script
echo ====================================================
echo.

:: Get script directory
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

echo Stopping Kyaro Anti-DDoS services...

:: Check for any running kyaro_antiddos processes
tasklist /FI "IMAGENAME eq kyaro_antiddos.exe" 2>NUL | find /I /N "kyaro_antiddos.exe" >NUL
if "%ERRORLEVEL%"=="0" (
    echo - Found running Kyaro Anti-DDoS processes
    echo - Attempting to stop processes...
    taskkill /F /IM kyaro_antiddos.exe >NUL 2>&1
    echo - Processes stopped
) else (
    echo - No running Kyaro Anti-DDoS processes found
)

echo.
echo All services stopped (simulated).
echo.
echo Note: This is a placeholder stop script.
echo You need to implement the actual functionality.
echo.

pause 