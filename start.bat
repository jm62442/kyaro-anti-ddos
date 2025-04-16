@echo off
title Kyaro Anti-DDoS Startup
color 0a

echo ====================================================
echo             Kyaro Anti-DDoS Startup Script
echo ====================================================
echo.

:: Get script directory
set "SCRIPT_DIR=%~dp0"
:: Remove trailing backslash
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

:: Make logs directory if it doesn't exist
if not exist "%SCRIPT_DIR%\logs" mkdir "%SCRIPT_DIR%\logs"

echo Starting Kyaro Anti-DDoS...
echo.

:: Check if Rust executable exists
if exist "%SCRIPT_DIR%\kyaro_antiddos\target\release\kyaro_antiddos.exe" (
    :: Start the backend
    echo Starting backend service...
    echo The backend would normally be started here if fully implemented.
    :: Uncomment the following line when you have a working backend:
    :: start "" "%SCRIPT_DIR%\kyaro_antiddos\target\release\kyaro_antiddos.exe"
    echo - Backend service started (simulated)
) else (
    echo Backend executable not found.
    echo You may need to build the project with cargo first.
    echo.
    echo Example: cd kyaro_antiddos ^&^& cargo build --release
)

echo.
echo Note: This is a placeholder startup script.
echo You need to implement the actual functionality.
echo.
echo To stop the service, use the stop.bat script.
echo.

echo Press any key to exit this script (services will be simulated as still running)
echo.
pause 