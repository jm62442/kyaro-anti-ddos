@echo off
title Kyaro Anti-DDoS Installer
color 0a

echo ====================================================
echo         Kyaro Anti-DDoS Installation Script
echo ====================================================
echo.

:: Create directories
echo Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "kyaro_antiddos\src" mkdir kyaro_antiddos\src
if not exist "dashboard\src" mkdir dashboard\src
if not exist "ml_models" mkdir ml_models
echo - Directories created successfully.
echo.

:: Create a simple Rust project
echo Creating basic Rust project...
if not exist "kyaro_antiddos\Cargo.toml" (
    echo [package] > kyaro_antiddos\Cargo.toml
    echo name = "kyaro_antiddos" >> kyaro_antiddos\Cargo.toml
    echo version = "0.1.0" >> kyaro_antiddos\Cargo.toml
    echo edition = "2021" >> kyaro_antiddos\Cargo.toml
    echo. >> kyaro_antiddos\Cargo.toml
    echo [dependencies] >> kyaro_antiddos\Cargo.toml
    echo.
)

if not exist "kyaro_antiddos\src\main.rs" (
    echo fn main() { > kyaro_antiddos\src\main.rs
    echo     println!^("Hello from Kyaro Anti-DDoS!"^); >> kyaro_antiddos\src\main.rs
    echo } >> kyaro_antiddos\src\main.rs
)
echo - Basic Rust project created.
echo.

:: Create a basic dashboard
echo Creating basic dashboard...
if not exist "dashboard\package.json" (
    echo { > dashboard\package.json
    echo   "name": "kyaro-dashboard", >> dashboard\package.json
    echo   "version": "0.1.0", >> dashboard\package.json
    echo   "description": "Dashboard for Kyaro Anti-DDoS", >> dashboard\package.json
    echo   "scripts": { >> dashboard\package.json
    echo     "start": "echo Starting dashboard..." >> dashboard\package.json
    echo   } >> dashboard\package.json
    echo } >> dashboard\package.json
)
echo - Basic dashboard created.
echo.

:: Create basic start and stop scripts
echo Creating start.bat...
if not exist "start.bat" (
    echo @echo off > start.bat
    echo title Kyaro Anti-DDoS Startup >> start.bat
    echo color 0a >> start.bat
    echo. >> start.bat
    echo echo ==================================================== >> start.bat
    echo echo             Kyaro Anti-DDoS Startup Script >> start.bat
    echo echo ==================================================== >> start.bat
    echo echo. >> start.bat
    echo. >> start.bat
    echo :: Get script directory >> start.bat
    echo set "SCRIPT_DIR=%%~dp0" >> start.bat
    echo :: Remove trailing backslash >> start.bat
    echo set "SCRIPT_DIR=%%SCRIPT_DIR:~0,-1%%" >> start.bat
    echo. >> start.bat
    echo :: Make logs directory if it doesn't exist >> start.bat
    echo if not exist "%%SCRIPT_DIR%%\logs" mkdir "%%SCRIPT_DIR%%\logs" >> start.bat
    echo. >> start.bat
    echo echo Starting Kyaro Anti-DDoS... >> start.bat
    echo echo. >> start.bat
    echo. >> start.bat
    echo :: Check if Rust executable exists >> start.bat
    echo if exist "%%SCRIPT_DIR%%\kyaro_antiddos\target\release\kyaro_antiddos.exe" ( >> start.bat
    echo     :: Start the backend >> start.bat
    echo     echo Starting backend service... >> start.bat
    echo     start "" "%%SCRIPT_DIR%%\kyaro_antiddos\target\release\kyaro_antiddos.exe" --config "%%SCRIPT_DIR%%\config.json" >> start.bat
    echo     echo - Backend service started >> start.bat
    echo ^) else ( >> start.bat
    echo     echo Backend executable not found. >> start.bat
    echo     echo You need to build the project with cargo first. >> start.bat
    echo     echo. >> start.bat
    echo     echo Example: cd kyaro_antiddos ^&^& cargo build --release >> start.bat
    echo ^) >> start.bat
    echo. >> start.bat
    echo echo. >> start.bat
    echo echo To stop the service, use the stop.bat script. >> start.bat
    echo echo. >> start.bat
    echo. >> start.bat
    echo echo Press any key to exit this script ^(services will continue running in background^) >> start.bat
    echo echo. >> start.bat
    echo pause >> start.bat
)
echo - Created start.bat

echo Creating stop.bat...
if not exist "stop.bat" (
    echo @echo off > stop.bat
    echo title Kyaro Anti-DDoS Stop >> stop.bat
    echo color 0a >> stop.bat
    echo. >> stop.bat
    echo echo ==================================================== >> stop.bat
    echo echo              Kyaro Anti-DDoS Stop Script >> stop.bat
    echo echo ==================================================== >> stop.bat
    echo echo. >> stop.bat
    echo. >> stop.bat
    echo :: Get script directory >> stop.bat
    echo set "SCRIPT_DIR=%%~dp0" >> stop.bat
    echo :: Remove trailing backslash >> stop.bat
    echo set "SCRIPT_DIR=%%SCRIPT_DIR:~0,-1%%" >> stop.bat
    echo. >> stop.bat
    echo echo Stopping Kyaro Anti-DDoS services... >> stop.bat
    echo. >> stop.bat
    echo :: Check for any running kyaro_antiddos processes >> stop.bat
    echo tasklist /FI "IMAGENAME eq kyaro_antiddos.exe" 2^>NUL ^| find /I /N "kyaro_antiddos.exe" ^>NUL >> stop.bat
    echo if "%%ERRORLEVEL%%"=="0" ( >> stop.bat
    echo     echo - Found running Kyaro Anti-DDoS processes >> stop.bat
    echo     echo - Stopping processes... >> stop.bat
    echo     taskkill /F /IM kyaro_antiddos.exe ^>NUL 2^>^&1 >> stop.bat
    echo     echo - Processes stopped >> stop.bat
    echo ^) else ( >> stop.bat
    echo     echo - No running Kyaro Anti-DDoS processes found >> stop.bat
    echo ^) >> stop.bat
    echo. >> stop.bat
    echo echo. >> stop.bat
    echo echo All services stopped. >> stop.bat
    echo echo. >> stop.bat
    echo. >> stop.bat
    echo pause >> stop.bat
)
echo - Created stop.bat
echo.

:: Create a default config file
echo Creating default config.json...
if not exist "config.json" (
   if exist "kyaro_antiddos\config.example.json" (
      copy /Y "kyaro_antiddos\config.example.json" "config.json" >nul 2>&1
      echo - Default configuration created from example.
   ) else (
      echo @echo off > create_config.bat
      echo echo { > config.json >> create_config.bat
      echo echo   "layer3": { >> create_config.bat
      echo echo     "enable_ip_blacklisting": true, >> create_config.bat
      echo echo     "enable_geo_blocking": true, >> create_config.bat
      echo echo     "enable_rate_limiting": true, >> create_config.bat
      echo echo     "max_packet_size": 65535, >> create_config.bat
      echo echo     "min_ttl": 5, >> create_config.bat
      echo echo     "blocked_countries": [], >> create_config.bat
      echo echo     "rate_limit_threshold": 1000, >> create_config.bat
      echo echo     "blacklist_duration_minutes": 30, >> create_config.bat
      echo echo     "whitelist_ips": [ >> create_config.bat
      echo echo       "127.0.0.1", >> create_config.bat
      echo echo       "::1" >> create_config.bat
      echo echo     ] >> create_config.bat
      echo echo   }, >> create_config.bat
      echo echo   "layer4": { >> create_config.bat
      echo echo     "enable_syn_protection": true, >> create_config.bat
      echo echo     "enable_udp_protection": true, >> create_config.bat
      echo echo     "enable_icmp_protection": true, >> create_config.bat
      echo echo     "syn_rate_threshold": 500, >> create_config.bat
      echo echo     "udp_rate_threshold": 500, >> create_config.bat
      echo echo     "icmp_rate_threshold": 100, >> create_config.bat
      echo echo     "max_concurrent_connections": 1000, >> create_config.bat
      echo echo     "blocked_ports": [], >> create_config.bat
      echo echo     "connection_timeout_seconds": 120 >> create_config.bat
      echo echo   }, >> create_config.bat
      echo echo   "layer7": { >> create_config.bat
      echo echo     "enable_http_protection": true, >> create_config.bat
      echo echo     "enable_dns_protection": true, >> create_config.bat
      echo echo     "enable_ssl_protection": true, >> create_config.bat
      echo echo     "http_request_rate_threshold": 100, >> create_config.bat
      echo echo     "blocked_http_methods": ["TRACE", "TRACK"], >> create_config.bat
      echo echo     "challenge_suspicious_requests": true, >> create_config.bat
      echo echo     "bot_protection_level": 2, >> create_config.bat
      echo echo     "waf_rules_enabled": true, >> create_config.bat
      echo echo     "allowed_user_agents": [], >> create_config.bat
      echo echo     "blocked_user_agents": [ >> create_config.bat
      echo echo       "curl", >> create_config.bat
      echo echo       "wget", >> create_config.bat
      echo echo       "python-requests", >> create_config.bat
      echo echo       "go-http-client" >> create_config.bat
      echo echo     ] >> create_config.bat
      echo echo   }, >> create_config.bat
      echo echo   "ml": { >> create_config.bat
      echo echo     "enable_ml": true, >> create_config.bat
      echo echo     "training_interval_hours": 24, >> create_config.bat
      echo echo     "detection_threshold": 0.85, >> create_config.bat
      echo echo     "anomaly_sensitivity": 0.7, >> create_config.bat
      echo echo     "feature_extraction_window_seconds": 60, >> create_config.bat
      echo echo     "model_save_path": "./ml_models", >> create_config.bat
      echo echo     "use_gpu_acceleration": false >> create_config.bat
      echo echo   }, >> create_config.bat
      echo echo   "notification": { >> create_config.bat
      echo echo     "enable_email_alerts": false, >> create_config.bat
      echo echo     "email_recipients": [], >> create_config.bat
      echo echo     "smtp_server": "", >> create_config.bat
      echo echo     "smtp_port": 587, >> create_config.bat
      echo echo     "smtp_username": "", >> create_config.bat
      echo echo     "smtp_password": "", >> create_config.bat
      echo echo     "alert_threshold_severity": "high", >> create_config.bat
      echo echo     "alert_cooldown_minutes": 15 >> create_config.bat
      echo echo   }, >> create_config.bat
      echo echo   "api": { >> create_config.bat
      echo echo     "port": 6868, >> create_config.bat
      echo echo     "enable_authentication": true, >> create_config.bat
      echo echo     "jwt_secret": "CHANGE_THIS_TO_A_SECURE_RANDOM_STRING", >> create_config.bat
      echo echo     "jwt_expiry_hours": 24, >> create_config.bat
      echo echo     "enable_cors": true, >> create_config.bat
      echo echo     "allowed_origins": ["*"], >> create_config.bat
      echo echo     "rate_limit_per_minute": 60 >> create_config.bat
      echo echo   }, >> create_config.bat
      echo echo   "monitoring": { >> create_config.bat
      echo echo     "metrics_retention_days": 30, >> create_config.bat
      echo echo     "logs_retention_days": 14, >> create_config.bat
      echo echo     "performance_logging_interval_seconds": 60, >> create_config.bat
      echo echo     "log_level": "info", >> create_config.bat
      echo echo     "log_file_path": "./logs/kyaro.log", >> create_config.bat
      echo echo     "enable_console_output": true >> create_config.bat
      echo echo   }, >> create_config.bat
      echo echo   "system": { >> create_config.bat
      echo echo     "worker_threads": 0, >> create_config.bat
      echo echo     "max_memory_usage_mb": 0, >> create_config.bat
      echo echo     "enable_auto_update": false, >> create_config.bat
      echo echo     "update_check_interval_hours": 24 >> create_config.bat
      echo echo   } >> create_config.bat
      echo echo } >> create_config.bat
      call create_config.bat
      del create_config.bat
      echo - Created default configuration file.
   )
) else (
   echo - Config file already exists, skipping creation.
)
echo.

echo Checking for Rust installation...
where rustc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
   echo - Rust not found. Please install Rust from https://rustup.rs/
   echo - After installing Rust, you can build the project with: cd kyaro_antiddos ^&^& cargo build --release
) else (
   echo - Rust is installed.
   echo - You can build the project with: cd kyaro_antiddos ^&^& cargo build --release
)
echo.

echo Installation completed!
echo.
echo Next steps:
echo 1. Review and customize config.json for your environment
echo 2. Build the Rust backend with: cd kyaro_antiddos ^&^& cargo build --release
echo 3. Start the service with start.bat
echo.
echo Thank you for using Kyaro Anti-DDoS!
echo.

pause 