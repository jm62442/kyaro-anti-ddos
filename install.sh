#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===================================================="
echo "        Kyaro Anti-DDoS Installation Script"
echo -e "====================================================${NC}"
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Installing in: $SCRIPT_DIR"
echo ""

# Create directories
echo "Creating necessary directories..."
mkdir -p "$SCRIPT_DIR/logs"
mkdir -p "$SCRIPT_DIR/kyaro_antiddos/src"
mkdir -p "$SCRIPT_DIR/dashboard/src"
mkdir -p "$SCRIPT_DIR/ml_models"
echo "- Directories created successfully."
echo ""

# Create a simple Rust project
echo "Creating basic Rust project..."
if [ ! -f "$SCRIPT_DIR/kyaro_antiddos/Cargo.toml" ]; then
    cat > "$SCRIPT_DIR/kyaro_antiddos/Cargo.toml" << EOF
[package]
name = "kyaro_antiddos"
version = "0.1.0"
edition = "2021"

[dependencies]
EOF
fi

if [ ! -f "$SCRIPT_DIR/kyaro_antiddos/src/main.rs" ]; then
    cat > "$SCRIPT_DIR/kyaro_antiddos/src/main.rs" << EOF
fn main() {
    println!("Hello from Kyaro Anti-DDoS!");
}
EOF
fi
echo "- Basic Rust project created."
echo ""

# Create a basic dashboard
echo "Creating basic dashboard..."
if [ ! -f "$SCRIPT_DIR/dashboard/package.json" ]; then
    cat > "$SCRIPT_DIR/dashboard/package.json" << EOF
{
  "name": "kyaro-dashboard",
  "version": "0.1.0",
  "description": "Dashboard for Kyaro Anti-DDoS",
  "scripts": {
    "start": "echo Starting dashboard..."
  }
}
EOF
fi
echo "- Basic dashboard created."
echo ""

# Create basic start and stop scripts
echo "Creating start.sh..."
if [ ! -f "$SCRIPT_DIR/start.sh" ]; then
    cat > "$SCRIPT_DIR/start.sh" << EOF
#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

echo -e "\${GREEN}===================================================="
echo "            Kyaro Anti-DDoS Startup Script"
echo -e "====================================================\${NC}"
echo ""

# Check for root privileges but don't require them
if [ "\$EUID" -eq 0 ]; then
  echo -e "Running with root privileges."
else
  echo -e "\${YELLOW}WARNING: Running without root privileges. Some operations may fail.\${NC}"
  echo -e "Consider running with sudo if you encounter permission issues."
fi
echo ""

# Make logs directory if it doesn't exist
mkdir -p "\$SCRIPT_DIR/logs"

# Check if Rust executable exists
if [ -f "\$SCRIPT_DIR/kyaro_antiddos/target/release/kyaro_antiddos" ]; then
    # Start the backend
    echo "Starting backend service..."
    "\$SCRIPT_DIR/kyaro_antiddos/target/release/kyaro_antiddos" --config "\$SCRIPT_DIR/config.json" > "\$SCRIPT_DIR/logs/backend.log" 2>&1 &
    BACKEND_PID=\$!
    echo "\$BACKEND_PID" > "\$SCRIPT_DIR/logs/backend.pid"
    echo "- Backend service started with PID \$BACKEND_PID"
else
    echo -e "\${RED}Backend executable not found.\${NC}"
    echo "You need to build the project with cargo first."
    echo ""
    echo "Example: cd kyaro_antiddos && cargo build --release"
    exit 1
fi

echo ""
echo "To stop the service, use the stop.sh script."
echo ""

# Keep terminal open
echo "Press Ctrl+C to exit this script (services will continue running in background)"
echo ""
tail -f "\$SCRIPT_DIR/logs/backend.log"
EOF
    chmod +x "$SCRIPT_DIR/start.sh"
fi
echo "- Created start.sh"

echo "Creating stop.sh..."
if [ ! -f "$SCRIPT_DIR/stop.sh" ]; then
    cat > "$SCRIPT_DIR/stop.sh" << EOF
#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"

echo -e "\${GREEN}===================================================="
echo "            Kyaro Anti-DDoS Stop Script"
echo -e "====================================================\${NC}"
echo ""

echo "Stopping Kyaro Anti-DDoS services..."

# Check for PID file first
if [ -f "\$SCRIPT_DIR/logs/backend.pid" ]; then
    PID=\$(cat "\$SCRIPT_DIR/logs/backend.pid")
    echo "- Found PID file with PID: \$PID"
    if ps -p \$PID > /dev/null; then
        echo "- Process is running, stopping it..."
        kill \$PID
        sleep 1
        # Force kill if still running
        if ps -p \$PID > /dev/null; then
            echo "- Process still running, force killing..."
            kill -9 \$PID
        fi
        echo "- Process stopped"
        rm "\$SCRIPT_DIR/logs/backend.pid"
    else
        echo "- Process not running, cleaning up PID file"
        rm "\$SCRIPT_DIR/logs/backend.pid"
    fi
# Fallback to process name search
elif pgrep -f "kyaro_antiddos" > /dev/null; then
    echo "- Found running Kyaro Anti-DDoS processes"
    echo "- Stopping processes..."
    pkill -f "kyaro_antiddos" 2>/dev/null
    sleep 1
    # Force kill if any processes are still running
    if pgrep -f "kyaro_antiddos" > /dev/null; then
        echo "- Some processes still running, force killing..."
        pkill -9 -f "kyaro_antiddos" 2>/dev/null
    fi
    echo "- Processes stopped"
else
    echo -e "\${YELLOW}- No running Kyaro Anti-DDoS processes found\${NC}"
fi

echo ""
echo -e "\${GREEN}All services stopped.\${NC}"
echo ""
EOF
    chmod +x "$SCRIPT_DIR/stop.sh"
fi
echo "- Created stop.sh"
echo ""

# Create a sample config file
echo "Creating config.json..."
if [ ! -f "$SCRIPT_DIR/config.json" ]; then
    if [ -f "$SCRIPT_DIR/kyaro_antiddos/config.example.json" ]; then
        cp "$SCRIPT_DIR/kyaro_antiddos/config.example.json" "$SCRIPT_DIR/config.json"
        echo "- Default configuration created from example."
    else
        # Directly create the config file
        cat > "$SCRIPT_DIR/config.json" << EOF
{
  "layer3": {
    "enable_ip_blacklisting": true,
    "enable_geo_blocking": true,
    "enable_rate_limiting": true,
    "max_packet_size": 65535,
    "min_ttl": 5,
    "blocked_countries": [],
    "rate_limit_threshold": 1000,
    "blacklist_duration_minutes": 30,
    "whitelist_ips": [
      "127.0.0.1",
      "::1"
    ]
  },
  "layer4": {
    "enable_syn_protection": true,
    "enable_udp_protection": true,
    "enable_icmp_protection": true,
    "syn_rate_threshold": 500,
    "udp_rate_threshold": 500,
    "icmp_rate_threshold": 100,
    "max_concurrent_connections": 1000,
    "blocked_ports": [],
    "connection_timeout_seconds": 120
  },
  "layer7": {
    "enable_http_protection": true,
    "enable_dns_protection": true,
    "enable_ssl_protection": true,
    "http_request_rate_threshold": 100,
    "blocked_http_methods": ["TRACE", "TRACK"],
    "challenge_suspicious_requests": true,
    "bot_protection_level": 2,
    "waf_rules_enabled": true,
    "allowed_user_agents": [],
    "blocked_user_agents": [
      "curl",
      "wget",
      "python-requests",
      "go-http-client"
    ]
  },
  "ml": {
    "enable_ml": true,
    "training_interval_hours": 24,
    "detection_threshold": 0.85,
    "anomaly_sensitivity": 0.7,
    "feature_extraction_window_seconds": 60,
    "model_save_path": "./ml_models",
    "use_gpu_acceleration": false
  },
  "notification": {
    "enable_email_alerts": false,
    "email_recipients": [],
    "smtp_server": "",
    "smtp_port": 587,
    "smtp_username": "",
    "smtp_password": "",
    "alert_threshold_severity": "high",
    "alert_cooldown_minutes": 15
  },
  "api": {
    "port": 6868,
    "enable_authentication": true,
    "jwt_secret": "CHANGE_THIS_TO_A_SECURE_RANDOM_STRING",
    "jwt_expiry_hours": 24,
    "enable_cors": true,
    "allowed_origins": ["*"],
    "rate_limit_per_minute": 60
  },
  "monitoring": {
    "metrics_retention_days": 30,
    "logs_retention_days": 14,
    "performance_logging_interval_seconds": 60,
    "log_level": "info",
    "log_file_path": "./logs/kyaro.log",
    "enable_console_output": true
  },
  "system": {
    "worker_threads": 0,
    "max_memory_usage_mb": 0,
    "enable_auto_update": false,
    "update_check_interval_hours": 24
  }
}
EOF
        echo "- Created default configuration file."
    fi
else
    echo "- Config file already exists, skipping creation."
fi
echo ""

# Check for Rust installation
echo "Checking for Rust installation..."
if ! command -v rustc &> /dev/null; then
    echo -e "${YELLOW}- Rust not found. Installing Rust is recommended.${NC}"
    echo "- You can install Rust with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    echo "- After installing Rust, you can build the project with: cd kyaro_antiddos && cargo build --release"
else
    echo "- Rust is installed ($(rustc --version))"
    echo "- You can build the project with: cd kyaro_antiddos && cargo build --release"
fi
echo ""

echo -e "${GREEN}Installation completed!${NC}"
echo
echo "Next steps:"
echo "1. Review and customize config.json for your environment"
echo "2. Build the Rust backend with: cd kyaro_antiddos && cargo build --release"
echo "3. Start the service with ./start.sh"
echo
echo "Thank you for using Kyaro Anti-DDoS!"
echo 