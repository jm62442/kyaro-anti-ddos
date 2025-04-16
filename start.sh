#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}===================================================="
echo "            Kyaro Anti-DDoS Startup Script"
echo -e "====================================================${NC}"
echo ""

# Check for root privileges but don't require them
if [ "$EUID" -eq 0 ]; then
  echo -e "Running with root privileges."
else
  echo -e "${YELLOW}WARNING: Running without root privileges. Some operations may fail.${NC}"
  echo -e "Consider running with sudo if you encounter permission issues."
fi
echo ""

# Check if Rust executable exists
if [ -f "$SCRIPT_DIR/kyaro_antiddos/target/release/kyaro_antiddos" ]; then
    # Start the backend
    echo "Starting backend service..."
    echo "The backend would normally be started here if fully implemented."
    # Uncomment the following line when you have a working backend:
    # "$SCRIPT_DIR/kyaro_antiddos/target/release/kyaro_antiddos" > "$SCRIPT_DIR/logs/backend.log" 2>&1 &
    echo "- Backend service started (simulated)"
else
    echo "Backend executable not found."
    echo "You may need to build the project with cargo first."
    echo ""
    echo "Example: cd kyaro_antiddos && cargo build --release"
fi

# Make logs directory if it doesn't exist
mkdir -p "$SCRIPT_DIR/logs"

echo ""
echo "Note: This is a placeholder startup script."
echo "You need to implement the actual functionality."
echo ""
echo "To stop the service, use the stop.sh script."
echo ""

# Keep terminal open to simulate running service
echo "Press Ctrl+C to exit this script (services will be simulated as still running)"
echo ""
# Simple infinite loop that does nothing to keep the script running
tail -f /dev/null 