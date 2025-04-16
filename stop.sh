#!/bin/bash

# ANSI color codes
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${GREEN}===================================================="
echo "            Kyaro Anti-DDoS Stop Script"
echo -e "====================================================${NC}"
echo ""

echo "Stopping Kyaro Anti-DDoS services..."

# Check for any running kyaro_antiddos processes
if pgrep -f "kyaro_antiddos" > /dev/null; then
    echo "- Found running Kyaro Anti-DDoS processes"
    echo "- Attempting to stop processes..."
    pkill -f "kyaro_antiddos" 2>/dev/null
    echo "- Processes stopped"
else
    echo -e "${YELLOW}- No running Kyaro Anti-DDoS processes found${NC}"
fi

echo ""
echo -e "${GREEN}All services stopped (simulated).${NC}"
echo ""
echo "Note: This is a placeholder stop script."
echo "You need to implement the actual functionality."
echo "" 