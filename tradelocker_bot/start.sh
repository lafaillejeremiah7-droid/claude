#!/bin/bash
# GOLD VORTEX v5 — Live Signal Bot Launcher
set -e
cd "$(dirname "$0")"

echo "GOLD VORTEX v5 — Live Signal Bot"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Install with: sudo apt install python3 python3-venv"
    exit 1
fi

echo "Python: $(python3 --version)"

# Create venv if needed
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR" || { echo "Install python3-venv: sudo apt install python3-venv"; exit 1; }
fi

# Activate
source "$VENV_DIR/bin/activate"

# Install deps
echo "Checking dependencies..."
if ! python -c "import fastapi, uvicorn, httpx, pandas, numpy" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install --upgrade pip
    pip install fastapi uvicorn httpx pandas numpy
    echo ""
fi

echo ""
echo "Starting bot..."
echo "Dashboard: http://localhost:5000"
echo "Signals also sent to Telegram"
echo ""
echo "Press Ctrl+C to stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python live_bot.py
