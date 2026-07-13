#!/bin/bash
# Quick start script for TradeLocker Signal Bot
# Optimized for Linux/Chromebook (handles PEP 668 externally-managed environments)

set -e

echo "🚀 Starting TradeLocker XAUUSD Signal Bot..."
echo ""

# Move to the directory this script lives in
cd "$(dirname "$0")"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo "On Chromebook Linux, install with:"
    echo "   sudo apt update && sudo apt install python3 python3-venv python3-pip"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

VENV_DIR="venv"

# Create the virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment (first run only)..."
    if ! python3 -m venv "$VENV_DIR" 2>/dev/null; then
        echo "❌ Failed to create venv. You may need python3-venv:"
        echo "   sudo apt install python3-venv python3-full"
        exit 1
    fi
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"

# Install / verify dependencies inside the venv
echo "📦 Checking dependencies..."
if ! python -c "import httpx, fastapi, uvicorn, pandas, requests, websocket" 2>/dev/null; then
    echo "⚠️  Installing dependencies into venv..."
    echo ""
    pip install --upgrade pip
    pip install fastapi uvicorn httpx websocket-client pandas requests
    echo ""
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

echo ""
echo "🎯 Starting live terminal..."
echo "📊 Dashboard: http://localhost:5000/terminal"
echo ""
echo "⚠️  REMEMBER: This is a SIGNAL-ONLY bot"
echo "    It will NOT place trades automatically."
echo "    You must manually execute trades based on signals."
echo ""
echo "Press Ctrl+C to stop the bot"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Start the bot using the venv's Python
python live_terminal.py
