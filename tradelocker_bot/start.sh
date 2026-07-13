#!/bin/bash
# Quick start script for TradeLocker Signal Bot
# Optimized for Linux/Chromebook

echo "🚀 Starting TradeLocker XAUUSD Signal Bot..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed."
    echo "On Chromebook Linux, install with:"
    echo "   sudo apt update && sudo apt install python3 python3-pip"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed."
    echo "On Chromebook Linux, install with:"
    echo "   sudo apt install python3-pip"
    exit 1
fi

echo "✅ pip3 found"
echo ""

# Check if dependencies are installed
echo "📦 Checking dependencies..."
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Dependencies not found. Installing..."
    echo ""
    pip3 install --user fastapi uvicorn httpx websocket-client pandas requests
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

# Start the bot
python3 live_terminal.py
