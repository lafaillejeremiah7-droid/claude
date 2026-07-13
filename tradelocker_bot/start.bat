@echo off
REM Quick start script for TradeLocker Signal Bot (Windows)

echo 🚀 Starting TradeLocker XAUUSD Signal Bot...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Check if dependencies are installed
echo 📦 Checking dependencies...
python -c "import fastapi" 2>nul
if errorlevel 1 (
    echo ⚠️  Dependencies not found. Installing...
    pip install fastapi uvicorn httpx websocket-client pandas requests
) else (
    echo ✅ Dependencies installed
)

echo.
echo 🎯 Starting live terminal...
echo 📊 Dashboard will be available at: http://localhost:5000/terminal
echo.
echo Press Ctrl+C to stop the bot
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo.

REM Start the bot
python live_terminal.py
pause
