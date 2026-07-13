# 🚀 Running TradeLocker Signal Bot Locally

## Prerequisites

- **Python 3.8+** installed
- **Git** installed
- Terminal/Command Prompt access

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/lafaillejeremiah7-droid/claude.git
cd claude/tradelocker_bot
```

---

## Step 2: Install Dependencies

```bash
pip install fastapi uvicorn httpx websocket-client pandas requests
```

Or if you prefer using a virtual environment (recommended):

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Mac/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn httpx websocket-client pandas requests
```

---

## Step 3: Configure TradeLocker Credentials

Open `live_terminal.py` and find this section near the top:

```python
# TradeLocker API Config (your AquaFunded account)
TL_SERVER = "https://demo.tradelocker.com"  # or live: https://live.tradelocker.com
TL_EMAIL = "your_email@example.com"
TL_PASSWORD = "your_password"
TL_ACC_NUM = 12345  # your account number
```

**Update with your actual credentials:**
- Set `TL_SERVER` to your TradeLocker server URL
- Set `TL_EMAIL` to your TradeLocker account email
- Set `TL_PASSWORD` to your TradeLocker password
- Set `TL_ACC_NUM` to your account number

---

## Step 4: Start the Live Terminal

```bash
python3 live_terminal.py
```

You should see:
```
Starting XAUUSD ASWP Live Terminal...
Open http://localhost:5000/terminal
INFO:     Uvicorn running on http://0.0.0.0:5000
```

---

## Step 5: Open the Dashboard

Open your browser and go to:

**http://localhost:5000/terminal**

You'll see:
- 📊 Live XAUUSD price chart
- 📈 Current signal (BUY/SELL/NEUTRAL)
- ⚡ Signal strength meter
- 📋 Recent signal history
- 🔔 Alert panel

---

## 🎯 What the Bot Does

✅ **Monitors XAUUSD price** from TradeLocker API (your actual broker)  
✅ **Generates BUY/SELL signals** based on ASWP strategy logic  
✅ **Displays signals** in real-time dashboard  
✅ **Logs all signals** to `real_trades.json`  

❌ **Does NOT place trades automatically** (signal-only bot)

---

## 📊 Data Sources

1. **TradeLocker API** → Live bid/ask prices (primary, every 10s)
2. **TradingView WebSocket** → Real-time backup feed
3. **Forexite API** → Historical 5m bars for indicators

---

## 🔧 Troubleshooting

### Port 5000 already in use?
Edit `live_terminal.py` and change the port:

```python
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)  # Use 8000 instead
```

Then access: http://localhost:8000/terminal

### Can't connect to TradeLocker API?
- Verify your credentials are correct
- Check if your IP needs to be whitelisted
- Ensure you're using the correct server URL (demo vs live)

### Missing dependencies?
```bash
pip install --upgrade fastapi uvicorn httpx websocket-client pandas requests
```

---

## 📱 Optional: Telegram Notifications

To receive signal alerts on Telegram:

1. Create a Telegram bot via [@BotFather](https://t.me/botfather)
2. Get your bot token
3. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
4. Add to `live_terminal.py`:

```python
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"
```

---

## 🛑 Stopping the Bot

Press **Ctrl+C** in the terminal to stop the server gracefully.

---

## 📂 File Structure

```
tradelocker_bot/
├── live_terminal.py              # Main server (run this)
├── replay_terminal.py            # Backtesting terminal
├── modules/
│   └── xauusd_aswp_engine.py    # Strategy logic
├── dashboard/
│   └── frontend/
│       └── signal_terminal.html  # Dashboard UI
├── real_trades.json              # Signal log
└── XAUUSD_STRATEGY_LOGIC.md     # Strategy documentation
```

---

## 🎓 Next Steps

1. **Monitor signals** in the dashboard
2. **Manually place trades** based on the signals you trust
3. **Review `real_trades.json`** to track signal history
4. **Read `XAUUSD_STRATEGY_LOGIC.md`** to understand the strategy

---

## 🆘 Need Help?

If you encounter issues, check:
- Python version: `python3 --version` (should be 3.8+)
- Dependencies installed: `pip list | grep fastapi`
- Server logs in the terminal where you ran `live_terminal.py`

---

**Remember: This is a SIGNAL-ONLY bot. Always review signals before manually placing trades!**
