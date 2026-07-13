# 🐧 Running TradeLocker Signal Bot on Chromebook (Linux)

## Step 1: Enable Linux on Your Chromebook

If you haven't already enabled Linux:

1. Click the **clock** in bottom-right corner
2. Click the **Settings** gear icon
3. Scroll to **Advanced** → **Developers**
4. Click **Turn On** next to Linux (Beta)
5. Follow the setup wizard (allocate at least 10GB disk space)

---

## Step 2: Install Python and Git

Open the Linux Terminal app and run:

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv python3-full git
```

> **Note:** Newer Debian/Python versions (3.11+) block system-wide pip installs (PEP 668). The `start.sh` script automatically creates and uses a **virtual environment** to handle this cleanly — no `--break-system-packages` hacks needed.

Verify installation:
```bash
python3 --version  # Should show Python 3.8+
git --version
```

---

## Step 3: Clone the Repository

```bash
cd ~
git clone https://github.com/lafaillejeremiah7-droid/claude.git
cd claude/tradelocker_bot
```

---

## Step 4: Configure Your TradeLocker Credentials

Open the file in a text editor:

```bash
nano live_terminal.py
```

Find this section near the top and update with your credentials:

```python
# TradeLocker API Config (your AquaFunded account)
TL_SERVER = "https://demo.tradelocker.com"  # or live: https://live.tradelocker.com
TL_EMAIL = "your_email@example.com"
TL_PASSWORD = "your_password"
TL_ACC_NUM = 12345  # your account number
```

**Save and exit:** Press `Ctrl+X`, then `Y`, then `Enter`

---

## Step 5: Run the Bot

### Option A: Use the Quick Start Script (Recommended)

```bash
chmod +x start.sh
./start.sh
```

The script will:
- ✅ Check if Python is installed
- ✅ Auto-install dependencies if needed
- ✅ Start the live terminal
- ✅ Show you the dashboard URL

### Option B: Manual Start

```bash
pip3 install --user fastapi uvicorn httpx websocket-client pandas requests
python3 live_terminal.py
```

---

## Step 6: Open the Dashboard

Once the bot is running, you'll see:
```
🚀 Starting TradeLocker XAUUSD Signal Bot...
📊 Dashboard: http://localhost:5000/terminal
INFO:     Uvicorn running on http://0.0.0.0:5000
```

**Open your Chromebook browser** and go to:

**http://localhost:5000/terminal**

---

## 📊 What You'll See

- **Live XAUUSD Price Chart** → Real-time price from your TradeLocker account
- **Current Signal** → BUY / SELL / NEUTRAL with strength indicator
- **Recent Signals** → History of the last 20 signals
- **Alert Panel** → Important messages and warnings

---

## 🎯 How to Use It

1. **Monitor the dashboard** for BUY/SELL signals
2. **Review the signal strength** (0-100%)
3. **Manually place trades** in your TradeLocker account when you see a strong signal
4. The bot **NEVER places trades automatically** — you're always in control

---

## 🛑 Stopping the Bot

Press **Ctrl+C** in the terminal to stop the server.

---

## 🔧 Chromebook-Specific Tips

### Keep Terminal Running
The bot needs to stay running to generate signals. Don't close the Linux Terminal window!

### Access from Android Apps?
Unfortunately, Android apps on Chromebook can't access `localhost`. You must use the Chrome browser.

### Run on Startup?
To auto-start the bot when Linux starts, create a systemd service:

```bash
nano ~/.config/systemd/user/tradelocker-bot.service
```

Add:
```ini
[Unit]
Description=TradeLocker Signal Bot

[Service]
WorkingDirectory=/home/YOUR_USERNAME/claude/tradelocker_bot
ExecStart=/usr/bin/python3 live_terminal.py
Restart=always

[Install]
WantedBy=default.target
```

Then:
```bash
systemctl --user enable tradelocker-bot.service
systemctl --user start tradelocker-bot.service
```

---

## ⚡ Performance Tips

Chromebook Linux containers can be resource-limited. If the bot is slow:

1. **Increase Linux container resources:**
   - Settings → Developers → Linux → Disk Size (increase to 20GB+)
   
2. **Close unused Chrome tabs** to free up RAM

3. **Use Power mode** (not Battery Saver) when trading

---

## 🐛 Troubleshooting

### "Permission denied" when running start.sh
```bash
chmod +x start.sh
```

### "ModuleNotFoundError" (e.g. No module named 'httpx')
This happens when deps didn't install into the venv. Fix it:
```bash
cd ~/claude/tradelocker_bot
rm -rf venv
./start.sh
```
The script rebuilds the virtual environment and reinstalls everything.

### "error: externally-managed-environment"
This is PEP 668 blocking system-wide installs. The updated `start.sh` handles it automatically by using a venv. Just make sure you have venv support:
```bash
sudo apt install python3-venv python3-full
```

### "Failed to create venv"
```bash
sudo apt install python3-venv python3-full
```

### Can't access http://localhost:5000
Make sure:
- The bot is actually running (check terminal for errors)
- You're using **Chrome browser** (not Android app)
- Port 5000 isn't blocked by firewall

### Terminal closes unexpectedly
The Linux container may have stopped. Reopen the Terminal app and restart the bot.

---

## 📱 Optional: Get Signals on Your Phone

Set up Telegram notifications to receive signals on your phone:

1. Create a Telegram bot: Message [@BotFather](https://t.me/botfather)
2. Get your chat ID: Message [@userinfobot](https://t.me/userinfobot)
3. Edit `live_terminal.py` and add:

```python
TELEGRAM_BOT_TOKEN = "your_bot_token_here"
TELEGRAM_CHAT_ID = "your_chat_id_here"
```

---

## 🆘 Need Help?

**Common Commands:**
```bash
# Check if bot is running
ps aux | grep live_terminal

# View logs
tail -f ~/claude/tradelocker_bot/real_trades.json

# Restart the bot
pkill -f live_terminal.py
./start.sh
```

---

**Remember: This is a SIGNAL-ONLY bot. It will NEVER place trades automatically. You must manually execute all trades!**

Happy trading! 📈
