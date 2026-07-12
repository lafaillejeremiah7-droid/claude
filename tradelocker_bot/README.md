# TradeLocker Multi-Timeframe Trading Bot

An autonomous Python trading bot for **BTC/USD** and **XAU/USD** that connects to TradeLocker's Live API. It implements a multi-timeframe EMA trend-following strategy with liquidity sweep detection, market structure breaks, and comprehensive risk management.

---

## Strategy Overview

### Entry Logic (All Must Confirm)

| # | Confirmation | Description |
|---|---|---|
| 1 | **4H Trend** | 50 EMA sloping in direction + price above/below it |
| 2 | **30M Confirmation** | 50 EMA vs 200 EMA alignment + price position |
| 3 | **5M Pullback** | Price retraces to 20 EMA or VWAP |
| 4 | **RSI Zone** | Longs: 45-60, Shorts: 40-55 (avoid >70 or <30) |
| 5 | **Liquidity Sweep** | Price sweeps swing high/low then reverses |
| 6 | **Structure Break** | Close above lower high (buy) / below higher low (sell) |
| 7 | **Candle Pattern** | Engulfing, hammer, shooting star, or strong rejection |
| 8 | **Volume** | Current bar volume > 20-period average |

### Risk Management

- **2% risk per trade** (position sized by SL distance)
- **Stop Loss**: Wider of swing high/low or 1 ATR(14)
- **Take Profit**: 1.5R minimum, 2R when trend is strong
- **Breakeven**: SL moves to entry at 1R profit
- **Max 2 trades per day**
- **4% daily drawdown limit** → stops trading for the day
- **4% weekly drawdown limit** → stops trading for the week
- **2 consecutive losses** → stops trading for the day

### Session Filter

- **BTC/USD**: London (07-16 UTC), New York (12-21 UTC), Overlap (12-16 UTC)
- **XAU/USD**: London + New York only (no Asian session)
- **News Avoidance**: No trades within 30 minutes of high-impact events

---

## Project Structure

```
tradelocker_bot/
├── main.py                    # Bot entry point & orchestrator
├── config.py                  # All configuration & settings
├── requirements.txt           # Python dependencies
├── .env.example               # Template for credentials
├── .env                       # Your credentials (DO NOT COMMIT)
├── modules/
│   ├── __init__.py
│   ├── api_client.py          # TradeLocker REST API client
│   ├── indicators.py          # Technical indicator calculations
│   ├── trend_analysis.py      # Multi-timeframe trend analysis
│   ├── entry_signals.py       # Entry signal detection (6 confirmations)
│   ├── risk_management.py     # Position sizing & risk limits
│   ├── session_filter.py      # Trading session & news filter
│   └── trade_manager.py       # Position lifecycle management
├── logs/                      # Daily log files & stats
│   ├── bot_YYYY-MM-DD.log
│   ├── daily_stats.json
│   └── active_positions.json
└── journal/                   # Trade journal (JSONL per day)
    └── journal_YYYY-MM-DD.jsonl
```

---

## Setup Instructions

### Prerequisites

- Python 3.11 or higher
- A TradeLocker account with a broker that supports TradeLocker
- Your broker server name (yours is `AQUA`)

### 1. Clone the Repository

```bash
git clone https://github.com/lafaillejeremiah7-droid/claude.git
cd claude/tradelocker_bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Credentials

Copy the example environment file and fill in your details:

```bash
cp .env.example .env
```

Edit `.env` with your TradeLocker credentials:

```env
TL_EMAIL=your_email@example.com
TL_PASSWORD=your_tradelocker_password
TL_SERVER=AQUA
TL_ENVIRONMENT=live
```

> **IMPORTANT**: Never commit your `.env` file. It's already in `.gitignore`.

### 4. Finding Your Credentials

| Credential | Where to Find It |
|---|---|
| Email | Your TradeLocker login email |
| Password | Your TradeLocker login password |
| Server | Login screen server name (yours: `AQUA`) |
| Environment | `live` for real money, `demo` for paper trading |

### 5. Run the Bot

```bash
# Dry run first (no real trades, just logs signals)
python main.py --dry

# Check current status
python main.py --status

# Run live (real trades!)
python main.py
```

---

## Running Modes

### Dry Run (Recommended First)

```bash
python main.py --dry
```

Runs the full analysis pipeline but **does NOT execute trades**. Use this to:
- Verify your credentials work
- See what signals the bot detects
- Confirm session timing is correct
- Validate the strategy logic

### Status Check

```bash
python main.py --status
```

Shows current risk limits, open positions, and session status then exits.

### Live Trading

```bash
python main.py
```

Full autonomous mode. The bot will:
1. Scan every 60 seconds
2. Analyze trends across 4H, 30M, and 5M charts
3. Execute trades when all conditions align
4. Manage positions (breakeven moves, monitoring)
5. Enforce all risk limits automatically

---

## Running 24/7 (VPS/Server)

To run the bot continuously on a VPS or server:

```bash
# Using screen
screen -S tradebot
python main.py
# Detach with Ctrl+A, D

# Using nohup
nohup python main.py > /dev/null 2>&1 &

# Using systemd (recommended)
# Create /etc/systemd/system/tradebot.service
```

Example systemd service:

```ini
[Unit]
Description=TradeLocker Trading Bot
After=network.target

[Service]
Type=simple
User=your_user
WorkingDirectory=/path/to/tradelocker_bot
ExecStart=/usr/bin/python3 main.py
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

---

## Configuration

All settings are in `config.py` and can be overridden via `.env`:

| Setting | Default | Description |
|---|---|---|
| `RISK_PERCENT` | 2.0 | % of equity risked per trade |
| `MAX_TRADES_PER_DAY` | 2 | Maximum trades allowed per day |
| `DAILY_DRAWDOWN_LIMIT` | 4.0 | % drawdown to stop daily trading |
| `WEEKLY_DRAWDOWN_LIMIT` | 4.0 | % drawdown to stop weekly trading |
| `INSTRUMENTS` | BTCUSD,XAUUSD | Comma-separated symbols |
| `SCAN_INTERVAL_SECONDS` | 60 | How often to scan (seconds) |
| `LOG_LEVEL` | INFO | Logging verbosity |

---

## Trading Journal

Every trade action is logged to `journal/journal_YYYY-MM-DD.jsonl`:

- Entry reasons and all confirmations
- SL/TP levels and method used
- Breakeven moves
- Partial profit takes
- Exit details with P&L and R-multiple

Review these regularly to assess strategy performance.

---

## Safety Features

1. **Dry run mode** - Test without risking capital
2. **Graceful shutdown** - Ctrl+C stops cleanly (positions remain open with SL/TP)
3. **Position persistence** - Bot remembers positions across restarts
4. **Token auto-refresh** - Handles expired auth tokens automatically
5. **Error resilience** - Continues running after API errors
6. **Daily/weekly locks** - Automatically stops trading at drawdown limits
7. **Never widens SL** - Stop loss only moves to breakeven, never further away

---

## Disclaimer

This bot trades with real money when not in dry-run mode. Trading carries significant risk of loss. Past performance does not guarantee future results. Always:

- Start with dry-run mode
- Test on a demo account first (`TL_ENVIRONMENT=demo`)
- Monitor the bot regularly
- Never risk more than you can afford to lose
- Review your trade journal after every 100-300 trades

---

## License

MIT
