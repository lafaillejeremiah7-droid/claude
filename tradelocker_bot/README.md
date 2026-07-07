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
│   ├── reporting.py           # Performance reporting engine (daily/weekly/monthly)
├── logs/                      # Daily log files & stats
│   ├── bot_YYYY-MM-DD.log
│   ├── daily_stats.json
│   ├── active_positions.json
│   ├── adaptive_config.json
│   ├── trade_features.jsonl
│   └── reports/               # Machine-readable performance reports (see below)
│       ├── daily_YYYY-MM-DD.json
│       ├── weekly_YYYY-Www.json
│       ├── monthly_YYYY-MM.json
│       ├── history.jsonl
│       └── .report_state.json
├── journal/                   # Trade journal (JSONL per day)
│   └── journal_YYYY-MM-DD.jsonl
└── tests/                     # Pytest suite
    └── test_reporting.py
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

## Performance Reports

The bot includes a **performance reporting engine** (`modules/reporting.py`,
`PerformanceReporter`) that produces both human-readable log summaries and
machine-readable JSON files. All times are **UTC**.

Reports are emitted automatically on time-boundary rollovers. Once per scan
cycle the bot calls `reporter.maybe_emit(now_utc)`, which detects whether a
day, week, or month has rolled over since the last report and emits whatever is
due. Detection is robust to the bot being offline across a boundary — the
pending report fires on the next start. The last-reported periods are tracked
in `logs/reports/.report_state.json`, so no report is ever emitted twice.

The reporter is **read-mostly**: it reads the bot's existing state
(`daily_stats.json` / weekly stats, `adaptive_config.json`,
`trade_features.jsonl`, and the per-day `journal/*.jsonl` files) and only ever
**writes inside `logs/reports/`**. It never overwrites the bot's live state.

### What is produced

| Report | Trigger | Contents |
|---|---|---|
| **Daily** | UTC day rollover | P&L ($ and %), trades taken, W/L, win rate, best & worst trade, average R |
| **Weekly** | UTC (ISO) week rollover | Weekly P&L ($ and %), total trades, win rate, average R, max drawdown, plus a **"What to improve"** self-adaptation section |
| **Monthly** | UTC month rollover | Monthly P&L ($ and %), total trades, win rate, best & worst day |

Example daily log line:

```
=== DAILY REPORT 2024-06-10 UTC ===  P&L: +$142.30 (+1.42%) | 2 trades | 1W/1L (50%) | Best +$210 | Worst -$68 | Avg R 0.34
```

### "What to improve" (weekly self-adaptation insight)

The weekly report derives concrete suggestions from the adaptive engine data
and the week's trades — nothing is hard-coded. Insights only surface when the
sample size is meaningful. Examples of the kinds of bullets generated:

```
IMPROVE: - Win rate in 08:00-09:00 UTC is 22% (12 trades) - consider avoid_hours.
IMPROVE: - 'doji' pattern avg R -0.4 over 9 trades (win rate 33%) - down-weight.
IMPROVE: - Low-confidence trades [8.0-8.5) win 30% vs 80% for [9.0-10.0) - consider raising min_confidence.
```

Sources analysed: worst-performing UTC hours, lowest-performing candle
patterns/features, confidence-band win rates, drawdown / consecutive-loss lock
incidents, and the adaptive engine's win-rate/avg-R trend versus the prior week.

### Where the files live (for the dashboard)

The dashboard reads the `logs/reports/` directory:

- `daily_YYYY-MM-DD.json` — one file per day
- `weekly_YYYY-Www.json` — one file per ISO week
- `monthly_YYYY-MM.json` — one file per month
- `history.jsonl` — one appended line per daily report (used for weekly/monthly aggregation)

### Live vs paper (mode-aware)

`PerformanceReporter` is mode-aware: it accepts a stats source so it can report
on **live** stats (default: `daily_stats.json` / weekly) or **paper** stats
once paper-trading files exist. Missing paper files never hard-fail — the
reporter degrades gracefully.

### Configuration

| Setting | Default | Description |
|---|---|---|
| `REPORTS_DIR` | `logs/reports` | Directory for machine-readable reports |
| `REPORT_MIN_SAMPLE` | 5 | Min trades in a bucket before an improvement suggestion surfaces |
| `REPORT_WEAK_WIN_RATE` | 0.40 | Win-rate threshold below which an hour/pattern is flagged |

### Tests

```bash
pip install pytest hypothesis
python -m pytest tests/test_reporting.py -q
```

The suite covers P&L/return math (including the `starting_equity == 0` → 0.00%
edge case), best/worst extraction, hour-bucket and confidence-band win rates,
the improvement-suggestion generator, and rollover detection (day-only,
day+week, day+week+month, and no double-emit), plus property-based invariants.

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
