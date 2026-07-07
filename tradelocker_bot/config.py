"""
Configuration for TradeLocker Trading Bot
"""
import os
from dotenv import load_dotenv

load_dotenv()

# === TradeLocker API ===
TL_EMAIL = os.getenv("TL_EMAIL", "")
TL_PASSWORD = os.getenv("TL_PASSWORD", "")
TL_SERVER = os.getenv("TL_SERVER", "AQUA")
TL_ENVIRONMENT = os.getenv("TL_ENVIRONMENT", "live")

# Base URLs
BASE_URLS = {
    "live": "https://live.tradelocker.com/backend-api",
    "demo": "https://demo.tradelocker.com/backend-api",
}
BASE_URL = BASE_URLS.get(TL_ENVIRONMENT, BASE_URLS["live"])

# === Instruments ===
INSTRUMENTS = [s.strip() for s in os.getenv("INSTRUMENTS", "BTCUSD,XAUUSD").split(",")]

# === Timeframes ===
# TradeLocker API timeframe codes
TIMEFRAMES = {
    "5m": 300,
    "30m": 1800,
    "4h": 14400,
}

# === Indicator Settings ===
# 4H Trend
EMA_4H_PERIOD = 50
EMA_4H_SLOPE_LOOKBACK = 3  # Number of bars to measure slope

# 30M Trend Confirmation
EMA_30M_FAST = 50
EMA_30M_SLOW = 200

# 5M Entry
EMA_5M_PULLBACK = 20
RSI_PERIOD = 14
RSI_LONG_MIN = 45
RSI_LONG_MAX = 60
RSI_SHORT_MIN = 40
RSI_SHORT_MAX = 55
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_AVG_PERIOD = 20
ATR_PERIOD = 14

# Liquidity Sweep
SWING_LOOKBACK = 10  # Bars to identify swing highs/lows

# === Risk Management ===
# RISK_PERCENT is kept as a fallback (used when no confidence score is provided,
# e.g. legacy/backward-compatible sizing paths).
RISK_PERCENT = float(os.getenv("RISK_PERCENT", "2.0"))
# Confidence-scaled position sizing bounds. Risk % scales linearly with the
# adaptive confidence score between the gate (8.0) and the max (10.0):
#   conf 8.0 -> MIN_RISK_PERCENT, conf 9.0 -> midpoint, conf 10.0 -> MAX_RISK_PERCENT
MIN_RISK_PERCENT = float(os.getenv("MIN_RISK_PERCENT", "1.0"))
MAX_RISK_PERCENT = float(os.getenv("MAX_RISK_PERCENT", "3.0"))
MAX_TRADES_PER_DAY = int(os.getenv("MAX_TRADES_PER_DAY", "2"))
DAILY_DRAWDOWN_LIMIT = float(os.getenv("DAILY_DRAWDOWN_LIMIT", "4.0"))
WEEKLY_DRAWDOWN_LIMIT = float(os.getenv("WEEKLY_DRAWDOWN_LIMIT", "4.0"))
MIN_RR_RATIO = 1.5  # Minimum risk:reward
PREFERRED_RR_RATIO = 2.0  # Preferred risk:reward
BREAKEVEN_TRIGGER = 1.0  # Move SL to breakeven at 1R profit

# === Session Times (UTC) ===
SESSIONS = {
    "BTCUSD": {
        "london_open": 7,
        "london_close": 16,
        "ny_open": 12,
        "ny_close": 21,
    },
    "XAUUSD": {
        "london_open": 7,
        "london_close": 16,
        "ny_open": 12,
        "ny_close": 21,
    },
}

# News avoidance buffer (minutes)
NEWS_BUFFER_MINUTES = 30

# === Paper Trading (--dry mode) ===
# Starting equity used by the paper-trading engine. Paper current_equity is
# computed as PAPER_STARTING_EQUITY + realized paper PnL, and paper positions
# are sized off that paper equity (never the live account balance).
PAPER_STARTING_EQUITY = float(os.getenv("PAPER_STARTING_EQUITY", "10000.0"))

# === Bot Settings ===
SCAN_INTERVAL_SECONDS = 60  # How often to check for signals (1 minute)
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
