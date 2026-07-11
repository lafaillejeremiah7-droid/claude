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
# BTCUSD: Extended to near-24h (6-22 UTC) since crypto trades 24/7.
# We still avoid the low-liquidity dead zone (22-06 UTC) where spreads widen.
# XAUUSD: Restricted to London + NY sessions (traditional forex hours).
SESSIONS = {
    "BTCUSD": {
        "london_open": 6,
        "london_close": 22,
        "ny_open": 12,
        "ny_close": 22,
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

# === API Rate-Limiting / Resilience ===
# Minimum spacing (seconds) enforced between consecutive TradeLocker HTTP
# requests. The DEMO server rate-limits aggressively (HTTP 429), so we space
# calls out to stay under its per-second budget. Env-overridable.
API_MIN_REQUEST_INTERVAL = float(os.getenv("API_MIN_REQUEST_INTERVAL", "0.5"))

# Max number of retry attempts on a transient failure (HTTP 429 / 5xx) before
# giving up and returning empty. Total tries = 1 initial + API_MAX_RETRIES.
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "4"))

# Base backoff (seconds) for exponential backoff: wait = base * 2**attempt,
# plus a little jitter. Honors the Retry-After response header when present.
API_BACKOFF_BASE = float(os.getenv("API_BACKOFF_BASE", "1.0"))
# Upper bound (seconds) on any single backoff sleep so we never stall the loop.
API_BACKOFF_MAX = float(os.getenv("API_BACKOFF_MAX", "30.0"))

# Short-lived in-memory cache TTL (seconds) for price-history bars, keyed by
# (tradableInstrumentId, resolution). Prevents refetching the same bars every
# 60s scan cycle. Keys are the resolution in seconds; values are TTL seconds.
# Env override format (optional): "300:60,1800:300,14400:900".
def _parse_cache_ttl(raw: str) -> dict:
    mapping = {}
    for part in raw.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        res_str, ttl_str = part.split(":", 1)
        try:
            mapping[int(res_str)] = float(ttl_str)
        except (ValueError, TypeError):
            continue
    return mapping


_DEFAULT_HISTORY_CACHE_TTL = {
    300: 60.0,     # 5m bars  -> refresh at most once/minute
    1800: 300.0,   # 30m bars -> refresh at most once/5min
    14400: 900.0,  # 4h bars  -> refresh at most once/15min
}
_env_cache_ttl = os.getenv("HISTORY_CACHE_TTL", "")
HISTORY_CACHE_TTL = _parse_cache_ttl(_env_cache_ttl) if _env_cache_ttl else dict(_DEFAULT_HISTORY_CACHE_TTL)
# Fallback TTL (seconds) used for any resolution not explicitly listed above.
HISTORY_CACHE_TTL_DEFAULT = float(os.getenv("HISTORY_CACHE_TTL_DEFAULT", "60.0"))

# === Trading Filter Overrides (for testing / immediate trading) ===
# Set to "true" to skip the news-event block filter (useful for testing)
SKIP_NEWS_FILTER = os.getenv("SKIP_NEWS_FILTER", "false").lower() in ("true", "1", "yes")
# Override avoid_hours (empty = trade all hours, comma-separated UTC hours to skip).
# If AVOID_HOURS env var is explicitly set (even to empty), it overrides the
# adaptive engine's learned avoid_hours. If not set at all, the adaptive engine's
# value from adaptive_config.json stands.
AVOID_HOURS_STR = os.getenv("AVOID_HOURS", "")

# === Graduated Conviction ===
# When True, allows trades when 4H has direction but 30M is neutral (not opposing).
# Confidence is capped at 8.5 for these trades (smaller position size).
# When False, requires full 4H+30M alignment (original strict behavior).
ALLOW_PARTIAL_ALIGNMENT = os.getenv("ALLOW_PARTIAL_ALIGNMENT", "true").lower() in ("true", "1", "yes")

# === Performance Reporting ===
# Directory (relative to the bot root) where machine-readable performance
# reports are written. The reporter only ever WRITES inside this directory;
# it never overwrites daily_stats.json / journal / adaptive_config.json.
REPORTS_DIR = os.getenv("REPORTS_DIR", "logs/reports")

# Minimum number of trades in a bucket (hour / pattern / confidence band)
# before the weekly "what to improve" engine will surface a suggestion.
# Keeps insights statistically meaningful instead of reacting to noise.
REPORT_MIN_SAMPLE = int(os.getenv("REPORT_MIN_SAMPLE", "5"))

# Win-rate threshold (fraction) below which an hour/pattern is flagged as
# under-performing in the weekly improvement section.
REPORT_WEAK_WIN_RATE = float(os.getenv("REPORT_WEAK_WIN_RATE", "0.40"))
