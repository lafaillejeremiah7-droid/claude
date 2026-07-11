"""
XAUUSD Trading Bot Configuration
All parameters tuned specifically for Gold (XAU/USD) characteristics.
"""

# ============================================================
# INSTRUMENT SETTINGS
# ============================================================
SYMBOL = "XAUUSD"
PIP_SIZE = 0.01  # 1 pip = $0.01 movement in gold
PIP_VALUE_PER_LOT = 1.0  # $1.00 per pip per standard lot (100oz)
CONTRACT_SIZE = 100  # 1 standard lot = 100 troy ounces

# ============================================================
# SESSION WINDOWS (UTC)
# ============================================================
SESSIONS = {
    "asian": {"start": "00:00", "end": "07:00", "mode": "RANGE"},
    "london": {"start": "07:00", "end": "12:00", "mode": "TREND"},
    "overlap": {"start": "12:00", "end": "16:00", "mode": "TREND_AGGRESSIVE"},
    "new_york": {"start": "16:00", "end": "21:00", "mode": "TREND"},
    "dead_zone": {"start": "21:00", "end": "00:00", "mode": "IDLE"},
}

# ============================================================
# STATE MACHINE STATES
# ============================================================
STATES = [
    "IDLE",
    "SESSION_ANALYSIS",
    "RANGE_MODE",
    "TREND_MODE",
    "NEWS_AVOID",
    "ENTRY_SEARCH",
    "POSITION_ACTIVE",
    "COOLDOWN",
]

# ============================================================
# INDICATOR PARAMETERS (Tuned for XAUUSD)
# ============================================================
INDICATORS = {
    # EMAs for trend direction
    "ema_fast": 20,
    "ema_slow": 50,
    "ema_bias": 200,

    # ADX for trend strength
    "adx_period": 14,
    "adx_trend_threshold": 25,  # Above = trending
    "adx_range_threshold": 20,  # Below = ranging

    # ATR for volatility measurement
    "atr_period": 14,
    "atr_sl_multiplier": 1.5,   # SL = 1.5x ATR
    "atr_tp_multiplier": 3.0,   # TP = 3.0x ATR (1:2 RR)
    "atr_expansion_threshold": 1.3,  # ATR must be 1.3x its own 20-period average

    # RSI for mean-reversion
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,

    # Bollinger Bands for range identification
    "bb_period": 20,
    "bb_std": 2.0,
    "bb_squeeze_threshold": 0.5,  # Squeeze when bandwidth < 50% of average

    # Stochastic for range mode entries
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
    "stoch_overbought": 80,
    "stoch_oversold": 20,
}

# ============================================================
# MULTI-FACTOR SCORING WEIGHTS
# ============================================================
TREND_FACTORS = {
    "ema_alignment": 0.25,      # EMA 20 > 50 > 200 (or inverse)
    "adx_momentum": 0.25,       # ADX > 25 with DI+/DI- confirmation
    "atr_expansion": 0.20,      # Volatility expanding
    "dxy_confluence": 0.15,     # DXY moving inversely
    "structure_break": 0.15,    # Break of session high/low
}

RANGE_FACTORS = {
    "range_identified": 0.30,   # ADX < 20, BB squeeze
    "sr_touch": 0.30,           # Price at range boundary
    "reversal_signal": 0.20,    # RSI divergence or candle pattern
    "stoch_confirmation": 0.20, # Stochastic at extreme
}

# Minimum score to trigger entry (0.0 - 1.0)
ENTRY_THRESHOLD = 0.70

# ============================================================
# RISK MANAGEMENT
# ============================================================
RISK = {
    "max_risk_per_trade": 0.01,     # 1% of account per trade
    "max_daily_loss": 0.03,         # 3% daily loss → shutdown
    "max_concurrent_positions": 1,   # Only 1 gold position at a time
    "min_risk_reward": 2.0,         # Minimum 1:2 RR
    "trailing_stop_activation": 1.0, # Activate trailing after 1:1 RR achieved
    "trailing_stop_distance_atr": 1.0,  # Trail by 1x ATR
    "max_spread_pips": 30,          # Don't trade if spread > 30 pips ($0.30)
    "slippage_buffer_pips": 5,      # Account for 5 pips slippage
}

# ============================================================
# NEWS FILTER
# ============================================================
NEWS = {
    "blackout_minutes_before": 15,  # No new trades 15min before
    "blackout_minutes_after": 15,   # No new trades 15min after
    "high_impact_events": [
        "NFP",          # Non-Farm Payrolls
        "CPI",          # Consumer Price Index
        "FOMC",         # Federal Reserve Decision
        "PCE",          # Personal Consumption Expenditures
        "GDP",          # Gross Domestic Product
        "UNEMPLOYMENT", # Unemployment Claims
        "PPI",          # Producer Price Index
        "RETAIL_SALES", # Retail Sales
        "FED_SPEECH",   # Fed Chair Speech
    ],
}

# ============================================================
# DXY CORRELATION FILTER
# ============================================================
DXY = {
    "symbol": "DXY",            # Dollar Index
    "correlation_period": 20,    # Look-back for correlation calc
    "min_inverse_strength": -0.5,  # Minimum inverse correlation to confirm
    "divergence_threshold": 0.3,   # Flag divergence if correlation breaks
}

# ============================================================
# TIMEFRAMES
# ============================================================
TIMEFRAMES = {
    "entry": "M15",     # 15-minute for entry signals
    "trend": "H1",      # 1-hour for trend direction
    "bias": "H4",       # 4-hour for daily bias
}

# ============================================================
# COOLDOWN SETTINGS
# ============================================================
COOLDOWN = {
    "after_loss_minutes": 30,       # Wait 30min after a loss
    "after_win_minutes": 15,        # Wait 15min after a win
    "after_daily_limit_hours": 24,  # Full day if daily limit hit
    "max_trades_per_session": 3,    # Max 3 trades per session
    "max_trades_per_day": 6,        # Max 6 trades per day
}
