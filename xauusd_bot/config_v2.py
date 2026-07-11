"""
XAUUSD Signal Bot Configuration V2
Win Rate Maximized — Reversal at Range Boundaries

Philosophy:
    - ONE trade type: Reversals at range boundaries
    - 1:1 Risk Reward (fixed, no trailing, set and forget)
    - Win rate > 70% = low variance = consistent
    - Condition identification IS the edge
    - Higher position sizing (2-3%) because low drawdown from high WR
"""

# ============================================================
# INSTRUMENT
# ============================================================
SYMBOL = "XAUUSD"
PIP_SIZE = 0.01
PIP_VALUE_PER_LOT = 1.0
CONTRACT_SIZE = 100

# ============================================================
# STRATEGY — Reversal at Range Boundaries
# ============================================================
STRATEGY = {
    # Range identification (THE most important part)
    "adx_range_max": 20,           # ADX must be below this = ranging
    "bb_squeeze_threshold": 0.8,    # BB width below 80% of avg = squeeze
    "min_range_bars": 10,           # Range must exist for 10+ bars

    # Entry conditions
    "rsi_oversold": 30,            # RSI below = oversold (BUY zone)
    "rsi_overbought": 70,          # RSI above = overbought (SELL zone)
    "boundary_proximity_pct": 0.15, # Within 15% of BB range from boundary

    # Trade parameters — FIXED 1:1, no management
    "risk_reward": 1.0,            # FIXED 1:1 — the key to high win rate
    "sl_buffer_pips": 5,           # SL placed 5 pips beyond boundary
    "max_sl_pips": 120,            # Max stop loss size (tighter)
    "min_sl_pips": 30,             # Min stop loss size (avoid noise)

    # Confluence requirements
    "min_confluences": 3,          # Need 3 of 6 confluences to trade
}

# ============================================================
# RISK MANAGEMENT — Aggressive because high WR = low DD
# ============================================================
RISK_V2 = {
    "risk_per_trade": 0.02,        # 2% per trade (safe with 70%+ WR)
    "max_daily_loss": 0.04,        # 4% daily max (prop firm standard)
    "max_daily_trades": 3,         # Quality > quantity
    "max_concurrent": 1,           # One trade at a time
    "max_spread_pips": 25,         # Don't trade if spread too wide

    # NO trailing stop. NO management. SET AND FORGET.
    "trailing_stop": False,
    "move_to_breakeven": False,
    "partial_close": False,
}

# ============================================================
# SESSION WINDOWS (UTC) — When to look for setups
# ============================================================
SESSIONS_V2 = {
    # Asian session = PRIME TIME for range reversals
    # Gold ranges most during Asian hours
    "asian": {
        "start": "00:00", "end": "07:00",
        "active": True,
        "note": "Best ranging conditions for gold",
    },
    # London open often breaks the Asian range — wait for NEW range to form
    "london_early": {
        "start": "07:00", "end": "09:00",
        "active": False,  # Avoid — range breaking period
        "note": "Range break zone, avoid",
    },
    # London mid — new structure forms, look for reversals
    "london_mid": {
        "start": "09:00", "end": "12:00",
        "active": True,
        "note": "New ranges form after London breakout",
    },
    # Overlap — high vol, still tradeable if ranging
    "overlap": {
        "start": "12:00", "end": "14:00",
        "active": True,
        "note": "Only if still ranging",
    },
    # NY afternoon + dead zone — avoid
    "ny_afternoon": {
        "start": "14:00", "end": "21:00",
        "active": False,
        "note": "Low quality, news risk",
    },
    "dead_zone": {
        "start": "21:00", "end": "00:00",
        "active": False,
        "note": "No liquidity",
    },
}

# ============================================================
# INDICATORS (same computation, different usage)
# ============================================================
INDICATORS_V2 = {
    "ema_fast": 20,
    "ema_slow": 50,
    "ema_bias": 200,
    "adx_period": 14,
    "atr_period": 14,
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,
}

# ============================================================
# NEWS FILTER (same as before)
# ============================================================
NEWS_V2 = {
    "blackout_minutes_before": 30,  # Wider buffer — we're patient
    "blackout_minutes_after": 30,
    "high_impact_events": [
        "NFP", "CPI", "FOMC", "PCE", "GDP",
        "UNEMPLOYMENT", "PPI", "RETAIL_SALES", "FED_SPEECH",
    ],
}

# ============================================================
# COOLDOWN — Longer because we want QUALITY not quantity
# ============================================================
COOLDOWN_V2 = {
    "after_loss_minutes": 60,       # Full hour after loss (avoid revenge)
    "after_win_minutes": 30,        # 30 min after win (avoid overtrading)
    "max_trades_per_session": 2,    # Max 2 per session
    "max_trades_per_day": 3,        # Max 3 per day (quality > quantity)
}
