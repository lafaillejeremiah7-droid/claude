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

# ============================================================
# V3 ADAPTIVE SYSTEMS CONFIGURATION
# ============================================================

# System 1: Adaptive R:R Based on Market Condition
ADAPTIVE_RR = {
    "rr_good": 1.35,               # Good condition target R:R
    "rr_okay": 1.0,                # Standard 1:1
    "rr_choppy": 0.75,             # Choppy = quick TP
    "good_condition_threshold": 0.7,  # Score above = GOOD
    "choppy_condition_threshold": 0.35,  # Score below = CHOPPY
}

# System 2: Adaptive Position Size (Kelly Criterion Lite)
ADAPTIVE_SIZE = {
    "base_risk_pct": 0.02,         # Base 2% risk
    "max_risk_pct": 0.03,          # Max 3% on hot streak
    "min_risk_pct": 0.01,          # Min 1% on cold streak
    "streak_scale_up": 0.005,      # Scale up per win streak trade
    "streak_scale_down": 0.005,    # Scale down per loss streak trade
}

# System 3: Tight SL Reality (Partial Losses)
TIGHT_SL = {
    "tight_sl_factor": 0.6,        # Real SL is tighter than max
    "partial_loss_min": 0.3,       # Min loss factor (best case)
    "partial_loss_max": 0.7,       # Max loss factor (worst case)
}

# System 4: Early Exit Based on Probability
EARLY_EXIT = {
    "adx_breakout_threshold": 22,  # ADX above = range breaking (lowered)
    "momentum_shift_threshold": 0.5,
    "min_profit_for_early_exit": 0.25,  # Must be 0.25R+ profit to exit early
    "max_loss_for_early_cut": 0.4,     # Cut at 0.4R loss if conditions bad
}

# System 5: Sentiment Scanner
SENTIMENT = {
    "mode": "backtest",            # "backtest" or "live"
    "signal_decay_minutes": 45,    # How long signals last
    "danger_decay_minutes": 60,    # How long danger signals last
    "bullish_threshold": 0.5,      # Net score above = bullish (raised to reduce noise)
    "bearish_threshold": -0.5,     # Net score below = bearish
    "danger_threshold": 0.75,      # Danger score above = avoid (raised)
}

# System 6: Influencer Flow (uses same scanner, tracked separately)
INFLUENCER_FLOW = {
    "min_signals_for_confidence": 3,
    "flow_weight": 0.15,           # Weight in final decision
}

# V3 Risk Management (overrides V2 where specified)
RISK_V3 = {
    "risk_per_trade": 0.02,        # Base (adaptive engine adjusts)
    "max_daily_loss": 0.05,        # 5% daily max
    "max_daily_trades": 6,         # 6 trades/day (optimal from backtest)
    "max_concurrent": 1,           # One trade at a time
    "max_spread_pips": 25,         # Don't trade if spread too wide
    "trailing_stop": False,
    "move_to_breakeven": False,
    "partial_close": False,
}

# V3 Session Windows (NY session added)
SESSIONS_V3 = {
    "asian": {
        "start": "00:00", "end": "07:00",
        "active": True,
        "note": "Best ranging conditions for gold",
    },
    "london_early": {
        "start": "07:00", "end": "09:00",
        "active": False,
        "note": "Range break zone, avoid",
    },
    "london_mid": {
        "start": "09:00", "end": "12:00",
        "active": True,
        "note": "New ranges form after London breakout",
    },
    "overlap": {
        "start": "12:00", "end": "14:00",
        "active": True,
        "note": "Only if still ranging",
    },
    "ny_session": {
        "start": "14:00", "end": "18:00",
        "active": True,
        "note": "NY session added - good reversals after initial move",
    },
    "ny_late": {
        "start": "18:00", "end": "21:00",
        "active": False,
        "note": "Low quality late NY",
    },
    "dead_zone": {
        "start": "21:00", "end": "00:00",
        "active": False,
        "note": "No liquidity",
    },
}

# V3 Cooldown (optimized from backtest results)
COOLDOWN_V3 = {
    "after_loss_minutes": 30,       # 30 min after loss (was 60)
    "after_win_minutes": 10,        # 10 min after win (was 30)
    "max_trades_per_session": 3,    # Max 3 per session
    "max_trades_per_day": 6,        # Max 6 per day (optimal)
}
