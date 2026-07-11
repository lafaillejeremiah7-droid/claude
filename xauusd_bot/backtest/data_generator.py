"""
Synthetic XAUUSD Data Generator for Backtesting.

Generates realistic M15 price data that mimics gold's behavior:
- Session-based volatility (high London/NY, low Asian)
- Trending periods followed by ranges
- News spikes
- Realistic ATR (150-300 pips daily range)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_xauusd_data(start_date: str = "2025-01-01",
                          days: int = 90,
                          base_price: float = 4100.0,
                          seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic XAUUSD M15 data.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    """
    np.random.seed(seed)

    bars_per_day = 96  # 24 hours * 4 bars per hour (M15)
    total_bars = days * bars_per_day

    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    start = datetime.strptime(start_date, "%Y-%m-%d")
    price = base_price

    # Trend phases (simulate multi-week trends)
    trend_direction = 1  # 1 = bull, -1 = bear
    trend_bars_remaining = np.random.randint(200, 600)
    trend_strength = np.random.uniform(0.02, 0.08)

    for i in range(total_bars):
        timestamp = start + timedelta(minutes=15 * i)
        hour = timestamp.hour

        # Skip weekends
        if timestamp.weekday() >= 5:
            continue

        # Session-based volatility multiplier
        if 0 <= hour < 7:       # Asian
            vol_mult = 0.3
            session_bias = 0.0  # Range-bound — very tight
        elif 7 <= hour < 12:    # London
            vol_mult = 1.2
            session_bias = trend_direction * trend_strength * 0.5
        elif 12 <= hour < 16:   # Overlap
            vol_mult = 1.5
            session_bias = trend_direction * trend_strength * 0.8
        elif 16 <= hour < 21:   # New York
            vol_mult = 1.0
            session_bias = trend_direction * trend_strength * 0.3
        else:                   # Dead zone
            vol_mult = 0.2
            session_bias = 0.0

        # Base volatility for XAUUSD (scaled for M15)
        base_vol = 2.5  # ~$2.50 per bar avg movement

        # Generate bar
        noise = np.random.normal(0, base_vol * vol_mult)
        bar_move = noise + session_bias

        # Occasional news spikes (every ~500 bars)
        if np.random.random() < 0.002:
            spike = np.random.choice([-1, 1]) * np.random.uniform(15, 40)
            bar_move += spike

        open_price = price
        close_price = price + bar_move

        # High and low
        bar_range = abs(bar_move) + np.random.uniform(0.5, 3.0) * vol_mult
        if bar_move >= 0:
            high_price = close_price + np.random.uniform(0, bar_range * 0.3)
            low_price = open_price - np.random.uniform(0, bar_range * 0.3)
        else:
            high_price = open_price + np.random.uniform(0, bar_range * 0.3)
            low_price = close_price - np.random.uniform(0, bar_range * 0.3)

        # Ensure OHLC integrity
        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        # Volume (higher during liquid sessions)
        volume = int(np.random.uniform(500, 3000) * vol_mult)

        timestamps.append(timestamp)
        opens.append(round(open_price, 2))
        highs.append(round(high_price, 2))
        lows.append(round(low_price, 2))
        closes.append(round(close_price, 2))
        volumes.append(volume)

        price = close_price

        # Manage trend phases
        trend_bars_remaining -= 1
        if trend_bars_remaining <= 0:
            # Flip or continue with new parameters
            if np.random.random() < 0.6:
                trend_direction *= -1  # Reverse
            trend_strength = np.random.uniform(0.01, 0.06)
            trend_bars_remaining = np.random.randint(150, 500)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    return df


def generate_dxy_data(gold_df: pd.DataFrame, seed: int = 123) -> np.ndarray:
    """
    Generate synthetic DXY data inversely correlated with gold.
    """
    np.random.seed(seed)
    n = len(gold_df)

    # Start DXY around 104
    dxy = np.zeros(n)
    dxy[0] = 104.0

    gold_returns = gold_df["close"].pct_change().fillna(0).values

    for i in range(1, n):
        # Inverse relationship with noise
        inverse_component = -gold_returns[i] * 0.3  # Scaled inverse
        noise = np.random.normal(0, 0.0003)
        dxy_return = inverse_component + noise
        dxy[i] = dxy[i - 1] * (1 + dxy_return)

    return dxy
