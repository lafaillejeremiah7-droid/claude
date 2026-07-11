"""
Realistic XAUUSD Data Generator — Models ACTUAL gold behavior.

Key differences from pure random:
- Mean-reverting within ranges (price bounces off boundaries)
- Clear trend/range regime switching
- Session-based volatility that matches real gold
- Institutional-level support/resistance that holds 70%+ of the time
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_realistic_xauusd(start_date: str = "2025-10-01",
                               days: int = 90,
                               base_price: float = 4100.0,
                               seed: int = 42) -> pd.DataFrame:
    """
    Generate XAUUSD data that models real gold behavior:
    - 60% of time in range (mean-reverting)
    - 40% of time trending
    - Range boundaries hold ~70% of the time (like real S/R)
    - Asian session = tight range, London = breakout/new trend
    """
    np.random.seed(seed)

    bars_per_day = 96
    total_bars = days * bars_per_day

    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []

    start = datetime.strptime(start_date, "%Y-%m-%d")
    price = base_price

    # Regime management
    regime = "RANGE"  # "RANGE" or "TREND"
    regime_bars_left = np.random.randint(150, 400)

    # Range parameters (when in RANGE regime)
    range_center = price
    range_width = np.random.uniform(8, 20)  # $8-$20 range width
    range_upper = range_center + range_width / 2
    range_lower = range_center - range_width / 2
    boundary_hold_rate = 0.72  # 72% of time, boundaries hold (realistic)

    # Trend parameters (when in TREND regime)
    trend_direction = np.random.choice([-1, 1])
    trend_speed = np.random.uniform(0.3, 0.8)  # $/bar

    for i in range(total_bars):
        timestamp = start + timedelta(minutes=15 * i)
        hour = timestamp.hour

        # Skip weekends
        if timestamp.weekday() >= 5:
            continue

        # Session volatility
        if 0 <= hour < 7:       # Asian — tight, range-bound
            vol = 0.8
            mean_revert_strength = 0.15  # Strong mean reversion
        elif 7 <= hour < 9:     # London open — breakout risk
            vol = 2.0
            mean_revert_strength = 0.02  # Weak mean reversion (can break)
        elif 9 <= hour < 12:    # London mid
            vol = 1.5
            mean_revert_strength = 0.08
        elif 12 <= hour < 16:   # Overlap
            vol = 2.0
            mean_revert_strength = 0.05
        elif 16 <= hour < 21:   # NY
            vol = 1.2
            mean_revert_strength = 0.06
        else:                   # Dead zone
            vol = 0.3
            mean_revert_strength = 0.1

        # Generate price movement based on regime
        noise = np.random.normal(0, vol)

        if regime == "RANGE":
            # Mean-reverting behavior: price pulled back toward center
            distance_from_center = price - range_center
            mean_revert = -distance_from_center * mean_revert_strength

            # Boundary interaction
            if price >= range_upper:
                # At resistance — 72% chance of rejection
                if np.random.random() < boundary_hold_rate:
                    # Rejection! Push price back down
                    noise = -abs(noise) * 1.5
                else:
                    # Breakout! Switch to trend
                    regime = "TREND"
                    trend_direction = 1
                    trend_speed = np.random.uniform(0.3, 0.7)
                    regime_bars_left = np.random.randint(50, 200)

            elif price <= range_lower:
                # At support — 72% chance of bounce
                if np.random.random() < boundary_hold_rate:
                    # Bounce! Push price back up
                    noise = abs(noise) * 1.5
                else:
                    # Breakdown! Switch to trend
                    regime = "TREND"
                    trend_direction = -1
                    trend_speed = np.random.uniform(0.3, 0.7)
                    regime_bars_left = np.random.randint(50, 200)

            bar_move = noise + mean_revert

        elif regime == "TREND":
            # Trending: drift + noise
            drift = trend_direction * trend_speed
            bar_move = drift + noise * 0.8  # Slightly less noise in trends

        # Apply move
        open_price = price
        close_price = price + bar_move

        # High/Low
        bar_range = abs(bar_move) + np.random.uniform(0.3, 1.5) * vol
        if bar_move >= 0:
            high_price = close_price + np.random.uniform(0, bar_range * 0.25)
            low_price = open_price - np.random.uniform(0, bar_range * 0.25)
        else:
            high_price = open_price + np.random.uniform(0, bar_range * 0.25)
            low_price = close_price - np.random.uniform(0, bar_range * 0.25)

        high_price = max(high_price, open_price, close_price)
        low_price = min(low_price, open_price, close_price)

        timestamps.append(timestamp)
        opens.append(round(open_price, 2))
        highs.append(round(high_price, 2))
        lows.append(round(low_price, 2))
        closes.append(round(close_price, 2))

        price = close_price

        # Regime management
        regime_bars_left -= 1
        if regime_bars_left <= 0:
            if regime == "RANGE":
                # Switch to trend or reset range
                if np.random.random() < 0.4:
                    regime = "TREND"
                    trend_direction = np.random.choice([-1, 1])
                    trend_speed = np.random.uniform(0.2, 0.6)
                    regime_bars_left = np.random.randint(50, 150)
                else:
                    # New range
                    range_center = price
                    range_width = np.random.uniform(8, 18)
                    range_upper = range_center + range_width / 2
                    range_lower = range_center - range_width / 2
                    regime_bars_left = np.random.randint(150, 400)

            elif regime == "TREND":
                # Switch to range
                regime = "RANGE"
                range_center = price
                range_width = np.random.uniform(8, 18)
                range_upper = range_center + range_width / 2
                range_lower = range_center - range_width / 2
                regime_bars_left = np.random.randint(150, 400)
                boundary_hold_rate = np.random.uniform(0.68, 0.78)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
    })
    return df
