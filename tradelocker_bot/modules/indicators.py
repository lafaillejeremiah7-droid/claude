"""
Technical Indicator Calculations Module

Computes all indicators needed for the strategy:
- EMA (20, 50, 200)
- RSI (14)
- ATR (14)
- VWAP
- Volume moving average
- Swing highs/lows for liquidity sweep detection
"""
import numpy as np
import pandas as pd
from typing import Optional

from config import (
    EMA_4H_PERIOD,
    EMA_4H_SLOPE_LOOKBACK,
    EMA_30M_FAST,
    EMA_30M_SLOW,
    EMA_5M_PULLBACK,
    RSI_PERIOD,
    ATR_PERIOD,
    VOLUME_AVG_PERIOD,
    SWING_LOOKBACK,
)


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        series: Price series (typically close prices)
        period: RSI lookback period (default 14)

    Returns:
        RSI values as a pandas Series
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """
    Calculate Average True Range.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)

    Returns:
        ATR values as a pandas Series
    """
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    return atr


def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Calculate Volume Weighted Average Price.
    Resets daily (uses date grouping).

    Args:
        df: DataFrame with 'high', 'low', 'close', 'volume', 'timestamp' columns

    Returns:
        VWAP values as a pandas Series
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, np.nan).fillna(1)

    # If we have timestamps, reset VWAP daily
    if "timestamp" in df.columns:
        df_calc = pd.DataFrame({
            "tp": typical_price,
            "vol": vol,
            "tp_vol": typical_price * vol,
            "date": df["timestamp"].dt.date,
        })
        cumulative_tp_vol = df_calc.groupby("date")["tp_vol"].cumsum()
        cumulative_vol = df_calc.groupby("date")["vol"].cumsum()
        vwap = cumulative_tp_vol / cumulative_vol
    else:
        # Simple cumulative VWAP without daily reset
        cumulative_tp_vol = (typical_price * vol).cumsum()
        cumulative_vol = vol.cumsum()
        vwap = cumulative_tp_vol / cumulative_vol

    return vwap


def calculate_volume_avg(volume: pd.Series, period: int = VOLUME_AVG_PERIOD) -> pd.Series:
    """Calculate simple moving average of volume."""
    return volume.rolling(window=period).mean()


def calculate_ema_slope(ema_series: pd.Series, lookback: int = EMA_4H_SLOPE_LOOKBACK) -> pd.Series:
    """
    Calculate the slope direction of an EMA.
    Returns positive values for upward slope, negative for downward.

    Args:
        ema_series: EMA values
        lookback: Number of bars to measure slope over

    Returns:
        Slope values (positive = up, negative = down)
    """
    return ema_series.diff(lookback)


def find_swing_highs(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.Series:
    """
    Identify swing highs in price data.
    A swing high is a high that is higher than 'lookback' bars on each side.

    Returns:
        Series with swing high prices (NaN where no swing high)
    """
    highs = df["high"]
    swing_highs = pd.Series(np.nan, index=df.index)

    for i in range(lookback, len(df) - lookback):
        window_left = highs.iloc[i - lookback:i]
        window_right = highs.iloc[i + 1:i + lookback + 1]
        if highs.iloc[i] >= window_left.max() and highs.iloc[i] >= window_right.max():
            swing_highs.iloc[i] = highs.iloc[i]

    return swing_highs


def find_swing_lows(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> pd.Series:
    """
    Identify swing lows in price data.
    A swing low is a low that is lower than 'lookback' bars on each side.

    Returns:
        Series with swing low prices (NaN where no swing low)
    """
    lows = df["low"]
    swing_lows = pd.Series(np.nan, index=df.index)

    for i in range(lookback, len(df) - lookback):
        window_left = lows.iloc[i - lookback:i]
        window_right = lows.iloc[i + 1:i + lookback + 1]
        if lows.iloc[i] <= window_left.min() and lows.iloc[i] <= window_right.min():
            swing_lows.iloc[i] = lows.iloc[i]

    return swing_lows


def get_recent_swing_high(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> Optional[float]:
    """Get the most recent swing high price."""
    swing_highs = find_swing_highs(df, lookback)
    valid = swing_highs.dropna()
    if len(valid) > 0:
        return valid.iloc[-1]
    return None


def get_recent_swing_low(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> Optional[float]:
    """Get the most recent swing low price."""
    swing_lows = find_swing_lows(df, lookback)
    valid = swing_lows.dropna()
    if len(valid) > 0:
        return valid.iloc[-1]
    return None


def add_all_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Add all relevant indicators to a DataFrame based on timeframe.

    Args:
        df: OHLCV DataFrame
        timeframe: One of '4h', '30m', '5m'

    Returns:
        DataFrame with indicator columns added
    """
    df = df.copy()

    if timeframe == "4h":
        df["ema_50"] = calculate_ema(df["close"], EMA_4H_PERIOD)
        df["ema_50_slope"] = calculate_ema_slope(df["ema_50"], EMA_4H_SLOPE_LOOKBACK)

    elif timeframe == "30m":
        df["ema_50"] = calculate_ema(df["close"], EMA_30M_FAST)
        df["ema_200"] = calculate_ema(df["close"], EMA_30M_SLOW)

    elif timeframe == "5m":
        df["ema_20"] = calculate_ema(df["close"], EMA_5M_PULLBACK)
        df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
        df["atr"] = calculate_atr(df, ATR_PERIOD)
        df["vwap"] = calculate_vwap(df)
        df["volume_avg"] = calculate_volume_avg(df["volume"], VOLUME_AVG_PERIOD)
        df["swing_highs"] = find_swing_highs(df, SWING_LOOKBACK)
        df["swing_lows"] = find_swing_lows(df, SWING_LOOKBACK)

    return df


def identify_candlestick_pattern(df: pd.DataFrame, index: int = -1) -> Optional[str]:
    """
    Identify candlestick patterns at the given index.

    Patterns detected:
    - bullish_engulfing
    - bearish_engulfing
    - hammer
    - shooting_star
    - bullish_rejection (strong body, long lower wick)
    - bearish_rejection (strong body, long upper wick)

    Args:
        df: OHLCV DataFrame
        index: Bar index to check (default: last bar)

    Returns:
        Pattern name or None
    """
    if len(df) < 2:
        return None

    idx = index if index >= 0 else len(df) + index

    if idx < 1 or idx >= len(df):
        return None

    curr_open = df["open"].iloc[idx]
    curr_close = df["close"].iloc[idx]
    curr_high = df["high"].iloc[idx]
    curr_low = df["low"].iloc[idx]

    prev_open = df["open"].iloc[idx - 1]
    prev_close = df["close"].iloc[idx - 1]

    body = abs(curr_close - curr_open)
    candle_range = curr_high - curr_low

    if candle_range == 0:
        return None

    body_ratio = body / candle_range
    upper_wick = curr_high - max(curr_open, curr_close)
    lower_wick = min(curr_open, curr_close) - curr_low

    # Bullish Engulfing: current bullish candle engulfs previous bearish candle
    if (curr_close > curr_open and prev_close < prev_open and
            curr_open <= prev_close and curr_close >= prev_open):
        return "bullish_engulfing"

    # Bearish Engulfing: current bearish candle engulfs previous bullish candle
    if (curr_close < curr_open and prev_close > prev_open and
            curr_open >= prev_close and curr_close <= prev_open):
        return "bearish_engulfing"

    # Hammer: small body at top, long lower wick (bullish reversal)
    if (body_ratio < 0.35 and lower_wick >= body * 2 and
            upper_wick < body * 0.5 and curr_close >= curr_open):
        return "hammer"

    # Shooting Star: small body at bottom, long upper wick (bearish reversal)
    if (body_ratio < 0.35 and upper_wick >= body * 2 and
            lower_wick < body * 0.5 and curr_close <= curr_open):
        return "shooting_star"

    # Bullish Rejection: strong bullish body with significant lower wick
    if (curr_close > curr_open and body_ratio >= 0.5 and
            lower_wick >= body * 0.7):
        return "bullish_rejection"

    # Bearish Rejection: strong bearish body with significant upper wick
    if (curr_close < curr_open and body_ratio >= 0.5 and
            upper_wick >= body * 0.7):
        return "bearish_rejection"

    # Strong bullish candle (large body)
    if curr_close > curr_open and body_ratio >= 0.65:
        return "strong_bullish"

    # Strong bearish candle (large body)
    if curr_close < curr_open and body_ratio >= 0.65:
        return "strong_bearish"

    return None
