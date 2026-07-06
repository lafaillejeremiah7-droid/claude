"""
Entry Signal Detection Module

Implements the 5-minute chart entry logic:
1. Pullback to value (20 EMA or VWAP)
2. RSI in appropriate zone (45-60 for longs, 40-55 for shorts)
3. Liquidity sweep detection (price sweeps swing high/low then reverses)
4. Market structure break (close above recent lower high / below recent higher low)
5. Candlestick pattern confirmation (engulfing, hammer, shooting star, etc.)
6. Volume above 20-period average

All confirmations must be present for a valid entry signal.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from modules.indicators import (
    add_all_indicators,
    identify_candlestick_pattern,
    get_recent_swing_high,
    get_recent_swing_low,
)
from modules.trend_analysis import TrendDirection
from config import (
    RSI_LONG_MIN,
    RSI_LONG_MAX,
    RSI_SHORT_MIN,
    RSI_SHORT_MAX,
    RSI_OVERBOUGHT,
    RSI_OVERSOLD,
    SWING_LOOKBACK,
)

logger = logging.getLogger(__name__)


@dataclass
class EntrySignal:
    """Complete entry signal with all confirmation details."""
    valid: bool = False
    direction: Optional[str] = None  # 'buy' or 'sell'
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    confidence_score: float = 0.0

    # Individual confirmations
    pullback_confirmed: bool = False
    rsi_confirmed: bool = False
    liquidity_sweep_detected: bool = False
    structure_break_confirmed: bool = False
    candle_pattern_confirmed: bool = False
    volume_confirmed: bool = False

    # Metadata
    candle_pattern: Optional[str] = None
    rsi_value: float = 0.0
    volume_ratio: float = 0.0
    sweep_level: float = 0.0
    structure_break_level: float = 0.0
    reasons: list = field(default_factory=list)
    rejections: list = field(default_factory=list)

    @property
    def confirmation_count(self) -> int:
        """Count how many confirmations are met."""
        confirmations = [
            self.pullback_confirmed,
            self.rsi_confirmed,
            self.liquidity_sweep_detected,
            self.structure_break_confirmed,
            self.candle_pattern_confirmed,
            self.volume_confirmed,
        ]
        return sum(confirmations)

    @property
    def all_confirmed(self) -> bool:
        """All confirmations must be met for highest-probability entry."""
        return all([
            self.pullback_confirmed,
            self.rsi_confirmed,
            self.liquidity_sweep_detected,
            self.structure_break_confirmed,
            self.candle_pattern_confirmed,
            self.volume_confirmed,
        ])


def check_pullback_to_value(
    df: pd.DataFrame, trend_direction: TrendDirection
) -> tuple[bool, list]:
    """
    Check if price has pulled back to the 20 EMA or VWAP zone.

    For longs: price should be near/touching EMA20 or VWAP from above
    For shorts: price should be near/touching EMA20 or VWAP from below

    The pullback is valid when the low (for longs) touches or comes within
    0.1% of the EMA/VWAP, or the high (for shorts) touches or comes within 0.1%.

    Args:
        df: 5M DataFrame with ema_20 and vwap columns
        trend_direction: Current confirmed trend direction

    Returns:
        Tuple of (confirmed, reasons list)
    """
    if len(df) < 3:
        return False, ["Insufficient data"]

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    ema_20 = latest["ema_20"]
    vwap = latest["vwap"]
    close = latest["close"]
    low = latest["low"]
    high = latest["high"]

    reasons = []
    threshold_pct = 0.002  # 0.2% proximity threshold

    if trend_direction == TrendDirection.BULLISH:
        # For longs: price should pull back DOWN to EMA20/VWAP
        ema_proximity = abs(low - ema_20) / ema_20 if ema_20 > 0 else 1
        vwap_proximity = abs(low - vwap) / vwap if vwap > 0 else 1

        # Check if low touched or came close to EMA20
        near_ema = ema_proximity <= threshold_pct or low <= ema_20
        # Check if low touched or came close to VWAP
        near_vwap = vwap_proximity <= threshold_pct or low <= vwap

        # Also check if close is still above (bouncing off support)
        bouncing = close > ema_20 or close > vwap

        if (near_ema or near_vwap) and bouncing:
            if near_ema:
                reasons.append(f"Pullback to EMA20 ({ema_20:.2f}), low={low:.2f}")
            if near_vwap:
                reasons.append(f"Pullback to VWAP ({vwap:.2f}), low={low:.2f}")
            return True, reasons

        # Alternative: check last 3 candles for recent pullback
        for i in range(-3, 0):
            if abs(i) <= len(df):
                bar = df.iloc[i]
                if bar["low"] <= ema_20 * (1 + threshold_pct) and bar["close"] > ema_20:
                    reasons.append(f"Recent pullback to EMA20 ({i} bars ago)")
                    return True, reasons
                if bar["low"] <= vwap * (1 + threshold_pct) and bar["close"] > vwap:
                    reasons.append(f"Recent pullback to VWAP ({i} bars ago)")
                    return True, reasons

    elif trend_direction == TrendDirection.BEARISH:
        # For shorts: price should pull back UP to EMA20/VWAP
        ema_proximity = abs(high - ema_20) / ema_20 if ema_20 > 0 else 1
        vwap_proximity = abs(high - vwap) / vwap if vwap > 0 else 1

        near_ema = ema_proximity <= threshold_pct or high >= ema_20
        near_vwap = vwap_proximity <= threshold_pct or high >= vwap

        bouncing = close < ema_20 or close < vwap

        if (near_ema or near_vwap) and bouncing:
            if near_ema:
                reasons.append(f"Pullback to EMA20 ({ema_20:.2f}), high={high:.2f}")
            if near_vwap:
                reasons.append(f"Pullback to VWAP ({vwap:.2f}), high={high:.2f}")
            return True, reasons

        for i in range(-3, 0):
            if abs(i) <= len(df):
                bar = df.iloc[i]
                if bar["high"] >= ema_20 * (1 - threshold_pct) and bar["close"] < ema_20:
                    reasons.append(f"Recent pullback to EMA20 ({i} bars ago)")
                    return True, reasons
                if bar["high"] >= vwap * (1 - threshold_pct) and bar["close"] < vwap:
                    reasons.append(f"Recent pullback to VWAP ({i} bars ago)")
                    return True, reasons

    return False, ["No pullback to value zone detected"]


def check_rsi_zone(df: pd.DataFrame, trend_direction: TrendDirection) -> tuple[bool, float, list]:
    """
    Check if RSI is in the appropriate continuation zone.

    Longs: RSI between 45-60 (not overbought, showing strength)
    Shorts: RSI between 40-55 (not oversold, showing weakness)
    Avoid: RSI > 70 (overbought) or RSI < 30 (oversold)

    Returns:
        Tuple of (confirmed, rsi_value, reasons)
    """
    if "rsi" not in df.columns or len(df) < 1:
        return False, 0, ["RSI not available"]

    rsi = df["rsi"].iloc[-1]
    reasons = []

    # Absolute rejection zones
    if rsi >= RSI_OVERBOUGHT:
        return False, rsi, [f"RSI overbought ({rsi:.1f} >= {RSI_OVERBOUGHT}) - exhaustion risk"]
    if rsi <= RSI_OVERSOLD:
        return False, rsi, [f"RSI oversold ({rsi:.1f} <= {RSI_OVERSOLD}) - exhaustion risk"]

    if trend_direction == TrendDirection.BULLISH:
        if RSI_LONG_MIN <= rsi <= RSI_LONG_MAX:
            reasons.append(f"RSI in bullish zone ({rsi:.1f}, range {RSI_LONG_MIN}-{RSI_LONG_MAX})")
            return True, rsi, reasons
        else:
            return False, rsi, [f"RSI outside bullish zone ({rsi:.1f}, need {RSI_LONG_MIN}-{RSI_LONG_MAX})"]

    elif trend_direction == TrendDirection.BEARISH:
        if RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX:
            reasons.append(f"RSI in bearish zone ({rsi:.1f}, range {RSI_SHORT_MIN}-{RSI_SHORT_MAX})")
            return True, rsi, reasons
        else:
            return False, rsi, [f"RSI outside bearish zone ({rsi:.1f}, need {RSI_SHORT_MIN}-{RSI_SHORT_MAX})"]

    return False, rsi, ["No valid trend direction for RSI check"]


def detect_liquidity_sweep(
    df: pd.DataFrame, trend_direction: TrendDirection, lookback: int = SWING_LOOKBACK
) -> tuple[bool, float, list]:
    """
    Detect a liquidity sweep pattern.

    A liquidity sweep occurs when price briefly moves beyond a recent
    swing high (for shorts) or swing low (for longs) to trigger stop
    losses, then reverses back into the prevailing trend.

    For BULLISH:
    - Price dips below recent swing low (sweeps sell-side liquidity)
    - Then closes back above the swing low (reversal)

    For BEARISH:
    - Price spikes above recent swing high (sweeps buy-side liquidity)
    - Then closes back below the swing high (reversal)

    Args:
        df: 5M DataFrame with swing_highs and swing_lows columns
        trend_direction: Confirmed trend direction
        lookback: Number of recent bars to check for sweep

    Returns:
        Tuple of (detected, sweep_level, reasons)
    """
    if len(df) < lookback + 5:
        return False, 0.0, ["Insufficient data for liquidity sweep detection"]

    reasons = []
    check_window = 5  # Check last 5 bars for the sweep event

    if trend_direction == TrendDirection.BULLISH:
        # Look for sweep below recent swing low
        recent_swing_low = get_recent_swing_low(df.iloc[:-check_window], lookback)

        if recent_swing_low is None:
            return False, 0.0, ["No recent swing low found"]

        # Check if any of the last bars swept below then closed above
        for i in range(-check_window, 0):
            bar = df.iloc[i]
            swept_below = bar["low"] < recent_swing_low
            closed_above = bar["close"] > recent_swing_low

            if swept_below and closed_above:
                reasons.append(
                    f"Liquidity sweep: low ({bar['low']:.2f}) swept below "
                    f"swing low ({recent_swing_low:.2f}), closed above ({bar['close']:.2f})"
                )
                return True, recent_swing_low, reasons

        # Also check if a previous bar swept and current bar confirms reversal
        for i in range(-check_window, -1):
            bar = df.iloc[i]
            if bar["low"] < recent_swing_low:
                # Check if subsequent bars show reversal (closing above)
                for j in range(i + 1, 0):
                    next_bar = df.iloc[j]
                    if next_bar["close"] > recent_swing_low:
                        reasons.append(
                            f"Liquidity sweep confirmed: swept low ({bar['low']:.2f}) "
                            f"below {recent_swing_low:.2f}, reversal confirmed"
                        )
                        return True, recent_swing_low, reasons

    elif trend_direction == TrendDirection.BEARISH:
        # Look for sweep above recent swing high
        recent_swing_high = get_recent_swing_high(df.iloc[:-check_window], lookback)

        if recent_swing_high is None:
            return False, 0.0, ["No recent swing high found"]

        for i in range(-check_window, 0):
            bar = df.iloc[i]
            swept_above = bar["high"] > recent_swing_high
            closed_below = bar["close"] < recent_swing_high

            if swept_above and closed_below:
                reasons.append(
                    f"Liquidity sweep: high ({bar['high']:.2f}) swept above "
                    f"swing high ({recent_swing_high:.2f}), closed below ({bar['close']:.2f})"
                )
                return True, recent_swing_high, reasons

        for i in range(-check_window, -1):
            bar = df.iloc[i]
            if bar["high"] > recent_swing_high:
                for j in range(i + 1, 0):
                    next_bar = df.iloc[j]
                    if next_bar["close"] < recent_swing_high:
                        reasons.append(
                            f"Liquidity sweep confirmed: swept high ({bar['high']:.2f}) "
                            f"above {recent_swing_high:.2f}, reversal confirmed"
                        )
                        return True, recent_swing_high, reasons

    return False, 0.0, ["No liquidity sweep detected"]


def detect_market_structure_break(
    df: pd.DataFrame, trend_direction: TrendDirection
) -> tuple[bool, float, list]:
    """
    Detect a market structure break on the 5-minute chart.

    For BULLISH entry:
    - Find the most recent lower high in the pullback
    - Price must close ABOVE this lower high

    For BEARISH entry:
    - Find the most recent higher low in the pullback
    - Price must close BELOW this higher low

    Args:
        df: 5M OHLCV DataFrame
        trend_direction: Confirmed trend direction

    Returns:
        Tuple of (confirmed, break_level, reasons)
    """
    if len(df) < 20:
        return False, 0.0, ["Insufficient data for structure break"]

    reasons = []
    # Look at last 15 bars to find structure
    window = df.iloc[-20:]
    current_close = df["close"].iloc[-1]

    if trend_direction == TrendDirection.BULLISH:
        # Find the most recent lower high in the pullback phase
        # A lower high is a local high that is lower than the previous local high
        local_highs = []
        for i in range(2, len(window) - 1):
            if (window["high"].iloc[i] > window["high"].iloc[i - 1] and
                    window["high"].iloc[i] > window["high"].iloc[i + 1]):
                local_highs.append((i, window["high"].iloc[i]))

        if len(local_highs) >= 2:
            # Find a lower high (second high is lower than first)
            for j in range(len(local_highs) - 1, 0, -1):
                if local_highs[j][1] < local_highs[j - 1][1]:
                    lower_high = local_highs[j][1]
                    # Check if current close breaks above it
                    if current_close > lower_high:
                        reasons.append(
                            f"Structure break: close ({current_close:.2f}) above "
                            f"recent lower high ({lower_high:.2f})"
                        )
                        return True, lower_high, reasons
                    break

        # Alternative: simple check - close above highest high of last 3 pullback bars
        pullback_highs = window["high"].iloc[-6:-1]
        if len(pullback_highs) > 0:
            recent_high = pullback_highs.max()
            if current_close > recent_high:
                reasons.append(
                    f"Structure break (alt): close ({current_close:.2f}) above "
                    f"recent pullback high ({recent_high:.2f})"
                )
                return True, recent_high, reasons

    elif trend_direction == TrendDirection.BEARISH:
        # Find the most recent higher low in the pullback phase
        local_lows = []
        for i in range(2, len(window) - 1):
            if (window["low"].iloc[i] < window["low"].iloc[i - 1] and
                    window["low"].iloc[i] < window["low"].iloc[i + 1]):
                local_lows.append((i, window["low"].iloc[i]))

        if len(local_lows) >= 2:
            for j in range(len(local_lows) - 1, 0, -1):
                if local_lows[j][1] > local_lows[j - 1][1]:
                    higher_low = local_lows[j][1]
                    if current_close < higher_low:
                        reasons.append(
                            f"Structure break: close ({current_close:.2f}) below "
                            f"recent higher low ({higher_low:.2f})"
                        )
                        return True, higher_low, reasons
                    break

        # Alternative
        pullback_lows = window["low"].iloc[-6:-1]
        if len(pullback_lows) > 0:
            recent_low = pullback_lows.min()
            if current_close < recent_low:
                reasons.append(
                    f"Structure break (alt): close ({current_close:.2f}) below "
                    f"recent pullback low ({recent_low:.2f})"
                )
                return True, recent_low, reasons

    return False, 0.0, ["No market structure break detected"]


def check_candle_confirmation(
    df: pd.DataFrame, trend_direction: TrendDirection
) -> tuple[bool, Optional[str], list]:
    """
    Check for a confirming candlestick pattern.

    Bullish patterns: bullish_engulfing, hammer, bullish_rejection, strong_bullish
    Bearish patterns: bearish_engulfing, shooting_star, bearish_rejection, strong_bearish

    Returns:
        Tuple of (confirmed, pattern_name, reasons)
    """
    pattern = identify_candlestick_pattern(df, -1)

    if pattern is None:
        return False, None, ["No recognizable candlestick pattern"]

    bullish_patterns = {"bullish_engulfing", "hammer", "bullish_rejection", "strong_bullish"}
    bearish_patterns = {"bearish_engulfing", "shooting_star", "bearish_rejection", "strong_bearish"}

    if trend_direction == TrendDirection.BULLISH and pattern in bullish_patterns:
        return True, pattern, [f"Bullish candle confirmation: {pattern}"]
    elif trend_direction == TrendDirection.BEARISH and pattern in bearish_patterns:
        return True, pattern, [f"Bearish candle confirmation: {pattern}"]

    return False, pattern, [f"Pattern '{pattern}' doesn't match trend direction"]


def check_volume_confirmation(df: pd.DataFrame) -> tuple[bool, float, list]:
    """
    Check if current volume is above the 20-period average.
    Confirms institutional participation.

    Returns:
        Tuple of (confirmed, volume_ratio, reasons)
    """
    if "volume_avg" not in df.columns or len(df) < 1:
        return False, 0.0, ["Volume average not available"]

    current_volume = df["volume"].iloc[-1]
    avg_volume = df["volume_avg"].iloc[-1]

    if avg_volume <= 0:
        return False, 0.0, ["Average volume is zero"]

    volume_ratio = current_volume / avg_volume

    if volume_ratio >= 1.0:
        return True, volume_ratio, [
            f"Volume confirmed: {current_volume:.0f} / avg {avg_volume:.0f} = {volume_ratio:.2f}x"
        ]

    return False, volume_ratio, [
        f"Volume below average: {current_volume:.0f} / avg {avg_volume:.0f} = {volume_ratio:.2f}x"
    ]


def scan_for_entry(
    df_5m: pd.DataFrame, trend_direction: TrendDirection
) -> EntrySignal:
    """
    Complete entry signal scan on the 5-minute chart.

    Checks all confirmations in order:
    1. Pullback to value (20 EMA / VWAP)
    2. RSI in appropriate zone
    3. Liquidity sweep detected
    4. Market structure break
    5. Candlestick pattern confirmation
    6. Volume above average

    ALL must be confirmed for a valid signal.

    Args:
        df_5m: 5-minute OHLCV DataFrame (with indicators already added)
        trend_direction: Confirmed trend direction from higher timeframes

    Returns:
        EntrySignal with all details
    """
    signal = EntrySignal()

    if trend_direction == TrendDirection.NEUTRAL:
        signal.rejections.append("No trade: trend is neutral")
        return signal

    direction = "buy" if trend_direction == TrendDirection.BULLISH else "sell"
    signal.direction = direction

    # Ensure indicators are calculated
    if "ema_20" not in df_5m.columns:
        df_5m = add_all_indicators(df_5m, "5m")

    # 1. Check pullback to value
    pullback_ok, pullback_reasons = check_pullback_to_value(df_5m, trend_direction)
    signal.pullback_confirmed = pullback_ok
    if pullback_ok:
        signal.reasons.extend(pullback_reasons)
    else:
        signal.rejections.extend(pullback_reasons)

    # 2. Check RSI zone
    rsi_ok, rsi_value, rsi_reasons = check_rsi_zone(df_5m, trend_direction)
    signal.rsi_confirmed = rsi_ok
    signal.rsi_value = rsi_value
    if rsi_ok:
        signal.reasons.extend(rsi_reasons)
    else:
        signal.rejections.extend(rsi_reasons)

    # 3. Detect liquidity sweep
    sweep_ok, sweep_level, sweep_reasons = detect_liquidity_sweep(df_5m, trend_direction)
    signal.liquidity_sweep_detected = sweep_ok
    signal.sweep_level = sweep_level
    if sweep_ok:
        signal.reasons.extend(sweep_reasons)
    else:
        signal.rejections.extend(sweep_reasons)

    # 4. Check market structure break
    break_ok, break_level, break_reasons = detect_market_structure_break(df_5m, trend_direction)
    signal.structure_break_confirmed = break_ok
    signal.structure_break_level = break_level
    if break_ok:
        signal.reasons.extend(break_reasons)
    else:
        signal.rejections.extend(break_reasons)

    # 5. Candlestick pattern
    candle_ok, pattern, candle_reasons = check_candle_confirmation(df_5m, trend_direction)
    signal.candle_pattern_confirmed = candle_ok
    signal.candle_pattern = pattern
    if candle_ok:
        signal.reasons.extend(candle_reasons)
    else:
        signal.rejections.extend(candle_reasons)

    # 6. Volume confirmation
    volume_ok, vol_ratio, volume_reasons = check_volume_confirmation(df_5m)
    signal.volume_confirmed = volume_ok
    signal.volume_ratio = vol_ratio
    if volume_ok:
        signal.reasons.extend(volume_reasons)
    else:
        signal.rejections.extend(volume_reasons)

    # Final determination
    signal.valid = signal.all_confirmed
    signal.entry_price = df_5m["close"].iloc[-1]

    # Calculate confidence based on confirmation count
    signal.confidence_score = signal.confirmation_count / 6.0

    if signal.valid:
        logger.info(
            f"VALID ENTRY SIGNAL: {direction.upper()} | "
            f"Price={signal.entry_price:.2f} | "
            f"Confirmations: {signal.confirmation_count}/6 | "
            f"Pattern={signal.candle_pattern} | "
            f"RSI={signal.rsi_value:.1f} | Vol={signal.volume_ratio:.2f}x"
        )
    else:
        logger.debug(
            f"Entry scan ({direction}): {signal.confirmation_count}/6 confirmations | "
            f"Missing: {signal.rejections[:3]}"
        )

    return signal
