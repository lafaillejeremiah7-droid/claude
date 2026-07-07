"""
Multi-Timeframe Trend Analysis Module

Implements the trend identification logic:
1. 4H Chart: 50 EMA slope + price position determines dominant trend
2. 30M Chart: 50 EMA vs 200 EMA alignment + price position confirms trend
3. Both must agree before any trade is considered

Trend States:
- BULLISH: Both 4H and 30M confirm uptrend
- BEARISH: Both 4H and 30M confirm downtrend
- NEUTRAL: No alignment = no trade
"""
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from modules.indicators import add_all_indicators

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class TrendState:
    """Complete trend state across all timeframes."""
    direction_4h: TrendDirection
    direction_30m: TrendDirection
    combined: TrendDirection
    ema_50_4h: float
    ema_50_slope_4h: float
    ema_50_30m: float
    ema_200_30m: float
    price_4h: float
    price_30m: float
    confidence: float  # 0-1 score of trend strength
    partial_alignment: bool = False  # True when 4H has direction but 30M is neutral

    @property
    def is_tradeable(self) -> bool:
        """Trade when both agree OR partial alignment (4H directed, 30M neutral)."""
        return self.combined != TrendDirection.NEUTRAL

    @property
    def trade_direction(self) -> Optional[str]:
        """Return 'buy' or 'sell' or None."""
        if self.combined == TrendDirection.BULLISH:
            return "buy"
        elif self.combined == TrendDirection.BEARISH:
            return "sell"
        return None


def analyze_4h_trend(df_4h: pd.DataFrame) -> tuple[TrendDirection, dict]:
    """
    Analyze the 4-hour chart trend.

    Rules:
    - 50 EMA sloping UP + price ABOVE 50 EMA = BULLISH
    - 50 EMA sloping DOWN + price BELOW 50 EMA = BEARISH
    - Otherwise = NEUTRAL

    Args:
        df_4h: 4H OHLCV DataFrame with indicators

    Returns:
        Tuple of (TrendDirection, metadata dict)
    """
    if df_4h is None or len(df_4h) < 55:
        logger.warning("Insufficient 4H data for trend analysis")
        return TrendDirection.NEUTRAL, {}

    # Add indicators if not already present
    if "ema_50" not in df_4h.columns:
        df_4h = add_all_indicators(df_4h, "4h")

    # Get latest values
    latest = df_4h.iloc[-1]
    ema_50 = latest["ema_50"]
    ema_slope = latest["ema_50_slope"]
    current_price = latest["close"]

    metadata = {
        "ema_50": ema_50,
        "ema_slope": ema_slope,
        "price": current_price,
        "price_vs_ema": current_price - ema_50,
    }

    # Determine trend
    if ema_slope > 0 and current_price > ema_50:
        direction = TrendDirection.BULLISH
        logger.debug(
            f"4H BULLISH: EMA50={ema_50:.2f}, slope={ema_slope:.4f}, "
            f"price={current_price:.2f} (above EMA)"
        )
    elif ema_slope < 0 and current_price < ema_50:
        direction = TrendDirection.BEARISH
        logger.debug(
            f"4H BEARISH: EMA50={ema_50:.2f}, slope={ema_slope:.4f}, "
            f"price={current_price:.2f} (below EMA)"
        )
    else:
        direction = TrendDirection.NEUTRAL
        logger.debug(
            f"4H NEUTRAL: EMA50={ema_50:.2f}, slope={ema_slope:.4f}, "
            f"price={current_price:.2f} (no clear direction)"
        )

    return direction, metadata


def analyze_30m_trend(df_30m: pd.DataFrame) -> tuple[TrendDirection, dict]:
    """
    Analyze the 30-minute chart trend confirmation.

    Rules:
    - 50 EMA ABOVE 200 EMA + price ABOVE 50 EMA = BULLISH
    - 50 EMA BELOW 200 EMA + price BELOW 50 EMA = BEARISH
    - Otherwise = NEUTRAL

    Args:
        df_30m: 30M OHLCV DataFrame with indicators

    Returns:
        Tuple of (TrendDirection, metadata dict)
    """
    if df_30m is None or len(df_30m) < 210:
        logger.warning("Insufficient 30M data for trend analysis")
        return TrendDirection.NEUTRAL, {}

    # Add indicators if not already present
    if "ema_50" not in df_30m.columns:
        df_30m = add_all_indicators(df_30m, "30m")

    # Get latest values
    latest = df_30m.iloc[-1]
    ema_50 = latest["ema_50"]
    ema_200 = latest["ema_200"]
    current_price = latest["close"]

    metadata = {
        "ema_50": ema_50,
        "ema_200": ema_200,
        "price": current_price,
        "ema_gap": ema_50 - ema_200,
        "price_vs_ema50": current_price - ema_50,
    }

    # Determine trend
    if ema_50 > ema_200 and current_price > ema_50:
        direction = TrendDirection.BULLISH
        logger.debug(
            f"30M BULLISH: EMA50={ema_50:.2f} > EMA200={ema_200:.2f}, "
            f"price={current_price:.2f} (above EMA50)"
        )
    elif ema_50 < ema_200 and current_price < ema_50:
        direction = TrendDirection.BEARISH
        logger.debug(
            f"30M BEARISH: EMA50={ema_50:.2f} < EMA200={ema_200:.2f}, "
            f"price={current_price:.2f} (below EMA50)"
        )
    else:
        direction = TrendDirection.NEUTRAL
        logger.debug(
            f"30M NEUTRAL: EMA50={ema_50:.2f}, EMA200={ema_200:.2f}, "
            f"price={current_price:.2f} (no clear alignment)"
        )

    return direction, metadata


def calculate_trend_confidence(
    direction_4h: TrendDirection,
    direction_30m: TrendDirection,
    meta_4h: dict,
    meta_30m: dict,
) -> float:
    """
    Calculate a confidence score (0-1) for the current trend.

    Higher confidence when:
    - Both timeframes strongly agree
    - Price is well above/below EMAs (not hugging them)
    - EMA slope is steep
    - EMA gap on 30M is wide

    Args:
        direction_4h: 4H trend direction
        direction_30m: 30M trend direction
        meta_4h: 4H analysis metadata
        meta_30m: 30M analysis metadata

    Returns:
        Confidence score between 0 and 1
    """
    if direction_4h == TrendDirection.NEUTRAL or direction_30m == TrendDirection.NEUTRAL:
        return 0.0

    if direction_4h != direction_30m:
        return 0.0

    confidence = 0.5  # Base for agreement

    # Bonus for price distance from EMA on 4H
    price_4h = meta_4h.get("price", 0)
    ema_4h = meta_4h.get("ema_50", 0)
    if ema_4h > 0:
        distance_pct_4h = abs(price_4h - ema_4h) / ema_4h
        confidence += min(distance_pct_4h * 10, 0.15)  # Max +0.15

    # Bonus for EMA slope strength on 4H
    slope = abs(meta_4h.get("ema_slope", 0))
    if ema_4h > 0:
        slope_pct = slope / ema_4h
        confidence += min(slope_pct * 50, 0.15)  # Max +0.15

    # Bonus for EMA gap on 30M
    ema_gap = abs(meta_30m.get("ema_gap", 0))
    ema_200 = meta_30m.get("ema_200", 0)
    if ema_200 > 0:
        gap_pct = ema_gap / ema_200
        confidence += min(gap_pct * 10, 0.1)  # Max +0.1

    # Bonus for price distance from 50 EMA on 30M
    price_30m = meta_30m.get("price", 0)
    ema_50_30m = meta_30m.get("ema_50", 0)
    if ema_50_30m > 0:
        distance_pct_30m = abs(price_30m - ema_50_30m) / ema_50_30m
        confidence += min(distance_pct_30m * 10, 0.1)  # Max +0.1

    return min(confidence, 1.0)


def get_trend_state(df_4h: pd.DataFrame, df_30m: pd.DataFrame) -> TrendState:
    """
    Full multi-timeframe trend analysis.

    Combines 4H and 30M analysis to determine if conditions
    are suitable for trading and in which direction.

    Graduated Conviction:
    - If 4H and 30M agree: full confidence, normal trade
    - If 4H has direction but 30M is NEUTRAL: partial alignment, capped confidence
    - If 4H and 30M OPPOSE each other: no trade (genuine conflict)
    - If 4H is NEUTRAL: no trade

    Args:
        df_4h: 4-hour OHLCV DataFrame
        df_30m: 30-minute OHLCV DataFrame

    Returns:
        TrendState with complete trend information
    """
    import os
    allow_partial = os.environ.get("ALLOW_PARTIAL_ALIGNMENT", "true").lower() in ("true", "1", "yes")

    # Analyze each timeframe
    direction_4h, meta_4h = analyze_4h_trend(df_4h)
    direction_30m, meta_30m = analyze_30m_trend(df_30m)

    partial_alignment = False

    # Determine combined direction
    if direction_4h == direction_30m and direction_4h != TrendDirection.NEUTRAL:
        # Full alignment
        combined = direction_4h
        logger.info(
            f"TREND ALIGNED: {combined.value.upper()} | "
            f"4H={direction_4h.value} 30M={direction_30m.value}"
        )
    elif (allow_partial and
          direction_4h != TrendDirection.NEUTRAL and
          direction_30m == TrendDirection.NEUTRAL):
        # Partial alignment: 4H has direction, 30M is neutral (not opposing)
        combined = direction_4h
        partial_alignment = True
        logger.info(
            f"TREND PARTIAL ALIGN: {combined.value.upper()} | "
            f"4H={direction_4h.value} 30M={direction_30m.value} (neutral, not opposing)"
        )
    elif (direction_4h != TrendDirection.NEUTRAL and
          direction_30m != TrendDirection.NEUTRAL and
          direction_4h != direction_30m):
        # Opposing trends: genuine conflict, reject
        combined = TrendDirection.NEUTRAL
        logger.info(
            f"TREND OPPOSING: No trade | "
            f"4H={direction_4h.value} 30M={direction_30m.value} (conflict)"
        )
    else:
        combined = TrendDirection.NEUTRAL
        logger.info(
            f"TREND NOT ALIGNED: No trade | "
            f"4H={direction_4h.value} 30M={direction_30m.value}"
        )

    # Calculate confidence
    confidence = calculate_trend_confidence(
        direction_4h, direction_30m, meta_4h, meta_30m
    )

    # For partial alignment, provide a reduced but non-zero confidence
    if partial_alignment and confidence == 0.0:
        # Base partial confidence: weaker than full alignment
        confidence = 0.35
        # Small bonus for 4H strength
        price_4h = meta_4h.get("price", 0)
        ema_4h = meta_4h.get("ema_50", 0)
        if ema_4h > 0:
            distance_pct_4h = abs(price_4h - ema_4h) / ema_4h
            confidence += min(distance_pct_4h * 8, 0.1)
        slope = abs(meta_4h.get("ema_slope", 0))
        if ema_4h > 0:
            slope_pct = slope / ema_4h
            confidence += min(slope_pct * 30, 0.1)
        confidence = min(confidence, 0.55)  # Cap for partial

    return TrendState(
        direction_4h=direction_4h,
        direction_30m=direction_30m,
        combined=combined,
        ema_50_4h=meta_4h.get("ema_50", 0),
        ema_50_slope_4h=meta_4h.get("ema_slope", 0),
        ema_50_30m=meta_30m.get("ema_50", 0),
        ema_200_30m=meta_30m.get("ema_200", 0),
        price_4h=meta_4h.get("price", 0),
        price_30m=meta_30m.get("price", 0),
        confidence=confidence,
        partial_alignment=partial_alignment,
    )
