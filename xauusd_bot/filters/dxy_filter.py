"""
DXY (Dollar Index) Correlation Filter for XAUUSD.

Gold has a strong inverse correlation with the US Dollar Index.
This filter confirms gold trade direction by checking if DXY is moving
in the opposite direction.

Logic:
    - XAUUSD BUY signal → DXY should be falling (confirms gold strength)
    - XAUUSD SELL signal → DXY should be rising (confirms gold weakness)
    - Divergence detected → Flag as caution (gold moving WITH dollar = unusual)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class DXYSignal:
    """DXY filter result."""
    confirms: bool = False
    correlation: float = 0.0
    dxy_direction: str = "FLAT"  # "UP", "DOWN", "FLAT"
    divergence_detected: bool = False
    reason: str = ""


class DXYFilter:
    """
    Monitors DXY movement to confirm or deny XAUUSD trade direction.
    Uses rolling correlation and short-term DXY trend.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.correlation_period = config.get("correlation_period", 20)
        self.min_inverse_strength = config.get("min_inverse_strength", -0.5)
        self.divergence_threshold = config.get("divergence_threshold", 0.3)

    def evaluate(self, gold_closes: np.ndarray, dxy_closes: np.ndarray,
                 proposed_direction: str) -> DXYSignal:
        """
        Evaluate whether DXY confirms the proposed gold trade direction.

        Args:
            gold_closes: Recent XAUUSD close prices (numpy array)
            dxy_closes: Recent DXY close prices (same length)
            proposed_direction: "BUY" or "SELL" for gold
        """
        signal = DXYSignal()

        if len(gold_closes) < self.correlation_period or len(dxy_closes) < self.correlation_period:
            signal.reason = "Insufficient data for correlation"
            signal.confirms = True  # Don't block trade if no data
            return signal

        # Calculate rolling correlation
        correlation = self._calculate_correlation(gold_closes, dxy_closes)
        signal.correlation = correlation

        # Determine DXY short-term direction
        dxy_direction = self._get_dxy_direction(dxy_closes)
        signal.dxy_direction = dxy_direction

        # Evaluate confirmation
        if proposed_direction == "BUY":
            # Gold BUY = expect DXY falling
            if dxy_direction == "DOWN":
                signal.confirms = True
                signal.reason = "DXY falling confirms gold BUY"
            elif dxy_direction == "FLAT":
                signal.confirms = True  # Neutral = don't block
                signal.reason = "DXY flat, not blocking"
            else:
                # DXY rising while gold wants to buy — divergence
                signal.confirms = False
                signal.divergence_detected = True
                signal.reason = "DXY rising contradicts gold BUY"

        elif proposed_direction == "SELL":
            # Gold SELL = expect DXY rising
            if dxy_direction == "UP":
                signal.confirms = True
                signal.reason = "DXY rising confirms gold SELL"
            elif dxy_direction == "FLAT":
                signal.confirms = True
                signal.reason = "DXY flat, not blocking"
            else:
                signal.confirms = False
                signal.divergence_detected = True
                signal.reason = "DXY falling contradicts gold SELL"

        # Check if correlation has broken down (unusual regime)
        if correlation > self.divergence_threshold:
            signal.divergence_detected = True
            signal.reason += " | WARNING: Positive correlation detected (unusual)"

        return signal

    def _calculate_correlation(self, gold: np.ndarray, dxy: np.ndarray) -> float:
        """Calculate Pearson correlation between gold and DXY returns."""
        period = self.correlation_period

        # Use returns (percentage changes) for correlation
        gold_returns = np.diff(gold[-period - 1:]) / gold[-period - 1:-1]
        dxy_returns = np.diff(dxy[-period - 1:]) / dxy[-period - 1:-1]

        if len(gold_returns) < 2 or len(dxy_returns) < 2:
            return 0.0

        # Pearson correlation
        gold_mean = np.mean(gold_returns)
        dxy_mean = np.mean(dxy_returns)

        numerator = np.sum((gold_returns - gold_mean) * (dxy_returns - dxy_mean))
        gold_std = np.sqrt(np.sum((gold_returns - gold_mean) ** 2))
        dxy_std = np.sqrt(np.sum((dxy_returns - dxy_mean) ** 2))

        if gold_std == 0 or dxy_std == 0:
            return 0.0

        correlation = numerator / (gold_std * dxy_std)
        return round(correlation, 4)

    def _get_dxy_direction(self, dxy_closes: np.ndarray) -> str:
        """Determine short-term DXY direction using last 5 bars."""
        if len(dxy_closes) < 5:
            return "FLAT"

        recent = dxy_closes[-5:]
        change_pct = (recent[-1] - recent[0]) / recent[0] * 100

        if change_pct > 0.05:  # DXY up by >0.05%
            return "UP"
        elif change_pct < -0.05:  # DXY down by >0.05%
            return "DOWN"
        else:
            return "FLAT"
