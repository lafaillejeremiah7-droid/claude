"""
Multi-Factor Confirmation Scoring System.

Evaluates market conditions against weighted factors and produces
a confidence score (0.0 - 1.0) for entry decisions.

Two scoring modes:
  - TREND: Used during London/NY sessions (EMA alignment, ADX, ATR, DXY, structure)
  - RANGE: Used during Asian session (range ID, S/R touch, reversal, stochastic)
"""

from dataclasses import dataclass
from typing import Optional
from .technical import IndicatorSnapshot


@dataclass
class ScoreResult:
    """Detailed scoring breakdown."""
    total_score: float = 0.0
    direction: str = "NONE"  # "BUY", "SELL", or "NONE"
    factors: dict = None     # Individual factor scores
    passed: bool = False     # Whether total_score >= threshold

    def __post_init__(self):
        if self.factors is None:
            self.factors = {}


class MultiFactorScorer:
    """
    Scores entry conditions using weighted multi-factor analysis.
    Each factor returns 0.0 (no signal) to 1.0 (perfect signal).
    Weighted sum produces overall confidence.
    """

    def __init__(self, trend_weights: dict, range_weights: dict, threshold: float):
        self.trend_weights = trend_weights
        self.range_weights = range_weights
        self.threshold = threshold

    # ==================================================================
    # TREND MODE SCORING
    # ==================================================================

    def score_trend(self, snap: IndicatorSnapshot,
                    dxy_confirms: bool = False,
                    structure_broken: bool = False) -> ScoreResult:
        """
        Score trend conditions for a potential entry.
        Returns score + direction (BUY/SELL).
        """
        factors = {}
        direction = self._determine_trend_direction(snap)

        if direction == "NONE":
            return ScoreResult(total_score=0.0, direction="NONE", factors=factors, passed=False)

        # Factor 1: EMA Alignment (0.0 - 1.0)
        factors["ema_alignment"] = self._score_ema_alignment(snap, direction)

        # Factor 2: ADX Momentum (0.0 - 1.0)
        factors["adx_momentum"] = self._score_adx_momentum(snap, direction)

        # Factor 3: ATR Expansion (0.0 - 1.0)
        factors["atr_expansion"] = self._score_atr_expansion(snap)

        # Factor 4: DXY Confluence (0.0 - 1.0)
        factors["dxy_confluence"] = 1.0 if dxy_confirms else 0.0

        # Factor 5: Structure Break (0.0 - 1.0)
        factors["structure_break"] = 1.0 if structure_broken else 0.0

        # Weighted sum
        total = sum(
            factors[key] * self.trend_weights[key]
            for key in self.trend_weights
            if key in factors
        )

        return ScoreResult(
            total_score=round(total, 4),
            direction=direction,
            factors=factors,
            passed=total >= self.threshold
        )

    def _determine_trend_direction(self, snap: IndicatorSnapshot) -> str:
        """Determine if trend is bullish, bearish, or unclear."""
        price = snap.current_price

        # Strong bullish: price > EMA fast > EMA slow, DI+ > DI-
        if price > snap.ema_fast > snap.ema_slow and snap.di_plus > snap.di_minus:
            return "BUY"

        # Strong bearish: price < EMA fast < EMA slow, DI- > DI+
        if price < snap.ema_fast < snap.ema_slow and snap.di_minus > snap.di_plus:
            return "SELL"

        return "NONE"

    def _score_ema_alignment(self, snap: IndicatorSnapshot, direction: str) -> float:
        """Score EMA alignment quality (0-1)."""
        score = 0.0

        if direction == "BUY":
            # Perfect: price > fast > slow > bias
            if snap.current_price > snap.ema_fast > snap.ema_slow > snap.ema_bias:
                score = 1.0
            elif snap.current_price > snap.ema_fast > snap.ema_slow:
                score = 0.75
            elif snap.current_price > snap.ema_fast:
                score = 0.4
        elif direction == "SELL":
            # Perfect: price < fast < slow < bias
            if snap.current_price < snap.ema_fast < snap.ema_slow < snap.ema_bias:
                score = 1.0
            elif snap.current_price < snap.ema_fast < snap.ema_slow:
                score = 0.75
            elif snap.current_price < snap.ema_fast:
                score = 0.4

        return score

    def _score_adx_momentum(self, snap: IndicatorSnapshot, direction: str) -> float:
        """Score ADX trend strength (0-1)."""
        if snap.adx < 20:
            return 0.0  # No trend
        if snap.adx < 25:
            return 0.3  # Weak trend

        # ADX > 25 = trending. Score based on DI separation
        di_diff = abs(snap.di_plus - snap.di_minus)

        # Confirm direction alignment
        if direction == "BUY" and snap.di_plus <= snap.di_minus:
            return 0.0
        if direction == "SELL" and snap.di_minus <= snap.di_plus:
            return 0.0

        # Score: more separation = stronger signal
        if di_diff > 20:
            return 1.0
        elif di_diff > 10:
            return 0.8
        elif di_diff > 5:
            return 0.6
        else:
            return 0.4

    def _score_atr_expansion(self, snap: IndicatorSnapshot) -> float:
        """Score volatility expansion (0-1). Want expanding ATR for trend entries."""
        if snap.atr_avg == 0:
            return 0.5

        ratio = snap.atr / snap.atr_avg

        if ratio >= 1.5:
            return 1.0
        elif ratio >= 1.3:
            return 0.8
        elif ratio >= 1.1:
            return 0.6
        elif ratio >= 0.9:
            return 0.3  # Normal volatility, not ideal for breakout
        else:
            return 0.0  # Contracting — not a trend entry

    # ==================================================================
    # RANGE MODE SCORING
    # ==================================================================

    def score_range(self, snap: IndicatorSnapshot,
                    at_support: bool = False,
                    at_resistance: bool = False) -> ScoreResult:
        """
        Score range/mean-reversion conditions for a potential entry.
        Returns score + direction (BUY at support, SELL at resistance).
        """
        factors = {}

        # Determine direction from S/R position
        if at_support:
            direction = "BUY"
        elif at_resistance:
            direction = "SELL"
        else:
            return ScoreResult(total_score=0.0, direction="NONE", factors=factors, passed=False)

        # Factor 1: Range Identified (ADX low + BB squeeze)
        factors["range_identified"] = self._score_range_identification(snap)

        # Factor 2: S/R Touch (proximity to Bollinger boundary)
        factors["sr_touch"] = self._score_sr_touch(snap, direction)

        # Factor 3: Reversal Signal (RSI at extreme)
        factors["reversal_signal"] = self._score_reversal_signal(snap, direction)

        # Factor 4: Stochastic Confirmation
        factors["stoch_confirmation"] = self._score_stochastic(snap, direction)

        # Weighted sum
        total = sum(
            factors[key] * self.range_weights[key]
            for key in self.range_weights
            if key in factors
        )

        return ScoreResult(
            total_score=round(total, 4),
            direction=direction,
            factors=factors,
            passed=total >= self.threshold
        )

    def _score_range_identification(self, snap: IndicatorSnapshot) -> float:
        """Score how well a range is identified (0-1)."""
        score = 0.0

        # ADX below 20 = ranging
        if snap.adx < 15:
            score += 0.5
        elif snap.adx < 20:
            score += 0.3

        # Bollinger Band squeeze (bandwidth below average)
        if snap.bb_bandwidth_avg > 0:
            bw_ratio = snap.bb_bandwidth / snap.bb_bandwidth_avg
            if bw_ratio < 0.5:
                score += 0.5  # Strong squeeze
            elif bw_ratio < 0.8:
                score += 0.3  # Moderate squeeze

        return min(score, 1.0)

    def _score_sr_touch(self, snap: IndicatorSnapshot, direction: str) -> float:
        """Score proximity to support/resistance (0-1)."""
        price = snap.current_price

        if direction == "BUY":
            # How close to lower BB
            bb_range = snap.bb_upper - snap.bb_lower
            if bb_range == 0:
                return 0.0
            distance_from_lower = (price - snap.bb_lower) / bb_range
            if distance_from_lower <= 0.05:
                return 1.0  # Right at support
            elif distance_from_lower <= 0.15:
                return 0.7
            elif distance_from_lower <= 0.25:
                return 0.4
            return 0.0

        elif direction == "SELL":
            # How close to upper BB
            bb_range = snap.bb_upper - snap.bb_lower
            if bb_range == 0:
                return 0.0
            distance_from_upper = (snap.bb_upper - price) / bb_range
            if distance_from_upper <= 0.05:
                return 1.0  # Right at resistance
            elif distance_from_upper <= 0.15:
                return 0.7
            elif distance_from_upper <= 0.25:
                return 0.4
            return 0.0

        return 0.0

    def _score_reversal_signal(self, snap: IndicatorSnapshot, direction: str) -> float:
        """Score RSI reversal signal (0-1)."""
        if direction == "BUY":
            if snap.rsi <= 25:
                return 1.0  # Deeply oversold
            elif snap.rsi <= 30:
                return 0.8
            elif snap.rsi <= 35:
                return 0.5
            return 0.0

        elif direction == "SELL":
            if snap.rsi >= 75:
                return 1.0  # Deeply overbought
            elif snap.rsi >= 70:
                return 0.8
            elif snap.rsi >= 65:
                return 0.5
            return 0.0

        return 0.0

    def _score_stochastic(self, snap: IndicatorSnapshot, direction: str) -> float:
        """Score stochastic confirmation (0-1)."""
        if direction == "BUY":
            if snap.stoch_k <= 20 and snap.stoch_d <= 20:
                return 1.0  # Both in oversold
            elif snap.stoch_k <= 30:
                return 0.6
            # Bullish crossover (K crossing above D from oversold)
            elif snap.stoch_k > snap.stoch_d and snap.stoch_k < 40:
                return 0.7
            return 0.0

        elif direction == "SELL":
            if snap.stoch_k >= 80 and snap.stoch_d >= 80:
                return 1.0  # Both in overbought
            elif snap.stoch_k >= 70:
                return 0.6
            # Bearish crossover (K crossing below D from overbought)
            elif snap.stoch_k < snap.stoch_d and snap.stoch_k > 60:
                return 0.7
            return 0.0

        return 0.0
