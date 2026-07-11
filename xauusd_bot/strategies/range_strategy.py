"""
Range/Mean-Reversion Strategy for XAUUSD (Asian Session).

Entry Logic:
    1. Market confirmed as ranging (ADX < 20, BB squeeze)
    2. Price touches support or resistance zone (Bollinger boundaries)
    3. RSI at extreme + Stochastic confirmation
    4. Multi-factor score >= threshold
    5. Risk manager validates position

Exit Logic:
    - Hard SL: Just outside range boundary (tighter than trend)
    - TP: Opposite side of range (BB middle or other boundary)
    - Quick exit if range breaks (ADX spikes above 25)
"""

from dataclasses import dataclass
from typing import Optional
from ..indicators.technical import IndicatorSnapshot
from ..indicators.scoring import MultiFactorScorer, ScoreResult
from ..core.risk_manager import RiskManager, TradeParams


@dataclass
class RangeSignal:
    """Output of range strategy evaluation."""
    has_signal: bool = False
    direction: str = "NONE"       # "BUY" or "SELL"
    entry_type: str = "NONE"      # "SUPPORT_BOUNCE" or "RESISTANCE_FADE"
    score: float = 0.0
    score_details: dict = None
    trade_params: Optional[TradeParams] = None
    reason: str = ""

    def __post_init__(self):
        if self.score_details is None:
            self.score_details = {}


class RangeStrategy:
    """
    Mean-reversion strategy for ranging conditions.
    Buys at support, sells at resistance with tight stops.
    """

    def __init__(self, scorer: MultiFactorScorer, risk_manager: RiskManager,
                 config: dict):
        self.scorer = scorer
        self.risk = risk_manager
        self.cfg = config
        # Range SL is tighter: 1.0x ATR instead of 1.5x
        self._range_sl_multiplier = 1.0
        # Range TP targets the middle band or opposite boundary
        self._range_tp_multiplier = 2.0

    def evaluate(self, snap: IndicatorSnapshot) -> RangeSignal:
        """
        Evaluate current conditions for a range entry.
        """
        signal = RangeSignal()

        # Step 1: Determine if price is at support or resistance
        at_support = self._is_at_support(snap)
        at_resistance = self._is_at_resistance(snap)

        if not at_support and not at_resistance:
            signal.reason = "Price not at range boundary"
            return signal

        # Step 2: Score multi-factor conditions
        score_result = self.scorer.score_range(
            snap, at_support=at_support, at_resistance=at_resistance
        )

        signal.score = score_result.total_score
        signal.score_details = score_result.factors
        signal.direction = score_result.direction

        # Step 3: Check threshold
        if not score_result.passed:
            signal.reason = f"Score {score_result.total_score:.2f} below threshold"
            return signal

        # Step 4: Set entry type
        if at_support:
            signal.entry_type = "SUPPORT_BOUNCE"
        else:
            signal.entry_type = "RESISTANCE_FADE"

        # Step 5: Calculate trade parameters (tighter for range)
        trade_params = self._calculate_range_trade(snap, score_result.direction)

        if not trade_params.valid:
            signal.reason = f"Risk check failed (R:R={trade_params.risk_reward_ratio:.2f})"
            return signal

        # All checks passed
        signal.has_signal = True
        signal.trade_params = trade_params
        signal.reason = (f"{signal.entry_type}: {score_result.direction} "
                         f"score={score_result.total_score:.2f}")

        return signal

    def _is_at_support(self, snap: IndicatorSnapshot) -> bool:
        """Check if price is at or near lower Bollinger Band (support zone)."""
        if snap.bb_lower == 0:
            return False

        bb_range = snap.bb_upper - snap.bb_lower
        if bb_range == 0:
            return False

        # Within 10% of band range from lower band
        distance_pct = (snap.current_price - snap.bb_lower) / bb_range
        return distance_pct <= 0.10

    def _is_at_resistance(self, snap: IndicatorSnapshot) -> bool:
        """Check if price is at or near upper Bollinger Band (resistance zone)."""
        if snap.bb_upper == 0:
            return False

        bb_range = snap.bb_upper - snap.bb_lower
        if bb_range == 0:
            return False

        # Within 10% of band range from upper band
        distance_pct = (snap.bb_upper - snap.current_price) / bb_range
        return distance_pct <= 0.10

    def _calculate_range_trade(self, snap: IndicatorSnapshot, direction: str) -> TradeParams:
        """Calculate trade params with range-specific tighter stops."""
        params = TradeParams(direction=direction, entry_price=snap.current_price)

        # Tighter SL for range trades
        sl_distance = snap.atr * self._range_sl_multiplier

        # TP targets the middle band (conservative) or opposite band
        if direction == "BUY":
            params.stop_loss = snap.current_price - sl_distance
            # TP = middle band (conservative target within the range)
            tp_distance = snap.bb_middle - snap.current_price
            if tp_distance <= 0:
                # Fallback: use ATR multiplier
                tp_distance = snap.atr * self._range_tp_multiplier
            params.take_profit = snap.current_price + tp_distance

        elif direction == "SELL":
            params.stop_loss = snap.current_price + sl_distance
            tp_distance = snap.current_price - snap.bb_middle
            if tp_distance <= 0:
                tp_distance = snap.atr * self._range_tp_multiplier
            params.take_profit = snap.current_price - tp_distance

        # Calculate pips and R:R
        params.sl_pips = sl_distance / 0.01
        params.tp_pips = tp_distance / 0.01

        if sl_distance > 0:
            params.risk_reward_ratio = tp_distance / sl_distance

        # Position sizing via risk manager
        risk_amount = self.risk.equity * self.risk.cfg["max_risk_per_trade"]
        params.risk_amount = risk_amount

        if params.sl_pips > 0:
            params.lot_size = round(risk_amount / (params.sl_pips * 1.0), 2)
            params.lot_size = max(params.lot_size, 0.01)

        params.reward_amount = params.lot_size * params.tp_pips * 1.0

        # Validate R:R (range trades can accept 1.5:1 minimum)
        params.valid = params.risk_reward_ratio >= 1.5 and params.lot_size > 0

        return params

    # ------------------------------------------------------------------
    # POSITION MANAGEMENT
    # ------------------------------------------------------------------

    def should_exit(self, direction: str, entry_price: float,
                    current_price: float, current_sl: float,
                    current_tp: float, snap: IndicatorSnapshot) -> tuple[bool, str]:
        """
        Check if an open range position should be exited.
        Range trades have an additional exit: range breakout detection.
        """
        # Hard stop loss
        if direction == "BUY" and current_price <= current_sl:
            return True, "STOP_LOSS"
        if direction == "SELL" and current_price >= current_sl:
            return True, "STOP_LOSS"

        # Take profit
        if direction == "BUY" and current_price >= current_tp:
            return True, "TAKE_PROFIT"
        if direction == "SELL" and current_price <= current_tp:
            return True, "TAKE_PROFIT"

        # Range breakout detection: ADX suddenly spikes (range is breaking)
        if snap.adx > 30:
            # Range is breaking — exit immediately to protect capital
            if direction == "BUY" and snap.di_minus > snap.di_plus:
                return True, "RANGE_BREAK_AGAINST"
            if direction == "SELL" and snap.di_plus > snap.di_minus:
                return True, "RANGE_BREAK_AGAINST"

        # Price broke through opposite Bollinger Band (beyond TP zone)
        if direction == "BUY" and current_price >= snap.bb_upper:
            return True, "UPPER_BAND_HIT"
        if direction == "SELL" and current_price <= snap.bb_lower:
            return True, "LOWER_BAND_HIT"

        return False, ""
