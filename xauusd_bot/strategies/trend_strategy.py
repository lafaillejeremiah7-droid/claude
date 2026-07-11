"""
Trend-Following Strategy for XAUUSD (London/NY Sessions).

Entry Logic:
    1. Multi-factor score >= threshold (EMA alignment, ADX, ATR expansion, DXY, structure)
    2. Price pulls back to EMA20 (buy the dip in uptrend / sell the rally in downtrend)
       OR price breaks and retests session high/low
    3. Risk manager validates position size and R:R

Exit Logic:
    - Hard SL: ATR-based stop loss
    - TP: ATR-based take profit (min 1:2 R:R)
    - Trailing stop: Activated after 1:1 R:R achieved
    - Time exit: Close if near session end with open profit
"""

from dataclasses import dataclass
from typing import Optional
from ..indicators.technical import IndicatorSnapshot
from ..indicators.scoring import MultiFactorScorer, ScoreResult
from ..core.risk_manager import RiskManager, TradeParams


@dataclass
class TrendSignal:
    """Output of trend strategy evaluation."""
    has_signal: bool = False
    direction: str = "NONE"       # "BUY" or "SELL"
    entry_type: str = "NONE"      # "PULLBACK" or "BREAKOUT"
    score: float = 0.0
    score_details: dict = None
    trade_params: Optional[TradeParams] = None
    reason: str = ""

    def __post_init__(self):
        if self.score_details is None:
            self.score_details = {}


class TrendStrategy:
    """
    Trend-following strategy optimized for XAUUSD during high-volatility sessions.
    Uses pullback-to-EMA and break-and-retest patterns.
    """

    def __init__(self, scorer: MultiFactorScorer, risk_manager: RiskManager,
                 config: dict):
        self.scorer = scorer
        self.risk = risk_manager
        self.cfg = config
        self._pullback_zone_pct = 0.3  # Within 30% of ATR from EMA = pullback zone

    def evaluate(self, snap: IndicatorSnapshot,
                 dxy_confirms: bool = False,
                 session_high: Optional[float] = None,
                 session_low: Optional[float] = None,
                 previous_session_high: Optional[float] = None,
                 previous_session_low: Optional[float] = None) -> TrendSignal:
        """
        Evaluate current conditions for a trend entry.

        Args:
            snap: Current indicator snapshot
            dxy_confirms: Whether DXY is moving inversely (confirming gold direction)
            session_high/low: Current session extremes
            previous_session_high/low: Previous session extremes for structure breaks
        """
        signal = TrendSignal()

        # Step 1: Score multi-factor conditions
        structure_broken = self._check_structure_break(
            snap, session_high, session_low, previous_session_high, previous_session_low
        )

        score_result = self.scorer.score_trend(
            snap, dxy_confirms=dxy_confirms, structure_broken=structure_broken
        )

        signal.score = score_result.total_score
        signal.score_details = score_result.factors
        signal.direction = score_result.direction

        # Step 2: Check if score passes threshold
        if not score_result.passed:
            signal.reason = f"Score {score_result.total_score:.2f} below threshold"
            return signal

        # Step 3: Identify entry type (pullback or breakout)
        entry_type = self._identify_entry_type(snap, score_result.direction,
                                                session_high, session_low)
        if entry_type == "NONE":
            signal.reason = "No valid entry pattern (waiting for pullback or breakout)"
            return signal

        signal.entry_type = entry_type

        # Step 4: Calculate trade parameters via risk manager
        trade_params = self.risk.calculate_trade(
            direction=score_result.direction,
            entry_price=snap.current_price,
            atr=snap.atr,
        )

        if not trade_params.valid:
            signal.reason = f"Risk check failed (R:R={trade_params.risk_reward_ratio:.2f})"
            return signal

        # All checks passed — valid signal
        signal.has_signal = True
        signal.trade_params = trade_params
        signal.reason = (f"{entry_type} entry: {score_result.direction} "
                         f"score={score_result.total_score:.2f}")

        return signal

    def _identify_entry_type(self, snap: IndicatorSnapshot, direction: str,
                             session_high: Optional[float],
                             session_low: Optional[float]) -> str:
        """
        Determine if price is at a valid entry point:
        - PULLBACK: Price has pulled back to EMA20 zone
        - BREAKOUT: Price has broken session high/low and is retesting
        """
        price = snap.current_price
        pullback_zone = snap.atr * self._pullback_zone_pct

        if direction == "BUY":
            # Pullback: price near EMA fast (within pullback zone above it)
            distance_to_ema = price - snap.ema_fast
            if 0 <= distance_to_ema <= pullback_zone:
                return "PULLBACK"

            # Breakout: price just broke above session high
            if session_high and price > session_high:
                retest_distance = price - session_high
                if retest_distance <= pullback_zone:
                    return "BREAKOUT"

        elif direction == "SELL":
            # Pullback: price near EMA fast (within pullback zone below it)
            distance_to_ema = snap.ema_fast - price
            if 0 <= distance_to_ema <= pullback_zone:
                return "PULLBACK"

            # Breakout: price just broke below session low
            if session_low and price < session_low:
                retest_distance = session_low - price
                if retest_distance <= pullback_zone:
                    return "BREAKOUT"

        return "NONE"

    def _check_structure_break(self, snap: IndicatorSnapshot,
                               session_high: Optional[float],
                               session_low: Optional[float],
                               prev_session_high: Optional[float],
                               prev_session_low: Optional[float]) -> bool:
        """Check if price has broken a significant structural level."""
        price = snap.current_price

        # Break of current session high/low
        if session_high and price > session_high:
            return True
        if session_low and price < session_low:
            return True

        # Break of previous session high/low (stronger signal)
        if prev_session_high and price > prev_session_high:
            return True
        if prev_session_low and price < prev_session_low:
            return True

        return False

    # ------------------------------------------------------------------
    # POSITION MANAGEMENT (while in POSITION_ACTIVE state)
    # ------------------------------------------------------------------

    def should_exit(self, direction: str, entry_price: float,
                    current_price: float, current_sl: float,
                    current_tp: float, snap: IndicatorSnapshot,
                    near_session_end: bool = False) -> tuple[bool, str]:
        """
        Check if an open trend position should be exited.

        Returns:
            (should_exit: bool, reason: str)
        """
        # Check hard stop loss hit
        if direction == "BUY" and current_price <= current_sl:
            return True, "STOP_LOSS"
        if direction == "SELL" and current_price >= current_sl:
            return True, "STOP_LOSS"

        # Check take profit hit
        if direction == "BUY" and current_price >= current_tp:
            return True, "TAKE_PROFIT"
        if direction == "SELL" and current_price <= current_tp:
            return True, "TAKE_PROFIT"

        # Trend reversal: EMA fast crosses against direction
        if direction == "BUY" and snap.ema_fast < snap.ema_slow:
            # Only exit if we're in profit (don't exit on noise)
            if current_price > entry_price:
                return True, "TREND_REVERSAL"
        if direction == "SELL" and snap.ema_fast > snap.ema_slow:
            if current_price < entry_price:
                return True, "TREND_REVERSAL"

        # Session end with open profit — take what we have
        if near_session_end:
            if direction == "BUY" and current_price > entry_price:
                return True, "SESSION_END_PROFIT"
            if direction == "SELL" and current_price < entry_price:
                return True, "SESSION_END_PROFIT"

        return False, ""
