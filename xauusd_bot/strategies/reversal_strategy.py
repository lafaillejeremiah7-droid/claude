"""
XAUUSD Reversal Strategy — Win Rate Maximized

Philosophy (from the video):
    - ONE trade type: Reversals at range boundaries in range-bound conditions
    - 1:1 Risk Reward (or less) — accept smaller wins for MUCH higher win rate
    - Set and Forget — no trailing, no managing. TP or SL, done.
    - Condition identification is the EDGE, not the pattern itself
    - Win rate > 70% = low variance = consistent = bigger position sizing

The Setup:
    1. CONDITION: Market is range-bound (ADX < 20, clear support/resistance)
    2. TIMING: 30 minutes into the hour (based on data showing highest WR at reversals)
    3. ENTRY: Price hits range boundary + reversal confirmation (RSI extreme + rejection candle)
    4. SL: Just beyond the range boundary (tight)
    5. TP: 1:1 from entry (middle of range or equal distance to SL)
    6. SET AND FORGET: No management, no trailing, no early exit

Why This Works for XAUUSD:
    - Gold ranges 60%+ of the time (Asian + consolidation phases)
    - Range boundaries on gold are STRONG (institutional levels, round numbers)
    - 1:1 TP means price only needs to go a small distance in your favor
    - High win rate = low drawdown = can risk 2-3% per trade safely
    - Fewer trades (quality > quantity) = less exposure to news/spikes
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ..indicators.technical import IndicatorSnapshot


@dataclass
class ReversalSignal:
    """Output of reversal strategy evaluation."""
    has_signal: bool = False
    direction: str = "NONE"       # "BUY" or "SELL"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.0
    risk_pips: float = 0.0
    confidence: float = 0.0       # 0-1, how many confluences align
    condition_quality: float = 0.0  # How good is the range condition
    reason: str = ""
    confluences: dict = None

    def __post_init__(self):
        if self.confluences is None:
            self.confluences = {}


class ReversalStrategy:
    """
    High Win Rate Reversal Strategy for XAUUSD.

    Core Logic:
        1. Is the market RANGING? (condition identification = the real edge)
        2. Is price at a BOUNDARY? (support or resistance)
        3. Is there a REJECTION? (candle pattern + RSI extreme)
        4. If yes to all 3 → ENTER with 1:1 RR, set and forget

    This is deliberately simple. The edge is in WHEN you trade (condition),
    not in some complex indicator combination.
    """

    def __init__(self, config: dict):
        self.cfg = config

        # Range identification thresholds
        self.adx_range_max = config.get("adx_range_max", 20)
        self.bb_squeeze_threshold = config.get("bb_squeeze_threshold", 0.8)
        self.min_range_bars = config.get("min_range_bars", 10)

        # Entry thresholds
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.boundary_proximity_pct = config.get("boundary_proximity_pct", 0.15)

        # Risk management
        self.risk_reward = config.get("risk_reward", 1.0)  # Fixed 1:1
        self.sl_buffer_pips = config.get("sl_buffer_pips", 5)  # Buffer beyond boundary
        self.max_sl_pips = config.get("max_sl_pips", 200)  # Max stop size
        self.min_sl_pips = config.get("min_sl_pips", 30)   # Min stop size

        # Confidence thresholds
        self.min_confluences = config.get("min_confluences", 3)  # Need 3 of 5 to trade

    def evaluate(self, snap: IndicatorSnapshot,
                 current_bar_open: float,
                 previous_bar_high: float,
                 previous_bar_low: float,
                 previous_bar_close: float,
                 previous_bar_open: float,
                 minute_of_hour: int = 0,
                 range_high: Optional[float] = None,
                 range_low: Optional[float] = None) -> ReversalSignal:
        """
        Evaluate current conditions for a reversal entry.

        The logic is deliberately simple — the edge is condition identification,
        not complex signal generation.
        """
        signal = ReversalSignal()
        confluences = {}

        # ================================================================
        # STEP 1: Is the market RANGING? (The most important check)
        # ================================================================
        range_quality = self._assess_range_condition(snap)
        confluences["range_condition"] = range_quality > 0.5
        signal.condition_quality = range_quality

        if range_quality < 0.3:
            signal.reason = f"Not ranging (quality={range_quality:.2f}, ADX={snap.adx:.1f})"
            signal.confluences = confluences
            return signal

        # ================================================================
        # STEP 2: Is price at a BOUNDARY?
        # ================================================================
        at_support, at_resistance = self._check_boundary(
            snap, range_high, range_low
        )

        if not at_support and not at_resistance:
            signal.reason = "Price not at range boundary"
            signal.confluences = confluences
            return signal

        direction = "BUY" if at_support else "SELL"
        signal.direction = direction
        confluences["at_boundary"] = True

        # ================================================================
        # STEP 3: RSI Extreme (confirmation of exhaustion)
        # ================================================================
        rsi_confirms = self._check_rsi_extreme(snap, direction)
        confluences["rsi_extreme"] = rsi_confirms

        # ================================================================
        # STEP 4: Rejection Candle (shows buyers/sellers stepping in)
        # ================================================================
        rejection = self._check_rejection_candle(
            direction, snap.current_price, current_bar_open,
            previous_bar_high, previous_bar_low,
            previous_bar_close, previous_bar_open
        )
        confluences["rejection_candle"] = rejection

        # ================================================================
        # STEP 5: Stochastic Extreme (bonus confluence)
        # ================================================================
        stoch_confirms = self._check_stochastic(snap, direction)
        confluences["stochastic_extreme"] = stoch_confirms

        # ================================================================
        # STEP 6: Timing (30 min into hour = highest WR zone)
        # ================================================================
        timing_good = 25 <= minute_of_hour <= 45
        confluences["timing_optimal"] = timing_good

        # ================================================================
        # COUNT CONFLUENCES — Need minimum to trade
        # ================================================================
        active_confluences = sum(1 for v in confluences.values() if v is True)
        signal.confidence = active_confluences / len(confluences)
        signal.confluences = confluences

        if active_confluences < self.min_confluences:
            signal.reason = (f"Only {active_confluences}/{self.min_confluences} "
                             f"confluences ({confluences})")
            return signal

        # ================================================================
        # STEP 7: Calculate FIXED 1:1 trade parameters
        # ================================================================
        trade = self._calculate_trade(snap, direction, range_high, range_low)

        if trade is None:
            signal.reason = "Trade params invalid (SL too wide or too tight)"
            return signal

        # ALL CHECKS PASSED — VALID SIGNAL
        signal.has_signal = True
        signal.entry_price = trade["entry"]
        signal.stop_loss = trade["sl"]
        signal.take_profit = trade["tp"]
        signal.risk_pips = trade["risk_pips"]
        signal.reason = (f"REVERSAL {direction} | {active_confluences} confluences | "
                         f"RR=1:{self.risk_reward} | SL={trade['risk_pips']:.0f}pips")

        return signal

    # ==================================================================
    # CONDITION IDENTIFICATION (This is THE edge)
    # ==================================================================

    def _assess_range_condition(self, snap: IndicatorSnapshot) -> float:
        """
        Score how clearly the market is ranging (0.0 = trending, 1.0 = perfect range).
        This is the MOST IMPORTANT function — condition ID is the real edge.
        """
        score = 0.0

        # ADX below threshold = ranging
        if snap.adx <= 15:
            score += 0.4  # Very clearly ranging
        elif snap.adx <= 20:
            score += 0.3
        elif snap.adx <= 25:
            score += 0.1
        else:
            return 0.0  # ADX > 25 = trending, DO NOT TRADE

        # Bollinger Band width relative to average (tight = ranging)
        if snap.bb_bandwidth_avg > 0:
            bw_ratio = snap.bb_bandwidth / snap.bb_bandwidth_avg
            if bw_ratio < 0.5:
                score += 0.3  # Very tight squeeze
            elif bw_ratio < 0.8:
                score += 0.2
            elif bw_ratio < 1.0:
                score += 0.1

        # DI+ and DI- close together (no dominant direction)
        di_diff = abs(snap.di_plus - snap.di_minus)
        if di_diff < 5:
            score += 0.3  # No dominant direction at all
        elif di_diff < 10:
            score += 0.2
        elif di_diff < 15:
            score += 0.1

        return min(score, 1.0)

    def _check_boundary(self, snap: IndicatorSnapshot,
                        range_high: Optional[float],
                        range_low: Optional[float]) -> tuple[bool, bool]:
        """Check if price is at support or resistance boundary."""
        price = snap.current_price

        # Use Bollinger Bands as dynamic range boundaries
        bb_range = snap.bb_upper - snap.bb_lower
        if bb_range <= 0:
            return False, False

        # At support: price within X% of lower boundary
        distance_from_lower = (price - snap.bb_lower) / bb_range
        at_support = distance_from_lower <= self.boundary_proximity_pct

        # At resistance: price within X% of upper boundary
        distance_from_upper = (snap.bb_upper - price) / bb_range
        at_resistance = distance_from_upper <= self.boundary_proximity_pct

        # Also check explicit range levels if provided
        if range_low and not at_support:
            if abs(price - range_low) <= bb_range * self.boundary_proximity_pct:
                at_support = True
        if range_high and not at_resistance:
            if abs(price - range_high) <= bb_range * self.boundary_proximity_pct:
                at_resistance = True

        return at_support, at_resistance

    def _check_rsi_extreme(self, snap: IndicatorSnapshot, direction: str) -> bool:
        """Check if RSI is at an extreme level confirming exhaustion."""
        if direction == "BUY":
            return snap.rsi <= self.rsi_oversold
        elif direction == "SELL":
            return snap.rsi >= self.rsi_overbought
        return False

    def _check_rejection_candle(self, direction: str, current_close: float,
                                current_open: float, prev_high: float,
                                prev_low: float, prev_close: float,
                                prev_open: float) -> bool:
        """
        Check for a rejection candle pattern (pin bar / engulfing).
        Shows that buyers/sellers are stepping in at the boundary.
        """
        prev_body = abs(prev_close - prev_open)
        prev_range = prev_high - prev_low

        if prev_range == 0:
            return False

        if direction == "BUY":
            # Bullish rejection: long lower wick (buyers rejecting lower prices)
            lower_wick = min(prev_open, prev_close) - prev_low
            wick_ratio = lower_wick / prev_range
            if wick_ratio >= 0.5:  # Lower wick is 50%+ of candle range
                return True
            # Or: current bar closing above previous bar high (engulfing)
            if current_close > prev_high and current_close > current_open:
                return True

        elif direction == "SELL":
            # Bearish rejection: long upper wick (sellers rejecting higher prices)
            upper_wick = prev_high - max(prev_open, prev_close)
            wick_ratio = upper_wick / prev_range
            if wick_ratio >= 0.5:  # Upper wick is 50%+ of candle range
                return True
            # Or: current bar closing below previous bar low (engulfing)
            if current_close < prev_low and current_close < current_open:
                return True

        return False

    def _check_stochastic(self, snap: IndicatorSnapshot, direction: str) -> bool:
        """Stochastic at extreme as bonus confirmation."""
        if direction == "BUY":
            return snap.stoch_k <= 20 or snap.stoch_d <= 20
        elif direction == "SELL":
            return snap.stoch_k >= 80 or snap.stoch_d >= 80
        return False

    # ==================================================================
    # TRADE CALCULATION — Fixed 1:1, Set and Forget
    # ==================================================================

    def _calculate_trade(self, snap: IndicatorSnapshot, direction: str,
                         range_high: Optional[float],
                         range_low: Optional[float]) -> Optional[dict]:
        """
        Calculate trade parameters with FIXED 1:1 risk reward.
        SL just beyond the range boundary.
        TP = equal distance from entry (1:1).
        NO trailing. NO management. Set and forget.
        """
        price = snap.current_price
        buffer = self.sl_buffer_pips * 0.01  # Convert pips to price

        if direction == "BUY":
            # SL below the lower boundary (or current low + buffer)
            sl_level = snap.bb_lower - buffer
            if range_low:
                sl_level = min(sl_level, range_low - buffer)

            risk_distance = price - sl_level
            tp_distance = risk_distance * self.risk_reward
            tp_level = price + tp_distance

        elif direction == "SELL":
            # SL above the upper boundary (or current high + buffer)
            sl_level = snap.bb_upper + buffer
            if range_high:
                sl_level = max(sl_level, range_high + buffer)

            risk_distance = sl_level - price
            tp_distance = risk_distance * self.risk_reward
            tp_level = price - tp_distance

        else:
            return None

        # Validate SL size
        risk_pips = risk_distance / 0.01
        if risk_pips > self.max_sl_pips or risk_pips < self.min_sl_pips:
            return None

        return {
            "entry": price,
            "sl": round(sl_level, 2),
            "tp": round(tp_level, 2),
            "risk_pips": risk_pips,
            "risk_distance": risk_distance,
        }

    # ==================================================================
    # EXIT — Set and Forget (NO management needed)
    # ==================================================================

    def check_exit(self, direction: str, current_price: float,
                   stop_loss: float, take_profit: float) -> tuple[bool, str]:
        """
        Simple exit check: did price hit SL or TP?
        NO trailing. NO early exit. NO management.
        Set and forget = true edge execution.
        """
        if direction == "BUY":
            if current_price <= stop_loss:
                return True, "STOP_LOSS"
            if current_price >= take_profit:
                return True, "TAKE_PROFIT"

        elif direction == "SELL":
            if current_price >= stop_loss:
                return True, "STOP_LOSS"
            if current_price <= take_profit:
                return True, "TAKE_PROFIT"

        return False, ""
