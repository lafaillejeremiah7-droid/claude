"""
XAUUSD Adaptive Engine - 6 Adaptive Systems

Systems:
1. Adaptive R:R Based on Market Condition
2. Adaptive Position Size Based on Winning Streaks (Kelly Criterion Lite)
3. Tight SL Reality (Partial Losses)
4. Early Exit Based on Probability
5. X/Twitter Sentiment Scanner Integration
6. Influencer Flow Tracking Integration

These systems work together to dynamically adjust risk, reward, and trade
management based on real-time market conditions and external signals.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta


@dataclass
class MarketCondition:
    """Assessment of current market condition for adaptive R:R."""
    quality: str = "OKAY"        # "GOOD", "OKAY", "CHOPPY"
    score: float = 0.5           # 0.0 = worst, 1.0 = best
    range_clarity: float = 0.0   # How clean are the boundaries
    volatility_state: str = "NORMAL"  # "LOW", "NORMAL", "HIGH"
    adx_value: float = 0.0
    bb_squeeze: float = 0.0
    reason: str = ""


@dataclass
class AdaptiveRR:
    """Adaptive risk-reward output."""
    rr_ratio: float = 1.0
    condition: str = "OKAY"
    tp_multiplier: float = 1.0
    reason: str = ""


@dataclass
class AdaptiveSize:
    """Adaptive position sizing output."""
    risk_pct: float = 0.02       # Default 2%
    streak_state: str = "NEUTRAL"  # "HOT", "NEUTRAL", "COLD"
    kelly_factor: float = 1.0
    reason: str = ""


@dataclass
class EarlyExitSignal:
    """Early exit recommendation."""
    should_exit: bool = False
    exit_type: str = ""          # "PARTIAL_PROFIT", "EARLY_LOSS", ""
    probability_shift: float = 0.0
    reason: str = ""


@dataclass
class SentimentBias:
    """Sentiment analysis output."""
    bias: str = "NEUTRAL"        # "BULLISH", "BEARISH", "NEUTRAL", "DANGER"
    strength: float = 0.0        # 0-1
    tp_stretch: float = 1.0      # Multiplier for TP (1.0 = no change)
    size_modifier: float = 1.0   # Multiplier for position size
    should_avoid: bool = False   # DANGER signal = don't trade
    sources: list = field(default_factory=list)
    reason: str = ""


class AdaptiveEngine:
    """
    Master adaptive engine combining all 6 systems.

    Usage:
        engine = AdaptiveEngine(config)
        # On each bar:
        condition = engine.assess_market_condition(snap)
        rr = engine.get_adaptive_rr(condition)
        size = engine.get_adaptive_size(recent_results)
        # During trade:
        exit_signal = engine.check_early_exit(position, snap)
    """

    def __init__(self, config: dict):
        self.cfg = config

        # System 1: Adaptive R:R parameters
        self.rr_good = config.get("rr_good", 1.35)
        self.rr_okay = config.get("rr_okay", 1.0)
        self.rr_choppy = config.get("rr_choppy", 0.75)
        self.good_threshold = config.get("good_condition_threshold", 0.7)
        self.choppy_threshold = config.get("choppy_condition_threshold", 0.35)

        # System 2: Adaptive sizing parameters
        self.base_risk = config.get("base_risk_pct", 0.02)
        self.max_risk = config.get("max_risk_pct", 0.03)
        self.min_risk = config.get("min_risk_pct", 0.01)
        self.streak_scale_up = config.get("streak_scale_up", 0.005)
        self.streak_scale_down = config.get("streak_scale_down", 0.005)

        # System 3: Tight SL parameters
        self.tight_sl_factor = config.get("tight_sl_factor", 0.6)
        self.partial_loss_min = config.get("partial_loss_min", 0.3)
        self.partial_loss_max = config.get("partial_loss_max", 0.7)

        # System 4: Early exit parameters
        self.adx_breakout_threshold = config.get("adx_breakout_threshold", 25)
        self.momentum_shift_threshold = config.get("momentum_shift_threshold", 0.6)
        self.min_profit_for_early_exit = config.get("min_profit_for_early_exit", 0.3)
        self.max_loss_for_early_cut = config.get("max_loss_for_early_cut", 0.5)

        # Tracking state
        self._recent_results: list[str] = []  # "WIN" or "LOSS"
        self._daily_results: list[str] = []
        self._current_date: str = ""
        self._win_streak: int = 0
        self._loss_streak: int = 0

    # ==================================================================
    # SYSTEM 1: Adaptive R:R Based on Market Condition
    # ==================================================================

    def assess_market_condition(self, adx: float, bb_bandwidth: float,
                                bb_bandwidth_avg: float, di_plus: float,
                                di_minus: float, atr: float,
                                atr_avg: float) -> MarketCondition:
        """
        Assess market condition quality for adaptive R:R.

        GOOD condition (1.2-1.5 R:R):
            - Strong range (ADX < 15)
            - Clean boundaries (BB tight, DI balanced)
            - Low volatility relative to average

        OKAY condition (1:1 R:R):
            - Moderate range (ADX 15-20)
            - Boundaries somewhat clear

        CHOPPY condition (0.7-0.8 R:R):
            - Weak range (ADX 20-25)
            - Boundaries unclear
            - High relative volatility
        """
        condition = MarketCondition()
        condition.adx_value = adx
        score = 0.0

        # ADX scoring (lower = better for reversal)
        if adx <= 12:
            score += 0.35
        elif adx <= 15:
            score += 0.30
        elif adx <= 18:
            score += 0.20
        elif adx <= 22:
            score += 0.10
        else:
            score += 0.0

        # BB squeeze scoring (tighter = cleaner range)
        if bb_bandwidth_avg > 0:
            bw_ratio = bb_bandwidth / bb_bandwidth_avg
            condition.bb_squeeze = bw_ratio
            if bw_ratio < 0.5:
                score += 0.25
            elif bw_ratio < 0.7:
                score += 0.20
            elif bw_ratio < 0.9:
                score += 0.12
            elif bw_ratio < 1.1:
                score += 0.05
            else:
                score += 0.0

        # DI balance (closer = no dominant direction)
        di_diff = abs(di_plus - di_minus)
        if di_diff < 3:
            score += 0.20
            condition.range_clarity = 0.9
        elif di_diff < 7:
            score += 0.15
            condition.range_clarity = 0.7
        elif di_diff < 12:
            score += 0.08
            condition.range_clarity = 0.4
        else:
            score += 0.0
            condition.range_clarity = 0.2

        # Volatility state (ATR vs average)
        if atr_avg > 0:
            vol_ratio = atr / atr_avg
            if vol_ratio < 0.7:
                condition.volatility_state = "LOW"
                score += 0.20
            elif vol_ratio < 1.2:
                condition.volatility_state = "NORMAL"
                score += 0.10
            else:
                condition.volatility_state = "HIGH"
                score += 0.0

        # Classify condition
        condition.score = min(score, 1.0)

        if score >= self.good_threshold:
            condition.quality = "GOOD"
            condition.reason = f"Strong range (ADX={adx:.1f}, DI_diff={di_diff:.1f})"
        elif score >= self.choppy_threshold:
            condition.quality = "OKAY"
            condition.reason = f"Moderate range (ADX={adx:.1f})"
        else:
            condition.quality = "CHOPPY"
            condition.reason = f"Weak/unclear range (ADX={adx:.1f}, vol={condition.volatility_state})"

        return condition

    def get_adaptive_rr(self, condition: MarketCondition) -> AdaptiveRR:
        """
        Get adaptive R:R based on market condition.

        Good condition -> Target 1.2-1.5 R:R (stretch TP)
        Okay condition -> Stay at 1:1
        Choppy condition -> Target 0.7-0.8 R:R (take quick profit)
        """
        result = AdaptiveRR()
        result.condition = condition.quality

        if condition.quality == "GOOD":
            # Scale between 1.2 and 1.5 based on score
            scale = (condition.score - self.good_threshold) / (1.0 - self.good_threshold)
            result.rr_ratio = 1.2 + scale * 0.3
            result.tp_multiplier = result.rr_ratio
            result.reason = f"GOOD condition -> stretched TP to {result.rr_ratio:.2f}R"

        elif condition.quality == "CHOPPY":
            # Scale between 0.7 and 0.8
            scale = condition.score / self.choppy_threshold
            result.rr_ratio = 0.7 + scale * 0.1
            result.tp_multiplier = result.rr_ratio
            result.reason = f"CHOPPY condition -> quick TP at {result.rr_ratio:.2f}R"

        else:  # OKAY
            result.rr_ratio = self.rr_okay
            result.tp_multiplier = 1.0
            result.reason = "OKAY condition -> standard 1:1 R:R"

        return result

    # ==================================================================
    # SYSTEM 2: Adaptive Position Size Based on Winning Streaks
    # ==================================================================

    def record_trade_result(self, result: str, timestamp: datetime):
        """Record a trade result for streak tracking."""
        current_date = timestamp.strftime("%Y-%m-%d")

        # Reset daily on new day
        if current_date != self._current_date:
            self._current_date = current_date
            self._daily_results = []

        self._recent_results.append(result)
        self._daily_results.append(result)

        # Keep last 20 results
        if len(self._recent_results) > 20:
            self._recent_results = self._recent_results[-20:]

        # Update streaks
        if result == "WIN":
            self._win_streak += 1
            self._loss_streak = 0
        else:
            self._loss_streak += 1
            self._win_streak = 0

    def get_adaptive_size(self) -> AdaptiveSize:
        """
        Get adaptive position size based on recent performance.

        Winning streak -> Scale UP: 2% -> 2.5% -> 3%
        Losing streak -> Scale DOWN: 2% -> 1.5% -> 1%

        Uses Kelly Criterion lite: f = (bp - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        result = AdaptiveSize()

        if len(self._recent_results) < 3:
            result.risk_pct = self.base_risk
            result.reason = "Insufficient history, using base risk"
            return result

        # Calculate recent win rate
        recent = self._recent_results[-10:]
        wins = sum(1 for r in recent if r == "WIN")
        win_rate = wins / len(recent)

        # Kelly Criterion lite (simplified)
        # f = win_rate - (1 - win_rate) / avg_payoff_ratio
        # For 1:1 RR, avg_payoff = 1.0, so f = 2*win_rate - 1
        kelly = 2.0 * win_rate - 1.0
        kelly = max(0.0, min(kelly, 0.5))  # Cap between 0 and 0.5
        result.kelly_factor = kelly

        # Streak-based adjustment
        if self._win_streak >= 3:
            # Hot streak - scale up
            scale_up = min(self._win_streak - 2, 4) * self.streak_scale_up
            result.risk_pct = min(self.base_risk + scale_up, self.max_risk)
            result.streak_state = "HOT"
            result.reason = (f"Win streak {self._win_streak} -> "
                           f"risk {result.risk_pct*100:.1f}%")

        elif self._loss_streak >= 2:
            # Cold streak - scale down
            scale_down = min(self._loss_streak - 1, 4) * self.streak_scale_down
            result.risk_pct = max(self.base_risk - scale_down, self.min_risk)
            result.streak_state = "COLD"
            result.reason = (f"Loss streak {self._loss_streak} -> "
                           f"risk {result.risk_pct*100:.1f}%")

        else:
            # Neutral - use Kelly-adjusted base
            kelly_adj = self.base_risk * (0.8 + kelly * 0.4)
            result.risk_pct = max(self.min_risk, min(kelly_adj, self.max_risk))
            result.streak_state = "NEUTRAL"
            result.reason = f"Neutral (Kelly={kelly:.2f}) -> risk {result.risk_pct*100:.1f}%"

        # Daily P&L guardrail: if 2+ daily losses, reduce further
        daily_losses = sum(1 for r in self._daily_results if r == "LOSS")
        if daily_losses >= 2:
            result.risk_pct = self.min_risk
            result.reason = f"Daily loss limit ({daily_losses} losses) -> min risk"

        return result

    # ==================================================================
    # SYSTEM 3: Tight SL Reality (Partial Losses)
    # ==================================================================

    def calculate_actual_risk(self, entry_price: float, boundary_price: float,
                              direction: str, atr: float) -> dict:
        """
        Calculate ACTUAL risk distance from entry to boundary.

        Real losses are often 0.3-0.5x what the "full stop" would be because:
        - Price rarely goes from entry straight to SL without bouncing
        - Tight SL at actual boundary (not arbitrary distance)
        - ATR-based reality check

        Returns dict with:
            - actual_sl: The real SL level
            - actual_risk_pips: Actual risk in pips
            - risk_factor: What fraction of "full stop" this represents (0.3-0.7)
            - reason: Explanation
        """
        pip_size = 0.01

        if direction == "BUY":
            # Distance from entry to lower boundary
            raw_distance = entry_price - boundary_price
        else:
            # Distance from entry to upper boundary
            raw_distance = boundary_price - entry_price

        raw_risk_pips = raw_distance / pip_size

        # ATR-based reality: actual risk is often tighter than max SL
        # If entry is very close to boundary, the real risk is small
        atr_pips = atr / pip_size
        tight_factor = min(raw_risk_pips / atr_pips, 1.0) if atr_pips > 0 else 1.0

        # Apply tight SL factor (real losses are 0.3-0.7x of full stop)
        actual_factor = self.partial_loss_min + (
            (self.partial_loss_max - self.partial_loss_min) * tight_factor
        )

        # The actual SL is placed at boundary + small buffer
        buffer_pips = 3.0  # Tight buffer
        if direction == "BUY":
            actual_sl = boundary_price - (buffer_pips * pip_size)
            actual_risk_pips = (entry_price - actual_sl) / pip_size
        else:
            actual_sl = boundary_price + (buffer_pips * pip_size)
            actual_risk_pips = (actual_sl - entry_price) / pip_size

        return {
            "actual_sl": actual_sl,
            "actual_risk_pips": actual_risk_pips,
            "risk_factor": actual_factor,
            "full_risk_pips": raw_risk_pips,
            "reason": (f"Tight SL at boundary: {actual_risk_pips:.0f} pips "
                      f"(factor={actual_factor:.2f}, full would be {raw_risk_pips:.0f})")
        }

    # ==================================================================
    # SYSTEM 4: Early Exit Based on Probability
    # ==================================================================

    def check_early_exit(self, direction: str, entry_price: float,
                         current_price: float, stop_loss: float,
                         take_profit: float, adx: float,
                         di_plus: float, di_minus: float,
                         rsi: float, prev_adx: float = 0.0) -> EarlyExitSignal:
        """
        Check if probability of continuation has shifted enough to exit early.

        In profit: If range is breaking (ADX rising, momentum shifting) -> take partial profit
        In loss: If recovery probability is low -> cut early with small loss

        Returns EarlyExitSignal with recommendation.
        """
        signal = EarlyExitSignal()
        pip_size = 0.01

        # Calculate current P&L position
        if direction == "BUY":
            pnl_pips = (current_price - entry_price) / pip_size
            risk_pips = (entry_price - stop_loss) / pip_size
            reward_pips = (take_profit - entry_price) / pip_size
        else:
            pnl_pips = (entry_price - current_price) / pip_size
            risk_pips = (stop_loss - entry_price) / pip_size
            reward_pips = (entry_price - take_profit) / pip_size

        if risk_pips <= 0:
            return signal

        pnl_ratio = pnl_pips / risk_pips  # How much of risk have we made/lost

        # --- IN PROFIT: Check if should exit early ---
        if pnl_ratio >= self.min_profit_for_early_exit:
            # ADX rising = range breaking = our reversal thesis is weakening
            adx_rising = adx > prev_adx + 3 if prev_adx > 0 else adx > self.adx_breakout_threshold

            # Momentum shifting against us
            if direction == "BUY":
                momentum_against = di_minus > di_plus + 5
            else:
                momentum_against = di_plus > di_minus + 5

            # RSI no longer at extreme (moved to middle = losing edge)
            rsi_neutral = 40 <= rsi <= 60

            # Count probability shift signals
            shift_signals = sum([adx_rising, momentum_against, rsi_neutral])

            if shift_signals >= 2:
                signal.should_exit = True
                signal.exit_type = "PARTIAL_PROFIT"
                signal.probability_shift = shift_signals / 3.0
                signal.reason = (f"Take profit early at {pnl_ratio:.1f}R "
                               f"(ADX_rising={adx_rising}, momentum_against={momentum_against})")

        # --- IN LOSS: Check if should cut early ---
        elif pnl_ratio < 0 and abs(pnl_ratio) >= self.max_loss_for_early_cut * 0.5:
            # If already lost 25%+ of risk AND conditions deteriorating
            adx_breaking = adx > self.adx_breakout_threshold + 5

            # Strong momentum against position
            if direction == "BUY":
                strong_against = di_minus > di_plus + 10
            else:
                strong_against = di_plus > di_minus + 10

            if adx_breaking and strong_against:
                signal.should_exit = True
                signal.exit_type = "EARLY_LOSS"
                signal.probability_shift = 0.8
                signal.reason = (f"Cut loss early at {pnl_ratio:.1f}R "
                               f"(range breaking, strong momentum against)")

        return signal

    # ==================================================================
    # SYSTEM 5 & 6: Sentiment & Influencer Integration Points
    # ==================================================================

    def apply_sentiment_bias(self, base_rr: float, base_size: float,
                             sentiment: SentimentBias) -> tuple[float, float, bool]:
        """
        Apply sentiment bias to trade parameters.

        Returns: (adjusted_rr, adjusted_size, should_avoid)
        """
        if sentiment.should_avoid:
            return base_rr, base_size, True

        adjusted_rr = base_rr * sentiment.tp_stretch
        adjusted_size = base_size * sentiment.size_modifier

        # Cap adjustments
        adjusted_rr = max(0.5, min(adjusted_rr, 2.0))
        adjusted_size = max(self.min_risk, min(adjusted_size, self.max_risk))

        return adjusted_rr, adjusted_size, False

    def get_combined_adjustment(self, condition: MarketCondition,
                                sentiment: SentimentBias) -> dict:
        """
        Get combined adaptive adjustments from all systems.
        """
        rr = self.get_adaptive_rr(condition)
        size = self.get_adaptive_size()

        # Apply sentiment
        final_rr, final_size, avoid = self.apply_sentiment_bias(
            rr.rr_ratio, size.risk_pct, sentiment
        )

        return {
            "rr_ratio": final_rr,
            "risk_pct": final_size,
            "condition": condition.quality,
            "streak_state": size.streak_state,
            "sentiment_bias": sentiment.bias,
            "should_avoid": avoid,
            "rr_reason": rr.reason,
            "size_reason": size.reason,
            "sentiment_reason": sentiment.reason,
        }
