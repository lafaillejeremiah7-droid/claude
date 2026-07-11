"""
XAUUSD Signal Bot V3 - Adaptive Engine

Built on V2 (Win Rate Maximized) + 6 Adaptive Systems:
1. Adaptive R:R Based on Market Condition
2. Adaptive Position Size Based on Winning Streaks
3. Tight SL Reality (Partial Losses)
4. Early Exit Based on Probability
5. X/Twitter Sentiment Scanner
6. Influencer Flow Tracking

Key improvements over V2:
- Dynamic R:R (0.75-1.5x) based on market condition quality
- Kelly Criterion lite position sizing (1%-3%)
- Early exits when probability shifts (not just set and forget)
- Sentiment-aware bias and TP stretching
- 6 trades/day with NY session added
- 30min loss cooldown / 10min win cooldown (optimized)
"""

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Callable

from .config_v2 import (
    STRATEGY, RISK_V3, SESSIONS_V3, INDICATORS_V2,
    NEWS_V2, COOLDOWN_V3, ADAPTIVE_RR, ADAPTIVE_SIZE,
    TIGHT_SL, EARLY_EXIT, SENTIMENT, INFLUENCER_FLOW,
)
from .indicators.technical import TechnicalIndicators, IndicatorSnapshot
from .strategies.reversal_strategy import ReversalStrategy, ReversalSignal
from .filters.news_filter import NewsFilter
from .filters.sentiment_scanner import SentimentScanner
from .core.adaptive_engine import (
    AdaptiveEngine, MarketCondition, AdaptiveRR,
    AdaptiveSize, EarlyExitSignal, SentimentBias,
)


@dataclass
class PositionV3:
    """An open position with adaptive parameters."""
    direction: str = "NONE"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.0
    entry_time: Optional[datetime] = None
    risk_pips: float = 0.0
    confluences: dict = None
    # V3 adaptive fields
    adaptive_rr: float = 1.0
    risk_pct_used: float = 0.02
    condition_quality: str = "OKAY"
    sentiment_bias: str = "NEUTRAL"
    prev_adx: float = 0.0         # For early exit detection


@dataclass
class SignalOutputV3:
    """Signal output for Telegram/Discord delivery."""
    pair: str = "XAUUSD"
    direction: str = ""
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: str = "1:1"
    confidence: str = ""
    condition: str = ""
    timestamp: Optional[datetime] = None
    # V3 fields
    adaptive_rr: float = 1.0
    market_condition: str = "OKAY"
    sentiment: str = "NEUTRAL"
    position_size_pct: float = 2.0


class XAUUSDBotV3:
    """
    Adaptive XAUUSD Signal Bot V3.

    Combines V2 reversal strategy with 6 adaptive systems for
    improved risk-adjusted returns.

    Usage:
        bot = XAUUSDBotV3(account_balance=10000.0)
        bot.initialize()
        result = bot.on_bar(timestamp, open, high, low, close)
    """

    def __init__(self, account_balance: float = 10000.0,
                 on_signal: Optional[Callable] = None,
                 on_close: Optional[Callable] = None):
        self.account_balance = account_balance
        self.equity = account_balance
        self._on_signal = on_signal
        self._on_close = on_close

        # Components
        self.indicators: Optional[TechnicalIndicators] = None
        self.strategy: Optional[ReversalStrategy] = None
        self.news_filter: Optional[NewsFilter] = None
        self.sentiment_scanner: Optional[SentimentScanner] = None
        self.adaptive_engine: Optional[AdaptiveEngine] = None

        # State
        self.position: Optional[PositionV3] = None
        self.daily_trades: int = 0
        self.daily_losses: int = 0
        self.daily_pnl: float = 0.0
        self.last_trade_date: str = ""
        self.cooldown_until: Optional[datetime] = None

        # Price history
        self._opens: list = []
        self._highs: list = []
        self._lows: list = []
        self._closes: list = []
        self._max_history = 100  # Reduced from 200 for performance
        self._bar_count: int = 0  # Total bars processed

        # Indicator state for early exit
        self._prev_adx: float = 0.0

        # Stats
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.equity_history: list = []
        self.trade_log: list = []

        # V3 stats
        self.early_exits_profit = 0
        self.early_exits_loss = 0
        self.sentiment_avoids = 0
        self.adaptive_rr_sum = 0.0
        self.adaptive_size_sum = 0.0

    def initialize(self):
        """Initialize bot components."""
        self.indicators = TechnicalIndicators(INDICATORS_V2)
        self.strategy = ReversalStrategy(STRATEGY)
        self.news_filter = NewsFilter(NEWS_V2)

        # V3: Adaptive systems
        adaptive_config = {}
        adaptive_config.update(ADAPTIVE_RR)
        adaptive_config.update(ADAPTIVE_SIZE)
        adaptive_config.update(TIGHT_SL)
        adaptive_config.update(EARLY_EXIT)
        self.adaptive_engine = AdaptiveEngine(adaptive_config)

        self.sentiment_scanner = SentimentScanner(SENTIMENT)

    def on_bar(self, timestamp: datetime,
               open_price: float, high: float, low: float, close: float,
               spread_pips: float = 15.0) -> Optional[SignalOutputV3]:
        """
        Process a new M15 bar. Returns a SignalOutputV3 if a trade is triggered.
        """
        # Update buffers
        self._opens.append(open_price)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        self._bar_count += 1

        if len(self._closes) > self._max_history:
            self._opens = self._opens[-self._max_history:]
            self._highs = self._highs[-self._max_history:]
            self._lows = self._lows[-self._max_history:]
            self._closes = self._closes[-self._max_history:]

        # Need minimum data
        if len(self._closes) < 50:
            return None

        # Daily reset
        current_date = timestamp.strftime("%Y-%m-%d")
        if current_date != self.last_trade_date:
            self.daily_trades = 0
            self.daily_losses = 0
            self.daily_pnl = 0.0
            self.last_trade_date = current_date

        # Track equity (sample every 4th bar to reduce memory)
        if self._bar_count % 4 == 0:
            self.equity_history.append((timestamp, self.equity))

        # ---- CHECK OPEN POSITION (fast path: no indicators needed for SL/TP) ----
        if self.position is not None:
            # Quick SL/TP check without full indicator computation
            exited, reason = self.strategy.check_exit(
                self.position.direction, close,
                self.position.stop_loss, self.position.take_profit
            )
            if exited:
                self._close_position(close, reason, timestamp)
                return None

            # Early exit check only every 4th bar for performance
            if self._bar_count % 4 == 0:
                highs_arr = np.array(self._highs)
                lows_arr = np.array(self._lows)
                closes_arr = np.array(self._closes)
                snap = self.indicators.compute_all(highs_arr, lows_arr, closes_arr)

                # Update sentiment
                self.sentiment_scanner.update(timestamp, close, high, low, snap.atr)

                # System 4: Early exit
                exit_signal = self.adaptive_engine.check_early_exit(
                    direction=self.position.direction,
                    entry_price=self.position.entry_price,
                    current_price=close,
                    stop_loss=self.position.stop_loss,
                    take_profit=self.position.take_profit,
                    adx=snap.adx,
                    di_plus=snap.di_plus,
                    di_minus=snap.di_minus,
                    rsi=snap.rsi,
                    prev_adx=self.position.prev_adx,
                )

                if exit_signal.should_exit:
                    if exit_signal.exit_type == "PARTIAL_PROFIT":
                        self.early_exits_profit += 1
                        self._close_position(close, "EARLY_PROFIT", timestamp)
                    elif exit_signal.exit_type == "EARLY_LOSS":
                        self.early_exits_loss += 1
                        self._close_position(close, "EARLY_LOSS", timestamp)

                if self.position is not None:
                    self.position.prev_adx = snap.adx

            return None

        # ---- PRE-CHECKS (fast: no indicator computation) ----
        if not self._should_trade(timestamp, spread_pips):
            return None

        # ---- COMPUTE INDICATORS (only when actively looking for trades) ----
        highs_arr = np.array(self._highs)
        lows_arr = np.array(self._lows)
        closes_arr = np.array(self._closes)
        snap = self.indicators.compute_all(highs_arr, lows_arr, closes_arr)

        # Update sentiment scanner
        self.sentiment_scanner.update(timestamp, close, high, low, snap.atr)

        # ---- CHECK SENTIMENT (System 5: avoid DANGER) ----
        sentiment = self.sentiment_scanner.get_sentiment(timestamp, "")
        if sentiment.should_avoid:
            self.sentiment_avoids += 1
            return None

        # ---- GET PREVIOUS BAR DATA ----
        if len(self._closes) < 3:
            return None

        prev_high = self._highs[-2]
        prev_low = self._lows[-2]
        prev_close = self._closes[-2]
        prev_open = self._opens[-2]
        minute_of_hour = timestamp.minute

        # ---- EVALUATE REVERSAL STRATEGY ----
        signal = self.strategy.evaluate(
            snap=snap,
            current_bar_open=open_price,
            previous_bar_high=prev_high,
            previous_bar_low=prev_low,
            previous_bar_close=prev_close,
            previous_bar_open=prev_open,
            minute_of_hour=minute_of_hour,
        )

        if not signal.has_signal:
            self._prev_adx = snap.adx
            return None

        # ---- APPLY ADAPTIVE SYSTEMS ----
        return self._open_adaptive_position(signal, timestamp, snap, sentiment)

    # ==================================================================
    # ADAPTIVE POSITION OPENING
    # ==================================================================

    def _open_adaptive_position(self, signal: ReversalSignal,
                                timestamp: datetime,
                                snap: IndicatorSnapshot,
                                sentiment: SentimentBias) -> Optional[SignalOutputV3]:
        """Open position with all adaptive systems applied."""

        # System 1: Adaptive R:R based on market condition
        condition = self.adaptive_engine.assess_market_condition(
            adx=snap.adx,
            bb_bandwidth=snap.bb_bandwidth,
            bb_bandwidth_avg=snap.bb_bandwidth_avg,
            di_plus=snap.di_plus,
            di_minus=snap.di_minus,
            atr=snap.atr,
            atr_avg=snap.atr_avg,
        )
        adaptive_rr = self.adaptive_engine.get_adaptive_rr(condition)

        # System 2: Adaptive position size
        adaptive_size = self.adaptive_engine.get_adaptive_size()

        # System 5: Apply sentiment bias to RR and size
        final_rr, final_size, should_avoid = self.adaptive_engine.apply_sentiment_bias(
            adaptive_rr.rr_ratio, adaptive_size.risk_pct, sentiment
        )

        if should_avoid:
            self.sentiment_avoids += 1
            return None

        # System 3: Tight SL reality
        if signal.direction == "BUY":
            boundary = snap.bb_lower
        else:
            boundary = snap.bb_upper

        tight_sl = self.adaptive_engine.calculate_actual_risk(
            entry_price=signal.entry_price,
            boundary_price=boundary,
            direction=signal.direction,
            atr=snap.atr,
        )

        # Recalculate TP with adaptive R:R
        actual_risk_distance = tight_sl["actual_risk_pips"] * 0.01
        tp_distance = actual_risk_distance * final_rr

        if signal.direction == "BUY":
            adjusted_tp = signal.entry_price + tp_distance
            adjusted_sl = tight_sl["actual_sl"]
        else:
            adjusted_tp = signal.entry_price - tp_distance
            adjusted_sl = tight_sl["actual_sl"]

        # Validate adjusted SL
        adjusted_risk_pips = tight_sl["actual_risk_pips"]
        if adjusted_risk_pips > 150 or adjusted_risk_pips < 15:
            return None

        # Calculate lot size with adaptive risk
        risk_amount = self.equity * final_size
        lot_size = risk_amount / (adjusted_risk_pips * 1.0)
        lot_size = round(max(lot_size, 0.01), 2)

        # Open position
        self.position = PositionV3(
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=adjusted_sl,
            take_profit=adjusted_tp,
            lot_size=lot_size,
            entry_time=timestamp,
            risk_pips=adjusted_risk_pips,
            confluences=signal.confluences,
            adaptive_rr=final_rr,
            risk_pct_used=final_size,
            condition_quality=condition.quality,
            sentiment_bias=sentiment.bias,
            prev_adx=snap.adx,
        )

        self.daily_trades += 1
        self.adaptive_rr_sum += final_rr
        self.adaptive_size_sum += final_size

        # Build signal output
        active_conf = sum(1 for v in signal.confluences.values() if v)
        total_conf = len(signal.confluences)

        output = SignalOutputV3(
            pair="XAUUSD",
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=adjusted_sl,
            take_profit=adjusted_tp,
            risk_reward=f"1:{final_rr:.2f}",
            confidence=f"{active_conf}/{total_conf}",
            condition="Range Reversal (Adaptive)",
            timestamp=timestamp,
            adaptive_rr=final_rr,
            market_condition=condition.quality,
            sentiment=sentiment.bias,
            position_size_pct=final_size * 100,
        )

        if self._on_signal:
            self._on_signal(output)

        return output

    def _close_position(self, close_price: float, reason: str,
                        timestamp: datetime):
        """Close position and record result."""
        if self.position is None:
            return

        # Determine exit price
        if reason == "STOP_LOSS":
            exit_price = self.position.stop_loss
        elif reason == "TAKE_PROFIT":
            exit_price = self.position.take_profit
        elif reason in ("EARLY_PROFIT", "EARLY_LOSS"):
            exit_price = close_price
        else:
            exit_price = close_price

        # Calculate P&L
        if self.position.direction == "BUY":
            pnl_pips = (exit_price - self.position.entry_price) / 0.01
        else:
            pnl_pips = (self.position.entry_price - exit_price) / 0.01

        pnl_dollars = pnl_pips * self.position.lot_size * 1.0

        # System 3: Apply tight SL reality to losses
        # Real losses are partial (0.3-0.7x of full stop)
        if reason == "STOP_LOSS" and pnl_dollars < 0:
            # Simulate that real SL hit is often partial
            # The tight SL was already applied, so loss is naturally smaller
            pass  # Already handled by tighter SL placement

        # Update stats
        self.total_trades += 1
        self.total_pnl += pnl_dollars
        self.equity += pnl_dollars
        self.daily_pnl += pnl_dollars

        if pnl_dollars >= 0:
            self.wins += 1
            result = "WIN"
        else:
            self.losses += 1
            self.daily_losses += 1
            result = "LOSS"

        # Record result for adaptive sizing
        self.adaptive_engine.record_trade_result(result, timestamp)

        # Log trade
        self.trade_log.append({
            "entry_time": self.position.entry_time,
            "exit_time": timestamp,
            "direction": self.position.direction,
            "entry_price": self.position.entry_price,
            "exit_price": exit_price,
            "sl": self.position.stop_loss,
            "tp": self.position.take_profit,
            "lot_size": self.position.lot_size,
            "pnl_pips": round(pnl_pips, 1),
            "pnl_dollars": round(pnl_dollars, 2),
            "result": result,
            "reason": reason,
            "confluences": self.position.confluences,
            # V3 fields
            "adaptive_rr": self.position.adaptive_rr,
            "risk_pct": self.position.risk_pct_used,
            "condition": self.position.condition_quality,
            "sentiment": self.position.sentiment_bias,
        })

        # Set cooldown (V3: shorter cooldowns)
        if result == "LOSS":
            self.cooldown_until = timestamp + timedelta(
                minutes=COOLDOWN_V3["after_loss_minutes"])
        else:
            self.cooldown_until = timestamp + timedelta(
                minutes=COOLDOWN_V3["after_win_minutes"])

        # Callback
        if self._on_close:
            self._on_close(self.trade_log[-1])

        # Clear position
        self.position = None

    # ==================================================================
    # PRE-CHECKS
    # ==================================================================

    def _should_trade(self, timestamp: datetime, spread_pips: float) -> bool:
        """All the reasons NOT to trade."""
        if not self._is_active_session(timestamp):
            return False

        if self.daily_trades >= RISK_V3["max_daily_trades"]:
            return False

        daily_loss_pct = abs(self.daily_pnl) / self.equity if self.daily_pnl < 0 else 0
        if daily_loss_pct >= RISK_V3["max_daily_loss"]:
            return False

        if spread_pips > RISK_V3["max_spread_pips"]:
            return False

        if self.cooldown_until and timestamp < self.cooldown_until:
            return False

        if self.news_filter.is_blackout(timestamp):
            return False

        if timestamp.weekday() >= 5:
            return False

        return True

    def _is_active_session(self, timestamp: datetime) -> bool:
        """Check if current time is in an active trading session."""
        hour = timestamp.hour
        minute = timestamp.minute
        current_minutes = hour * 60 + minute

        for session_name, session in SESSIONS_V3.items():
            if not session["active"]:
                continue

            start_parts = session["start"].split(":")
            end_parts = session["end"].split(":")
            start_min = int(start_parts[0]) * 60 + int(start_parts[1])
            end_min = int(end_parts[0]) * 60 + int(end_parts[1])

            if end_min == 0:
                end_min = 24 * 60

            if start_min <= current_minutes < end_min:
                return True

        return False

    # ==================================================================
    # STATISTICS
    # ==================================================================

    def get_stats(self) -> dict:
        """Get bot performance statistics."""
        win_rate = (self.wins / self.total_trades * 100) if self.total_trades > 0 else 0

        # Profit factor
        winners = [t["pnl_dollars"] for t in self.trade_log if t["pnl_dollars"] > 0]
        losers = [t["pnl_dollars"] for t in self.trade_log if t["pnl_dollars"] < 0]
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        peak = self.account_balance
        max_dd = 0
        for _, eq in self.equity_history:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

        # Sharpe ratio
        if len(self.equity_history) > 96:
            equities = [e[1] for e in self.equity_history]
            daily_eq = equities[::96]
            daily_rets = [(daily_eq[i] - daily_eq[i-1]) / daily_eq[i-1]
                          for i in range(1, len(daily_eq)) if daily_eq[i-1] != 0]
            sharpe = (np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)
                      if daily_rets and np.std(daily_rets) > 0 else 0)
        else:
            sharpe = 0.0

        # Average adaptive R:R used
        avg_rr = self.adaptive_rr_sum / self.total_trades if self.total_trades > 0 else 1.0
        avg_size = self.adaptive_size_sum / self.total_trades if self.total_trades > 0 else 0.02

        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": f"{win_rate:.1f}%",
            "profit_factor": f"{pf:.2f}",
            "total_pnl": f"${self.total_pnl:.2f}",
            "total_return": f"{(self.equity - self.account_balance) / self.account_balance * 100:.1f}%",
            "max_drawdown": f"{max_dd * 100:.1f}%",
            "avg_winner": f"${np.mean(winners):.2f}" if winners else "$0",
            "avg_loser": f"${np.mean(losers):.2f}" if losers else "$0",
            "equity": f"${self.equity:.2f}",
            "sharpe": f"{sharpe:.2f}",
            # V3 specific
            "avg_adaptive_rr": f"{avg_rr:.2f}",
            "avg_position_size": f"{avg_size*100:.1f}%",
            "early_exits_profit": self.early_exits_profit,
            "early_exits_loss": self.early_exits_loss,
            "sentiment_avoids": self.sentiment_avoids,
        }
