"""
XAUUSD Signal Bot V2 — Win Rate Maximized

ONE trade type: Reversals at range boundaries with 1:1 RR.
Set and forget. No management. Condition identification is the edge.

This bot is SIMPLER than V1 because:
- One strategy, not two
- No trailing stops, no trade management
- Fewer decisions = fewer mistakes = more consistency
- The edge is knowing WHEN to trade (condition), not complex signals
"""

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional, Callable

from .config_v2 import (
    STRATEGY, RISK_V2, SESSIONS_V2, INDICATORS_V2,
    NEWS_V2, COOLDOWN_V2
)
from .indicators.technical import TechnicalIndicators, IndicatorSnapshot
from .strategies.reversal_strategy import ReversalStrategy, ReversalSignal
from .filters.news_filter import NewsFilter


@dataclass
class PositionV2:
    """An open position — set and forget."""
    direction: str = "NONE"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.0
    entry_time: Optional[datetime] = None
    risk_pips: float = 0.0
    confluences: dict = None


@dataclass
class SignalOutput:
    """Signal output for Telegram/Discord delivery."""
    pair: str = "XAUUSD"
    direction: str = ""          # "BUY" or "SELL"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    risk_reward: str = "1:1"
    confidence: str = ""         # "3/5", "4/5", "5/5"
    condition: str = ""          # "Range Reversal"
    timestamp: Optional[datetime] = None


class XAUUSDBotV2:
    """
    Win Rate Maximized XAUUSD Signal Bot.

    Usage:
        bot = XAUUSDBotV2(account_balance=10000.0)
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

        # State
        self.position: Optional[PositionV2] = None
        self.daily_trades: int = 0
        self.daily_losses: int = 0
        self.daily_pnl: float = 0.0
        self.last_trade_date: str = ""
        self.cooldown_until: Optional[datetime] = None

        # Price history
        self._opens: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []
        self._max_history = 200

        # Stats
        self.total_trades = 0
        self.wins = 0
        self.losses = 0
        self.total_pnl = 0.0
        self.equity_history: list[tuple[datetime, float]] = []
        self.trade_log: list[dict] = []

    def initialize(self):
        """Initialize bot components."""
        self.indicators = TechnicalIndicators(INDICATORS_V2)
        self.strategy = ReversalStrategy(STRATEGY)
        self.news_filter = NewsFilter(NEWS_V2)

    def on_bar(self, timestamp: datetime,
               open_price: float, high: float, low: float, close: float,
               spread_pips: float = 15.0) -> Optional[SignalOutput]:
        """
        Process a new M15 bar. Returns a SignalOutput if a trade is triggered.
        """
        # Update buffers
        self._opens.append(open_price)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

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

        # Track equity
        self.equity_history.append((timestamp, self.equity))

        # ---- CHECK OPEN POSITION (Set and Forget) ----
        if self.position is not None:
            exited, reason = self.strategy.check_exit(
                self.position.direction, close,
                self.position.stop_loss, self.position.take_profit
            )
            if exited:
                self._close_position(close, reason, timestamp)
            return None  # Don't look for new signals while in a trade

        # ---- PRE-CHECKS (should we even be looking?) ----
        if not self._should_trade(timestamp, spread_pips):
            return None

        # ---- COMPUTE INDICATORS ----
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        closes = np.array(self._closes)

        snap = self.indicators.compute_all(highs, lows, closes)

        # ---- GET PREVIOUS BAR DATA (for rejection candle check) ----
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
            return None

        # ---- OPEN POSITION ----
        return self._open_position(signal, timestamp)

    # ==================================================================
    # POSITION MANAGEMENT
    # ==================================================================

    def _open_position(self, signal: ReversalSignal,
                       timestamp: datetime) -> Optional[SignalOutput]:
        """Open a new position based on reversal signal."""
        # Calculate lot size (fixed % risk)
        risk_amount = self.equity * RISK_V2["risk_per_trade"]
        lot_size = risk_amount / (signal.risk_pips * 1.0)  # $1 per pip per lot
        lot_size = round(max(lot_size, 0.01), 2)

        self.position = PositionV2(
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            lot_size=lot_size,
            entry_time=timestamp,
            risk_pips=signal.risk_pips,
            confluences=signal.confluences,
        )

        self.daily_trades += 1

        # Build signal output for Telegram/Discord
        active_conf = sum(1 for v in signal.confluences.values() if v)
        total_conf = len(signal.confluences)

        output = SignalOutput(
            pair="XAUUSD",
            direction=signal.direction,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            risk_reward="1:1",
            confidence=f"{active_conf}/{total_conf}",
            condition="Range Reversal",
            timestamp=timestamp,
        )

        # Callback
        if self._on_signal:
            self._on_signal(output)

        return output

    def _close_position(self, close_price: float, reason: str,
                        timestamp: datetime):
        """Close position and record result."""
        if self.position is None:
            return

        # Use SL/TP price for exit (not the current bar close)
        # This simulates proper stop execution
        if reason == "STOP_LOSS":
            exit_price = self.position.stop_loss
        elif reason == "TAKE_PROFIT":
            exit_price = self.position.take_profit
        else:
            exit_price = close_price

        # Calculate P&L
        if self.position.direction == "BUY":
            pnl_pips = (exit_price - self.position.entry_price) / 0.01
        else:
            pnl_pips = (self.position.entry_price - exit_price) / 0.01

        pnl_dollars = pnl_pips * self.position.lot_size * 1.0

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
        })

        # Set cooldown
        if result == "LOSS":
            self.cooldown_until = timestamp + timedelta(
                minutes=COOLDOWN_V2["after_loss_minutes"])
        else:
            self.cooldown_until = timestamp + timedelta(
                minutes=COOLDOWN_V2["after_win_minutes"])

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

        # Session check
        if not self._is_active_session(timestamp):
            return False

        # Daily trade limit
        if self.daily_trades >= RISK_V2["max_daily_trades"]:
            return False

        # Daily loss limit
        daily_loss_pct = abs(self.daily_pnl) / self.equity if self.daily_pnl < 0 else 0
        if daily_loss_pct >= RISK_V2["max_daily_loss"]:
            return False

        # Spread check
        if spread_pips > RISK_V2["max_spread_pips"]:
            return False

        # Cooldown
        if self.cooldown_until and timestamp < self.cooldown_until:
            return False

        # News blackout
        if self.news_filter.is_blackout(timestamp):
            return False

        # Weekend
        if timestamp.weekday() >= 5:
            return False

        return True

    def _is_active_session(self, timestamp: datetime) -> bool:
        """Check if current time is in an active trading session."""
        hour = timestamp.hour
        minute = timestamp.minute
        current_minutes = hour * 60 + minute

        for session_name, session in SESSIONS_V2.items():
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

        return {
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": f"{win_rate:.1f}%",
            "profit_factor": f"{pf:.2f}",
            "total_pnl": f"${self.total_pnl:.2f}",
            "total_return": f"{(self.equity - self.account_balance) / self.account_balance * 100:.2f}%",
            "max_drawdown": f"{max_dd * 100:.2f}%",
            "avg_winner": f"${np.mean(winners):.2f}" if winners else "$0",
            "avg_loser": f"${np.mean(losers):.2f}" if losers else "$0",
            "equity": f"${self.equity:.2f}",
        }
