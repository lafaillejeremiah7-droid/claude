"""
XAUUSD Trading Bot — Main Orchestrator

This is the central engine that ties all components together:
    - State Machine (controls bot behavior flow)
    - Session Manager (determines which mode to use)
    - Technical Indicators (computes signals from price data)
    - Multi-Factor Scorer (evaluates entry quality)
    - Trend Strategy (London/NY entries)
    - Range Strategy (Asian session entries)
    - DXY Filter (correlation confirmation)
    - News Filter (event avoidance)
    - Risk Manager (position sizing, SL/TP, daily limits)

The bot operates on a tick/bar loop:
    1. Receive new price data
    2. Update session and indicators
    3. State machine determines what action to take
    4. Execute appropriate strategy logic
    5. Manage open positions
    6. Log everything

Designed for integration with MT5, cTrader, or any broker API.
"""

import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Callable

from .config import (
    SESSIONS, INDICATORS, TREND_FACTORS, RANGE_FACTORS,
    ENTRY_THRESHOLD, RISK, NEWS, DXY, COOLDOWN, TIMEFRAMES
)
from .core.state_machine import StateMachine, State, BotContext
from .core.session_manager import SessionManager
from .core.risk_manager import RiskManager, TradeParams
from .indicators.technical import TechnicalIndicators, IndicatorSnapshot
from .indicators.scoring import MultiFactorScorer
from .strategies.trend_strategy import TrendStrategy, TrendSignal
from .strategies.range_strategy import RangeStrategy, RangeSignal
from .filters.news_filter import NewsFilter
from .filters.dxy_filter import DXYFilter, DXYSignal


@dataclass
class Position:
    """An open trade position."""
    direction: str = "NONE"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.0
    entry_time: Optional[datetime] = None
    strategy: str = ""  # "TREND" or "RANGE"
    trailing_active: bool = False


@dataclass
class BotStatus:
    """Current bot status for monitoring/logging."""
    state: str = "IDLE"
    session: str = "dead_zone"
    session_mode: str = "IDLE"
    has_position: bool = False
    position_direction: str = "NONE"
    position_pnl: float = 0.0
    daily_pnl: float = 0.0
    daily_trades: int = 0
    last_signal_score: float = 0.0
    last_signal_direction: str = "NONE"
    equity: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class LogEntry:
    """A log entry for audit trail."""
    timestamp: datetime
    level: str  # "INFO", "SIGNAL", "TRADE", "WARNING", "ERROR"
    message: str
    data: dict = field(default_factory=dict)


class XAUUSDBot:
    """
    Main trading bot orchestrator.

    Usage:
        bot = XAUUSDBot(account_balance=10000.0)
        bot.initialize()

        # On each new bar/tick:
        bot.on_bar(timestamp, high, low, close, open_price, spread,
                   dxy_closes=dxy_data)
    """

    def __init__(self, account_balance: float = 10000.0,
                 on_trade_open: Optional[Callable] = None,
                 on_trade_close: Optional[Callable] = None,
                 on_log: Optional[Callable] = None):
        """
        Args:
            account_balance: Starting account balance in USD
            on_trade_open: Callback when a trade should be opened
                           Signature: (TradeParams) -> bool (success)
            on_trade_close: Callback when a trade should be closed
                           Signature: (Position, reason) -> float (pnl)
            on_log: Callback for log entries
                    Signature: (LogEntry) -> None
        """
        self.account_balance = account_balance
        self._on_trade_open = on_trade_open
        self._on_trade_close = on_trade_close
        self._on_log = on_log

        # Components (initialized in .initialize())
        self.state_machine: Optional[StateMachine] = None
        self.session_manager: Optional[SessionManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.indicators: Optional[TechnicalIndicators] = None
        self.scorer: Optional[MultiFactorScorer] = None
        self.trend_strategy: Optional[TrendStrategy] = None
        self.range_strategy: Optional[RangeStrategy] = None
        self.news_filter: Optional[NewsFilter] = None
        self.dxy_filter: Optional[DXYFilter] = None

        # State
        self.position: Optional[Position] = None
        self.context: BotContext = BotContext()
        self.last_snapshot: Optional[IndicatorSnapshot] = None
        self.logs: list[LogEntry] = []

        # Price history buffers
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._closes: list[float] = []
        self._max_history = 300  # Keep last 300 bars

        # Previous session tracking (for structure breaks)
        self._prev_session_high: Optional[float] = None
        self._prev_session_low: Optional[float] = None
        self._last_session_name: str = ""

    def initialize(self):
        """Initialize all bot components. Call once before starting."""
        self.state_machine = StateMachine()
        self.session_manager = SessionManager(SESSIONS)
        self.risk_manager = RiskManager(RISK, self.account_balance)
        self.indicators = TechnicalIndicators(INDICATORS)
        self.scorer = MultiFactorScorer(TREND_FACTORS, RANGE_FACTORS, ENTRY_THRESHOLD)
        self.trend_strategy = TrendStrategy(self.scorer, self.risk_manager, INDICATORS)
        self.range_strategy = RangeStrategy(self.scorer, self.risk_manager, INDICATORS)
        self.news_filter = NewsFilter(NEWS)
        self.dxy_filter = DXYFilter(DXY)

        self._log("INFO", "Bot initialized", {
            "account_balance": self.account_balance,
            "entry_threshold": ENTRY_THRESHOLD,
            "max_risk_per_trade": RISK["max_risk_per_trade"],
            "max_daily_loss": RISK["max_daily_loss"],
        })

    # ==================================================================
    # MAIN LOOP — Called on every new bar
    # ==================================================================

    def on_bar(self, timestamp: datetime,
               high: float, low: float, close: float, open_price: float,
               spread_pips: float = 10.0,
               dxy_closes: Optional[np.ndarray] = None) -> BotStatus:
        """
        Process a new price bar. This is the main entry point called
        on every candle close (M15 timeframe recommended).

        Args:
            timestamp: Bar timestamp (UTC)
            high, low, close, open_price: OHLC data
            spread_pips: Current spread in pips
            dxy_closes: Optional DXY close price array for correlation filter

        Returns:
            BotStatus with current state info
        """
        # Update price buffers
        self._update_price_history(high, low, close)

        # Skip if not enough data
        if len(self._closes) < 50:
            return self._build_status(timestamp)

        # ---- STEP 1: Update Session ----
        session_info = self.session_manager.update(timestamp, close)

        # Track previous session levels for structure breaks
        if session_info.name != self._last_session_name and self._last_session_name:
            self._prev_session_high = self.session_manager.current_session.session_high
            self._prev_session_low = self.session_manager.current_session.session_low
        self._last_session_name = session_info.name

        # ---- STEP 2: Compute Indicators ----
        highs = np.array(self._highs)
        lows = np.array(self._lows)
        closes = np.array(self._closes)

        self.last_snapshot = self.indicators.compute_all(
            highs, lows, closes,
            session_high=session_info.session_high,
            session_low=session_info.session_low,
        )

        # ---- STEP 3: Check News Filter ----
        news_result = self.news_filter.check(timestamp)

        # ---- STEP 4: Update Bot Context ----
        self._update_context(timestamp, session_info.mode, spread_pips, news_result.is_blackout)

        # ---- STEP 5: Check Daily Reset ----
        self._check_daily_reset(timestamp)

        # ---- STEP 6: State Machine Transition ----
        new_state = self.state_machine.try_transition(self.context)
        current_state = self.state_machine.get_state()

        # ---- STEP 7: Execute State Logic ----
        if current_state == State.ENTRY_SEARCH:
            self._execute_entry_search(dxy_closes)

        elif current_state == State.POSITION_ACTIVE:
            self._manage_position(timestamp, close)

        elif current_state == State.NEWS_AVOID:
            self._log("INFO", f"News blackout: {news_result.reason}")

        elif current_state == State.COOLDOWN:
            pass  # Waiting — state machine handles exit

        elif current_state == State.IDLE:
            pass  # Not trading

        # ---- STEP 8: Return Status ----
        return self._build_status(timestamp)

    # ==================================================================
    # ENTRY SEARCH LOGIC
    # ==================================================================

    def _execute_entry_search(self, dxy_closes: Optional[np.ndarray]):
        """Scan for entry signals based on current mode."""
        snap = self.last_snapshot
        mode = self.session_manager.get_mode()

        if mode in ("TREND", "TREND_AGGRESSIVE"):
            self._search_trend_entry(snap, dxy_closes)
        elif mode == "RANGE":
            self._search_range_entry(snap)

    def _search_trend_entry(self, snap: IndicatorSnapshot,
                            dxy_closes: Optional[np.ndarray]):
        """Look for trend-following entry."""
        # Check DXY confirmation
        dxy_confirms = False
        if dxy_closes is not None and len(dxy_closes) >= 20:
            gold_closes = np.array(self._closes[-25:])
            dxy_signal = self.dxy_filter.evaluate(
                gold_closes, dxy_closes[-25:], "BUY"  # Direction determined inside
            )
            dxy_confirms = dxy_signal.confirms

        # Get session levels
        session_high, session_low = self.session_manager.get_session_range()

        # Evaluate trend strategy
        signal = self.trend_strategy.evaluate(
            snap,
            dxy_confirms=dxy_confirms,
            session_high=session_high,
            session_low=session_low,
            previous_session_high=self._prev_session_high,
            previous_session_low=self._prev_session_low,
        )

        self.context.entry_score = signal.score

        if signal.has_signal:
            self._open_position(signal.trade_params, "TREND", signal.reason)
        else:
            self._log("INFO", f"Trend scan: {signal.reason}", signal.score_details)

    def _search_range_entry(self, snap: IndicatorSnapshot):
        """Look for range/mean-reversion entry."""
        signal = self.range_strategy.evaluate(snap)
        self.context.entry_score = signal.score

        if signal.has_signal:
            self._open_position(signal.trade_params, "RANGE", signal.reason)
        else:
            self._log("INFO", f"Range scan: {signal.reason}", signal.score_details)

    # ==================================================================
    # POSITION MANAGEMENT
    # ==================================================================

    def _open_position(self, params: TradeParams, strategy: str, reason: str):
        """Open a new position."""
        if self.position is not None:
            return  # Already in a trade

        # Check near session end (don't open within 15 min of session close)
        if self.session_manager.is_near_session_end(self.context.current_time, 15):
            self._log("INFO", "Skipping entry — too close to session end")
            return

        self.position = Position(
            direction=params.direction,
            entry_price=params.entry_price,
            stop_loss=params.stop_loss,
            take_profit=params.take_profit,
            lot_size=params.lot_size,
            entry_time=self.context.current_time,
            strategy=strategy,
        )

        self.context.has_open_position = True
        self.session_manager.record_trade()

        self._log("TRADE", f"OPENED {params.direction} @ {params.entry_price:.2f}", {
            "strategy": strategy,
            "reason": reason,
            "sl": params.stop_loss,
            "tp": params.take_profit,
            "lot_size": params.lot_size,
            "risk_amount": params.risk_amount,
            "rr_ratio": params.risk_reward_ratio,
        })

        # Execute via callback
        if self._on_trade_open:
            self._on_trade_open(params)

        # Force state machine to POSITION_ACTIVE
        self.state_machine.force_state(State.POSITION_ACTIVE, self.context)

    def _manage_position(self, timestamp: datetime, current_price: float):
        """Manage an open position — check exits and trailing stops."""
        if self.position is None:
            return

        snap = self.last_snapshot
        near_end = self.session_manager.is_near_session_end(timestamp, 15)

        # Check exit conditions based on strategy
        should_exit = False
        exit_reason = ""

        if self.position.strategy == "TREND":
            should_exit, exit_reason = self.trend_strategy.should_exit(
                self.position.direction,
                self.position.entry_price,
                current_price,
                self.position.stop_loss,
                self.position.take_profit,
                snap,
                near_session_end=near_end,
            )
        elif self.position.strategy == "RANGE":
            should_exit, exit_reason = self.range_strategy.should_exit(
                self.position.direction,
                self.position.entry_price,
                current_price,
                self.position.stop_loss,
                self.position.take_profit,
                snap,
            )

        if should_exit:
            self._close_position(current_price, exit_reason)
            return

        # Update trailing stop (trend trades only)
        if self.position.strategy == "TREND":
            new_sl = self.risk_manager.update_trailing_stop(
                self.position.direction,
                current_price,
                self.position.entry_price,
                snap.atr,
                self.position.stop_loss,
            )
            if new_sl is not None:
                old_sl = self.position.stop_loss
                self.position.stop_loss = new_sl
                self.position.trailing_active = True
                self._log("INFO", f"Trailing SL moved: {old_sl:.2f} → {new_sl:.2f}")

    def _close_position(self, close_price: float, reason: str):
        """Close the current position and record results."""
        if self.position is None:
            return

        # Calculate P&L
        if self.position.direction == "BUY":
            pnl_pips = (close_price - self.position.entry_price) / 0.01
        else:
            pnl_pips = (self.position.entry_price - close_price) / 0.01

        pnl_dollars = pnl_pips * self.position.lot_size * 1.0  # $1 per pip per lot

        # Record in risk manager
        self.risk_manager.record_trade_result(pnl_dollars)

        # Determine result
        result = "WIN" if pnl_dollars >= 0 else "LOSS"

        self._log("TRADE", f"CLOSED {self.position.direction} @ {close_price:.2f}", {
            "reason": reason,
            "entry": self.position.entry_price,
            "exit": close_price,
            "pnl_pips": round(pnl_pips, 1),
            "pnl_dollars": round(pnl_dollars, 2),
            "result": result,
            "duration": str(self.context.current_time - self.position.entry_time)
            if self.position.entry_time else "N/A",
        })

        # Execute via callback
        if self._on_trade_close:
            self._on_trade_close(self.position, reason)

        # Update context
        self.context.has_open_position = False
        self.context.last_trade_result = result
        self.context.daily_loss_pct = self.risk_manager.get_daily_loss_pct()
        self.context.daily_trade_count = self.risk_manager.get_trade_count()

        # Set cooldown
        cooldown_duration = self.risk_manager.get_cooldown_duration(result, COOLDOWN)
        self.context.cooldown_expires = self.context.current_time + cooldown_duration

        # Clear position
        self.position = None

        # Transition to COOLDOWN
        self.state_machine.force_state(State.COOLDOWN, self.context)

    # ==================================================================
    # HELPERS
    # ==================================================================

    def _update_price_history(self, high: float, low: float, close: float):
        """Append to price buffers, maintain max length."""
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        if len(self._highs) > self._max_history:
            self._highs = self._highs[-self._max_history:]
            self._lows = self._lows[-self._max_history:]
            self._closes = self._closes[-self._max_history:]

    def _update_context(self, timestamp: datetime, session_mode: str,
                        spread_pips: float, news_blackout: bool):
        """Update the shared bot context for state machine evaluation."""
        self.context.current_time = timestamp
        self.context.session_mode = session_mode
        self.context.news_blackout_active = news_blackout
        self.context.spread_ok = self.risk_manager.is_spread_acceptable(spread_pips)
        self.context.daily_loss_pct = self.risk_manager.get_daily_loss_pct()
        self.context.daily_trade_count = self.risk_manager.get_trade_count()
        self.context.session_trade_count = self.session_manager.get_session_trade_count()
        self.context.has_open_position = self.position is not None

    def _check_daily_reset(self, timestamp: datetime):
        """Reset daily stats at midnight UTC."""
        current_date = timestamp.strftime("%Y-%m-%d")
        if self.risk_manager.daily_stats.date != current_date:
            self._log("INFO", f"New trading day: {current_date}")
            self.risk_manager.reset_daily()
            self.context.last_trade_result = None

    def _build_status(self, timestamp: datetime) -> BotStatus:
        """Build current bot status snapshot."""
        position_pnl = 0.0
        if self.position and self._closes:
            current = self._closes[-1]
            if self.position.direction == "BUY":
                position_pnl = (current - self.position.entry_price) * self.position.lot_size * 100
            else:
                position_pnl = (self.position.entry_price - current) * self.position.lot_size * 100

        return BotStatus(
            state=self.state_machine.get_state().value if self.state_machine else "UNINITIALIZED",
            session=self.session_manager.get_session_name() if self.session_manager else "N/A",
            session_mode=self.session_manager.get_mode() if self.session_manager else "N/A",
            has_position=self.position is not None,
            position_direction=self.position.direction if self.position else "NONE",
            position_pnl=round(position_pnl, 2),
            daily_pnl=round(self.risk_manager.daily_stats.gross_pnl, 2) if self.risk_manager else 0,
            daily_trades=self.risk_manager.get_trade_count() if self.risk_manager else 0,
            last_signal_score=self.context.entry_score,
            last_signal_direction=self.context.last_trade_result or "NONE",
            equity=round(self.risk_manager.equity, 2) if self.risk_manager else 0,
            timestamp=timestamp,
        )

    def _log(self, level: str, message: str, data: dict = None):
        """Create a log entry."""
        entry = LogEntry(
            timestamp=self.context.current_time or datetime.utcnow(),
            level=level,
            message=message,
            data=data or {},
        )
        self.logs.append(entry)

        # Keep log buffer manageable
        if len(self.logs) > 1000:
            self.logs = self.logs[-500:]

        # Callback
        if self._on_log:
            self._on_log(entry)

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def get_status(self) -> BotStatus:
        """Get current bot status."""
        return self._build_status(self.context.current_time or datetime.utcnow())

    def get_logs(self, last_n: int = 50, level: Optional[str] = None) -> list[LogEntry]:
        """Get recent log entries, optionally filtered by level."""
        logs = self.logs[-last_n:]
        if level:
            logs = [l for l in logs if l.level == level]
        return logs

    def add_news_event(self, name: str, timestamp: datetime):
        """Add a news event to the filter."""
        if self.news_filter:
            self.news_filter.add_event(name, timestamp)

    def force_close(self, reason: str = "MANUAL_CLOSE"):
        """Force close any open position (emergency/manual override)."""
        if self.position and self._closes:
            self._close_position(self._closes[-1], reason)

    def shutdown(self):
        """Gracefully shut down the bot."""
        if self.position:
            self.force_close("SHUTDOWN")
        self.state_machine.force_state(State.IDLE, self.context)
        self._log("INFO", "Bot shut down")

    def get_statistics(self) -> dict:
        """Get trading statistics summary."""
        stats = self.risk_manager.get_stats() if self.risk_manager else None
        if not stats:
            return {}

        win_rate = (stats.wins / stats.trades_taken * 100) if stats.trades_taken > 0 else 0

        return {
            "date": stats.date,
            "trades_taken": stats.trades_taken,
            "wins": stats.wins,
            "losses": stats.losses,
            "win_rate": f"{win_rate:.1f}%",
            "gross_pnl": f"${stats.gross_pnl:.2f}",
            "max_drawdown": f"{stats.max_drawdown_pct * 100:.2f}%",
            "equity": f"${self.risk_manager.equity:.2f}",
            "state": self.state_machine.get_state().value,
        }
