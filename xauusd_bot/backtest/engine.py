"""
Backtesting Engine for XAUUSD Signal Bot.

Runs the bot against historical data and collects:
- All trades (entry, exit, P&L, duration)
- Equity curve
- Per-session performance
- Drawdown tracking
- Signal accuracy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

from ..bot import XAUUSDBot, BotStatus, LogEntry
from ..core.risk_manager import TradeParams


@dataclass
class TradeRecord:
    """Record of a completed trade."""
    entry_time: datetime = None
    exit_time: datetime = None
    direction: str = ""
    strategy: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.0
    pnl_pips: float = 0.0
    pnl_dollars: float = 0.0
    exit_reason: str = ""
    duration_minutes: int = 0
    session: str = ""
    score: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results."""
    # Summary
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration_min: float = 0.0
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    # Data series
    equity_curve: list = field(default_factory=list)
    drawdown_curve: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    daily_pnl: dict = field(default_factory=dict)
    session_stats: dict = field(default_factory=dict)

    # Config
    starting_balance: float = 10000.0
    final_balance: float = 10000.0
    total_return_pct: float = 0.0
    test_period_days: int = 0


class BacktestEngine:
    """
    Runs the XAUUSD bot against historical data.
    """

    def __init__(self, starting_balance: float = 10000.0):
        self.starting_balance = starting_balance
        self.trades: list[TradeRecord] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.equity = starting_balance
        self.peak_equity = starting_balance

        # Track open position for recording
        self._current_trade: Optional[TradeRecord] = None

    def run(self, price_data: pd.DataFrame,
            dxy_data: Optional[np.ndarray] = None,
            spread_pips: float = 15.0) -> BacktestResult:
        """
        Run backtest on price data.

        Args:
            price_data: DataFrame with timestamp, open, high, low, close columns
            dxy_data: Optional numpy array of DXY closes (same length as price_data)
            spread_pips: Simulated spread in pips
        """
        # Initialize bot
        bot = XAUUSDBot(
            account_balance=self.starting_balance,
            on_trade_open=self._on_trade_open,
            on_trade_close=self._on_trade_close,
            on_log=self._on_log,
        )
        bot.initialize()

        # Run through each bar
        for idx, row in price_data.iterrows():
            timestamp = row["timestamp"]
            if isinstance(timestamp, str):
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

            # Get DXY slice if available
            dxy_slice = None
            if dxy_data is not None and idx >= 25:
                dxy_slice = dxy_data[max(0, idx - 25):idx + 1]

            # Feed bar to bot
            status = bot.on_bar(
                timestamp=timestamp,
                high=row["high"],
                low=row["low"],
                close=row["close"],
                open_price=row["open"],
                spread_pips=spread_pips,
                dxy_closes=dxy_slice,
            )

            # Track equity
            current_equity = bot.risk_manager.equity if bot.risk_manager else self.starting_balance
            self.equity_curve.append((timestamp, current_equity))

        # Compile results
        return self._compile_results(price_data)

    def _on_trade_open(self, params: TradeParams):
        """Callback when bot opens a trade."""
        self._current_trade = TradeRecord(
            direction=params.direction,
            entry_price=params.entry_price,
            stop_loss=params.stop_loss,
            take_profit=params.take_profit,
            lot_size=params.lot_size,
        )

    def _on_trade_close(self, position, reason: str):
        """Callback when bot closes a trade."""
        if self._current_trade is None:
            return

        # Calculate P&L
        if position.direction == "BUY":
            pnl_pips = (self.equity_curve[-1][1] - self._current_trade.entry_price) / 0.01 if self.equity_curve else 0
        else:
            pnl_pips = (self._current_trade.entry_price - self.equity_curve[-1][1]) / 0.01 if self.equity_curve else 0

        self._current_trade.exit_reason = reason
        self._current_trade.entry_time = position.entry_time
        self._current_trade.exit_time = self.equity_curve[-1][0] if self.equity_curve else None
        self._current_trade.strategy = position.strategy
        self._current_trade.exit_price = self.equity_curve[-1][1] if self.equity_curve else 0

        # Calculate actual P&L from the risk manager's record
        self.trades.append(self._current_trade)
        self._current_trade = None

    def _on_log(self, entry: LogEntry):
        """Capture trade logs for analysis."""
        if entry.level == "TRADE" and "CLOSED" in entry.message:
            # Update the last trade record with actual P&L data
            if self.trades and entry.data:
                trade = self.trades[-1]
                trade.pnl_pips = entry.data.get("pnl_pips", 0)
                trade.pnl_dollars = entry.data.get("pnl_dollars", 0)
                trade.exit_reason = entry.data.get("reason", "")
                trade.exit_price = entry.data.get("exit", 0)

    def _compile_results(self, price_data: pd.DataFrame) -> BacktestResult:
        """Compile all results into BacktestResult."""
        result = BacktestResult()
        result.starting_balance = self.starting_balance
        result.equity_curve = self.equity_curve
        result.trades = self.trades

        if not self.trades:
            return result

        # Basic counts
        result.total_trades = len(self.trades)
        result.winning_trades = sum(1 for t in self.trades if t.pnl_dollars > 0)
        result.losing_trades = sum(1 for t in self.trades if t.pnl_dollars <= 0)
        result.win_rate = (result.winning_trades / result.total_trades * 100
                           if result.total_trades > 0 else 0)

        # P&L stats
        pnls = [t.pnl_dollars for t in self.trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        result.total_pnl = sum(pnls)
        result.avg_trade_pnl = np.mean(pnls) if pnls else 0
        result.avg_winner = np.mean(winners) if winners else 0
        result.avg_loser = np.mean(losers) if losers else 0
        result.largest_win = max(winners) if winners else 0
        result.largest_loss = min(losers) if losers else 0

        # Profit factor
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Equity and drawdown
        equities = [e[1] for e in self.equity_curve]
        if equities:
            result.final_balance = equities[-1]
            result.total_return_pct = ((equities[-1] - self.starting_balance)
                                        / self.starting_balance * 100)

            # Max drawdown
            peak = equities[0]
            max_dd = 0
            max_dd_dollars = 0
            dd_curve = []
            for eq in equities:
                if eq > peak:
                    peak = eq
                dd = (peak - eq) / peak
                dd_dollars = peak - eq
                max_dd = max(max_dd, dd)
                max_dd_dollars = max(max_dd_dollars, dd_dollars)
                dd_curve.append(dd)

            result.max_drawdown_pct = max_dd * 100
            result.max_drawdown_dollars = max_dd_dollars
            result.drawdown_curve = dd_curve

        # Sharpe ratio (annualized, using daily returns)
        if len(equities) > 2:
            daily_returns = []
            prev = equities[0]
            for eq in equities[96::96]:  # Sample every 96 bars (1 day)
                daily_returns.append((eq - prev) / prev)
                prev = eq
            if daily_returns:
                mean_ret = np.mean(daily_returns)
                std_ret = np.std(daily_returns)
                result.sharpe_ratio = (mean_ret / std_ret * np.sqrt(252)
                                       if std_ret > 0 else 0)

        # Consecutive wins/losses
        consecutive_w = 0
        consecutive_l = 0
        max_cw = 0
        max_cl = 0
        for t in self.trades:
            if t.pnl_dollars > 0:
                consecutive_w += 1
                consecutive_l = 0
                max_cw = max(max_cw, consecutive_w)
            else:
                consecutive_l += 1
                consecutive_w = 0
                max_cl = max(max_cl, consecutive_l)

        result.max_consecutive_wins = max_cw
        result.max_consecutive_losses = max_cl

        # Duration
        durations = []
        for t in self.trades:
            if t.entry_time and t.exit_time:
                dur = (t.exit_time - t.entry_time).total_seconds() / 60
                t.duration_minutes = int(dur)
                durations.append(dur)
        result.avg_trade_duration_min = np.mean(durations) if durations else 0

        # Session stats
        session_map = {"asian": [], "london": [], "overlap": [], "new_york": []}
        for t in self.trades:
            if t.entry_time:
                h = t.entry_time.hour
                if 0 <= h < 7:
                    session_map["asian"].append(t.pnl_dollars)
                elif 7 <= h < 12:
                    session_map["london"].append(t.pnl_dollars)
                elif 12 <= h < 16:
                    session_map["overlap"].append(t.pnl_dollars)
                elif 16 <= h < 21:
                    session_map["new_york"].append(t.pnl_dollars)

        for session, pnls_list in session_map.items():
            if pnls_list:
                wins = sum(1 for p in pnls_list if p > 0)
                result.session_stats[session] = {
                    "trades": len(pnls_list),
                    "wins": wins,
                    "win_rate": wins / len(pnls_list) * 100,
                    "total_pnl": sum(pnls_list),
                }

        # Test period
        if len(price_data) > 0:
            first_ts = price_data.iloc[0]["timestamp"]
            last_ts = price_data.iloc[-1]["timestamp"]
            if isinstance(first_ts, str):
                first_ts = datetime.strptime(first_ts, "%Y-%m-%d %H:%M:%S")
                last_ts = datetime.strptime(last_ts, "%Y-%m-%d %H:%M:%S")
            result.test_period_days = (last_ts - first_ts).days

        # Daily P&L
        for t in self.trades:
            if t.exit_time:
                day = t.exit_time.strftime("%Y-%m-%d")
                result.daily_pnl[day] = result.daily_pnl.get(day, 0) + t.pnl_dollars

        return result
