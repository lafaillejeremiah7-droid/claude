"""
Risk Management Engine for XAUUSD Trading Bot.

Handles:
    - ATR-based position sizing (risk % of account per trade)
    - Stop loss / Take profit calculation
    - Trailing stop management
    - Daily P&L tracking and shutdown
    - Spread validation
    - Trade count limits
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class TradeParams:
    """Calculated trade parameters for an entry."""
    direction: str = "NONE"      # "BUY" or "SELL"
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.0
    risk_amount: float = 0.0     # Dollar risk
    reward_amount: float = 0.0   # Dollar potential reward
    risk_reward_ratio: float = 0.0
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    valid: bool = False          # Whether trade passes all risk checks


@dataclass
class DailyStats:
    """Daily performance tracking."""
    date: str = ""
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    peak_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    current_loss_pct: float = 0.0


class RiskManager:
    """
    Core risk management engine. Every trade must pass through here
    before execution.
    """

    def __init__(self, config: dict, account_balance: float):
        self.cfg = config
        self.account_balance = account_balance
        self.equity = account_balance
        self.daily_stats = DailyStats(date=datetime.utcnow().strftime("%Y-%m-%d"))
        self.daily_stats.peak_equity = account_balance

        # Active trade tracking
        self.trailing_stop: Optional[float] = None
        self.trailing_activated: bool = False

    # ------------------------------------------------------------------
    # POSITION SIZING
    # ------------------------------------------------------------------

    def calculate_trade(self, direction: str, entry_price: float,
                        atr: float, pip_value: float = 1.0) -> TradeParams:
        """
        Calculate full trade parameters based on ATR and risk settings.

        Args:
            direction: "BUY" or "SELL"
            entry_price: Current price for entry
            atr: Current ATR value (in price units, e.g., $15.50)
            pip_value: Dollar value per pip per lot ($1.00 for standard XAUUSD)
        """
        params = TradeParams(direction=direction, entry_price=entry_price)

        # Calculate SL distance (ATR * multiplier)
        sl_distance = atr * self.cfg["atr_sl_multiplier"]
        sl_distance += self.cfg["slippage_buffer_pips"] * 0.01  # Add slippage buffer

        # Calculate TP distance (ATR * multiplier)
        tp_distance = atr * self.cfg["atr_tp_multiplier"]

        # Set SL and TP levels
        if direction == "BUY":
            params.stop_loss = entry_price - sl_distance
            params.take_profit = entry_price + tp_distance
        elif direction == "SELL":
            params.stop_loss = entry_price + sl_distance
            params.take_profit = entry_price - tp_distance
        else:
            return params  # Invalid direction

        # Calculate pips
        params.sl_pips = sl_distance / 0.01  # Convert to pips
        params.tp_pips = tp_distance / 0.01

        # Risk/Reward ratio
        if sl_distance > 0:
            params.risk_reward_ratio = tp_distance / sl_distance
        else:
            return params  # Invalid

        # Position sizing: risk_amount = account * risk_pct
        risk_amount = self.equity * self.cfg["max_risk_per_trade"]
        params.risk_amount = risk_amount

        # Lot size = risk_amount / (SL in pips * pip_value)
        if params.sl_pips > 0 and pip_value > 0:
            params.lot_size = risk_amount / (params.sl_pips * pip_value)
            params.lot_size = round(params.lot_size, 2)  # Round to 0.01 lots
            params.lot_size = max(params.lot_size, 0.01)  # Minimum lot

        # Reward amount
        params.reward_amount = params.lot_size * params.tp_pips * pip_value

        # Validate
        params.valid = self._validate_trade(params)

        return params

    def _validate_trade(self, params: TradeParams) -> bool:
        """Run all risk checks on a proposed trade."""
        # Check minimum R:R
        if params.risk_reward_ratio < self.cfg["min_risk_reward"]:
            return False

        # Check daily loss limit
        if self.daily_stats.current_loss_pct >= self.cfg["max_daily_loss"]:
            return False

        # Check daily trade count
        if self.daily_stats.trades_taken >= 6:  # From COOLDOWN config
            return False

        # Check lot size is reasonable (not too large)
        max_lot = self.equity / 1000  # Rough safety cap
        if params.lot_size > max_lot:
            params.lot_size = round(max_lot, 2)

        return True

    # ------------------------------------------------------------------
    # SPREAD VALIDATION
    # ------------------------------------------------------------------

    def is_spread_acceptable(self, current_spread_pips: float) -> bool:
        """Check if current spread is within acceptable limits."""
        return current_spread_pips <= self.cfg["max_spread_pips"]

    # ------------------------------------------------------------------
    # TRAILING STOP MANAGEMENT
    # ------------------------------------------------------------------

    def update_trailing_stop(self, direction: str, current_price: float,
                             entry_price: float, atr: float,
                             current_sl: float) -> Optional[float]:
        """
        Update trailing stop if conditions are met.
        Returns new SL price if it should be moved, None otherwise.
        """
        activation_distance = atr * self.cfg["trailing_stop_activation"]
        trail_distance = atr * self.cfg["trailing_stop_distance_atr"]

        if direction == "BUY":
            # Check if price has moved enough to activate trailing
            profit_distance = current_price - entry_price
            if profit_distance >= activation_distance:
                self.trailing_activated = True
                new_sl = current_price - trail_distance
                # Only move SL up, never down
                if new_sl > current_sl:
                    self.trailing_stop = new_sl
                    return new_sl

        elif direction == "SELL":
            profit_distance = entry_price - current_price
            if profit_distance >= activation_distance:
                self.trailing_activated = True
                new_sl = current_price + trail_distance
                # Only move SL down (lower number = better for short)
                if new_sl < current_sl:
                    self.trailing_stop = new_sl
                    return new_sl

        return None

    # ------------------------------------------------------------------
    # P&L TRACKING
    # ------------------------------------------------------------------

    def record_trade_result(self, pnl: float):
        """Record a closed trade result and update daily stats."""
        self.equity += pnl
        self.daily_stats.gross_pnl += pnl
        self.daily_stats.trades_taken += 1

        if pnl >= 0:
            self.daily_stats.wins += 1
        else:
            self.daily_stats.losses += 1

        # Update peak equity and drawdown
        if self.equity > self.daily_stats.peak_equity:
            self.daily_stats.peak_equity = self.equity

        # Current loss from starting balance
        if self.account_balance > 0:
            self.daily_stats.current_loss_pct = max(
                0, (self.account_balance - self.equity) / self.account_balance
            )

        # Max drawdown from peak
        if self.daily_stats.peak_equity > 0:
            dd = (self.daily_stats.peak_equity - self.equity) / self.daily_stats.peak_equity
            self.daily_stats.max_drawdown_pct = max(self.daily_stats.max_drawdown_pct, dd)

        # Reset trailing stop
        self.trailing_stop = None
        self.trailing_activated = False

    def is_daily_limit_hit(self) -> bool:
        """Check if daily loss limit has been reached."""
        return self.daily_stats.current_loss_pct >= self.cfg["max_daily_loss"]

    def get_daily_loss_pct(self) -> float:
        return self.daily_stats.current_loss_pct

    def get_trade_count(self) -> int:
        return self.daily_stats.trades_taken

    # ------------------------------------------------------------------
    # COOLDOWN CALCULATION
    # ------------------------------------------------------------------

    def get_cooldown_duration(self, last_result: str, cooldown_config: dict) -> timedelta:
        """Calculate cooldown duration based on last trade result."""
        if last_result == "LOSS":
            return timedelta(minutes=cooldown_config["after_loss_minutes"])
        elif last_result == "WIN":
            return timedelta(minutes=cooldown_config["after_win_minutes"])
        return timedelta(minutes=15)  # Default

    # ------------------------------------------------------------------
    # DAILY RESET
    # ------------------------------------------------------------------

    def reset_daily(self):
        """Reset daily stats (call at start of new trading day)."""
        self.account_balance = self.equity  # New day starts with current equity
        self.daily_stats = DailyStats(
            date=datetime.utcnow().strftime("%Y-%m-%d"),
            peak_equity=self.equity,
        )

    def get_stats(self) -> DailyStats:
        return self.daily_stats
