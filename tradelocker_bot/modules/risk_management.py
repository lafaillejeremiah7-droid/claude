"""
Risk Management Module

The foundation of the entire strategy. Implements:
- 2% account risk per trade (configurable)
- Position sizing based on entry-to-stop-loss distance
- Stop loss placement (swing high/low or ATR-based, whichever is wider)
- Take profit at 1.5R minimum, 2R preferred
- Breakeven trigger at 1R
- Daily drawdown limit (4%)
- Weekly drawdown limit (4%)
- Max 2 trades per day
- Daily P&L tracking
"""
import logging
import json
import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    RISK_PERCENT,
    MIN_RISK_PERCENT,
    MAX_RISK_PERCENT,
    MAX_TRADES_PER_DAY,
    DAILY_DRAWDOWN_LIMIT,
    WEEKLY_DRAWDOWN_LIMIT,
    MIN_RR_RATIO,
    PREFERRED_RR_RATIO,
    BREAKEVEN_TRIGGER,
    ATR_PERIOD,
    SWING_LOOKBACK,
)
from modules.indicators import get_recent_swing_high, get_recent_swing_low, calculate_atr

logger = logging.getLogger(__name__)

# Path for persisting daily stats
STATS_FILE = Path(__file__).parent.parent / "logs" / "daily_stats.json"

# Default confidence gate below which a trade is not taken (mirrors the
# adaptive engine's min_confidence). Used as the lower anchor for scaling risk.
CONFIDENCE_GATE = 8.0
CONFIDENCE_MAX = 10.0


def confidence_to_risk_pct(
    confidence: float,
    gate: float = CONFIDENCE_GATE,
    min_pct: float = MIN_RISK_PERCENT,
    max_pct: float = MAX_RISK_PERCENT,
) -> float:
    """
    Map an adaptive confidence score to a risk percentage.

    Risk scales linearly between ``min_pct`` (at the gate) and ``max_pct`` (at a
    perfect 10.0 score):

        pct = min_pct + ((clamp(conf, gate, 10) - gate) / (10 - gate)) * (max_pct - min_pct)

    With the defaults (gate=8.0, min=1.0, max=3.0):
        - conf 8.0 -> 1.0%  ($100 on $10k)
        - conf 9.0 -> 2.0%  ($200 on $10k)
        - conf 10.0 -> 3.0% ($300 on $10k)

    Confidence is clamped to [gate, 10] so values below the gate still map to
    ``min_pct`` (never below) and values above 10 map to ``max_pct``.

    Args:
        confidence: Adaptive confidence score (0-10).
        gate: Lower confidence anchor (maps to min_pct).
        min_pct: Risk percent at the gate.
        max_pct: Risk percent at a perfect score.

    Returns:
        Risk percentage to use for position sizing.
    """
    span = CONFIDENCE_MAX - gate
    if span <= 0:
        return max_pct

    clamped = min(max(confidence, gate), CONFIDENCE_MAX)
    fraction = (clamped - gate) / span
    return min_pct + fraction * (max_pct - min_pct)


@dataclass
class TradeSetup:
    """Complete trade setup with risk parameters calculated."""
    symbol: str
    direction: str  # 'buy' or 'sell'
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float  # Lot size
    risk_amount: float  # Dollar amount at risk
    risk_reward_ratio: float
    sl_distance: float  # Distance in price from entry to SL
    tp_distance: float  # Distance in price from entry to TP
    sl_method: str  # 'swing' or 'atr'
    valid: bool = True
    rejection_reason: Optional[str] = None


@dataclass
class DailyStats:
    """Track daily trading statistics for risk limits."""
    date: str = ""
    trades_taken: int = 0
    wins: int = 0
    losses: int = 0
    consecutive_losses: int = 0
    realized_pnl: float = 0.0
    starting_equity: float = 0.0
    current_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    is_locked: bool = False  # True if daily limit hit
    lock_reason: str = ""
    trade_ids: list = field(default_factory=list)

    @property
    def daily_return_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        return ((self.current_equity - self.starting_equity) / self.starting_equity) * 100

    @property
    def drawdown_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        loss = self.starting_equity - self.current_equity
        if loss <= 0:
            return 0.0
        return (loss / self.starting_equity) * 100


@dataclass
class WeeklyStats:
    """Track weekly trading statistics."""
    week_start: str = ""
    starting_equity: float = 0.0
    current_equity: float = 0.0
    total_trades: int = 0
    is_locked: bool = False
    lock_reason: str = ""

    @property
    def drawdown_pct(self) -> float:
        if self.starting_equity <= 0:
            return 0.0
        loss = self.starting_equity - self.current_equity
        if loss <= 0:
            return 0.0
        return (loss / self.starting_equity) * 100


class RiskManager:
    """
    Manages all risk-related calculations and enforces trading limits.
    """

    def __init__(self, risk_percent: float = RISK_PERCENT, stats_file: Optional[Path] = None):
        self.risk_percent = risk_percent
        # Allow a custom stats file so parallel runs (e.g. the paper-trading
        # engine) can persist their own daily/weekly stats without ever
        # overwriting the live stats file.
        self.stats_file = Path(stats_file) if stats_file is not None else STATS_FILE
        self.daily_stats = DailyStats()
        self.weekly_stats = WeeklyStats()
        self._load_stats()

    # ========================================
    # POSITION SIZING
    # ========================================

    def calculate_position_size(
        self,
        account_equity: float,
        entry_price: float,
        stop_loss_price: float,
        pip_size: float = 0.01,
        lot_size: float = 1.0,
        min_lot: float = 0.01,
        lot_step: float = 0.01,
        risk_percent: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Calculate position size based on percentage risk.

        Position Size = (Account Equity * Risk%) / (Entry - SL distance)

        The dollar amount at risk stays constant regardless of SL distance.

        Args:
            account_equity: Current account equity
            entry_price: Planned entry price
            stop_loss_price: Planned stop loss price
            pip_size: Instrument pip size
            lot_size: Contract/lot size
            min_lot: Minimum lot size
            lot_step: Lot size increment
            risk_percent: Optional override for the risk percentage. When None,
                the manager's configured ``self.risk_percent`` is used (fixed
                behavior). Confidence-scaled sizing passes an explicit value here.

        Returns:
            Tuple of (lot_quantity, dollar_risk)
        """
        effective_risk_pct = self.risk_percent if risk_percent is None else risk_percent
        risk_amount = account_equity * (effective_risk_pct / 100.0)
        sl_distance = abs(entry_price - stop_loss_price)

        if sl_distance <= 0:
            logger.error("Stop loss distance is zero or negative")
            return min_lot, risk_amount

        # Calculate raw position size
        # risk_amount = position_size * sl_distance
        # position_size = risk_amount / sl_distance
        raw_size = risk_amount / sl_distance

        # Round down to nearest lot step
        position_size = max(
            min_lot,
            round(int(raw_size / lot_step) * lot_step, 8)
        )

        # Recalculate actual risk with rounded size
        actual_risk = position_size * sl_distance

        logger.info(
            f"Position sizing: equity={account_equity:.2f}, risk%={effective_risk_pct}%, "
            f"risk$={risk_amount:.2f}, SL dist={sl_distance:.5f}, "
            f"size={position_size:.4f}, actual_risk={actual_risk:.2f}"
        )

        return position_size, actual_risk

    # ========================================
    # STOP LOSS CALCULATION
    # ========================================

    def calculate_stop_loss(
        self,
        df_5m: pd.DataFrame,
        direction: str,
        entry_price: float,
    ) -> tuple[float, str]:
        """
        Calculate stop loss price.

        Uses the WIDER of:
        1. Beyond the most recent swing high (for shorts) or swing low (for longs)
        2. One ATR(14) from entry

        Never places SL too tight - always gives room for normal fluctuations.

        Args:
            df_5m: 5-minute DataFrame
            direction: 'buy' or 'sell'
            entry_price: Entry price

        Returns:
            Tuple of (stop_loss_price, method used)
        """
        # Calculate ATR-based SL
        atr = calculate_atr(df_5m, ATR_PERIOD)
        current_atr = atr.iloc[-1] if len(atr) > 0 else 0

        if direction == "buy":
            # SL below entry for longs
            atr_sl = entry_price - current_atr

            # Swing-based SL: below recent swing low
            swing_low = get_recent_swing_low(df_5m, SWING_LOOKBACK)
            if swing_low is not None:
                # Add small buffer below swing low (0.05% of price)
                swing_sl = swing_low - (entry_price * 0.0005)
            else:
                swing_sl = atr_sl  # Default to ATR if no swing found

            # Use whichever provides MORE protection (is further from entry)
            if swing_sl < atr_sl:
                return swing_sl, "swing"
            else:
                return atr_sl, "atr"

        else:  # sell
            # SL above entry for shorts
            atr_sl = entry_price + current_atr

            # Swing-based SL: above recent swing high
            swing_high = get_recent_swing_high(df_5m, SWING_LOOKBACK)
            if swing_high is not None:
                swing_sl = swing_high + (entry_price * 0.0005)
            else:
                swing_sl = atr_sl

            # Use whichever provides MORE protection (is further from entry)
            if swing_sl > atr_sl:
                return swing_sl, "swing"
            else:
                return atr_sl, "atr"

    # ========================================
    # TAKE PROFIT CALCULATION
    # ========================================

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss_price: float,
        direction: str,
        trend_confidence: float = 0.5,
    ) -> float:
        """
        Calculate take profit target.

        Minimum: 1.5R
        Preferred: 2R (used when trend is strong)

        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            direction: 'buy' or 'sell'
            trend_confidence: Trend confidence score (0-1)

        Returns:
            Take profit price
        """
        sl_distance = abs(entry_price - stop_loss_price)

        # Use 2R when trend confidence is high (>0.7), otherwise 1.5R
        if trend_confidence >= 0.7:
            rr_ratio = PREFERRED_RR_RATIO
        else:
            rr_ratio = MIN_RR_RATIO

        tp_distance = sl_distance * rr_ratio

        if direction == "buy":
            take_profit = entry_price + tp_distance
        else:
            take_profit = entry_price - tp_distance

        logger.info(
            f"TP calculated: {rr_ratio}R | entry={entry_price:.2f}, "
            f"SL={stop_loss_price:.2f}, TP={take_profit:.2f}, "
            f"SL_dist={sl_distance:.5f}, TP_dist={tp_distance:.5f}"
        )

        return take_profit

    # ========================================
    # BREAKEVEN CALCULATION
    # ========================================

    def calculate_breakeven_price(
        self, entry_price: float, direction: str, spread: float = 0.0
    ) -> float:
        """
        Calculate breakeven price (entry + spread to cover costs).

        Args:
            entry_price: Original entry price
            direction: 'buy' or 'sell'
            spread: Approximate spread to add for true breakeven

        Returns:
            Breakeven price
        """
        if direction == "buy":
            return entry_price + spread
        else:
            return entry_price - spread

    def should_move_to_breakeven(
        self,
        entry_price: float,
        current_price: float,
        stop_loss_price: float,
        direction: str,
    ) -> bool:
        """
        Check if price has reached 1R profit, triggering breakeven move.

        Args:
            entry_price: Original entry price
            current_price: Current market price
            stop_loss_price: Original stop loss price
            direction: 'buy' or 'sell'

        Returns:
            True if should move SL to breakeven
        """
        sl_distance = abs(entry_price - stop_loss_price)
        target_1r = sl_distance * BREAKEVEN_TRIGGER

        if direction == "buy":
            profit = current_price - entry_price
        else:
            profit = entry_price - current_price

        return profit >= target_1r

    # ========================================
    # COMPLETE TRADE SETUP
    # ========================================

    def create_trade_setup(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        df_5m: pd.DataFrame,
        account_equity: float,
        trend_confidence: float = 0.5,
        pip_size: float = 0.01,
        lot_size: float = 1.0,
        min_lot: float = 0.01,
        lot_step: float = 0.01,
        confidence: Optional[float] = None,
        confidence_gate: float = CONFIDENCE_GATE,
    ) -> TradeSetup:
        """
        Create a complete trade setup with all risk parameters.

        Args:
            symbol: Instrument symbol
            direction: 'buy' or 'sell'
            entry_price: Entry price
            df_5m: 5-minute DataFrame for SL calculation
            account_equity: Current equity
            trend_confidence: Confidence score from trend analysis
            pip_size: Instrument pip size
            lot_size: Contract size
            min_lot: Minimum lot
            lot_step: Lot increment
            confidence: Optional adaptive confidence score (0-10). When provided,
                the trade is sized using confidence-scaled risk
                (``confidence_to_risk_pct``) between MIN_RISK_PERCENT and
                MAX_RISK_PERCENT. When None, the manager falls back to the fixed
                ``self.risk_percent`` behavior (backward compatible).
            confidence_gate: Confidence gate used as the lower anchor for scaling.

        Returns:
            TradeSetup with all parameters calculated
        """
        # Calculate stop loss
        stop_loss, sl_method = self.calculate_stop_loss(df_5m, direction, entry_price)

        # Calculate take profit
        take_profit = self.calculate_take_profit(
            entry_price, stop_loss, direction, trend_confidence
        )

        # Determine risk percentage: confidence-scaled when a score is provided,
        # otherwise fall back to the fixed configured risk percent.
        risk_percent_override = None
        if confidence is not None:
            risk_percent_override = confidence_to_risk_pct(
                confidence, gate=confidence_gate
            )
            logger.info(
                f"Confidence-scaled risk: confidence={confidence:.2f} -> "
                f"{risk_percent_override:.2f}% (gate={confidence_gate})"
            )

        # Calculate position size
        position_size, risk_amount = self.calculate_position_size(
            account_equity, entry_price, stop_loss,
            pip_size, lot_size, min_lot, lot_step,
            risk_percent=risk_percent_override,
        )

        # RISK vs DAILY-DRAWDOWN INTERACTION:
        # The 4% daily drawdown lock (DAILY_DRAWDOWN_LIMIT) is enforced separately
        # in can_trade(). With confidence-scaled sizing a single 3% trade that
        # stops out consumes 3% of the 4% daily headroom, and combined with an
        # earlier loss it can trip the daily lock. We surface a WARNING when a
        # trade's risk exceeds the remaining daily-drawdown headroom so the
        # operator is aware a stop-out could halt trading for the day. We do NOT
        # shrink the trade or change DAILY_DRAWDOWN_LIMIT here.
        if risk_amount > 0 and self.daily_stats.starting_equity > 0:
            remaining_dd_pct = DAILY_DRAWDOWN_LIMIT - self.daily_stats.drawdown_pct
            remaining_headroom = self.daily_stats.starting_equity * (
                remaining_dd_pct / 100.0
            )
            if risk_amount > remaining_headroom:
                logger.warning(
                    f"Trade risk ${risk_amount:.2f} exceeds remaining daily "
                    f"drawdown headroom ${remaining_headroom:.2f} "
                    f"({remaining_dd_pct:.2f}% of {DAILY_DRAWDOWN_LIMIT}% limit). "
                    f"A stop-out on this trade could trip the daily drawdown lock."
                )

        # Calculate distances and R:R
        sl_distance = abs(entry_price - stop_loss)
        tp_distance = abs(take_profit - entry_price)
        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0

        # Validate minimum R:R
        if rr_ratio < MIN_RR_RATIO:
            return TradeSetup(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=0,
                risk_amount=0,
                risk_reward_ratio=rr_ratio,
                sl_distance=sl_distance,
                tp_distance=tp_distance,
                sl_method=sl_method,
                valid=False,
                rejection_reason=f"R:R too low ({rr_ratio:.2f} < {MIN_RR_RATIO})",
            )

        setup = TradeSetup(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            position_size=position_size,
            risk_amount=risk_amount,
            risk_reward_ratio=rr_ratio,
            sl_distance=sl_distance,
            tp_distance=tp_distance,
            sl_method=sl_method,
            valid=True,
        )

        logger.info(
            f"Trade setup: {direction.upper()} {symbol} | "
            f"Entry={entry_price:.2f} SL={stop_loss:.2f} TP={take_profit:.2f} | "
            f"Size={position_size:.4f} Risk=${risk_amount:.2f} R:R={rr_ratio:.2f} | "
            f"SL method={sl_method}"
        )

        return setup

    # ========================================
    # DAILY/WEEKLY LIMITS
    # ========================================

    def can_trade(self, current_equity: float) -> tuple[bool, str]:
        """
        Check if trading is allowed based on all risk limits.

        Checks:
        1. Max trades per day (2)
        2. Daily drawdown limit (4%)
        3. Weekly drawdown limit (4%)
        4. Consecutive loss lock (2 in a row)

        Args:
            current_equity: Current account equity

        Returns:
            Tuple of (can_trade, reason_if_not)
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Reset daily stats if new day
        if self.daily_stats.date != today:
            self._reset_daily_stats(today, current_equity)

        # Update current equity
        self.daily_stats.current_equity = current_equity

        # Check if already locked
        if self.daily_stats.is_locked:
            return False, f"Daily trading locked: {self.daily_stats.lock_reason}"

        # Check max trades per day
        if self.daily_stats.trades_taken >= MAX_TRADES_PER_DAY:
            self.daily_stats.is_locked = True
            self.daily_stats.lock_reason = f"Max daily trades reached ({MAX_TRADES_PER_DAY})"
            self._save_stats()
            return False, self.daily_stats.lock_reason

        # Check consecutive losses
        if self.daily_stats.consecutive_losses >= 2:
            self.daily_stats.is_locked = True
            self.daily_stats.lock_reason = "Two consecutive losses"
            self._save_stats()
            return False, self.daily_stats.lock_reason

        # Check daily drawdown
        daily_dd = self.daily_stats.drawdown_pct
        if daily_dd >= DAILY_DRAWDOWN_LIMIT:
            self.daily_stats.is_locked = True
            self.daily_stats.lock_reason = f"Daily drawdown limit hit ({daily_dd:.2f}% >= {DAILY_DRAWDOWN_LIMIT}%)"
            self._save_stats()
            return False, self.daily_stats.lock_reason

        # Check weekly drawdown
        self._update_weekly_stats(current_equity)
        weekly_dd = self.weekly_stats.drawdown_pct
        if weekly_dd >= WEEKLY_DRAWDOWN_LIMIT:
            self.weekly_stats.is_locked = True
            self.weekly_stats.lock_reason = f"Weekly drawdown limit hit ({weekly_dd:.2f}% >= {WEEKLY_DRAWDOWN_LIMIT}%)"
            self._save_stats()
            return False, f"Weekly trading locked: {self.weekly_stats.lock_reason}"

        if self.weekly_stats.is_locked:
            return False, f"Weekly trading locked: {self.weekly_stats.lock_reason}"

        return True, "Trading allowed"

    def record_trade_opened(self, trade_id: str):
        """Record that a trade has been opened."""
        self.daily_stats.trades_taken += 1
        self.daily_stats.trade_ids.append(trade_id)
        self.weekly_stats.total_trades += 1
        self._save_stats()
        logger.info(
            f"Trade recorded: #{self.daily_stats.trades_taken}/{MAX_TRADES_PER_DAY} today"
        )

    def record_trade_closed(self, pnl: float, is_win: bool):
        """Record trade result for drawdown tracking."""
        self.daily_stats.realized_pnl += pnl

        if is_win:
            self.daily_stats.wins += 1
            self.daily_stats.consecutive_losses = 0
        else:
            self.daily_stats.losses += 1
            self.daily_stats.consecutive_losses += 1

        self._save_stats()
        logger.info(
            f"Trade closed: PnL=${pnl:.2f} | "
            f"Day: {self.daily_stats.wins}W/{self.daily_stats.losses}L | "
            f"Consecutive losses: {self.daily_stats.consecutive_losses}"
        )

    # ========================================
    # PERSISTENCE
    # ========================================

    def _reset_daily_stats(self, date: str, equity: float):
        """Reset daily stats for a new trading day."""
        self.daily_stats = DailyStats(
            date=date,
            starting_equity=equity,
            current_equity=equity,
        )
        logger.info(f"New trading day: {date} | Starting equity: ${equity:.2f}")

    def _update_weekly_stats(self, current_equity: float):
        """Update or reset weekly stats."""
        now = datetime.now(timezone.utc)
        # Week starts Monday
        week_start = (now - timedelta(days=now.weekday())).strftime("%Y-%m-%d")

        if self.weekly_stats.week_start != week_start:
            self.weekly_stats = WeeklyStats(
                week_start=week_start,
                starting_equity=current_equity,
                current_equity=current_equity,
            )
            logger.info(f"New trading week: {week_start} | Starting equity: ${current_equity:.2f}")
        else:
            self.weekly_stats.current_equity = current_equity

    def _save_stats(self):
        """Persist stats to file."""
        try:
            data = {
                "daily": asdict(self.daily_stats),
                "weekly": asdict(self.weekly_stats),
            }
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save stats: {e}")

    def _load_stats(self):
        """Load persisted stats."""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, "r") as f:
                    data = json.load(f)

                daily_data = data.get("daily", {})
                self.daily_stats = DailyStats(**daily_data)

                weekly_data = data.get("weekly", {})
                self.weekly_stats = WeeklyStats(**weekly_data)

                logger.info(
                    f"Loaded stats: date={self.daily_stats.date}, "
                    f"trades={self.daily_stats.trades_taken}, "
                    f"locked={self.daily_stats.is_locked}"
                )
        except Exception as e:
            logger.warning(f"Failed to load stats (starting fresh): {e}")
            self.daily_stats = DailyStats()
            self.weekly_stats = WeeklyStats()

    def get_status_summary(self) -> dict:
        """Get current risk management status."""
        return {
            "date": self.daily_stats.date,
            "trades_today": f"{self.daily_stats.trades_taken}/{MAX_TRADES_PER_DAY}",
            "daily_pnl": f"${self.daily_stats.realized_pnl:.2f}",
            "daily_drawdown": f"{self.daily_stats.drawdown_pct:.2f}%",
            "weekly_drawdown": f"{self.weekly_stats.drawdown_pct:.2f}%",
            "consecutive_losses": self.daily_stats.consecutive_losses,
            "daily_locked": self.daily_stats.is_locked,
            "weekly_locked": self.weekly_stats.is_locked,
            "risk_per_trade": f"{self.risk_percent}%",
        }
