"""
Paper Trading Engine (for --dry mode)

Simulates the FULL trade lifecycle using REAL live prices, writing to PARALLEL
files that mirror the live schemas so the dashboard/reports can read them by
switching mode (MODE=paper). Live files are NEVER touched by this engine.

Parallel (namespaced) files:
- logs/paper_active_positions.json  -> mirrors logs/active_positions.json
- journal/paper_journal_YYYY-MM-DD.jsonl -> mirrors journal/journal_YYYY-MM-DD.jsonl
- logs/paper_daily_stats.json -> mirrors logs/daily_stats.json (daily + weekly)

Lifecycle handled here:
1. Open a PAPER position at the real latest price using a confidence-scaled setup.
2. Each scan cycle, manage open paper positions against the REAL live price:
   - Move SL to breakeven at 1R (BREAKEVEN journal entry).
   - CLOSE when price crosses SL or TP.
3. On close compute pnl / is_win / r_multiple, journal CLOSE, and update paper
   daily/weekly stats (realized_pnl, wins/losses, consecutive_losses, equity).

The engine respects max-trades/day, drawdown and consecutive-loss locks using
PAPER stats so the paper run mirrors the real trading constraints.
"""
import json
import logging
import uuid
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from config import PAPER_STARTING_EQUITY, RISK_PERCENT
from modules.risk_management import RiskManager, TradeSetup, CONFIDENCE_GATE
from modules.trade_manager import ManagedPosition

logger = logging.getLogger(__name__)

# Default parallel file locations (mirror the live layout under the project root)
_BASE_DIR = Path(__file__).parent.parent
DEFAULT_LOGS_DIR = _BASE_DIR / "logs"
DEFAULT_JOURNAL_DIR = _BASE_DIR / "journal"


class PaperTradeManager:
    """
    Simulates trade execution and management for --dry mode.

    Mirrors ``TradeManager``'s journal/close semantics but writes to paper
    (namespaced) files and sizes trades off a simulated paper equity.
    """

    def __init__(
        self,
        client=None,
        starting_equity: float = PAPER_STARTING_EQUITY,
        risk_percent: float = RISK_PERCENT,
        logs_dir: Optional[Path] = None,
        journal_dir: Optional[Path] = None,
    ):
        self.client = client
        self.starting_equity = float(starting_equity)

        self.logs_dir = Path(logs_dir) if logs_dir is not None else DEFAULT_LOGS_DIR
        self.journal_dir = Path(journal_dir) if journal_dir is not None else DEFAULT_JOURNAL_DIR

        self.positions_file = self.logs_dir / "paper_active_positions.json"
        self.stats_file = self.logs_dir / "paper_daily_stats.json"

        # Dedicated risk manager persisting to the PAPER stats file so paper
        # locks/drawdown are tracked independently of the live account.
        self.risk_manager = RiskManager(
            risk_percent=risk_percent, stats_file=self.stats_file
        )

        self.active_positions: dict[str, ManagedPosition] = {}
        self._load_positions()

    # ========================================
    # PAPER EQUITY
    # ========================================

    @property
    def current_equity(self) -> float:
        """Paper equity = starting equity + realized paper PnL."""
        return self.starting_equity + self.risk_manager.daily_stats.realized_pnl

    # ========================================
    # POSITION OPENING
    # ========================================

    def open_from_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        df_5m,
        trend_confidence: float = 0.5,
        confidence: Optional[float] = None,
        pip_size: float = 0.01,
        lot_size: float = 1.0,
        min_lot: float = 0.01,
        lot_step: float = 0.01,
        entry_reasons: Optional[list] = None,
    ) -> Optional[str]:
        """
        Build a confidence-scaled setup off PAPER equity and open a paper position.

        Returns the paper position id, or None if the setup is invalid or a
        paper trading lock (max trades / drawdown / consecutive losses) is active.
        """
        # Respect paper trading limits using paper stats.
        allowed, reason = self.risk_manager.can_trade(self.current_equity)
        if not allowed:
            logger.info(f"[PAPER] Trade blocked: {reason}")
            return None

        setup = self.risk_manager.create_trade_setup(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            df_5m=df_5m,
            account_equity=self.current_equity,
            trend_confidence=trend_confidence,
            pip_size=pip_size,
            lot_size=lot_size,
            min_lot=min_lot,
            lot_step=lot_step,
            confidence=confidence,
            confidence_gate=CONFIDENCE_GATE,
        )

        return self.open_position(setup, entry_reasons=entry_reasons)

    def open_position(
        self, setup: TradeSetup, entry_reasons: Optional[list] = None
    ) -> Optional[str]:
        """Open a paper position from a pre-built TradeSetup."""
        if not setup.valid:
            logger.warning(
                f"[PAPER] Cannot open position: setup invalid - {setup.rejection_reason}"
            )
            return None

        # Respect paper trading limits (in case open_position is called directly).
        allowed, reason = self.risk_manager.can_trade(self.current_equity)
        if not allowed:
            logger.info(f"[PAPER] Trade blocked: {reason}")
            return None

        position_id = f"paper-{uuid.uuid4()}"

        position = ManagedPosition(
            position_id=position_id,
            symbol=setup.symbol,
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            quantity=setup.position_size,
            risk_amount=setup.risk_amount,
            risk_reward_ratio=setup.risk_reward_ratio,
            entry_time=datetime.now(timezone.utc).isoformat(),
            sl_method=setup.sl_method,
            entry_reasons=entry_reasons or [],
            trend_confidence=0.0,
        )

        self.active_positions[position_id] = position
        self._save_positions()

        self.risk_manager.record_trade_opened(position_id)
        self._journal_entry("OPEN", position)

        logger.info(
            f"[PAPER] POSITION OPENED: {setup.direction.upper()} "
            f"{setup.position_size} {setup.symbol} | "
            f"Entry={setup.entry_price:.2f} SL={setup.stop_loss:.2f} "
            f"TP={setup.take_profit:.2f} | Risk=${setup.risk_amount:.2f} "
            f"R:R={setup.risk_reward_ratio:.2f} | Equity=${self.current_equity:.2f} | "
            f"ID={position_id}"
        )

        return position_id

    # ========================================
    # POSITION MANAGEMENT
    # ========================================

    def _current_price(self, position: ManagedPosition, price_overrides: Optional[dict]):
        """
        Resolve the current market price for a symbol.

        Uses price_overrides[symbol] when provided (hermetic tests), otherwise
        fetches the real live quote from the client. Longs exit on the bid,
        shorts exit on the ask (mirrors TradeManager).
        """
        if price_overrides and position.symbol in price_overrides:
            return float(price_overrides[position.symbol])

        if self.client is None:
            return None

        price_data = self.client.get_latest_price(position.symbol)
        if not price_data:
            return None

        if position.direction == "buy":
            return float(price_data.get("bid", price_data.get("mid", 0)))
        return float(price_data.get("ask", price_data.get("mid", 0)))

    def manage_open_positions(self, price_overrides: Optional[dict] = None) -> list:
        """
        Manage all open paper positions against the REAL live price.

        - Close positions whose price has crossed SL or TP (exit AT the level).
        - Move SL to breakeven once 1R profit is reached.

        Returns a list of dicts describing the trades closed this cycle (used to
        feed the adaptive engine / performance reporter).
        """
        closed_trades = []
        closed_ids = []

        for pos_id, position in list(self.active_positions.items()):
            current_price = self._current_price(position, price_overrides)
            if current_price is None:
                continue

            exit_price = self._check_exit(position, current_price)
            if exit_price is not None:
                closed = self._close_position(position, exit_price)
                closed_trades.append(closed)
                closed_ids.append(pos_id)
                continue

            # Still open - check for breakeven move at 1R.
            if not position.is_breakeven:
                self._check_breakeven(position, current_price)

        for pos_id in closed_ids:
            self.active_positions.pop(pos_id, None)

        if closed_ids:
            self._save_positions()

        return closed_trades

    def _check_exit(self, position: ManagedPosition, current_price: float) -> Optional[float]:
        """
        Determine whether the position should close given the current price.

        Returns the exit price (the SL or TP level crossed) or None if still open.
        """
        if position.direction == "buy":
            if current_price <= position.stop_loss:
                return position.stop_loss
            if current_price >= position.take_profit:
                return position.take_profit
        else:  # sell / short
            if current_price >= position.stop_loss:
                return position.stop_loss
            if current_price <= position.take_profit:
                return position.take_profit
        return None

    def _check_breakeven(self, position: ManagedPosition, current_price: float):
        """Move SL to breakeven (entry) once price reaches 1R profit."""
        should_be = self.risk_manager.should_move_to_breakeven(
            entry_price=position.entry_price,
            current_price=current_price,
            stop_loss_price=position.stop_loss,
            direction=position.direction,
        )
        if not should_be:
            return

        breakeven_price = position.entry_price
        position.is_breakeven = True
        position.stop_loss = breakeven_price
        self._save_positions()

        self._journal_entry(
            "BREAKEVEN",
            position,
            {
                "new_sl": breakeven_price,
                "current_price": current_price,
                "profit_at_move": abs(current_price - position.entry_price),
            },
        )
        logger.info(
            f"[PAPER] BREAKEVEN MOVED: {position.symbol} {position.direction} | "
            f"New SL={breakeven_price:.2f} | Current price={current_price:.2f}"
        )

    def _close_position(self, position: ManagedPosition, exit_price: float) -> dict:
        """Close a paper position, journal it, and update paper stats."""
        if position.direction == "buy":
            pnl = (exit_price - position.entry_price) * position.quantity
        else:
            pnl = (position.entry_price - exit_price) * position.quantity

        is_win = pnl > 0
        r_multiple = pnl / position.risk_amount if position.risk_amount > 0 else 0.0

        # Update paper daily/weekly stats.
        self.risk_manager.record_trade_closed(pnl, is_win)

        extra = {
            "exit_price": exit_price,
            "pnl": pnl,
            "is_win": is_win,
            "r_multiple": r_multiple,
            "was_breakeven": position.is_breakeven,
        }
        self._journal_entry("CLOSE", position, extra)

        result = "WIN" if is_win else "LOSS"
        logger.info(
            f"[PAPER] POSITION CLOSED ({result}): {position.symbol} {position.direction} | "
            f"Entry={position.entry_price:.2f} Exit={exit_price:.2f} | "
            f"PnL=${pnl:.2f} ({r_multiple:.2f}R) | Equity=${self.current_equity:.2f}"
        )

        return {
            "position_id": position.position_id,
            "symbol": position.symbol,
            "direction": position.direction,
            **extra,
        }

    def close_all_positions(self, reason: str = "Manual close", price_overrides: Optional[dict] = None):
        """Emergency-close all paper positions at the current price."""
        for pos_id, position in list(self.active_positions.items()):
            current_price = self._current_price(position, price_overrides)
            if current_price is None:
                current_price = position.stop_loss
            self._close_position(position, current_price)
            self._journal_entry("EMERGENCY_CLOSE", position, {"reason": reason})
        self.active_positions.clear()
        self._save_positions()

    # ========================================
    # JOURNAL
    # ========================================

    def _journal_entry(self, action: str, position: ManagedPosition, extra: dict = None):
        """Write a paper journal entry (same fields as the live journal)."""
        self.journal_dir.mkdir(parents=True, exist_ok=True)

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "symbol": position.symbol,
            "direction": position.direction,
            "position_id": position.position_id,
            "entry_price": position.entry_price,
            "stop_loss": position.stop_loss,
            "take_profit": position.take_profit,
            "quantity": position.quantity,
            "risk_amount": position.risk_amount,
            "risk_reward_ratio": position.risk_reward_ratio,
            "sl_method": position.sl_method,
            "is_breakeven": position.is_breakeven,
            "entry_reasons": position.entry_reasons,
            "mode": "paper",
        }
        if extra:
            entry.update(extra)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        journal_file = self.journal_dir / f"paper_journal_{today}.jsonl"

        try:
            with open(journal_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"[PAPER] Failed to write journal: {e}")

    # ========================================
    # PERSISTENCE
    # ========================================

    def _save_positions(self):
        """Save active paper positions to disk (paper file only)."""
        try:
            self.positions_file.parent.mkdir(parents=True, exist_ok=True)
            data = {pos_id: asdict(pos) for pos_id, pos in self.active_positions.items()}
            with open(self.positions_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[PAPER] Failed to save positions: {e}")

    def _load_positions(self):
        """Load active paper positions from disk."""
        try:
            if self.positions_file.exists():
                with open(self.positions_file, "r") as f:
                    data = json.load(f)
                for pos_id, pos_data in data.items():
                    self.active_positions[pos_id] = ManagedPosition(**pos_data)
                if self.active_positions:
                    logger.info(
                        f"[PAPER] Loaded {len(self.active_positions)} active paper positions"
                    )
        except Exception as e:
            logger.warning(f"[PAPER] Failed to load positions (starting fresh): {e}")
            self.active_positions = {}

    # ========================================
    # STATUS
    # ========================================

    def get_status(self) -> dict:
        """Get current paper trading status."""
        positions_summary = []
        for pos_id, pos in self.active_positions.items():
            positions_summary.append({
                "id": pos_id,
                "symbol": pos.symbol,
                "direction": pos.direction,
                "entry": pos.entry_price,
                "sl": pos.stop_loss,
                "tp": pos.take_profit,
                "qty": pos.quantity,
                "breakeven": pos.is_breakeven,
            })
        return {
            "mode": "paper",
            "starting_equity": self.starting_equity,
            "current_equity": self.current_equity,
            "realized_pnl": self.risk_manager.daily_stats.realized_pnl,
            "active_positions": len(self.active_positions),
            "positions": positions_summary,
        }
