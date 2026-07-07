"""
Trade Management Module

Manages the lifecycle of open positions:
1. Track all active positions with entry details
2. Move stop loss to breakeven at 1R profit
3. Handle partial profit taking at 1R
4. Monitor TP/SL hits
5. Log all trade actions to the trading journal
6. Provide position status updates
"""
import logging
import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from modules.api_client import TradeLockerClient
from modules.risk_management import RiskManager, TradeSetup

logger = logging.getLogger(__name__)

JOURNAL_DIR = Path(__file__).parent.parent / "journal"
POSITIONS_FILE = Path(__file__).parent.parent / "logs" / "active_positions.json"


@dataclass
class ManagedPosition:
    """Tracks a position throughout its lifecycle."""
    position_id: str
    symbol: str
    direction: str  # 'buy' or 'sell'
    entry_price: float
    stop_loss: float
    take_profit: float
    quantity: float
    risk_amount: float
    risk_reward_ratio: float

    # State tracking
    is_breakeven: bool = False
    partial_closed: bool = False
    partial_quantity: float = 0.0

    # Metadata
    entry_time: str = ""
    sl_method: str = ""
    entry_reasons: list = field(default_factory=list)
    trend_confidence: float = 0.0

    # Calculated
    @property
    def target_1r_price(self) -> float:
        """Price level at which 1R profit is reached."""
        sl_distance = abs(self.entry_price - self.stop_loss)
        if self.direction == "buy":
            return self.entry_price + sl_distance
        else:
            return self.entry_price - sl_distance

    @property
    def original_sl_distance(self) -> float:
        """Original distance from entry to SL."""
        return abs(self.entry_price - self.stop_loss)


class TradeManager:
    """
    Manages active positions with breakeven management,
    partial profits, and trade journaling.
    """

    def __init__(self, client: TradeLockerClient, risk_manager: RiskManager):
        self.client = client
        self.risk_manager = risk_manager
        self.active_positions: dict[str, ManagedPosition] = {}
        self._load_positions()

    # ========================================
    # POSITION OPENING
    # ========================================

    def open_position(self, setup: TradeSetup, entry_reasons: list = None) -> Optional[str]:
        """
        Execute a trade based on the calculated setup.

        Args:
            setup: Complete TradeSetup from risk management
            entry_reasons: List of reasons for the entry (for journal)

        Returns:
            Position/order ID if successful, None otherwise
        """
        if not setup.valid:
            logger.warning(f"Cannot open position: setup invalid - {setup.rejection_reason}")
            return None

        # Place the order via API
        order_id = self.client.create_order(
            symbol=setup.symbol,
            side=setup.direction,
            quantity=setup.position_size,
            order_type="market",
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
        )

        if not order_id:
            logger.error(f"Failed to place order for {setup.symbol}")
            return None

        # Create managed position
        position = ManagedPosition(
            position_id=order_id,
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

        # Track it
        self.active_positions[order_id] = position
        self._save_positions()

        # Record in risk manager
        self.risk_manager.record_trade_opened(order_id)

        # Journal entry
        self._journal_entry("OPEN", position)

        logger.info(
            f"POSITION OPENED: {setup.direction.upper()} {setup.position_size} {setup.symbol} | "
            f"Entry={setup.entry_price:.2f} SL={setup.stop_loss:.2f} TP={setup.take_profit:.2f} | "
            f"Risk=${setup.risk_amount:.2f} R:R={setup.risk_reward_ratio:.2f} | "
            f"ID={order_id}"
        )

        return order_id

    # ========================================
    # POSITION MONITORING
    # ========================================

    def manage_open_positions(self):
        """
        Check all open positions and apply management rules:
        1. Move SL to breakeven at 1R profit
        2. Handle partial profit at 1R (optional)
        3. Detect closed positions (hit TP/SL)
        """
        if not self.active_positions:
            return

        # Get current positions from API
        api_positions = self.client.get_open_positions()
        api_position_ids = set()

        for pos in api_positions:
            pos_id = str(pos.get("id", pos.get("positionId", "")))
            api_position_ids.add(pos_id)

        # Check each managed position
        closed_ids = []

        for pos_id, managed_pos in self.active_positions.items():
            # Check if position is still open
            if pos_id not in api_position_ids:
                # Position was closed (hit TP, SL, or manually)
                closed_ids.append(pos_id)
                self._handle_position_closed(managed_pos)
                continue

            # Position is still open - check for breakeven move
            if not managed_pos.is_breakeven:
                self._check_breakeven(managed_pos)

        # Remove closed positions from tracking
        for pos_id in closed_ids:
            del self.active_positions[pos_id]

        if closed_ids:
            self._save_positions()

    def _check_breakeven(self, position: ManagedPosition):
        """
        Check if position has reached 1R profit and move SL to breakeven.

        Once price reaches 1R in profit:
        - Move stop loss to entry price (breakeven)
        - This eliminates downside risk
        """
        # Get current price
        price_data = self.client.get_latest_price(position.symbol)
        if not price_data:
            return

        # Use appropriate price based on direction
        if position.direction == "buy":
            current_price = price_data["bid"]  # Exit price for longs
        else:
            current_price = price_data["ask"]  # Exit price for shorts

        # Check if 1R reached
        should_breakeven = self.risk_manager.should_move_to_breakeven(
            entry_price=position.entry_price,
            current_price=current_price,
            stop_loss_price=position.stop_loss,
            direction=position.direction,
        )

        if should_breakeven:
            # Calculate breakeven price (entry + small buffer for spread)
            spread = abs(price_data["ask"] - price_data["bid"])
            breakeven_price = self.risk_manager.calculate_breakeven_price(
                position.entry_price, position.direction, spread * 0.5
            )

            # Modify position SL
            success = self.client.modify_position(
                position_id=position.position_id,
                stop_loss=breakeven_price,
            )

            if success:
                position.is_breakeven = True
                position.stop_loss = breakeven_price
                self._save_positions()
                self._journal_entry("BREAKEVEN", position, {
                    "new_sl": breakeven_price,
                    "current_price": current_price,
                    "profit_at_move": abs(current_price - position.entry_price),
                })
                logger.info(
                    f"BREAKEVEN MOVED: {position.symbol} {position.direction} | "
                    f"New SL={breakeven_price:.2f} | "
                    f"Current price={current_price:.2f}"
                )
            else:
                logger.warning(
                    f"Failed to move SL to breakeven for {position.position_id}"
                )

    def _handle_position_closed(self, position: ManagedPosition):
        """Handle a position that has been closed (TP/SL hit or manual).

        LIMITATION (LIVE MODE): The TradeLocker REST API does not expose the
        exact fill price on a position close event via the positions endpoint.
        We estimate the exit price using the latest bid/ask quote. In volatile
        conditions (BTC can move 0.3%+ in the 60s scan interval), the actual
        fill may differ from this estimate. The order history endpoint is
        queried as a best-effort attempt to get the real fill price; if
        unavailable we fall back to the quote and log a WARNING.
        """
        # Attempt to get actual fill price from order history
        exit_price = None
        try:
            order_history = self.client.get_orders_history()
            for order in order_history:
                order_pos_id = str(order.get("positionId", ""))
                if order_pos_id == position.position_id:
                    fill = order.get("filledPrice") or order.get("avgPrice")
                    if fill:
                        exit_price = float(fill)
                        break
        except Exception as e:
            logger.debug(f"Could not query order history for fill price: {e}")

        # Fallback: use latest quote (estimated, not confirmed fill)
        if exit_price is None:
            price_data = self.client.get_latest_price(position.symbol)
            if price_data:
                if position.direction == "buy":
                    exit_price = price_data["bid"]
                else:
                    exit_price = price_data["ask"]
                logger.warning(
                    f"Exit price for {position.position_id} is ESTIMATED from "
                    f"latest quote (${exit_price:.2f}), not confirmed fill. "
                    f"Actual fill may differ due to slippage/scan delay."
                )

        if exit_price is not None:
            if position.direction == "buy":
                pnl = (exit_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - exit_price) * position.quantity

            is_win = pnl > 0

            # Calculate R multiple
            r_multiple = pnl / position.risk_amount if position.risk_amount > 0 else 0
        else:
            # Can't determine, assume loss for safety
            pnl = 0
            is_win = False
            exit_price = 0
            r_multiple = 0

        # Record in risk manager
        self.risk_manager.record_trade_closed(pnl, is_win)

        # Journal
        self._journal_entry("CLOSE", position, {
            "exit_price": exit_price,
            "pnl": pnl,
            "is_win": is_win,
            "r_multiple": r_multiple,
            "was_breakeven": position.is_breakeven,
        })

        result = "WIN" if is_win else "LOSS"
        logger.info(
            f"POSITION CLOSED ({result}): {position.symbol} {position.direction} | "
            f"Entry={position.entry_price:.2f} Exit≈{exit_price:.2f} | "
            f"PnL≈${pnl:.2f} ({r_multiple:.2f}R)"
        )

    # ========================================
    # PARTIAL PROFIT TAKING
    # ========================================

    def take_partial_profit(self, position_id: str, close_percent: float = 50.0) -> bool:
        """
        Close a portion of the position at 1R profit.

        Args:
            position_id: Position ID
            close_percent: Percentage of position to close (default 50%)

        Returns:
            True if successful
        """
        position = self.active_positions.get(position_id)
        if not position or position.partial_closed:
            return False

        partial_qty = position.quantity * (close_percent / 100.0)

        # Round to lot step (approximate)
        partial_qty = round(partial_qty, 2)

        if partial_qty <= 0:
            return False

        success = self.client.close_position(position_id, quantity=partial_qty)

        if success:
            position.partial_closed = True
            position.partial_quantity = partial_qty
            position.quantity -= partial_qty
            self._save_positions()

            self._journal_entry("PARTIAL_CLOSE", position, {
                "closed_qty": partial_qty,
                "remaining_qty": position.quantity,
                "close_percent": close_percent,
            })

            logger.info(
                f"PARTIAL PROFIT: Closed {partial_qty} of {position.symbol} | "
                f"Remaining: {position.quantity}"
            )
            return True

        return False

    # ========================================
    # MANUAL CLOSE
    # ========================================

    def close_all_positions(self, reason: str = "Manual close"):
        """Emergency close all positions."""
        for pos_id, position in list(self.active_positions.items()):
            success = self.client.close_position(pos_id)
            if success:
                self._journal_entry("EMERGENCY_CLOSE", position, {"reason": reason})
                logger.warning(f"EMERGENCY CLOSE: {position.symbol} - {reason}")

        self.active_positions.clear()
        self._save_positions()

    # ========================================
    # JOURNAL
    # ========================================

    def _journal_entry(self, action: str, position: ManagedPosition, extra: dict = None):
        """
        Write a journal entry for trade actions.

        Creates a JSON log of every trade action for review.
        """
        JOURNAL_DIR.mkdir(parents=True, exist_ok=True)

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
        }

        if extra:
            entry.update(extra)

        # Append to daily journal file
        # NOTE: JSONL appends are not fully atomic. On partial writes, the
        # reader (reporting engine) already skips malformed lines, which handles
        # the corruption case gracefully.
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        journal_file = JOURNAL_DIR / f"journal_{today}.jsonl"

        try:
            with open(journal_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write journal: {e}")

    # ========================================
    # PERSISTENCE
    # ========================================

    def _save_positions(self):
        """Save active positions to disk (atomic write via temp + os.replace)."""
        try:
            POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
            data = {}
            for pos_id, pos in self.active_positions.items():
                data[pos_id] = asdict(pos)

            import tempfile
            tmp_file = POSITIONS_FILE.with_suffix(".tmp")
            with open(tmp_file, "w") as f:
                json.dump(data, f, indent=2)
            import os as _os
            _os.replace(tmp_file, POSITIONS_FILE)
        except Exception as e:
            logger.warning(f"Failed to save positions: {e}")

    def _load_positions(self):
        """Load active positions from disk."""
        try:
            if POSITIONS_FILE.exists():
                with open(POSITIONS_FILE, "r") as f:
                    data = json.load(f)

                for pos_id, pos_data in data.items():
                    self.active_positions[pos_id] = ManagedPosition(**pos_data)

                if self.active_positions:
                    logger.info(
                        f"Loaded {len(self.active_positions)} active positions from disk"
                    )
        except Exception as e:
            logger.warning(f"Failed to load positions (starting fresh): {e}")
            self.active_positions = {}

    # ========================================
    # STATUS
    # ========================================

    def get_status(self) -> dict:
        """Get current trade management status."""
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
            "active_positions": len(self.active_positions),
            "positions": positions_summary,
        }
