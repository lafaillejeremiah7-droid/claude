"""
P&L Computation module.

Requirement 4:
  - Daily realized P&L from daily_stats.json realized_pnl field.
  - Daily return % from (current_equity - starting_equity) / starting_equity * 100.
  - Cumulative realized P&L = sum of pnl values from all CLOSE actions in journals.
  - Unrealized P&L per open position from entry price, quantity, direction, live price.
  - Color coding: positive=green, negative=red, zero=neutral.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dashboard_backend.readers.api_reader import Quote


@dataclass
class DailyPnL:
    """Daily P&L data from daily_stats.json."""
    realized_pnl: Optional[float] = None
    daily_return_pct: Optional[float] = None
    starting_equity: Optional[float] = None
    current_equity: Optional[float] = None
    error: str = ""


@dataclass
class CumulativePnL:
    """Cumulative realized P&L from all journal CLOSE actions."""
    total: float = 0.0
    trade_count: int = 0


@dataclass
class PositionPnL:
    """Unrealized P&L for a single open position."""
    position_id: str = ""
    symbol: str = ""
    unrealized_pnl: Optional[float] = None
    distance_to_sl: Optional[float] = None
    distance_to_tp: Optional[float] = None
    error: str = ""


@dataclass
class PnLSummary:
    """Complete P&L summary for the dashboard."""
    daily: DailyPnL = field(default_factory=DailyPnL)
    cumulative: CumulativePnL = field(default_factory=CumulativePnL)
    unrealized_total: Optional[float] = None
    positions_pnl: List[PositionPnL] = field(default_factory=list)


def compute_daily_pnl(daily_stats: Dict[str, Any]) -> DailyPnL:
    """
    Compute daily P&L from daily_stats.json (Req 4.1, 4.2, 4.3, 4.8, 4.9).
    """
    daily = daily_stats.get("daily", {})
    if not isinstance(daily, dict):
        return DailyPnL(error="daily object missing")

    # Req 4.1: realized_pnl
    raw_pnl = daily.get("realized_pnl")
    realized_pnl: Optional[float] = None
    if raw_pnl is not None:
        try:
            realized_pnl = float(raw_pnl)
        except (TypeError, ValueError):
            pass  # Req 4.8: display "no data" if non-numeric

    # Req 4.2, 4.3: daily return %
    raw_starting = daily.get("starting_equity")
    raw_current = daily.get("current_equity")
    starting_equity: Optional[float] = None
    current_equity: Optional[float] = None
    daily_return_pct: Optional[float] = None

    try:
        starting_equity = float(raw_starting) if raw_starting is not None else None
    except (TypeError, ValueError):
        starting_equity = None

    try:
        current_equity = float(raw_current) if raw_current is not None else None
    except (TypeError, ValueError):
        current_equity = None

    if starting_equity is not None and current_equity is not None:
        if starting_equity > 0:
            # Req 4.2
            daily_return_pct = ((current_equity - starting_equity) / starting_equity) * 100
        elif starting_equity == 0:
            # Req 4.3
            daily_return_pct = 0.0
        # If starting_equity < 0, leave as None (Req 4.9)

    return DailyPnL(
        realized_pnl=realized_pnl,
        daily_return_pct=daily_return_pct,
        starting_equity=starting_equity,
        current_equity=current_equity,
    )


def compute_cumulative_pnl(journal_entries: List[Dict[str, Any]]) -> CumulativePnL:
    """
    Compute cumulative realized P&L from all CLOSE actions (Req 4.4).
    Excludes entries with absent/null/non-numeric pnl.
    Reports 0 when no valid CLOSE actions exist.
    """
    total = 0.0
    count = 0

    for entry in journal_entries:
        if entry.get("action") != "CLOSE":
            continue

        raw_pnl = entry.get("pnl")
        if raw_pnl is None:
            continue

        try:
            pnl_value = float(raw_pnl)
            total += pnl_value
            count += 1
        except (TypeError, ValueError):
            continue  # Exclude non-numeric

    return CumulativePnL(total=total, trade_count=count)


def compute_unrealized_pnl(
    positions: Dict[str, Any],
    quotes: Dict[str, Quote],
    max_quote_age_seconds: float = 5.0,
) -> List[PositionPnL]:
    """
    Compute unrealized P&L for each open position (Req 4.6, 4.10, 9.3).

    Args:
        positions: Mapping of position_id -> position record from active_positions.json.
        quotes: Current quotes by symbol.
        max_quote_age_seconds: Max age for quote to be considered valid.

    Returns:
        List of PositionPnL entries.
    """
    import time

    results: List[PositionPnL] = []
    now = time.time()

    for pos_id, pos in positions.items():
        if not isinstance(pos, dict):
            continue

        symbol = pos.get("symbol", "")
        direction = pos.get("direction", "").upper()
        entry_price = _parse_float(pos.get("entry_price"))
        quantity = _parse_float(pos.get("quantity"))
        sl = _parse_float(pos.get("stop_loss"))
        tp = _parse_float(pos.get("take_profit"))

        pos_pnl = PositionPnL(position_id=str(pos_id), symbol=symbol)

        # Get live price
        quote = quotes.get(symbol)
        if quote is None or quote.last_price is None:
            pos_pnl.error = "No live price available"
            results.append(pos_pnl)
            continue

        # Check quote freshness (Req 4.10)
        if (now - quote.timestamp) > max_quote_age_seconds:
            pos_pnl.error = "Quote too old"
            results.append(pos_pnl)
            continue

        live_price = quote.last_price

        # Compute unrealized P&L (Req 4.6)
        if entry_price is not None and quantity is not None:
            if direction == "BUY" or direction == "LONG":
                pos_pnl.unrealized_pnl = (live_price - entry_price) * quantity
            elif direction == "SELL" or direction == "SHORT":
                pos_pnl.unrealized_pnl = (entry_price - live_price) * quantity
            else:
                pos_pnl.error = f"Unknown direction: {direction}"

        # Distance to SL and TP (Req 9.3)
        if sl is not None:
            pos_pnl.distance_to_sl = abs(live_price - sl)
        if tp is not None:
            pos_pnl.distance_to_tp = abs(live_price - tp)

        results.append(pos_pnl)

    return results


def _parse_float(value: Any) -> Optional[float]:
    """Safely parse a value to float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
