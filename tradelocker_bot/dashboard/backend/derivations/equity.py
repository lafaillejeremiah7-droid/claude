"""Equity source selection and PnL derivations (Req 3, 4)."""
from __future__ import annotations

from typing import Any, Iterable, Optional

from .common import is_number


def select_equity(api_state: Optional[dict], daily_stats: Optional[dict]) -> dict:
    """Deterministic equity-source fallback (Req 3.4, 3.5).

    Preference order:
      1. API equity when present, numeric, and non-negative -> source ``"api"``
         (balance and free margin are taken from the API as well).
      2. Otherwise the Daily_Stats ``current_equity`` when numeric ->
         source ``"daily_stats_fallback"`` and balance/free margin UNAVAILABLE.
      3. Otherwise "no data" -> everything unavailable.
    """
    result = {
        "equity": None,
        "equity_available": False,
        "equity_source": "none",
        "balance": None,
        "balance_available": False,
        "free_margin": None,
        "free_margin_available": False,
    }

    api_equity = None if api_state is None else api_state.get("equity")
    if is_number(api_equity) and api_equity >= 0:
        result["equity"] = float(api_equity)
        result["equity_available"] = True
        result["equity_source"] = "api"
        balance = api_state.get("balance")
        if is_number(balance):
            result["balance"] = float(balance)
            result["balance_available"] = True
        free_margin = api_state.get("free_margin")
        if is_number(free_margin):
            result["free_margin"] = float(free_margin)
            result["free_margin_available"] = True
        return result

    fallback_equity = None if daily_stats is None else daily_stats.get("current_equity")
    if is_number(fallback_equity):
        result["equity"] = float(fallback_equity)
        result["equity_available"] = True
        result["equity_source"] = "daily_stats_fallback"
        # Daily_Stats does not provide balance / free margin -> unavailable.
        return result

    return result


def daily_return_pct(starting_equity: Any, current_equity: Any) -> Optional[float]:
    """Daily return percentage (Req 4.2, 4.3, 4.9).

    - ``starting_equity == 0`` -> exactly ``0.00``.
    - ``starting_equity > 0`` and numeric current -> ``((cur - start)/start)*100``.
    - Missing / non-numeric / negative starting equity -> ``None`` (Unavailable).
    """
    if not is_number(starting_equity):
        return None
    start = float(starting_equity)
    if start == 0:
        return 0.00
    if start < 0:
        return None
    if not is_number(current_equity):
        return None
    return ((float(current_equity) - start) / start) * 100.0


def cumulative_realized_pnl(close_actions: Iterable[dict]) -> float:
    """Sum of numeric ``pnl`` across CLOSE actions (Req 4.4).

    Absent / null / non-numeric pnl values are ignored; ``0`` when none exist.
    """
    total = 0.0
    for action in close_actions:
        if not isinstance(action, dict):
            continue
        pnl = action.get("pnl")
        if is_number(pnl):
            total += float(pnl)
    return total
