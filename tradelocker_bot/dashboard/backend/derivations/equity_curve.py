"""Equity-curve construction (Req 10)."""
from __future__ import annotations

from typing import Iterable

from .common import is_number
from .ordering import order_closes


def equity_curve(close_actions: Iterable[dict], baseline: float) -> dict:
    """Build the cumulative equity curve from CLOSE actions.

    - Exactly one point per CLOSE action with a valid numeric ``pnl`` (invalid
      excluded).
    - Points are produced in deterministic ascending timestamp order (ties by
      source appearance key), each equity = ``baseline`` + running sum of valid
      pnl up to and including that action.
    - The final point equals ``baseline + cumulative_realized_pnl``.

    Returns ``{"baseline", "points", "state"}`` where ``state`` is
    ``"ok"`` (>=2 points) or ``"insufficient_data"`` (<2 points).
    """
    base = float(baseline) if is_number(baseline) else 0.0
    ordered = order_closes(close_actions)
    points = []
    running = 0.0
    for action in ordered:
        pnl = action.get("pnl")
        if not is_number(pnl):
            continue
        running += float(pnl)
        points.append({"timestamp_utc": action.get("timestamp"), "equity": base + running})

    state = "ok" if len(points) >= 2 else "insufficient_data"
    return {"baseline": base, "points": points, "state": state}
