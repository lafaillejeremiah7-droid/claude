"""
Equity Curve Builder.

Requirement 10:
  - Construct equity-curve time series by applying pnl of each CLOSE action
    cumulatively to a starting equity baseline, in ascending timestamp order.
  - Ties in timestamp: preserve file/line order (stable sort).
  - Exclude CLOSE actions with missing/non-numeric/invalid pnl.
  - Produces one equity data point per valid CLOSE action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EquityPoint:
    """A single point on the equity curve."""
    timestamp: str
    equity: float
    pnl: float  # The P&L that produced this point


@dataclass
class EquityCurve:
    """Complete equity curve data."""
    points: List[EquityPoint] = field(default_factory=list)
    starting_equity: float = 0.0
    insufficient_data: bool = False
    error: str = ""


def build_equity_curve(
    journal_entries: List[Dict[str, Any]],
    starting_equity: float = 0.0,
) -> EquityCurve:
    """
    Build equity curve from journal CLOSE actions (Req 10.1-10.7).

    Args:
        journal_entries: All journal entries (from read_all_journals).
        starting_equity: Baseline equity value to start from.

    Returns:
        EquityCurve with ordered data points.
    """
    # Filter CLOSE actions
    close_actions = [
        e for e in journal_entries
        if isinstance(e, dict) and e.get("action") == "CLOSE"
    ]

    if not close_actions:
        return EquityCurve(
            starting_equity=starting_equity,
            insufficient_data=True,
            error="No CLOSE actions available",
        )

    # Sort by timestamp ascending (Req 10.1)
    # Stable sort preserves file/line order for ties (Req 10.2)
    close_actions.sort(key=lambda e: e.get("timestamp", ""))

    # Build cumulative equity series
    points: List[EquityPoint] = []
    cumulative_equity = starting_equity

    for entry in close_actions:
        raw_pnl = entry.get("pnl")
        timestamp = entry.get("timestamp", "")

        # Req 10.3: exclude missing/non-numeric/invalid pnl
        if raw_pnl is None:
            continue

        try:
            pnl_value = float(raw_pnl)
        except (TypeError, ValueError):
            continue

        cumulative_equity += pnl_value
        points.append(EquityPoint(
            timestamp=timestamp,
            equity=cumulative_equity,
            pnl=pnl_value,
        ))

    # Req 10.5: fewer than 2 valid CLOSE actions = insufficient data
    if len(points) < 2:
        return EquityCurve(
            points=points,
            starting_equity=starting_equity,
            insufficient_data=True,
            error="Fewer than 2 valid CLOSE actions",
        )

    return EquityCurve(
        points=points,
        starting_equity=starting_equity,
    )
