"""
Pure derivation and parsing functions for the Trading Dashboard.

Every function in this package is PURE: no file/network I/O, no reads of the
system clock, no global mutable state. Callers pass in already-read data and
(where relevant) an explicit ``now`` timestamp. This is what makes the logic
exhaustively testable with property-based tests (see ``dashboard/tests``).
"""

from .common import is_number
from .timeutil import parse_iso_utc
from .formatting import format_money, format_win_rate, sign_color
from .equity import select_equity, daily_return_pct, cumulative_realized_pnl
from .positions import (
    unrealized_pnl,
    sl_tp_distances,
    is_price_fresh,
    derive_position,
    total_unrealized,
)
from .ordering import order_closes
from .streaks import compute_streaks, Streaks
from .confidence import parse_confidence, classify_gate
from .readers import parse_jsonl_line, parse_jsonl, valid_journal_entries
from .feed import build_feed
from .equity_curve import equity_curve
from .countdown import compute_countdown, resolve_bot_status, hour_in_avoid_hours
from .freshness import data_age, is_stale, is_bot_offline, seconds_since
from .instruments import monitored_instruments, gather_instrument_data
from .reconnect import plan_reconnects

__all__ = [
    "is_number",
    "parse_iso_utc",
    "format_money",
    "format_win_rate",
    "sign_color",
    "select_equity",
    "daily_return_pct",
    "cumulative_realized_pnl",
    "unrealized_pnl",
    "sl_tp_distances",
    "is_price_fresh",
    "derive_position",
    "total_unrealized",
    "order_closes",
    "compute_streaks",
    "Streaks",
    "parse_confidence",
    "classify_gate",
    "parse_jsonl_line",
    "parse_jsonl",
    "valid_journal_entries",
    "build_feed",
    "equity_curve",
    "compute_countdown",
    "resolve_bot_status",
    "hour_in_avoid_hours",
    "data_age",
    "is_stale",
    "is_bot_offline",
    "seconds_since",
    "monitored_instruments",
    "gather_instrument_data",
    "plan_reconnects",
]
