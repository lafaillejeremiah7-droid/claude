"""Next-scan countdown and bot-status precedence (Req 11, 14)."""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Iterable, Optional

SCAN_INTERVAL_SECONDS = 60

# Highest-priority first (Req 11, 14 precedence).
STATUS_PRECEDENCE = (
    "initializing",
    "bot_offline",
    "locked",
    "paused_avoid_hours",
    "paused_out_of_session",
    "no_scan_yet",
    "scanning",
)


def compute_countdown(last_scan: Optional[datetime], now: datetime) -> dict:
    """Countdown to the next scan (Req 11.2, 11.3, 11.6).

    - No scan yet -> ``state = "no_scan_yet"``.
    - ``next_scan = last_scan + 60s``.
    - ``now`` before next -> ``state = "scanning"`` with a non-negative whole
      number of ``seconds_remaining`` equal to the whole-second gap.
    - ``now`` at/after next -> ``state = "due"`` with ``seconds_remaining = 0``
      (never negative).
    """
    if last_scan is None:
        return {"state": "no_scan_yet", "seconds_remaining": None, "next_scan_utc": None}

    next_scan = last_scan + timedelta(seconds=SCAN_INTERVAL_SECONDS)
    remaining = (next_scan - now).total_seconds()
    if remaining <= 0:
        return {"state": "due", "seconds_remaining": 0, "next_scan_utc": next_scan}
    return {
        "state": "scanning",
        "seconds_remaining": int(math.floor(remaining)),
        "next_scan_utc": next_scan,
    }


def hour_in_avoid_hours(hour: int, avoid_hours: Optional[Iterable[int]]) -> bool:
    """Whether the given UTC hour is in the adaptive ``avoid_hours`` list (Req 11.5)."""
    if not avoid_hours:
        return False
    try:
        return int(hour) in {int(h) for h in avoid_hours}
    except (TypeError, ValueError):
        return False


def resolve_bot_status(
    *,
    initializing: bool,
    bot_offline: bool,
    is_locked: bool,
    in_avoid_hours: bool,
    out_of_session: bool,
    has_scan_activity: bool,
) -> str:
    """Resolve the bot/countdown status by strict precedence (Req 11.4, 11.5, 11.7, 14).

    Precedence (highest first): initializing > bot_offline > locked >
    paused_avoid_hours > paused_out_of_session > no_scan_yet > scanning.
    """
    if initializing:
        return "initializing"
    if bot_offline:
        return "bot_offline"
    if is_locked:
        return "locked"
    if in_avoid_hours:
        return "paused_avoid_hours"
    if out_of_session:
        return "paused_out_of_session"
    if not has_scan_activity:
        return "no_scan_yet"
    return "scanning"
