"""Data freshness and bot-offline thresholds (Req 12.4, 14.1)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

STALE_THRESHOLD_SECONDS = 15.0
OFFLINE_THRESHOLD_SECONDS = 90.0


def seconds_since(past: Optional[datetime], now: datetime) -> Optional[float]:
    """Elapsed seconds since ``past``; ``None`` when ``past`` is unavailable."""
    if past is None:
        return None
    return (now - past).total_seconds()


def data_age(last_update: Optional[datetime], now: datetime) -> Optional[float]:
    """Age of the most recent successful update in seconds (Req 12.4)."""
    return seconds_since(last_update, now)


def is_stale(last_update: Optional[datetime], now: datetime,
             threshold: float = STALE_THRESHOLD_SECONDS) -> bool:
    """Stale iff the data age strictly exceeds the threshold (Req 12.4).

    An unavailable ``last_update`` is treated as stale.
    """
    age = data_age(last_update, now)
    if age is None:
        return True
    return age > threshold


def is_bot_offline(
    last_file_mod: Optional[datetime],
    last_scan: Optional[datetime],
    now: datetime,
    threshold: float = OFFLINE_THRESHOLD_SECONDS,
) -> bool:
    """Offline iff BOTH file-mod age AND scan age exceed the threshold (Req 14.1).

    A missing file-mod or scan time counts as "exceeds threshold" (age = infinity).
    """
    file_age = seconds_since(last_file_mod, now)
    scan_age = seconds_since(last_scan, now)
    file_stale = file_age is None or file_age > threshold
    scan_stale = scan_age is None or scan_age > threshold
    return file_stale and scan_stale
