"""
Scan Countdown and Bot Status Detection.

Requirement 11:
  - Determine most recent scan time from latest scan-activity entry in Bot_Log_File.
  - Next scan time = most recent scan + SCAN_INTERVAL (60s).
  - Display countdown (remaining seconds) while bot is active.
  - Display lock reason text when bot is in Locked_State.
  - Display paused indication when in avoid_hours or outside trading session.
  - Display "due" if next scan time is in the past.
  - Display "no scan occurred" if no scan-activity entry exists.

Requirement 14:
  - Bot_Offline if no monitored file modified in 90s AND no scan in 90s.
  - Empty states for zero trade records.
  - Initializing state if no file ever created.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class CountdownState:
    """State of the next-scan countdown."""
    seconds_remaining: Optional[int] = None
    is_due: bool = False
    no_scan_yet: bool = False
    last_scan_time: Optional[str] = None
    next_scan_time: Optional[str] = None


@dataclass
class BotStatus:
    """Overall bot status."""
    is_online: bool = True
    is_locked: bool = False
    is_paused: bool = False
    is_initializing: bool = False
    lock_reason: str = ""
    pause_reason: str = ""
    offline_seconds: Optional[float] = None
    last_file_mtime: Optional[float] = None
    countdown: CountdownState = field(default_factory=CountdownState)


# Pattern for scan activity in bot log
# Common patterns: "Starting scan", "Scan complete", "Running scan cycle"
_SCAN_PATTERNS = [
    re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}).*(?:Starting scan|Scan complete|scan cycle|scanning)", re.IGNORECASE),
    re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}).*(?:Checking instruments|Running analysis)", re.IGNORECASE),
]


def compute_countdown(
    bot_log_content: str,
    scan_interval_seconds: int = 60,
) -> CountdownState:
    """
    Compute scan countdown from bot log (Req 11.1-11.7).
    """
    # Find the most recent scan timestamp
    last_scan_ts = _find_last_scan_time(bot_log_content)

    if last_scan_ts is None:
        # Req 11.7: no scan-activity entry found
        return CountdownState(no_scan_yet=True)

    # Compute next scan time (Req 11.2)
    try:
        # Parse the timestamp
        if "T" in last_scan_ts:
            last_scan_dt = datetime.fromisoformat(last_scan_ts.replace("Z", "+00:00"))
        else:
            last_scan_dt = datetime.strptime(last_scan_ts, "%Y-%m-%d %H:%M:%S")
            last_scan_dt = last_scan_dt.replace(tzinfo=timezone.utc)

        last_scan_epoch = last_scan_dt.timestamp()
        next_scan_epoch = last_scan_epoch + scan_interval_seconds
        now = time.time()

        remaining = next_scan_epoch - now

        if remaining <= 0:
            # Req 11.6: display "due"
            return CountdownState(
                is_due=True,
                last_scan_time=last_scan_ts,
                next_scan_time=datetime.fromtimestamp(
                    next_scan_epoch, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S"),
            )

        return CountdownState(
            seconds_remaining=int(remaining),
            last_scan_time=last_scan_ts,
            next_scan_time=datetime.fromtimestamp(
                next_scan_epoch, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S"),
        )

    except (ValueError, OSError):
        return CountdownState(no_scan_yet=True)


def compute_bot_status(
    latest_file_mtime: Optional[float],
    any_file_ever_existed: bool,
    daily_stats: Dict[str, Any],
    adaptive_config: Dict[str, Any],
    bot_log_content: str,
    scan_interval_seconds: int = 60,
    offline_threshold_seconds: float = 90.0,
) -> BotStatus:
    """
    Compute overall bot status (Req 11, 14).
    """
    now = time.time()
    status = BotStatus()

    # Req 14.5: initializing state (no file ever created)
    if not any_file_ever_existed:
        status.is_initializing = True
        status.is_online = False
        return status

    # Req 14.1: Bot offline detection
    if latest_file_mtime is not None:
        elapsed = now - latest_file_mtime
        if elapsed > offline_threshold_seconds:
            status.is_online = False
            status.offline_seconds = elapsed
            status.last_file_mtime = latest_file_mtime
    else:
        status.is_online = False
        status.offline_seconds = None

    # Compute countdown
    status.countdown = compute_countdown(bot_log_content, scan_interval_seconds)

    # Check for Locked_State (Req 11.4)
    lock_reason = _check_locked_state(daily_stats)
    if lock_reason:
        status.is_locked = True
        status.lock_reason = lock_reason

    # Check for paused state (Req 11.5)
    pause_reason = _check_paused_state(adaptive_config)
    if pause_reason:
        status.is_paused = True
        status.pause_reason = pause_reason

    return status


def _find_last_scan_time(bot_log_content: str) -> Optional[str]:
    """Find the most recent scan timestamp in bot log."""
    if not bot_log_content:
        return None

    last_ts: Optional[str] = None

    for line in bot_log_content.splitlines():
        for pattern in _SCAN_PATTERNS:
            match = pattern.search(line)
            if match:
                ts = match.group(1)
                if last_ts is None or ts > last_ts:
                    last_ts = ts
                break

    return last_ts


def _check_locked_state(daily_stats: Dict[str, Any]) -> str:
    """
    Check if bot is in locked state from daily_stats (Req 11.4).
    Returns lock reason string or empty string if not locked.
    """
    daily = daily_stats.get("daily", {})
    weekly = daily_stats.get("weekly", {})

    reasons: List[str] = []

    # Check daily lock conditions
    if daily.get("max_trades_reached"):
        reasons.append("Max daily trades reached")
    if daily.get("daily_drawdown_hit"):
        reasons.append("Daily drawdown limit hit")
    if daily.get("consecutive_losses_pause"):
        reasons.append("Paused after consecutive losses")

    # Check weekly lock conditions
    if weekly.get("weekly_drawdown_hit"):
        reasons.append("Weekly drawdown limit hit")

    # Generic locked flag
    if daily.get("locked") or daily.get("is_locked"):
        lock_reason = daily.get("lock_reason", "Trading paused")
        if lock_reason and lock_reason not in reasons:
            reasons.append(lock_reason)

    return "; ".join(reasons)


def _check_paused_state(adaptive_config: Dict[str, Any]) -> str:
    """
    Check if bot is paused due to avoid_hours or outside trading session (Req 11.5).
    Returns pause reason or empty string.
    """
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour

    # Check avoid_hours
    avoid_hours = adaptive_config.get("avoid_hours", [])
    if isinstance(avoid_hours, list) and current_hour in avoid_hours:
        return f"Avoid hours (current hour {current_hour} UTC is in avoid list)"

    # Check trading session
    session_start = adaptive_config.get("session_start_hour")
    session_end = adaptive_config.get("session_end_hour")

    if session_start is not None and session_end is not None:
        try:
            start_h = int(session_start)
            end_h = int(session_end)
            if start_h <= end_h:
                # Normal range: e.g., 8-22
                if current_hour < start_h or current_hour >= end_h:
                    return f"Outside trading session ({start_h}:00-{end_h}:00 UTC)"
            else:
                # Wrapping range: e.g., 22-8 (overnight)
                if end_h <= current_hour < start_h:
                    return f"Outside trading session ({start_h}:00-{end_h}:00 UTC)"
        except (TypeError, ValueError):
            pass

    return ""
