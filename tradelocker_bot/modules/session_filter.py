"""
Session & Timing Filter Module

Controls WHEN the bot is allowed to trade:
1. Session windows (London, New York, overlap only)
   - BTC/USD: London + NY + overlap
   - XAU/USD: London + NY (avoid Asian session)
2. News avoidance (30-minute buffer before/after high-impact events)
3. Weekend/market closure detection

All times are in UTC.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from dataclasses import dataclass

from config import SESSIONS, NEWS_BUFFER_MINUTES

logger = logging.getLogger(__name__)


@dataclass
class SessionStatus:
    """Current session state for a symbol."""
    symbol: str
    is_active_session: bool
    current_session: Optional[str]  # 'london', 'new_york', 'overlap', None
    next_session_open: Optional[datetime]
    minutes_until_open: Optional[int]
    news_blocked: bool
    news_event: Optional[str]
    reason: str


# High-impact economic events schedule
# In production, this would be fetched from an economic calendar API
# For now, we define recurring known events (UTC times)
# Format: (day_of_week, hour, minute, description)
# day_of_week: 0=Monday, 1=Tuesday, ..., 4=Friday
HIGH_IMPACT_EVENTS = [
    # US Non-Farm Payrolls - First Friday of month
    # US CPI - Usually 2nd Tuesday/Wednesday of month
    # FOMC Rate Decision - 8 scheduled per year
    # These are RECURRING TEMPLATES - actual dates vary
    # The bot should integrate with an economic calendar API for real-time data
]

# Known fixed-time recurring events (approximate, UTC)
RECURRING_EVENTS = {
    # Day of week -> list of (hour, minute, duration_minutes, description)
    0: [],  # Monday
    1: [
        (13, 30, 60, "US Economic Data Release Window"),
    ],
    2: [
        (13, 30, 60, "US Economic Data Release Window"),
        (19, 0, 60, "FOMC Minutes (when scheduled)"),
    ],
    3: [
        (12, 30, 60, "US Jobless Claims / Economic Data"),
    ],
    4: [
        (13, 30, 60, "US NFP / Economic Data Window"),
    ],
}


def get_current_utc() -> datetime:
    """Get current UTC time."""
    return datetime.now(timezone.utc)


def is_weekend() -> bool:
    """Check if it's currently the weekend (markets closed)."""
    now = get_current_utc()
    # Forex/Gold: closed from Friday 21:00 UTC to Sunday 21:00 UTC
    # BTC: trades 24/7 but lower liquidity on weekends

    if now.weekday() == 5:  # Saturday
        return True
    if now.weekday() == 6 and now.hour < 21:  # Sunday before 21:00
        return True
    if now.weekday() == 4 and now.hour >= 22:  # Friday after 22:00
        return True

    return False


def is_in_session(symbol: str, current_time: Optional[datetime] = None) -> tuple[bool, Optional[str]]:
    """
    Check if the current time is within an active trading session.

    BTC/USD: London (07-16), New York (12-21), Overlap (12-16)
    XAU/USD: London (07-16), New York (12-21) — avoid Asian

    Args:
        symbol: Instrument symbol (BTCUSD or XAUUSD)
        current_time: Time to check (default: now UTC)

    Returns:
        Tuple of (is_active, session_name)
    """
    if current_time is None:
        current_time = get_current_utc()

    hour = current_time.hour

    # Get session config for this symbol
    session_config = None
    for sym_key, config in SESSIONS.items():
        if sym_key.upper() in symbol.upper() or symbol.upper() in sym_key.upper():
            session_config = config
            break

    if session_config is None:
        # Default to standard forex sessions
        session_config = SESSIONS.get("BTCUSD", {
            "london_open": 7, "london_close": 16,
            "ny_open": 12, "ny_close": 21,
        })

    london_open = session_config["london_open"]
    london_close = session_config["london_close"]
    ny_open = session_config["ny_open"]
    ny_close = session_config["ny_close"]

    # Check overlap first (most liquid)
    if ny_open <= hour < london_close:
        return True, "overlap"

    # Check London session
    if london_open <= hour < london_close:
        return True, "london"

    # Check New York session
    if ny_open <= hour < ny_close:
        return True, "new_york"

    return False, None


def get_next_session_open(symbol: str, current_time: Optional[datetime] = None) -> tuple[Optional[datetime], Optional[int]]:
    """
    Calculate when the next trading session opens.

    Returns:
        Tuple of (next_open_datetime, minutes_until_open)
    """
    if current_time is None:
        current_time = get_current_utc()

    session_config = None
    for sym_key, config in SESSIONS.items():
        if sym_key.upper() in symbol.upper() or symbol.upper() in sym_key.upper():
            session_config = config
            break

    if session_config is None:
        return None, None

    london_open = session_config["london_open"]
    hour = current_time.hour

    # If before London open today
    if hour < london_open:
        next_open = current_time.replace(hour=london_open, minute=0, second=0, microsecond=0)
    else:
        # Next London open is tomorrow
        next_open = (current_time + timedelta(days=1)).replace(
            hour=london_open, minute=0, second=0, microsecond=0
        )

    # Skip weekends
    while next_open.weekday() >= 5:  # Saturday or Sunday
        next_open += timedelta(days=1)

    minutes_until = int((next_open - current_time).total_seconds() / 60)
    return next_open, minutes_until


def _event_block_window(
    now_utc: datetime,
    event: tuple,
    buffer_minutes: int,
) -> tuple[datetime, datetime, datetime, datetime]:
    """
    Derive the concrete UTC datetimes for a single recurring event, anchored to
    the DATE of ``now_utc``.

    ``event`` is ``(hour, minute, duration_minutes, description)``.

    Returns ``(event_start, event_end, block_start, block_end)`` where the block
    window extends ``buffer_minutes`` before the event start and after the event
    end.
    """
    event_hour, event_minute, duration, _description = event
    event_start = now_utc.replace(
        hour=event_hour, minute=event_minute, second=0, microsecond=0
    )
    event_end = event_start + timedelta(minutes=duration)
    block_start = event_start - timedelta(minutes=buffer_minutes)
    block_end = event_end + timedelta(minutes=buffer_minutes)
    return event_start, event_end, block_start, block_end


def is_within_news_block(
    now_utc: datetime,
    events,
    buffer_minutes: int = NEWS_BUFFER_MINUTES,
) -> tuple[bool, Optional[str]]:
    """
    Pure, side-effect-free news-block check.

    A news block is active ONLY when ``now_utc`` actually falls inside the block
    window of one of ``events`` — i.e. within
    ``[event_start - buffer, event_end + buffer]``.

    Convention: the block window is INCLUSIVE on BOTH edges. Given an event that
    runs 13:30-14:30 with a 30-minute buffer, the block window is 13:00-15:00 and
    both 13:00 and 15:00 are considered blocked; 12:59 and 15:01 are not.

    All datetimes are compared in UTC. Event times are anchored to the calendar
    date of ``now_utc`` (these are recurring intraday windows), so callers should
    pass the events scheduled for ``now_utc``'s weekday.

    Args:
        now_utc: The current (or evaluated) UTC time.
        events: Iterable of ``(hour, minute, duration_minutes, description)``.
        buffer_minutes: Minutes before/after the event to also block.

    Returns:
        Tuple of ``(is_blocked, reason)`` where ``reason`` is the event
        description (which contains the event name) when blocked, else ``None``.
    """
    for event in events:
        description = event[3]
        _start, _end, block_start, block_end = _event_block_window(
            now_utc, event, buffer_minutes
        )
        if block_start <= now_utc <= block_end:
            return True, description

    return False, None


def is_near_news_event(
    current_time: Optional[datetime] = None,
    buffer_minutes: int = NEWS_BUFFER_MINUTES,
) -> tuple[bool, Optional[str]]:
    """
    Check if we're within the buffer zone of a known high-impact news event.

    No trades within 30 minutes before or after major events. This is the
    runtime entry point: it resolves the real UTC clock (or an injected
    ``current_time`` for tests), picks the events scheduled for today, and
    delegates the actual time-window decision to the pure
    :func:`is_within_news_block` helper.

    Args:
        current_time: Time to check (default: now UTC)
        buffer_minutes: Minutes before/after to block (default: 30)

    Returns:
        Tuple of (is_blocked, event_description)
    """
    from config import SKIP_NEWS_FILTER
    if SKIP_NEWS_FILTER:
        return False, None

    if current_time is None:
        current_time = get_current_utc()

    # Recurring events are keyed by weekday and anchored to today's date.
    events_today = RECURRING_EVENTS.get(current_time.weekday(), [])

    blocked, reason = is_within_news_block(current_time, events_today, buffer_minutes)

    if blocked:
        # Only emit the informative WARNING when we are ACTUALLY inside the
        # block window (i.e. blocked is True). Re-derive the matched event's
        # window purely for the human-readable log line.
        for event in events_today:
            event_start, event_end, block_start, block_end = _event_block_window(
                current_time, event, buffer_minutes
            )
            if block_start <= current_time <= block_end:
                logger.warning(
                    f"NEWS BLOCK: {event[3]} | "
                    f"Event: {event_start.strftime('%H:%M')}-{event_end.strftime('%H:%M')} | "
                    f"Block window: {block_start.strftime('%H:%M')}-{block_end.strftime('%H:%M')}"
                )
                break

    return blocked, reason


def check_session_status(symbol: str, current_time: Optional[datetime] = None) -> SessionStatus:
    """
    Complete session check for a symbol.

    Combines all timing filters into a single status object.

    Args:
        symbol: Instrument symbol
        current_time: Time to check (default: now UTC)

    Returns:
        SessionStatus with all timing information
    """
    if current_time is None:
        current_time = get_current_utc()

    # Check weekend
    if is_weekend():
        next_open, mins = get_next_session_open(symbol, current_time)
        return SessionStatus(
            symbol=symbol,
            is_active_session=False,
            current_session=None,
            next_session_open=next_open,
            minutes_until_open=mins,
            news_blocked=False,
            news_event=None,
            reason="Weekend - markets closed",
        )

    # Check if in active session
    in_session, session_name = is_in_session(symbol, current_time)

    if not in_session:
        next_open, mins = get_next_session_open(symbol, current_time)
        return SessionStatus(
            symbol=symbol,
            is_active_session=False,
            current_session=None,
            next_session_open=next_open,
            minutes_until_open=mins,
            news_blocked=False,
            news_event=None,
            reason=f"Outside active session hours (next: {next_open.strftime('%H:%M UTC') if next_open else 'unknown'})",
        )

    # Check news events
    news_blocked, news_event = is_near_news_event(current_time)

    if news_blocked:
        return SessionStatus(
            symbol=symbol,
            is_active_session=True,
            current_session=session_name,
            next_session_open=None,
            minutes_until_open=None,
            news_blocked=True,
            news_event=news_event,
            reason=f"News block: {news_event} (±{NEWS_BUFFER_MINUTES}min buffer)",
        )

    # All clear
    return SessionStatus(
        symbol=symbol,
        is_active_session=True,
        current_session=session_name,
        next_session_open=None,
        minutes_until_open=None,
        news_blocked=False,
        news_event=None,
        reason=f"Active session: {session_name}",
    )


def can_trade_now(symbol: str) -> tuple[bool, str]:
    """
    Simple boolean check: can we trade this symbol right now?

    Args:
        symbol: Instrument symbol

    Returns:
        Tuple of (can_trade, reason)
    """
    status = check_session_status(symbol)

    if not status.is_active_session:
        return False, status.reason

    if status.news_blocked:
        return False, status.reason

    return True, f"Clear to trade ({status.current_session} session)"
