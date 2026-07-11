"""
Session Manager for XAUUSD Trading Bot.

Determines the current trading session based on UTC time and returns
the appropriate trading mode. Tracks session-level statistics (highs,
lows, trade count) and handles session transitions.

Sessions:
    Asian      (00:00 - 07:00 UTC) → RANGE mode
    London     (07:00 - 12:00 UTC) → TREND mode
    Overlap    (12:00 - 16:00 UTC) → TREND_AGGRESSIVE mode
    New York   (16:00 - 21:00 UTC) → TREND mode
    Dead Zone  (21:00 - 00:00 UTC) → IDLE mode
"""

from datetime import datetime, time, timedelta
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionInfo:
    """Current session state and metadata."""
    name: str = "dead_zone"
    mode: str = "IDLE"
    start_time: time = time(0, 0)
    end_time: time = time(0, 0)
    session_high: float = 0.0
    session_low: float = float("inf")
    session_open: float = 0.0
    trade_count: int = 0
    is_active: bool = False


@dataclass
class SessionDefinition:
    """Static definition of a trading session."""
    name: str
    start: time
    end: time
    mode: str


class SessionManager:
    """
    Manages session identification, mode switching, and session-level tracking.
    """

    def __init__(self, sessions_config: dict):
        self.sessions: list[SessionDefinition] = []
        self._parse_config(sessions_config)
        self.current_session: SessionInfo = SessionInfo()
        self._previous_session_name: str = ""

    def _parse_config(self, config: dict):
        """Parse session config dict into SessionDefinition objects."""
        for name, params in config.items():
            start_parts = params["start"].split(":")
            end_parts = params["end"].split(":")
            self.sessions.append(SessionDefinition(
                name=name,
                start=time(int(start_parts[0]), int(start_parts[1])),
                end=time(int(end_parts[0]), int(end_parts[1])),
                mode=params["mode"],
            ))

    def update(self, current_time: datetime, current_price: float) -> SessionInfo:
        """
        Update session state based on current UTC time.
        Call this on every tick/bar.
        Returns the current SessionInfo.
        """
        current_t = current_time.time()
        matched_session = self._find_session(current_t)

        if matched_session is None:
            # No session matched — dead zone
            self.current_session.name = "dead_zone"
            self.current_session.mode = "IDLE"
            self.current_session.is_active = False
            return self.current_session

        # Check if session changed
        if matched_session.name != self._previous_session_name:
            self._on_session_change(matched_session, current_price)

        # Update session high/low
        if current_price > self.current_session.session_high:
            self.current_session.session_high = current_price
        if current_price < self.current_session.session_low:
            self.current_session.session_low = current_price

        return self.current_session

    def _find_session(self, current_t: time) -> Optional[SessionDefinition]:
        """Find which session the current time falls into."""
        for session in self.sessions:
            if session.end == time(0, 0):
                # Wraps midnight: e.g., 21:00 - 00:00
                if current_t >= session.start:
                    return session
            else:
                if session.start <= current_t < session.end:
                    return session
        return None

    def _on_session_change(self, new_session: SessionDefinition, current_price: float):
        """Handle transition to a new session — reset tracking."""
        self._previous_session_name = new_session.name
        self.current_session = SessionInfo(
            name=new_session.name,
            mode=new_session.mode,
            start_time=new_session.start,
            end_time=new_session.end,
            session_high=current_price,
            session_low=current_price,
            session_open=current_price,
            trade_count=0,
            is_active=(new_session.mode != "IDLE"),
        )

    def get_mode(self) -> str:
        """Get the current trading mode string."""
        return self.current_session.mode

    def get_session_name(self) -> str:
        """Get the current session name."""
        return self.current_session.name

    def get_session_range(self) -> tuple[float, float]:
        """Get (session_high, session_low) for structure break detection."""
        return self.current_session.session_high, self.current_session.session_low

    def record_trade(self):
        """Increment session trade counter."""
        self.current_session.trade_count += 1

    def get_session_trade_count(self) -> int:
        return self.current_session.trade_count

    def is_session_active(self) -> bool:
        return self.current_session.is_active

    def time_remaining_in_session(self, current_time: datetime) -> Optional[timedelta]:
        """How much time is left in the current session."""
        if not self.current_session.is_active:
            return None

        end = self.current_session.end_time
        now = current_time.time()

        # Calculate remaining minutes
        end_minutes = end.hour * 60 + end.minute
        now_minutes = now.hour * 60 + now.minute

        if end_minutes == 0:
            end_minutes = 24 * 60  # Midnight wrap

        remaining = end_minutes - now_minutes
        if remaining < 0:
            return timedelta(0)

        return timedelta(minutes=remaining)

    def is_near_session_end(self, current_time: datetime, threshold_minutes: int = 15) -> bool:
        """Check if we're within X minutes of session end (avoid late entries)."""
        remaining = self.time_remaining_in_session(current_time)
        if remaining is None:
            return True
        return remaining <= timedelta(minutes=threshold_minutes)
