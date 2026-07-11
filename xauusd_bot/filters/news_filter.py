"""
News Filter for XAUUSD Trading Bot.

Prevents new entries during high-impact economic events that can
cause extreme volatility spikes in gold (NFP, CPI, FOMC, etc.).

Features:
    - Maintains a schedule of upcoming events
    - Blackout window before/after events (configurable)
    - Can be fed events from an economic calendar API
    - Manual event injection for known dates
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional


@dataclass
class NewsEvent:
    """A scheduled high-impact economic event."""
    name: str
    timestamp: datetime
    impact: str = "HIGH"         # "HIGH", "MEDIUM"
    currency: str = "USD"
    description: str = ""


@dataclass
class NewsFilterResult:
    """Result of news filter check."""
    is_blackout: bool = False
    reason: str = ""
    next_event: Optional[NewsEvent] = None
    minutes_until_event: Optional[int] = None
    minutes_since_event: Optional[int] = None


class NewsFilter:
    """
    Filters out trading during high-impact news events.
    Maintains a list of upcoming events and enforces blackout periods.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.blackout_before = config.get("blackout_minutes_before", 15)
        self.blackout_after = config.get("blackout_minutes_after", 15)
        self.high_impact_events = config.get("high_impact_events", [])
        self.scheduled_events: list[NewsEvent] = []

    def add_event(self, name: str, timestamp: datetime,
                  impact: str = "HIGH", currency: str = "USD"):
        """Add a known upcoming event to the schedule."""
        event = NewsEvent(
            name=name,
            timestamp=timestamp,
            impact=impact,
            currency=currency,
        )
        self.scheduled_events.append(event)
        # Keep sorted by time
        self.scheduled_events.sort(key=lambda e: e.timestamp)

    def add_events_bulk(self, events: list[dict]):
        """
        Add multiple events from a calendar API response.
        Expected format: [{"name": "NFP", "timestamp": datetime, "impact": "HIGH"}]
        """
        for event_data in events:
            self.add_event(
                name=event_data["name"],
                timestamp=event_data["timestamp"],
                impact=event_data.get("impact", "HIGH"),
                currency=event_data.get("currency", "USD"),
            )

    def check(self, current_time: datetime) -> NewsFilterResult:
        """
        Check if current time falls within a news blackout period.

        Returns:
            NewsFilterResult with blackout status and details.
        """
        result = NewsFilterResult()

        # Clean expired events (more than 1 hour past)
        self._cleanup_old_events(current_time)

        if not self.scheduled_events:
            return result  # No events scheduled, all clear

        for event in self.scheduled_events:
            blackout_start = event.timestamp - timedelta(minutes=self.blackout_before)
            blackout_end = event.timestamp + timedelta(minutes=self.blackout_after)

            # Check if we're in blackout window
            if blackout_start <= current_time <= blackout_end:
                result.is_blackout = True
                result.next_event = event

                # Determine if before or after event
                if current_time < event.timestamp:
                    diff = event.timestamp - current_time
                    result.minutes_until_event = int(diff.total_seconds() / 60)
                    result.reason = (f"BLACKOUT: {event.name} in "
                                     f"{result.minutes_until_event}min")
                else:
                    diff = current_time - event.timestamp
                    result.minutes_since_event = int(diff.total_seconds() / 60)
                    result.reason = (f"BLACKOUT: {event.name} was "
                                     f"{result.minutes_since_event}min ago")
                return result

            # Find next upcoming event
            if current_time < blackout_start and result.next_event is None:
                result.next_event = event
                diff = event.timestamp - current_time
                result.minutes_until_event = int(diff.total_seconds() / 60)

        return result

    def is_blackout(self, current_time: datetime) -> bool:
        """Simple boolean check for blackout status."""
        return self.check(current_time).is_blackout

    def get_next_event(self, current_time: datetime) -> Optional[NewsEvent]:
        """Get the next scheduled event."""
        for event in self.scheduled_events:
            if event.timestamp > current_time:
                return event
        return None

    def time_to_next_blackout(self, current_time: datetime) -> Optional[timedelta]:
        """How long until the next blackout starts."""
        for event in self.scheduled_events:
            blackout_start = event.timestamp - timedelta(minutes=self.blackout_before)
            if current_time < blackout_start:
                return blackout_start - current_time
        return None

    def _cleanup_old_events(self, current_time: datetime):
        """Remove events that are more than 1 hour past."""
        cutoff = current_time - timedelta(hours=1)
        self.scheduled_events = [
            e for e in self.scheduled_events if e.timestamp > cutoff
        ]

    def clear_events(self):
        """Clear all scheduled events."""
        self.scheduled_events.clear()

    def get_todays_events(self, current_time: datetime) -> list[NewsEvent]:
        """Get all events scheduled for today."""
        today = current_time.date()
        return [e for e in self.scheduled_events if e.timestamp.date() == today]
