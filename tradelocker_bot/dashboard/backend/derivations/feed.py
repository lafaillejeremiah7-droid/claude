"""Live trade/event feed assembly (Req 8)."""
from __future__ import annotations

from typing import Iterable, List

from .timeutil import parse_iso_utc

FEED_CAP = 100
_REQUIRED = ("timestamp", "action", "symbol", "direction")


def _non_empty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _is_emittable(entry: object) -> bool:
    if not isinstance(entry, dict):
        return False
    if not all(_non_empty(entry.get(field)) for field in _REQUIRED):
        return False
    # A non-empty timestamp must also be a parseable UTC instant to order it.
    return parse_iso_utc(entry.get("timestamp")) is not None


def build_feed(
    journal_entries: Iterable[dict], log_events: Iterable[dict]
) -> List[dict]:
    """Merge journal trade entries and log events into the display feed.

    - Only entries with non-empty ``timestamp``/``action``/``symbol``/``direction``
      (and a parseable timestamp) are emitted.
    - Ordered newest-first (non-increasing timestamp).
    - Capped at the most recent ``FEED_CAP`` (100) entries.
    """
    merged = [e for e in list(journal_entries) + list(log_events) if _is_emittable(e)]
    merged.sort(key=lambda e: parse_iso_utc(e.get("timestamp")), reverse=True)
    return merged[:FEED_CAP]
