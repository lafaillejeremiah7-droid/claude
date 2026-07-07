"""UTC-only timestamp parsing (Req 16)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def parse_iso_utc(s: object) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string and return a timezone-aware UTC datetime.

    - A string without an explicit timezone offset is interpreted as UTC
      (the bot writes naive-looking timestamps that are actually UTC).
    - A string with an explicit offset is converted to UTC.
    - Missing, empty, non-string, or non-ISO input returns ``None`` (Unavailable)
      and NEVER raises, so callers can simply exclude the value.
    """
    if not isinstance(s, str):
        return None
    text = s.strip()
    if not text:
        return None
    # ``datetime.fromisoformat`` in 3.11+ accepts 'Z', but normalise defensively.
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(normalized)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
