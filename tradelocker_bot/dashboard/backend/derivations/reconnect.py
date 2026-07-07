"""Pure model of the frontend SSE reconnection controller (Req 12.6).

The live controller wraps the browser ``EventSource``; this pure function models
the *schedule* it must follow so the invariants can be property-tested: reconnect
at intervals no greater than 5 seconds, at most 12 consecutive attempts, and keep
displaying the last successfully received snapshot throughout.
"""
from __future__ import annotations

from typing import Any, List

MAX_RECONNECT_ATTEMPTS = 12
MAX_RECONNECT_INTERVAL_SECONDS = 5.0


def plan_reconnects(
    interval_seconds: float,
    max_attempts: int = MAX_RECONNECT_ATTEMPTS,
    last_snapshot: Any = None,
) -> List[dict]:
    """Produce the bounded reconnection schedule.

    Returns a list of attempt descriptors, each ``{"attempt", "delay",
    "displayed_snapshot"}``. The number of attempts is capped at 12, each delay
    is clamped to at most 5 seconds, and every attempt keeps displaying the last
    received snapshot.
    """
    capped = max(0, min(int(max_attempts), MAX_RECONNECT_ATTEMPTS))
    delay = min(float(interval_seconds), MAX_RECONNECT_INTERVAL_SECONDS)
    return [
        {"attempt": i, "delay": delay, "displayed_snapshot": last_snapshot}
        for i in range(1, capped + 1)
    ]
