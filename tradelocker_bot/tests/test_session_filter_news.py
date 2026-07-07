"""
Hermetic tests for the news-block timing logic in ``modules.session_filter``.

These tests are fully deterministic and require no network / no live clock:
they inject an explicit ``now_utc`` into the pure helper
:func:`session_filter.is_within_news_block` and assert that a news block is
active ONLY when the current UTC time actually falls inside the block window
``[event_start - buffer, event_end + buffer]``.

Block-window convention under test: INCLUSIVE on both edges. For an event
running 13:30-14:30 with a 30-minute buffer, the block window is 13:00-15:00,
and both 13:00 and 15:00 count as blocked; 12:59 and 15:01 do not.
"""
from datetime import datetime, timezone

import pytest

from modules.session_filter import is_within_news_block, is_near_news_event


# A single event: 13:30-14:30 UTC, 60-minute duration.
# With a 30-minute buffer the effective block window is 13:00-15:00 UTC.
EVENT_NAME = "US Economic Data Release Window"
EVENTS = [(13, 30, 60, EVENT_NAME)]
BUFFER = 30


def _utc(hour, minute=0):
    # 2024-06-04 is a Tuesday; date is irrelevant to the pure helper because we
    # pass EVENTS explicitly, but we keep it fixed for determinism.
    return datetime(2024, 6, 4, hour, minute, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "hour,minute,expected_blocked",
    [
        (8, 0, False),    # well before the window -> NOT blocked (the reported bug)
        (12, 0, False),   # still before buffer start -> NOT blocked
        (12, 59, False),  # one minute before buffer start -> NOT blocked
        (13, 0, True),    # buffer start edge (inclusive) -> blocked
        (13, 45, True),   # squarely inside the event -> blocked
        (14, 30, True),   # event end -> blocked
        (15, 0, True),    # buffer end edge (inclusive) -> blocked
        (15, 1, False),   # one minute after buffer end -> NOT blocked
        (15, 30, False),  # after the window -> NOT blocked
    ],
)
def test_news_block_only_active_inside_window(hour, minute, expected_blocked):
    blocked, reason = is_within_news_block(_utc(hour, minute), EVENTS, BUFFER)
    assert blocked is expected_blocked
    if expected_blocked:
        assert reason is not None
        assert EVENT_NAME in reason
    else:
        assert reason is None


def test_reason_contains_event_name_when_blocked():
    blocked, reason = is_within_news_block(_utc(13, 45), EVENTS, BUFFER)
    assert blocked is True
    assert EVENT_NAME in reason


def test_no_events_never_blocks():
    blocked, reason = is_within_news_block(_utc(13, 45), [], BUFFER)
    assert blocked is False
    assert reason is None


def test_multiple_events_matches_the_active_one():
    events = [
        (13, 30, 60, EVENT_NAME),
        (19, 0, 60, "FOMC Minutes"),
    ]
    # 19:15 is inside the second event's window (18:30-20:30), not the first.
    blocked, reason = is_within_news_block(_utc(19, 15), events, BUFFER)
    assert blocked is True
    assert "FOMC Minutes" in reason

    # 16:00 is between both windows -> not blocked.
    blocked, reason = is_within_news_block(_utc(16, 0), events, BUFFER)
    assert blocked is False
    assert reason is None


def test_zero_buffer_uses_event_bounds_only():
    blocked, _ = is_within_news_block(_utc(13, 0), EVENTS, 0)
    assert blocked is False  # 13:00 is before the 13:30 event start with no buffer
    blocked, _ = is_within_news_block(_utc(13, 30), EVENTS, 0)
    assert blocked is True   # exactly at event start


def test_runtime_entry_point_respects_injected_clock():
    # is_near_news_event uses RECURRING_EVENTS for the weekday. 2024-06-04 is a
    # Tuesday, which has the 13:30-14:30 US data window. At 08:00 it must NOT
    # block (the originally reported symptom), and inside the window it must.
    blocked_early, _ = is_near_news_event(_utc(8, 0))
    assert blocked_early is False

    blocked_inside, reason = is_near_news_event(_utc(13, 45))
    assert blocked_inside is True
    assert reason is not None and EVENT_NAME in reason
