"""Deterministic CLOSE-action ordering (Req 6.1, 10.2)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, List

from .timeutil import parse_iso_utc

_MAX_DT = datetime.max.replace(tzinfo=timezone.utc)


def _sort_key(action: dict):
    ts = parse_iso_utc(action.get("timestamp"))
    # Entries with an unparseable timestamp sort last, still deterministically
    # ordered by their source appearance keys.
    ts_key = (0, ts) if ts is not None else (1, _MAX_DT)
    file_date = action.get("file_date")
    file_date = file_date if isinstance(file_date, str) else ""
    line_index = action.get("line_index")
    line_index = line_index if isinstance(line_index, int) and not isinstance(line_index, bool) else 0
    return (ts_key, file_date, line_index)


def order_closes(close_actions: Iterable[dict]) -> List[dict]:
    """Return CLOSE actions ordered ascending by UTC ``timestamp``.

    Ties (identical timestamps) are broken by source appearance order — earlier
    ``file_date`` first, then earlier ``line_index``. The result is a total order
    that is identical across repeated constructions of the same input.
    """
    return sorted(list(close_actions), key=_sort_key)
