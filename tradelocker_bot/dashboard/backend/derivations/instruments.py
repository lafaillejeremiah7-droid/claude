"""Instrument monitoring set and per-instrument failure isolation (Req 15)."""
from __future__ import annotations

from typing import Callable, Iterable, List

MAX_INSTRUMENTS = 2


def monitored_instruments(configured: Iterable[str]) -> List[str]:
    """The monitored set: exactly the configured instruments, capped at two.

    Preserves order, drops empty/non-string entries and duplicates, and excludes
    anything beyond the first two valid entries (Req 15.5).
    """
    result: List[str] = []
    for item in configured or []:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name in result:
            continue
        result.append(name)
        if len(result) >= MAX_INSTRUMENTS:
            break
    return result


def gather_instrument_data(
    monitored: Iterable[str], fetcher: Callable[[str], object]
) -> dict:
    """Fetch per-instrument data with failure isolation (Req 15.4).

    ``fetcher(symbol)`` returns the instrument's data, returns ``None`` to signal
    "no data", or raises to signal a failure. A failure/None for one instrument
    is reported for THAT instrument only; all succeeding instruments still yield
    results.

    Returns ``{symbol: {"data_available": bool, "data": <data or None>}}``.
    """
    out = {}
    for symbol in monitored:
        try:
            data = fetcher(symbol)
        except Exception:  # isolate: one instrument's failure must not affect others
            out[symbol] = {"data_available": False, "data": None}
            continue
        if data is None:
            out[symbol] = {"data_available": False, "data": None}
        else:
            out[symbol] = {"data_available": True, "data": data}
    return out
