"""Open-position derivations: unrealized PnL, SL/TP distances, freshness (Req 9)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Optional, Tuple

from .common import is_number
from .timeutil import parse_iso_utc

FRESH_PRICE_MAX_AGE_SECONDS = 5.0


def _direction_is_long(direction: Any) -> Optional[bool]:
    if not isinstance(direction, str):
        return None
    d = direction.strip().lower()
    if d in ("buy", "long"):
        return True
    if d in ("sell", "short"):
        return False
    return None


def unrealized_pnl(position: dict, live_price: dict) -> Optional[float]:
    """Direction-correct mark-to-market PnL (Req 4.6, 9.3).

    Long:  ``(bid - entry) * qty``
    Short: ``(entry - ask) * qty``
    Returns ``None`` when direction, entry, quantity or the relevant side of the
    quote is missing/non-numeric.
    """
    if not isinstance(position, dict) or not isinstance(live_price, dict):
        return None
    is_long = _direction_is_long(position.get("direction"))
    if is_long is None:
        return None
    entry = position.get("entry_price")
    qty = position.get("quantity")
    if not (is_number(entry) and is_number(qty)):
        return None
    if is_long:
        bid = live_price.get("bid")
        if not is_number(bid):
            return None
        return (float(bid) - float(entry)) * float(qty)
    ask = live_price.get("ask")
    if not is_number(ask):
        return None
    return (float(entry) - float(ask)) * float(qty)


def sl_tp_distances(
    live_price: Any, stop_loss: Any, take_profit: Any
) -> Tuple[Optional[float], Optional[float]]:
    """Absolute price distances to SL and TP (Req 9.3)."""
    dist_sl = abs(float(live_price) - float(stop_loss)) if (
        is_number(live_price) and is_number(stop_loss)
    ) else None
    dist_tp = abs(float(live_price) - float(take_profit)) if (
        is_number(live_price) and is_number(take_profit)
    ) else None
    return dist_sl, dist_tp


def is_price_fresh(
    price_timestamp: Optional[datetime],
    now: datetime,
    max_age_seconds: float = FRESH_PRICE_MAX_AGE_SECONDS,
) -> bool:
    """A live price is fresh iff it is no older than ``max_age_seconds`` (Req 4.10, 9.7)."""
    if price_timestamp is None:
        return False
    age = (now - price_timestamp).total_seconds()
    # "no older than max_age" -> age must not exceed the threshold. A price
    # captured essentially at/after ``now`` (age <= 0) is trivially not stale.
    return age <= max_age_seconds


def _mid_price(quote: dict, is_long: Optional[bool]) -> Optional[float]:
    """Price used for SL/TP distances: the side the position would close at."""
    if is_long is True and is_number(quote.get("bid")):
        return float(quote["bid"])
    if is_long is False and is_number(quote.get("ask")):
        return float(quote["ask"])
    for key in ("mid", "bid", "ask"):
        if is_number(quote.get(key)):
            return float(quote[key])
    return None


def derive_position(position: dict, quote: Optional[dict], now: datetime) -> dict:
    """Derive the display DTO for a single position with fresh-price exclusion.

    When no fresh quote is available, ``unrealized_pnl`` and both distances are
    marked unavailable while static fields are retained (Req 9.7).
    """
    out = {
        "position_id": position.get("position_id"),
        "symbol": position.get("symbol"),
        "direction": position.get("direction"),
        "entry_price": position.get("entry_price"),
        "stop_loss": position.get("stop_loss"),
        "take_profit": position.get("take_profit"),
        "quantity": position.get("quantity"),
        "risk_reward_ratio": position.get("risk_reward_ratio"),
        "is_breakeven": bool(position.get("is_breakeven", False)),
        "live_price_available": False,
        "unrealized_pnl": None,
        "unrealized_pnl_available": False,
        "distance_to_sl": None,
        "distance_to_tp": None,
        "distances_available": False,
    }
    if not isinstance(quote, dict):
        return out
    price_ts = parse_iso_utc(quote.get("timestamp")) if "timestamp" in quote else quote.get("_ts")
    if not is_price_fresh(price_ts, now):
        return out

    out["live_price_available"] = True
    upnl = unrealized_pnl(position, quote)
    if upnl is not None:
        out["unrealized_pnl"] = upnl
        out["unrealized_pnl_available"] = True
    is_long = _direction_is_long(position.get("direction"))
    price = _mid_price(quote, is_long)
    if price is not None:
        dist_sl, dist_tp = sl_tp_distances(price, position.get("stop_loss"), position.get("take_profit"))
        out["distance_to_sl"] = dist_sl
        out["distance_to_tp"] = dist_tp
        out["distances_available"] = dist_sl is not None and dist_tp is not None
    return out


def total_unrealized(
    positions: Iterable[dict], quotes: dict, now: datetime
) -> Optional[float]:
    """Total unrealized PnL over exactly the positions with a fresh price (Req 4.6, 4.10).

    ``quotes`` maps ``symbol -> quote dict`` (with a ``timestamp`` or ``_ts``).
    Returns ``None`` when no position has a fresh, computable price.
    """
    total = 0.0
    counted = 0
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        quote = quotes.get(pos.get("symbol"))
        if not isinstance(quote, dict):
            continue
        price_ts = parse_iso_utc(quote.get("timestamp")) if "timestamp" in quote else quote.get("_ts")
        if not is_price_fresh(price_ts, now):
            continue
        upnl = unrealized_pnl(pos, quote)
        if upnl is None:
            continue
        total += upnl
        counted += 1
    return total if counted > 0 else None
