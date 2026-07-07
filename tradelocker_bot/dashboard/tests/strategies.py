"""Custom Hypothesis strategies (generators) for the dashboard property tests.

These mirror the design's Testing Strategy: CLOSE-action lists with mixed
valid/invalid pnl & is_win and duplicate timestamps, positions + per-symbol
prices with staleness, JSONL blobs with interleaved malformed lines, confidence
strings, time pairs, and method/url + payload/secret generators.
"""
from __future__ import annotations

import json
import string
from datetime import datetime, timezone

from hypothesis import strategies as st

UTC = timezone.utc

# Sentinel meaning "omit this key entirely" when building an action dict.
MISSING = object()


# --------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------
def iso_timestamps():
    """Valid ISO-8601 UTC timestamp strings (naive-UTC or explicit +00:00)."""
    base = st.datetimes(
        min_value=datetime(2024, 1, 1, 0, 0, 0),
        max_value=datetime(2024, 12, 31, 23, 59, 59),
    )
    naive = base.map(lambda d: d.replace(microsecond=0).isoformat())
    offset = base.map(lambda d: d.replace(microsecond=0).isoformat() + "+00:00")
    return st.one_of(naive, offset)


def aware_datetimes():
    return st.datetimes(
        min_value=datetime(2024, 1, 1),
        max_value=datetime(2024, 12, 31),
    ).map(lambda d: d.replace(tzinfo=UTC))


# --------------------------------------------------------------------------
# pnl / is_win values (valid + invalid mixes)
# --------------------------------------------------------------------------
def numeric_pnl():
    return st.floats(min_value=-10000, max_value=10000, allow_nan=False, allow_infinity=False)


def invalid_pnl():
    return st.one_of(
        st.none(),
        st.text(alphabet=string.ascii_letters, max_size=4),
        st.booleans(),  # bool must NOT be treated as numeric
    )


def is_win_values():
    return st.one_of(
        st.booleans(),                      # valid classification
        st.none(),                          # excluded
        st.text(alphabet=string.ascii_letters, max_size=3),  # non-bool -> excluded
        st.integers(min_value=0, max_value=1),               # non-bool -> excluded
    )


@st.composite
def close_actions(draw, min_size=0, max_size=12):
    """A list of CLOSE actions with mixed valid/invalid pnl & is_win, duplicate
    timestamps, and unique ``(file_date, line_index)`` source keys."""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    ts_pool = draw(st.lists(iso_timestamps(), min_size=1, max_size=4))
    file_dates = ["2024-06-10", "2024-06-11", "2024-06-12"]
    actions = []
    for i in range(n):
        action = {"action": "CLOSE"}
        ts = draw(st.sampled_from(ts_pool))
        action["timestamp"] = ts
        action["file_date"] = draw(st.sampled_from(file_dates))
        action["line_index"] = i  # globally unique -> keys are unique
        pnl = draw(st.one_of(numeric_pnl(), invalid_pnl(), st.just(MISSING)))
        if pnl is not MISSING:
            action["pnl"] = pnl
        is_win = draw(st.one_of(is_win_values(), st.just(MISSING)))
        if is_win is not MISSING:
            action["is_win"] = is_win
        actions.append(action)
    return actions


# --------------------------------------------------------------------------
# Positions + per-symbol prices with staleness
# --------------------------------------------------------------------------
def price_floats():
    return st.floats(min_value=0.1, max_value=100000, allow_nan=False, allow_infinity=False)


@st.composite
def positions_with_quotes(draw, max_size=6):
    """List of (position, quote) pairs; each quote carries an age in seconds.

    Each position gets a unique symbol so quotes map 1:1.
    """
    n = draw(st.integers(min_value=0, max_value=max_size))
    pairs = []
    for i in range(n):
        direction = draw(st.sampled_from(["buy", "sell", "long", "short"]))
        pos = {
            "position_id": str(i),
            "symbol": f"SYM{i}",
            "direction": direction,
            "entry_price": draw(price_floats()),
            "quantity": draw(st.floats(min_value=0.01, max_value=100,
                                       allow_nan=False, allow_infinity=False)),
            "stop_loss": draw(price_floats()),
            "take_profit": draw(price_floats()),
        }
        quote = {
            "bid": draw(price_floats()),
            "ask": draw(price_floats()),
            "age": draw(st.floats(min_value=-2, max_value=30,
                                  allow_nan=False, allow_infinity=False)),
        }
        pairs.append((pos, quote))
    return pairs


# --------------------------------------------------------------------------
# JSONL blobs with interleaved malformed lines
# --------------------------------------------------------------------------
def _json_scalar():
    return st.one_of(
        st.integers(min_value=-1000, max_value=1000),
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.text(alphabet=string.printable, max_size=8).filter(lambda s: "\n" not in s and "\r" not in s),
        st.booleans(),
        st.none(),
    )


def json_object_lines():
    """A JSON *object* serialized to a single line (a valid JSONL record)."""
    return st.dictionaries(
        keys=st.text(alphabet=string.ascii_letters + "_", min_size=1, max_size=6),
        values=_json_scalar(),
        max_size=4,
    ).map(lambda d: json.dumps(d))


def malformed_lines():
    """Lines that are NOT valid JSON objects (must be skipped by tolerant readers)."""
    return st.one_of(
        st.just("{not valid json"),
        st.just("]["),
        st.just("42"),                 # valid JSON but not an object
        st.just("[1, 2, 3]"),          # valid JSON array, not an object
        st.just('"just a string"'),    # valid JSON string, not an object
        st.just(""),                   # blank line
        st.just("   "),                # whitespace only
        st.text(alphabet=string.ascii_letters + " ", max_size=10).filter(
            lambda s: not s.strip().startswith("{")
        ),
    )


# --------------------------------------------------------------------------
# Confidence strings
# --------------------------------------------------------------------------
def in_range_confidence():
    return st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)


# --------------------------------------------------------------------------
# Feed entries (valid + invalid mixes)
# --------------------------------------------------------------------------
@st.composite
def feed_entry(draw, valid=None):
    """A feed entry. When ``valid`` is True it has all required non-empty fields
    and a parseable timestamp; when False at least one required field is broken."""
    if valid is None:
        valid = draw(st.booleans())
    entry = {
        "kind": draw(st.sampled_from(["trade", "event"])),
        "action": draw(st.sampled_from(["OPEN", "CLOSE", "NEAR_MISS", "APPROVED"])),
        "symbol": draw(st.sampled_from(["BTCUSD", "XAUUSD"])),
        "direction": draw(st.sampled_from(["buy", "sell"])),
        "timestamp": draw(iso_timestamps()),
    }
    if not valid:
        broken = draw(st.sampled_from(["timestamp", "action", "symbol", "direction"]))
        entry[broken] = draw(st.sampled_from(["", None, "not-a-timestamp" if broken == "timestamp" else ""]))
        if broken == "timestamp":
            entry[broken] = draw(st.sampled_from(["", "garbage", None]))
    return entry


@st.composite
def feed_entries(draw, max_size=130):
    n = draw(st.integers(min_value=0, max_value=max_size))
    return [draw(feed_entry()) for _ in range(n)]


# --------------------------------------------------------------------------
# API method / URL generators (guard)
# --------------------------------------------------------------------------
API_BASE = "https://demo.tradelocker.com/backend-api"
TRADING_DATA_PATHS = [
    "/trade/accounts/1/state",
    "/trade/quotes",
    "/trade/accounts/1/instruments",
    "/trade/history",
]
AUTH_PATHS = ["/auth/jwt/token", "/auth/jwt/refresh"]
MUTATION_PATHS = [
    "/trade/orders",
    "/trade/orders/123",
    "/trade/positions/123",
    "/trade/accounts/1/orders",
]
ALL_PATHS = TRADING_DATA_PATHS + AUTH_PATHS + MUTATION_PATHS
HTTP_METHODS = ["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]


def http_methods():
    return st.sampled_from(HTTP_METHODS)


def api_paths():
    return st.sampled_from(ALL_PATHS)


# --------------------------------------------------------------------------
# Secret / payload generators (credentials)
# --------------------------------------------------------------------------
def secret_string():
    return st.text(alphabet=string.ascii_letters + string.digits, min_size=6, max_size=20)
