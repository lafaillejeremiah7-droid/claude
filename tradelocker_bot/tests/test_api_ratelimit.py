"""
Tests for TradeLocker API rate-limiting / resilience behavior in
modules/api_client.py.

Hermetic: no real network. We construct a TradeLockerClient, then replace its
`session` with a fake and stub out authentication + instrument/route lookups so
`get_price_history` exercises only the throttle / retry / cache / diagnostics
logic. `time.sleep` is monkeypatched to record calls (so backoff/throttle are
asserted without actually sleeping), and a controllable monotonic clock lets us
test cache TTL expiry deterministically.
"""
import sys
from pathlib import Path

import pytest

# Make the bot root importable regardless of pytest's CWD.
BOT_ROOT = Path(__file__).resolve().parent.parent
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

import modules.api_client as api_client  # noqa: E402
from modules.api_client import TradeLockerClient  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, status_code=200, json_data=None, headers=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.headers = headers or {}
        self.text = text

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise api_client.requests.exceptions.HTTPError(
                f"{self.status_code} Error"
            )


class FakeSession:
    """Session whose `request` returns queued responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # list of (method, url, kwargs)
        self.headers = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self._responses:
            return self._responses.pop(0)
        # Default to an empty 200 if we run out.
        return FakeResponse(200, {"d": {"bars": []}})

    # get() is only used by get_latest_price; not exercised here but kept for
    # completeness / safety.
    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)


def _make_bar(t):
    return {"t": t, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100}


BARS_PAYLOAD = {"d": {"bars": [_make_bar(1_000_000), _make_bar(1_300_000)]}}


@pytest.fixture
def client(monkeypatch):
    """A TradeLockerClient with auth + instrument lookups stubbed out."""
    c = TradeLockerClient()
    monkeypatch.setattr(c, "ensure_authenticated", lambda: True)
    monkeypatch.setattr(c, "get_instrument_id", lambda symbol: 4242)
    monkeypatch.setattr(c, "get_route_id", lambda symbol, rt="INFO": 77)
    c.acc_num = 1
    # Deterministic, non-zero interval so throttle math is observable.
    c._min_request_interval = 0.5
    c._backoff_base = 1.0
    c._max_retries = 4
    return c


@pytest.fixture
def sleeps(monkeypatch):
    """Record all time.sleep calls (in api_client) instead of sleeping."""
    recorded = []
    monkeypatch.setattr(api_client.time, "sleep", lambda s: recorded.append(s))
    return recorded


class FakeClock:
    """Controllable monotonic clock."""

    def __init__(self, start=1000.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture
def clock(monkeypatch):
    fc = FakeClock()
    monkeypatch.setattr(api_client.time, "monotonic", fc)
    return fc


# ---------------------------------------------------------------------------
# Retry / backoff on 429
# ---------------------------------------------------------------------------

def test_retries_on_429_then_succeeds(client, sleeps, clock):
    """A 429 followed by a 200 should retry and eventually return bars."""
    client.session = FakeSession([
        FakeResponse(429, headers={}),
        FakeResponse(200, BARS_PAYLOAD),
    ])

    df = client.get_price_history("XAUUSD", 300, lookback_bars=10)

    assert df is not None
    assert len(df) == 2
    # Exactly two HTTP attempts were made.
    assert len(client.session.calls) == 2
    # It backed off at least once (a positive sleep between the two attempts).
    assert any(s > 0 for s in sleeps)


def test_honors_retry_after_header(client, sleeps, clock):
    """When Retry-After is set on a 429, the backoff must use that value."""
    client.session = FakeSession([
        FakeResponse(429, headers={"Retry-After": "7"}),
        FakeResponse(200, BARS_PAYLOAD),
    ])

    df = client.get_price_history("XAUUSD", 300, lookback_bars=10)

    assert df is not None
    # The Retry-After value (7s) must appear among the recorded sleeps. Throttle
    # sleeps are < the 0.5s interval, so 7.0 is unambiguously the backoff.
    assert 7.0 in sleeps


def test_gives_up_after_max_retries_returns_none(client, sleeps, clock):
    """Persistent 429s exhaust retries and return None (no raise)."""
    # 1 initial + 4 retries = 5 attempts, all 429.
    client.session = FakeSession([FakeResponse(429) for _ in range(5)])

    df = client.get_price_history("XAUUSD", 300, lookback_bars=10)

    assert df is None
    assert len(client.session.calls) == 5  # initial + max_retries


def test_retries_on_5xx(client, sleeps, clock):
    """A 503 should also be retried."""
    client.session = FakeSession([
        FakeResponse(503),
        FakeResponse(200, BARS_PAYLOAD),
    ])

    df = client.get_price_history("BTCUSD", 1800, lookback_bars=10)

    assert df is not None
    assert len(client.session.calls) == 2


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def test_second_call_within_ttl_uses_cache(client, sleeps, clock):
    """A second call within TTL returns cached bars without a new HTTP call."""
    client.session = FakeSession([FakeResponse(200, BARS_PAYLOAD)])

    df1 = client.get_price_history("BTCUSD", 300, lookback_bars=10)
    assert df1 is not None
    assert len(client.session.calls) == 1

    # Advance less than the 5m TTL (60s) -> cache hit, no new request.
    clock.advance(30)
    df2 = client.get_price_history("BTCUSD", 300, lookback_bars=10)
    assert df2 is not None
    assert len(client.session.calls) == 1  # unchanged -> served from cache
    assert len(df2) == len(df1)


def test_cache_expires_after_ttl_refetches(client, sleeps, clock):
    """After TTL expiry the client hits the network again."""
    client.session = FakeSession([
        FakeResponse(200, BARS_PAYLOAD),
        FakeResponse(200, BARS_PAYLOAD),
    ])

    client.get_price_history("BTCUSD", 300, lookback_bars=10)
    assert len(client.session.calls) == 1

    # Advance beyond the 5m resolution TTL (60s).
    clock.advance(61)
    client.get_price_history("BTCUSD", 300, lookback_bars=10)
    assert len(client.session.calls) == 2  # refetched


def test_cache_is_keyed_by_resolution(client, sleeps, clock):
    """Different resolutions are cached independently."""
    client.session = FakeSession([
        FakeResponse(200, BARS_PAYLOAD),
        FakeResponse(200, BARS_PAYLOAD),
    ])

    client.get_price_history("BTCUSD", 300, lookback_bars=10)
    client.get_price_history("BTCUSD", 1800, lookback_bars=10)
    # Two distinct keys -> two network calls.
    assert len(client.session.calls) == 2


def test_cached_dataframe_is_isolated_copy(client, sleeps, clock):
    """Mutating a returned frame must not corrupt the cached copy."""
    client.session = FakeSession([FakeResponse(200, BARS_PAYLOAD)])

    df1 = client.get_price_history("BTCUSD", 300, lookback_bars=10)
    df1.drop(df1.index, inplace=True)  # wipe the caller's copy

    clock.advance(10)  # still within TTL
    df2 = client.get_price_history("BTCUSD", 300, lookback_bars=10)
    assert df2 is not None
    assert len(df2) == 2  # cache unaffected by caller mutation


# ---------------------------------------------------------------------------
# Throttle
# ---------------------------------------------------------------------------

def test_throttle_spaces_consecutive_requests():
    """
    Two consecutive throttled requests are spaced by >= the min interval.
    Uses a real (short) interval and real timing to validate spacing.
    """
    c = TradeLockerClient()
    c._min_request_interval = 0.2

    import time as real_time
    t0 = real_time.monotonic()
    c._throttle()  # first call: no wait (last_request_ts starts at 0-ish)
    c._throttle()  # second call: must wait ~interval since the first
    c._throttle()  # third call: must wait ~interval since the second
    elapsed = real_time.monotonic() - t0

    # At least two inter-request gaps of 0.2s each.
    assert elapsed >= 0.4 - 0.01


def test_throttle_sleeps_to_maintain_interval(monkeypatch):
    """
    With a monotonic clock that doesn't advance, the second throttle call must
    sleep for (approximately) the full interval.
    """
    c = TradeLockerClient()
    c._min_request_interval = 0.5

    fc = FakeClock(start=100.0)
    monkeypatch.setattr(api_client.time, "monotonic", fc)
    recorded = []
    monkeypatch.setattr(api_client.time, "sleep", lambda s: recorded.append(s))

    c._throttle()  # establishes last_request_ts = 100.0
    c._throttle()  # clock hasn't advanced -> must sleep ~0.5s
    assert recorded
    assert recorded[-1] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Empty-200 diagnostics
# ---------------------------------------------------------------------------

def test_empty_200_logs_diagnostic_and_returns_none(client, sleeps, clock, caplog):
    """A 200 with no bars logs a diagnostic (status + keys) and returns None."""
    import logging
    client.session = FakeSession([
        FakeResponse(200, {"d": {"bars": []}}, text='{"d":{"bars":[]}}'),
    ])

    with caplog.at_level(logging.WARNING, logger="modules.api_client"):
        df = client.get_price_history("BTCUSD", 300, lookback_bars=10)

    assert df is None
    # Diagnostic must mention the HTTP status and that no data was returned.
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "No price data returned" in joined
    assert "HTTP 200" in joined
    # No exception was raised; only one attempt (200 is terminal).
    assert len(client.session.calls) == 1


def test_empty_200_is_not_cached(client, sleeps, clock):
    """An empty response must not populate the cache (so we retry next cycle)."""
    client.session = FakeSession([
        FakeResponse(200, {"d": {"bars": []}}, text="{}"),
        FakeResponse(200, BARS_PAYLOAD),
    ])

    assert client.get_price_history("BTCUSD", 300, lookback_bars=10) is None
    # Next call (even immediately) must hit the network again.
    df = client.get_price_history("BTCUSD", 300, lookback_bars=10)
    assert df is not None
    assert len(client.session.calls) == 2
