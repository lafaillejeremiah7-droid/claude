"""
Tests for the TradeLocker /trade/history resolution-code fix in
modules/api_client.py.

Background: the API requires a STRING resolution code (e.g. "5m", "30m", "4H")
and returns an HTTP 200 error envelope ({"s":"error","errmsg":...}) when it is
sent integer seconds. These hermetic tests (no network) confirm that:
  * get_price_history converts integer-seconds resolutions to the correct
    string codes in the outgoing request params (300->"5m", 1800->"30m",
    14400->"4H"),
  * an {"s":"error"} 200 body is treated as an error (returns None, logs
    errmsg) and is NOT cached,
  * the success path still returns a DataFrame when bars are present.
"""
import sys
from pathlib import Path

import pytest

# Make the bot root importable regardless of pytest's CWD.
BOT_ROOT = Path(__file__).resolve().parent.parent
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

import modules.api_client as api_client  # noqa: E402
from modules.api_client import (  # noqa: E402
    TradeLockerClient,
    RESOLUTION_CODES,
    resolution_to_code,
)


# ---------------------------------------------------------------------------
# Test doubles (mirrors the pattern in test_api_ratelimit.py)
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
            raise api_client.requests.exceptions.HTTPError(f"{self.status_code} Error")


class FakeSession:
    """Session whose `request` returns queued responses in order and records calls."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []  # list of (method, url, kwargs)
        self.headers = {}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self._responses:
            return self._responses.pop(0)
        return FakeResponse(200, {"d": {"bars": []}})

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)


def _make_bar(t):
    return {"t": t, "o": 1.0, "h": 2.0, "l": 0.5, "c": 1.5, "v": 100}


BARS_PAYLOAD = {"d": {"bars": [_make_bar(1_000_000), _make_bar(1_300_000)]}}

# The exact error envelope the live API returns for a bad resolution param.
RESOLUTION_ERROR_PAYLOAD = {
    "s": "error",
    "errmsg": (
        "The parameter resolution must have a value among : "
        "[1m, 5m, 15m, 30m, 1H, 4H, 1D, 1W, 1M]"
    ),
}


@pytest.fixture
def client(monkeypatch):
    """A TradeLockerClient with auth + instrument lookups stubbed out."""
    c = TradeLockerClient()
    monkeypatch.setattr(c, "ensure_authenticated", lambda: True)
    monkeypatch.setattr(c, "get_instrument_id", lambda symbol: 4242)
    monkeypatch.setattr(c, "get_route_id", lambda symbol, rt="INFO": 77)
    c.acc_num = 1
    c._min_request_interval = 0.0  # no throttling delay in tests
    return c


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Never actually sleep during these tests."""
    monkeypatch.setattr(api_client.time, "sleep", lambda s: None)


def _last_resolution_param(session):
    """Extract the outgoing `resolution` param from the last recorded call."""
    assert session.calls, "expected at least one HTTP call"
    _, _, kwargs = session.calls[-1]
    return kwargs["params"]["resolution"]


# ---------------------------------------------------------------------------
# Pure mapping helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,code",
    [(60, "1m"), (300, "5m"), (900, "15m"), (1800, "30m"),
     (3600, "1H"), (14400, "4H"), (86400, "1D"), (604800, "1W")],
)
def test_resolution_codes_mapping(seconds, code):
    assert RESOLUTION_CODES[seconds] == code
    assert resolution_to_code(seconds) == code


def test_resolution_to_code_best_effort_and_invalid():
    # Unknown but clean multiples fall back to best-effort codes.
    assert resolution_to_code(120) == "2m"      # sub-hour minute multiple
    assert resolution_to_code(7200) == "2H"     # exact hour multiple
    # Nonsensical values yield None.
    assert resolution_to_code(0) is None
    assert resolution_to_code(-5) is None


# ---------------------------------------------------------------------------
# get_price_history sends STRING resolution codes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seconds,expected_code",
    [(300, "5m"), (1800, "30m"), (14400, "4H")],
)
def test_get_price_history_sends_string_resolution(client, seconds, expected_code):
    """The three resolutions the bot uses must be sent as STRING codes."""
    client.session = FakeSession([FakeResponse(200, BARS_PAYLOAD)])

    df = client.get_price_history("BTCUSD", seconds, lookback_bars=10)

    assert df is not None
    sent = _last_resolution_param(client.session)
    assert sent == expected_code
    assert isinstance(sent, str)


# ---------------------------------------------------------------------------
# Error envelope handling
# ---------------------------------------------------------------------------

def test_error_envelope_returns_none_and_logs_errmsg(client, caplog):
    """A 200 {"s":"error"} body returns None and logs the errmsg."""
    import logging
    client.session = FakeSession([FakeResponse(200, RESOLUTION_ERROR_PAYLOAD)])

    with caplog.at_level(logging.WARNING, logger="modules.api_client"):
        df = client.get_price_history("BTCUSD", 300, lookback_bars=10)

    assert df is None
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "must have a value among" in joined
    assert len(client.session.calls) == 1  # 200 is terminal, no retries


def test_error_envelope_is_not_cached(client):
    """An error envelope must not populate the cache (retry next cycle)."""
    client.session = FakeSession([
        FakeResponse(200, RESOLUTION_ERROR_PAYLOAD),
        FakeResponse(200, BARS_PAYLOAD),
    ])

    assert client.get_price_history("BTCUSD", 300, lookback_bars=10) is None
    # Next call must hit the network again (nothing cached).
    df = client.get_price_history("BTCUSD", 300, lookback_bars=10)
    assert df is not None
    assert len(client.session.calls) == 2


# ---------------------------------------------------------------------------
# Success path still parses bars
# ---------------------------------------------------------------------------

def test_success_path_returns_dataframe(client):
    """When bars are present the success path returns a parsed DataFrame."""
    client.session = FakeSession([FakeResponse(200, BARS_PAYLOAD)])

    df = client.get_price_history("BTCUSD", 300, lookback_bars=10)

    assert df is not None
    assert len(df) == 2
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert col in df.columns
    # Numeric OHLCV values parsed correctly.
    assert df["close"].iloc[0] == pytest.approx(1.5)
