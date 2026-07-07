"""Focused example/unit tests for concrete edge cases of the pure derivations.

These complement the property tests with specific, human-readable scenarios and
the exact schema shapes from the design's Data Models section.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from dashboard.backend.guard import ReadOnlyGuard, ReadOnlyViolation, TradingCallBlocked
from dashboard.backend import security
from dashboard.backend.derivations import (
    build_feed,
    classify_gate,
    compute_countdown,
    compute_streaks,
    cumulative_realized_pnl,
    daily_return_pct,
    equity_curve,
    format_money,
    format_win_rate,
    gather_instrument_data,
    is_bot_offline,
    monitored_instruments,
    order_closes,
    parse_confidence,
    parse_iso_utc,
    parse_jsonl,
    resolve_bot_status,
    select_equity,
    sign_color,
    unrealized_pnl,
    derive_position,
)

UTC = timezone.utc


# ---- formatting ---------------------------------------------------------
def test_format_money_two_decimals():
    assert format_money(10091.1) == "10091.10"
    assert format_money(-18.42) == "-18.42"
    assert format_money(142.875) == "142.88"
    assert format_money(0) == "0.00"
    assert format_money(-0.0001) == "0.00"  # no "-0.00"


def test_format_win_rate_one_decimal():
    assert format_win_rate(62.5) == "62.5"
    assert format_win_rate(100) == "100.0"


def test_sign_color():
    assert sign_color(5) == "green"
    assert sign_color(-5) == "red"
    assert sign_color(0) == "neutral"


# ---- equity / pnl -------------------------------------------------------
def test_select_equity_prefers_api():
    r = select_equity({"equity": 10091.16, "balance": 10109.58, "free_margin": 9800.0},
                      {"current_equity": 9000.0})
    assert r["equity_source"] == "api"
    assert r["equity"] == 10091.16
    assert r["balance"] == 10109.58


def test_select_equity_falls_back_and_hides_balance():
    r = select_equity({"equity": -1}, {"current_equity": 9000.0})
    assert r["equity_source"] == "daily_stats_fallback"
    assert r["equity"] == 9000.0
    assert r["balance_available"] is False
    assert r["free_margin_available"] is False


def test_select_equity_no_data():
    r = select_equity(None, None)
    assert r["equity_source"] == "none"
    assert r["equity_available"] is False


def test_daily_return_zero_start():
    assert daily_return_pct(0, 500) == 0.00


def test_daily_return_formula():
    assert daily_return_pct(10000, 10142.3) == pytest.approx(1.423)


def test_cumulative_ignores_invalid_pnl():
    closes = [
        {"pnl": 210.0}, {"pnl": None}, {"pnl": "x"}, {"pnl": True}, {"pnl": -68.0},
    ]
    assert cumulative_realized_pnl(closes) == pytest.approx(142.0)
    assert cumulative_realized_pnl([]) == 0


# ---- positions ----------------------------------------------------------
def test_unrealized_long_and_short():
    long_pos = {"direction": "buy", "entry_price": 100.0, "quantity": 2.0}
    assert unrealized_pnl(long_pos, {"bid": 105.0, "ask": 106.0}) == pytest.approx(10.0)
    short_pos = {"direction": "sell", "entry_price": 100.0, "quantity": 2.0}
    assert unrealized_pnl(short_pos, {"bid": 94.0, "ask": 95.0}) == pytest.approx(10.0)


def test_derive_position_stale_price_unavailable():
    now = datetime(2024, 6, 10, 12, 0, 0, tzinfo=UTC)
    pos = {"symbol": "BTCUSD", "direction": "buy", "entry_price": 100.0,
           "quantity": 1.0, "stop_loss": 90.0, "take_profit": 120.0}
    stale = {"bid": 110.0, "ask": 111.0, "_ts": now - timedelta(seconds=10)}
    d = derive_position(pos, stale, now)
    assert d["unrealized_pnl_available"] is False
    assert d["distances_available"] is False


# ---- streaks ------------------------------------------------------------
def test_streaks_trailing_run():
    closes = [
        {"timestamp": "2024-06-10T10:00:00", "is_win": False, "file_date": "2024-06-10", "line_index": 0},
        {"timestamp": "2024-06-10T11:00:00", "is_win": True, "file_date": "2024-06-10", "line_index": 1},
        {"timestamp": "2024-06-10T12:00:00", "is_win": True, "file_date": "2024-06-10", "line_index": 2},
    ]
    assert compute_streaks(order_closes(closes)) == (2, 0)


def test_streaks_non_bool_excluded_without_breaking_run():
    closes = [
        {"timestamp": "2024-06-10T10:00:00", "is_win": True, "file_date": "2024-06-10", "line_index": 0},
        {"timestamp": "2024-06-10T11:00:00", "is_win": "n/a", "file_date": "2024-06-10", "line_index": 1},
        {"timestamp": "2024-06-10T12:00:00", "is_win": True, "file_date": "2024-06-10", "line_index": 2},
    ]
    assert compute_streaks(order_closes(closes)) == (2, 0)


def test_streaks_empty():
    assert compute_streaks([]) == (0, 0)


# ---- confidence ---------------------------------------------------------
def test_parse_confidence_variants():
    assert parse_confidence("Confidence: 8.5/10") == 8.5
    assert parse_confidence("Confidence:  8 / 10 (need 8.0)") == 8.0
    assert parse_confidence("no score") is None
    assert parse_confidence("Confidence: 12/10") is None


def test_classify_gate():
    assert classify_gate(8.5, 8.0) == "met_gate"
    assert classify_gate(7.9, 8.0) == "near_miss"
    assert classify_gate(None, 8.0) == "unknown"


# ---- readers ------------------------------------------------------------
def test_parse_jsonl_skips_malformed():
    blob = '{"a": 1}\nnot json\n\n[1,2]\n{"b": 2}'
    assert parse_jsonl(blob) == [{"a": 1}, {"b": 2}]


# ---- feed ---------------------------------------------------------------
def test_build_feed_orders_and_filters():
    journal = [
        {"timestamp": "2024-06-10T10:00:00", "action": "OPEN", "symbol": "BTCUSD", "direction": "buy"},
        {"timestamp": "", "action": "CLOSE", "symbol": "BTCUSD", "direction": "buy"},  # invalid
    ]
    events = [
        {"timestamp": "2024-06-10T11:00:00", "action": "NEAR_MISS", "symbol": "XAUUSD", "direction": "sell"},
    ]
    feed = build_feed(journal, events)
    assert [e["action"] for e in feed] == ["NEAR_MISS", "OPEN"]


# ---- equity curve -------------------------------------------------------
def test_equity_curve_running_sum():
    closes = [
        {"timestamp": "2024-06-10T10:00:00", "pnl": 10.0, "file_date": "2024-06-10", "line_index": 0},
        {"timestamp": "2024-06-10T11:00:00", "pnl": -4.0, "file_date": "2024-06-10", "line_index": 1},
    ]
    curve = equity_curve(closes, 1000.0)
    assert [p["equity"] for p in curve["points"]] == [1010.0, 1006.0]
    assert curve["state"] == "ok"


def test_equity_curve_insufficient():
    curve = equity_curve([{"timestamp": "2024-06-10T10:00:00", "pnl": 10.0}], 1000.0)
    assert curve["state"] == "insufficient_data"


# ---- countdown / status -------------------------------------------------
def test_countdown_due_and_scanning():
    last = datetime(2024, 6, 10, 12, 0, 0, tzinfo=UTC)
    assert compute_countdown(last, last + timedelta(seconds=18))["seconds_remaining"] == 42
    assert compute_countdown(last, last + timedelta(seconds=75))["state"] == "due"
    assert compute_countdown(None, last)["state"] == "no_scan_yet"


def test_status_precedence_examples():
    assert resolve_bot_status(initializing=True, bot_offline=True, is_locked=True,
                              in_avoid_hours=True, out_of_session=True,
                              has_scan_activity=True) == "initializing"
    assert resolve_bot_status(initializing=False, bot_offline=False, is_locked=True,
                              in_avoid_hours=True, out_of_session=False,
                              has_scan_activity=True) == "locked"
    assert resolve_bot_status(initializing=False, bot_offline=False, is_locked=False,
                              in_avoid_hours=False, out_of_session=False,
                              has_scan_activity=False) == "no_scan_yet"


# ---- freshness ----------------------------------------------------------
def test_bot_offline_requires_both():
    now = datetime(2024, 6, 10, 12, 0, 0, tzinfo=UTC)
    # File fresh, scan old -> not offline.
    assert is_bot_offline(now - timedelta(seconds=10), now - timedelta(seconds=200), now) is False
    # Both old -> offline.
    assert is_bot_offline(now - timedelta(seconds=200), now - timedelta(seconds=200), now) is True


# ---- timeutil -----------------------------------------------------------
def test_parse_iso_utc_naive_is_utc():
    assert parse_iso_utc("2024-06-10T12:34:56") == datetime(2024, 6, 10, 12, 34, 56, tzinfo=UTC)
    assert parse_iso_utc("2024-06-10T12:34:56+00:00") == datetime(2024, 6, 10, 12, 34, 56, tzinfo=UTC)
    assert parse_iso_utc("") is None
    assert parse_iso_utc(None) is None
    assert parse_iso_utc("garbage") is None


# ---- instruments --------------------------------------------------------
def test_monitored_capped_at_two():
    assert monitored_instruments(["BTCUSD", "XAUUSD", "ETHUSD"]) == ["BTCUSD", "XAUUSD"]
    assert monitored_instruments([]) == []


def test_gather_instrument_isolation():
    def fetcher(sym):
        if sym == "XAUUSD":
            raise RuntimeError("boom")
        return {"bid": 1, "ask": 2}
    res = gather_instrument_data(["BTCUSD", "XAUUSD"], fetcher)
    assert res["BTCUSD"]["data_available"] is True
    assert res["XAUUSD"]["data_available"] is False


# ---- guard --------------------------------------------------------------
def test_guard_blocks_order_post():
    g = ReadOnlyGuard()
    with pytest.raises(TradingCallBlocked):
        g.assert_get_only("POST", "https://x/backend-api/trade/orders")
    assert g.errors[-1]["type"] == "blocked_request"


def test_guard_allows_auth_post_and_get():
    g = ReadOnlyGuard()
    assert g.assert_get_only("POST", "https://x/backend-api/auth/jwt/token") is True
    assert g.assert_get_only("GET", "https://x/backend-api/trade/quotes") is True


def test_guard_blocks_write_open(tmp_path):
    p = tmp_path / "daily_stats.json"
    p.write_text('{"daily": {}}')
    before = p.read_bytes()
    g = ReadOnlyGuard()
    with pytest.raises(ReadOnlyViolation):
        g.open_readonly(str(p), "w")
    assert p.read_bytes() == before
    assert g.errors[-1]["type"] == "blocked_write"


# ---- security -----------------------------------------------------------
def test_credential_status_ok_and_missing():
    full = {f: "value" for f in security.CREDENTIAL_FIELDS}
    assert security.credential_status(full)["status"] == "ok"
    del full["TL_PASSWORD"]
    status = security.credential_status(full)
    assert status["config_error_field"] == "TL_PASSWORD"


def test_redact_removes_tokens():
    payload = {"access_token": "abc", "equity": 100.0, "nested": {"TL_PASSWORD": "p"}}
    out = security.redact_secrets(payload, ["abc", "p"])
    assert "access_token" not in out
    assert "TL_PASSWORD" not in out["nested"]
    assert out["equity"] == 100.0
