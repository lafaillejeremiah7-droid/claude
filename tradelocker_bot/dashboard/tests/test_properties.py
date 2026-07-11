"""Property-based tests for the Trading Dashboard pure derivation/parsing logic.

EXACTLY ONE property-based test per design property (Properties 1-25). Each test
carries a ``# Feature: trading-dashboard, Property N: ...`` tag on the line above.
Each runs >= 100 examples via the ``dashboard`` hypothesis profile (conftest.py).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dashboard.backend import guard as guard_mod
from dashboard.backend import security
from dashboard.backend.guard import ReadOnlyGuard, ReadOnlyViolation, TradingCallBlocked
from dashboard.backend.derivations import (
    build_feed,
    classify_gate,
    compute_countdown,
    compute_streaks,
    cumulative_realized_pnl,
    daily_return_pct,
    data_age,
    derive_position,
    equity_curve,
    format_money,
    format_win_rate,
    gather_instrument_data,
    is_bot_offline,
    is_price_fresh,
    is_stale,
    monitored_instruments,
    order_closes,
    parse_confidence,
    parse_iso_utc,
    parse_jsonl,
    resolve_bot_status,
    seconds_since,
    select_equity,
    sign_color,
    sl_tp_distances,
    total_unrealized,
    unrealized_pnl,
)
from dashboard.backend.derivations.common import is_number

from . import strategies as gen

UTC = timezone.utc


# ==========================================================================
# Property 1
# ==========================================================================
# Feature: trading-dashboard, Property 1: Read-only API guard permits only GET to trading-data or POST to the two auth endpoints; all order/position mutations are blocked before transmission and recorded.
@given(method=gen.http_methods(), path=gen.api_paths())
def test_property_01_api_guard_permits_only_safe_requests(method, path):
    g = ReadOnlyGuard()
    url = gen.API_BASE + path
    is_auth = path in gen.AUTH_PATHS
    expected_allowed = (method == "GET") or (method == "POST" and is_auth)

    if expected_allowed:
        assert g.assert_get_only(method, url) is True
        assert g.errors == []
    else:
        with pytest.raises(TradingCallBlocked):
            g.assert_get_only(method, url)
        assert len(g.errors) == 1
        assert g.errors[0]["type"] == "blocked_request"
        assert g.errors[0]["url"] == url


# ==========================================================================
# Property 2
# ==========================================================================
# Feature: trading-dashboard, Property 2: File guard raises before the OS call for any write/modify/truncate mode, leaves the file's bytes and mtime unchanged, and records a blocked-write error naming the path.
@given(
    content=st.text(max_size=64),
    mode=st.sampled_from(["w", "a", "x", "r+", "w+", "a+", "wb", "ab"]),
)
@settings(max_examples=100)
def test_property_02_file_guard_never_mutates(content, mode):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(content)
        before_bytes = open(path, "rb").read()
        before_mtime = os.stat(path).st_mtime_ns

        g = ReadOnlyGuard()
        with pytest.raises(ReadOnlyViolation):
            g.open_readonly(path, mode)

        assert open(path, "rb").read() == before_bytes
        assert os.stat(path).st_mtime_ns == before_mtime
        assert len(g.errors) == 1
        assert g.errors[0]["type"] == "blocked_write"
        assert g.errors[0]["path"] == str(path)
    finally:
        os.remove(path)


# ==========================================================================
# Property 3
# ==========================================================================
# Feature: trading-dashboard, Property 3: Secrets never leave the server - no outbound payload contains TL_EMAIL/TL_PASSWORD/access/refresh tokens as values or keys, and client-supplied TL_* fields are ignored.
@given(
    email=gen.secret_string(),
    password=gen.secret_string(),
    access=gen.secret_string(),
    refresh=gen.secret_string(),
    extra=st.dictionaries(
        keys=st.text(alphabet="abcdef", min_size=1, max_size=4),
        values=st.text(max_size=8),
        max_size=4,
    ),
)
def test_property_03_secrets_never_leave(email, password, access, refresh, extra):
    # Real secrets (emails, JWTs) are long/distinctive; prefix the generated
    # values so they cannot coincide with structural key/value strings.
    email = "SECRET_EMAIL_" + email
    password = "SECRET_PWD_" + password
    access = "SECRET_ACCESS_" + access
    refresh = "SECRET_REFRESH_" + refresh
    secrets = [email, password, access, refresh]
    # An "internal state" payload that embeds secrets both as values and keys.
    payload = {
        "TL_EMAIL": email,
        "TL_PASSWORD": password,
        "access_token": access,
        "refresh_token": refresh,
        "account": {"equity": 10091.16, "note": email},
        "feed": [{"symbol": "BTCUSD", "token": access}],
        "extra": extra,
    }
    redacted = security.redact_secrets(payload, secrets)
    serialized = json.dumps(redacted)

    for secret in secrets:
        assert secret not in serialized
    for key in ("TL_EMAIL", "TL_PASSWORD", "access_token", "refresh_token"):
        assert key not in serialized

    # Client-supplied credential fields are ignored; stored creds unchanged.
    stored = {f: f"stored_{f}" for f in security.CREDENTIAL_FIELDS}
    stored_before = dict(stored)
    client_payload = {f: "attacker_value" for f in security.CREDENTIAL_FIELDS}
    client_payload["harmless"] = "ok"
    accepted = security.strip_credential_fields(client_payload)
    assert all(f not in accepted for f in security.CREDENTIAL_FIELDS)
    assert stored == stored_before


# ==========================================================================
# Property 4
# ==========================================================================
# Feature: trading-dashboard, Property 4: With exactly one missing credential, the status names the missing field, does not attempt auth, and exposes no credential value.
@given(missing=st.sampled_from(security.CREDENTIAL_FIELDS), values=st.lists(gen.secret_string(), min_size=4, max_size=4))
def test_property_04_missing_credential_names_field(missing, values):
    env = {f: v for f, v in zip(security.CREDENTIAL_FIELDS, values)}
    present_values = {f: env[f] for f in security.CREDENTIAL_FIELDS if f != missing}
    del env[missing]

    status = security.credential_status(env)
    assert status["status"] == "config_error"
    assert status["config_error_field"] == missing
    # No credential value appears anywhere in the status.
    serialized = json.dumps(status)
    for value in present_values.values():
        assert value not in serialized


# ==========================================================================
# Property 5
# ==========================================================================
# Feature: trading-dashboard, Property 5: A token refresh is required iff now >= expiry - 30s.
@given(
    expiry=gen.aware_datetimes(),
    offset=st.integers(min_value=-120, max_value=120),
)
def test_property_05_refresh_decision_tracks_window(expiry, offset):
    now = expiry + timedelta(seconds=offset)
    expected = now >= expiry - timedelta(seconds=30)
    assert security.refresh_required(now, expiry) is expected


# ==========================================================================
# Property 6
# ==========================================================================
# Feature: trading-dashboard, Property 6: format_money yields exactly two decimals parsing back to round(v,2); format_win_rate yields exactly one decimal.
@given(value=st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False))
def test_property_06_fixed_decimal_precision(value):
    money = format_money(value)
    assert re.fullmatch(r"-?\d+\.\d{2}", money)
    assert float(money) == pytest.approx(round(value, 2), abs=1e-9)

    wr = format_win_rate(value)
    assert re.fullmatch(r"-?\d+\.\d{1}", wr)
    assert float(wr) == pytest.approx(round(value, 1), abs=1e-9)


# ==========================================================================
# Property 7
# ==========================================================================
# Feature: trading-dashboard, Property 7: Equity source is API when its equity is numeric and non-negative, else daily_stats current_equity when numeric, else no-data; fallback marks balance/free-margin unavailable.
@given(
    api_equity=st.one_of(st.none(), st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), st.text(max_size=3)),
    balance=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    free_margin=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    daily_equity=st.one_of(st.none(), st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False), st.text(max_size=3)),
    has_api=st.booleans(),
    has_daily=st.booleans(),
)
def test_property_07_equity_fallback(api_equity, balance, free_margin, daily_equity, has_api, has_daily):
    api_state = {"equity": api_equity, "balance": balance, "free_margin": free_margin} if has_api else None
    daily_stats = {"current_equity": daily_equity} if has_daily else None

    result = select_equity(api_state, daily_stats)

    api_ok = has_api and is_number(api_equity) and api_equity >= 0
    daily_ok = has_daily and is_number(daily_equity)

    if api_ok:
        assert result["equity_source"] == "api"
        assert result["equity"] == pytest.approx(float(api_equity))
        assert result["balance_available"] is True
        assert result["free_margin_available"] is True
    elif daily_ok:
        assert result["equity_source"] == "daily_stats_fallback"
        assert result["equity"] == pytest.approx(float(daily_equity))
        assert result["balance_available"] is False
        assert result["free_margin_available"] is False
    else:
        assert result["equity_source"] == "none"
        assert result["equity_available"] is False


# ==========================================================================
# Property 8
# ==========================================================================
# Feature: trading-dashboard, Property 8: Daily return % equals ((current-start)/start)*100 for start>0, and exactly 0.00 when start==0.
@given(
    start=st.floats(min_value=1e-6, max_value=1e7, allow_nan=False, allow_infinity=False),
    current=st.floats(min_value=-1e7, max_value=1e7, allow_nan=False, allow_infinity=False),
)
def test_property_08_daily_return_formula(start, current):
    expected = ((current - start) / start) * 100.0
    assert daily_return_pct(start, current) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert daily_return_pct(0, current) == 0.00


# ==========================================================================
# Property 9
# ==========================================================================
# Feature: trading-dashboard, Property 9: Cumulative realized PnL sums exactly the present-and-numeric pnl values, ignoring absent/null/non-numeric, 0 when none.
@given(closes=gen.close_actions())
def test_property_09_cumulative_sums_valid_only(closes):
    expected = sum(float(a["pnl"]) for a in closes if is_number(a.get("pnl")))
    assert cumulative_realized_pnl(closes) == pytest.approx(expected, rel=1e-9, abs=1e-9)
    if not any(is_number(a.get("pnl")) for a in closes):
        assert cumulative_realized_pnl(closes) == 0


# ==========================================================================
# Property 10
# ==========================================================================
# Feature: trading-dashboard, Property 10: Positions with a fresh (<=5s) price get direction-correct unrealized PnL and abs SL/TP distances; stale-priced positions mark these unavailable; total equals the sum over fresh positions.
@given(pairs=gen.positions_with_quotes())
def test_property_10_position_derivations_and_freshness(pairs):
    now = datetime(2024, 6, 10, 12, 0, 0, tzinfo=UTC)
    positions = []
    quotes = {}
    expected_total = 0.0
    fresh_count = 0
    for pos, quote in pairs:
        ts = now - timedelta(seconds=quote["age"])
        q = {"bid": quote["bid"], "ask": quote["ask"], "_ts": ts}
        positions.append(pos)
        quotes[pos["symbol"]] = q

        fresh = quote["age"] <= 5
        derived = derive_position(pos, q, now)
        if fresh:
            is_long = pos["direction"] in ("buy", "long")
            if is_long:
                exp_pnl = (quote["bid"] - pos["entry_price"]) * pos["quantity"]
                price = quote["bid"]
            else:
                exp_pnl = (pos["entry_price"] - quote["ask"]) * pos["quantity"]
                price = quote["ask"]
            assert derived["unrealized_pnl_available"] is True
            assert derived["unrealized_pnl"] == pytest.approx(exp_pnl)
            assert derived["distance_to_sl"] == pytest.approx(abs(price - pos["stop_loss"]))
            assert derived["distance_to_tp"] == pytest.approx(abs(price - pos["take_profit"]))
            expected_total += exp_pnl
            fresh_count += 1
        else:
            assert derived["unrealized_pnl_available"] is False
            assert derived["distances_available"] is False

    total = total_unrealized(positions, quotes, now)
    if fresh_count == 0:
        assert total is None
    else:
        assert total == pytest.approx(expected_total)


# ==========================================================================
# Property 11
# ==========================================================================
# Feature: trading-dashboard, Property 11: Sign-to-colour mapping is >0 green, <0 red, ==0 neutral.
@given(value=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12))
def test_property_11_sign_color_consistent(value):
    color = sign_color(value)
    if value > 0:
        assert color == "green"
    elif value < 0:
        assert color == "red"
    else:
        assert color == "neutral"


# ==========================================================================
# Property 12
# ==========================================================================
# Feature: trading-dashboard, Property 12: Ordering CLOSE actions ascending by UTC timestamp with ties broken by (file_date,line_index) is a total order identical across repeated/permuted constructions.
@given(closes=gen.close_actions(min_size=0, max_size=15), seed=st.integers(0, 10_000))
def test_property_12_deterministic_close_ordering(closes, seed):
    import random

    shuffled = list(closes)
    random.Random(seed).shuffle(shuffled)

    order_a = order_closes(closes)
    order_b = order_closes(shuffled)

    def key(a):
        return (a.get("file_date"), a.get("line_index"))

    # Same total order regardless of input permutation.
    assert [key(a) for a in order_a] == [key(a) for a in order_b]
    # And repeated construction is identical.
    assert [key(a) for a in order_closes(closes)] == [key(a) for a in order_a]
    # Ascending by timestamp with (file_date, line_index) tie-break.
    keys = [
        ((parse_iso_utc(a.get("timestamp")) or datetime.max.replace(tzinfo=UTC)),
         a.get("file_date"), a.get("line_index"))
        for a in order_a
    ]
    assert keys == sorted(keys)


# ==========================================================================
# Property 13
# ==========================================================================
# Feature: trading-dashboard, Property 13: Streaks classify only boolean is_win, exclude non-bool without breaking runs, compute the trailing run, are (0,0) when none, and never both non-zero.
@given(closes=gen.close_actions())
def test_property_13_streaks_correct_and_robust(closes):
    ordered = order_closes(closes)
    win_streak, loss_streak = compute_streaks(ordered)

    classified = [a["is_win"] for a in ordered if a.get("is_win") is True or a.get("is_win") is False]
    if not classified:
        assert (win_streak, loss_streak) == (0, 0)
    else:
        last = classified[-1]
        run = 0
        for v in reversed(classified):
            if v == last:
                run += 1
            else:
                break
        if last is True:
            assert (win_streak, loss_streak) == (run, 0)
        else:
            assert (win_streak, loss_streak) == (0, run)
    # Never both non-zero.
    assert not (win_streak > 0 and loss_streak > 0)


# ==========================================================================
# Property 14
# ==========================================================================
# Feature: trading-dashboard, Property 14: Confidence: v/10 (with surrounding whitespace) parses back to v in [0,10]; text without an in-range token yields unavailable.
@given(value=gen.in_range_confidence())
def test_property_14_confidence_extraction(value):
    v = round(value, 2)
    text = f"  reason ... Confidence:  {v} / 10  (need 8.0)"
    assert parse_confidence(text) == pytest.approx(v)
    # Out of range -> unavailable (not fabricated).
    assert parse_confidence(f"Confidence: {v + 11.0}/10") is None
    # No token at all -> unavailable.
    assert parse_confidence("no confidence token present here") is None


# ==========================================================================
# Property 15
# ==========================================================================
# Feature: trading-dashboard, Property 15: An entry meets the gate iff confidence >= gate, else near_miss.
@given(
    confidence=st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
    gate=st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
)
def test_property_15_gate_classification(confidence, gate):
    result = classify_gate(confidence, gate)
    if confidence >= gate:
        assert result == "met_gate"
    else:
        assert result == "near_miss"


# ==========================================================================
# Property 16
# ==========================================================================
# Feature: trading-dashboard, Property 16: Tolerant JSONL parsing returns exactly the valid object records (empty when none) and never raises on missing/partial/malformed input.
@given(
    items=st.lists(
        st.one_of(
            gen.json_object_lines().map(lambda s: ("valid", s)),
            gen.malformed_lines().map(lambda s: ("invalid", s)),
        ),
        max_size=40,
    )
)
def test_property_16_robust_readers(items):
    lines = [text for _, text in items]
    blob = "\n".join(lines)
    expected = [json.loads(text) for kind, text in items if kind == "valid"]

    result = parse_jsonl(blob)
    assert result == expected


# ==========================================================================
# Property 17
# ==========================================================================
# Feature: trading-dashboard, Property 17: The merged feed is newest-first, capped at 100 most-recent, and every emitted entry has non-empty timestamp/action/symbol/direction.
@given(journal=gen.feed_entries(max_size=80), events=gen.feed_entries(max_size=80))
def test_property_17_feed_ordering_merge_cap(journal, events):
    feed = build_feed(journal, events)

    def is_valid(e):
        return (
            all(isinstance(e.get(f), str) and e.get(f).strip() for f in ("action", "symbol", "direction"))
            and parse_iso_utc(e.get("timestamp")) is not None
        )

    valid_all = [e for e in (journal + events) if is_valid(e)]

    assert len(feed) == min(len(valid_all), 100)
    # Newest-first (non-increasing) ordering.
    ts = [parse_iso_utc(e["timestamp"]) for e in feed]
    assert ts == sorted(ts, reverse=True)
    # Every emitted entry is valid.
    for e in feed:
        assert is_valid(e)
    # When under the cap, the feed is exactly the valid set.
    if len(valid_all) <= 100:
        assert sorted(id(e) for e in feed) == sorted(id(e) for e in valid_all)


# ==========================================================================
# Property 18
# ==========================================================================
# Feature: trading-dashboard, Property 18: The equity curve has one point per valid-pnl CLOSE, each equity is baseline + running sum in deterministic order, final == baseline + cumulative realized PnL.
@given(closes=gen.close_actions(), baseline=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
def test_property_18_equity_curve_construction(closes, baseline):
    curve = equity_curve(closes, baseline)
    points = curve["points"]

    ordered = order_closes(closes)
    valid = [a for a in ordered if is_number(a.get("pnl"))]

    assert len(points) == len(valid)

    running = 0.0
    for point, action in zip(points, valid):
        running += float(action["pnl"])
        assert point["equity"] == pytest.approx(baseline + running)

    if points:
        assert points[-1]["equity"] == pytest.approx(baseline + cumulative_realized_pnl(closes))


# ==========================================================================
# Property 19
# ==========================================================================
# Feature: trading-dashboard, Property 19: next_scan = last_scan + 60s; before it, seconds_remaining is a non-negative whole-second gap; at/after it the state is due with 0 remaining.
@given(last_scan=gen.aware_datetimes(), offset=st.integers(min_value=-30, max_value=180))
def test_property_19_countdown_computation(last_scan, offset):
    now = last_scan + timedelta(seconds=offset)
    result = compute_countdown(last_scan, now)

    remaining_real = 60 - offset
    if remaining_real <= 0:
        assert result["state"] == "due"
        assert result["seconds_remaining"] == 0
    else:
        assert result["state"] == "scanning"
        assert result["seconds_remaining"] == remaining_real
        assert result["seconds_remaining"] >= 0
    assert result["next_scan_utc"] == last_scan + timedelta(seconds=60)


# ==========================================================================
# Property 20
# ==========================================================================
# Feature: trading-dashboard, Property 20: Bot status is the highest-priority applicable state (initializing > bot_offline > locked > paused_avoid_hours > paused_out_of_session > no_scan_yet > scanning).
@given(
    initializing=st.booleans(),
    bot_offline=st.booleans(),
    is_locked=st.booleans(),
    in_avoid_hours=st.booleans(),
    out_of_session=st.booleans(),
    has_scan_activity=st.booleans(),
)
def test_property_20_status_precedence(initializing, bot_offline, is_locked, in_avoid_hours, out_of_session, has_scan_activity):
    result = resolve_bot_status(
        initializing=initializing,
        bot_offline=bot_offline,
        is_locked=is_locked,
        in_avoid_hours=in_avoid_hours,
        out_of_session=out_of_session,
        has_scan_activity=has_scan_activity,
    )
    if initializing:
        expected = "initializing"
    elif bot_offline:
        expected = "bot_offline"
    elif is_locked:
        expected = "locked"
    elif in_avoid_hours:
        expected = "paused_avoid_hours"
    elif out_of_session:
        expected = "paused_out_of_session"
    elif not has_scan_activity:
        expected = "no_scan_yet"
    else:
        expected = "scanning"
    assert result == expected


# ==========================================================================
# Property 21
# ==========================================================================
# Feature: trading-dashboard, Property 21: Stale indicator shows iff data age > 15s, and the displayed age equals the elapsed seconds.
@given(last_update=gen.aware_datetimes(), age=st.floats(min_value=-5, max_value=120, allow_nan=False, allow_infinity=False))
def test_property_21_stale_threshold(last_update, age):
    now = last_update + timedelta(seconds=age)
    # datetime has microsecond resolution, so sub-microsecond ages round-trip
    # with a tiny loss; compare within one microsecond.
    assert data_age(last_update, now) == pytest.approx(age, abs=1e-6)
    assert is_stale(last_update, now) is bool(age > 15)


# ==========================================================================
# Property 22
# ==========================================================================
# Feature: trading-dashboard, Property 22: The reconnection schedule reconnects at intervals <= 5s, at most 12 consecutive attempts, and retains the last snapshot throughout.
@given(
    interval=st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False),
    max_attempts=st.integers(min_value=1, max_value=12),
    last_snapshot=st.integers(),
)
def test_property_22_reconnection_schedule_bounded(interval, max_attempts, last_snapshot):
    # Pure model of the frontend reconnect controller.
    from dashboard.backend.derivations.reconnect import plan_reconnects

    attempts = plan_reconnects(interval_seconds=interval, max_attempts=max_attempts,
                               last_snapshot=last_snapshot)
    assert len(attempts) <= 12
    assert len(attempts) == max_attempts
    assert all(a["delay"] <= 5.0 for a in attempts)
    # Attempts are numbered consecutively and keep displaying the last snapshot.
    assert [a["attempt"] for a in attempts] == list(range(1, max_attempts + 1))
    assert all(a["displayed_snapshot"] == last_snapshot for a in attempts)


# ==========================================================================
# Property 23
# ==========================================================================
# Feature: trading-dashboard, Property 23: Bot-offline shows iff BOTH file-mod age and scan age exceed 90s; reported elapsed equals age since most recent file modification.
@given(
    file_age=st.one_of(st.none(), st.floats(min_value=0, max_value=300, allow_nan=False, allow_infinity=False)),
    scan_age=st.one_of(st.none(), st.floats(min_value=0, max_value=300, allow_nan=False, allow_infinity=False)),
)
def test_property_23_bot_offline_threshold(file_age, scan_age):
    now = datetime(2024, 6, 10, 12, 0, 0, tzinfo=UTC)
    last_file_mod = None if file_age is None else now - timedelta(seconds=file_age)
    last_scan = None if scan_age is None else now - timedelta(seconds=scan_age)

    offline = is_bot_offline(last_file_mod, last_scan, now)
    file_stale = file_age is None or file_age > 90
    scan_stale = scan_age is None or scan_age > 90
    assert offline is bool(file_stale and scan_stale)

    if last_file_mod is not None:
        assert seconds_since(last_file_mod, now) == pytest.approx(file_age, abs=1e-6)


# ==========================================================================
# Property 24
# ==========================================================================
# Feature: trading-dashboard, Property 24: Offset-less and +00:00 ISO renderings of a UTC instant parse to the same instant; missing/empty/non-ISO strings return unavailable without raising.
@given(dt=gen.aware_datetimes(), junk=st.one_of(st.none(), st.just(""), st.just("   "), st.text(alphabet="xyz/:- ", max_size=8)))
def test_property_24_utc_parse_and_robustness(dt, junk):
    dt = dt.replace(microsecond=0)
    naive_iso = dt.replace(tzinfo=None).isoformat()
    offset_iso = dt.isoformat()  # includes +00:00

    parsed_naive = parse_iso_utc(naive_iso)
    parsed_offset = parse_iso_utc(offset_iso)
    assert parsed_naive == parsed_offset == dt

    # Robustness: junk never raises and yields None (unless it happens to be ISO).
    result = parse_iso_utc(junk)
    if not (isinstance(junk, str) and result is not None):
        assert result is None


# ==========================================================================
# Property 25
# ==========================================================================
# Feature: trading-dashboard, Property 25: The monitored set equals the configured instruments capped at two (excluding unlisted); a per-instrument failure is isolated to that instrument while others still produce results.
@given(
    configured=st.lists(st.sampled_from(["BTCUSD", "XAUUSD", "ETHUSD", "EURUSD"]), max_size=6),
    failing=st.sets(st.sampled_from(["BTCUSD", "XAUUSD", "ETHUSD", "EURUSD"])),
)
def test_property_25_instrument_set_and_isolation(configured, failing):
    monitored = monitored_instruments(configured)

    assert len(monitored) <= 2
    assert all(sym in configured for sym in monitored)
    # equals dedup(configured) capped at two, preserving order
    seen = []
    for sym in configured:
        if sym not in seen:
            seen.append(sym)
    assert monitored == seen[:2]

    def fetcher(symbol):
        if symbol in failing:
            raise RuntimeError("quote failed")
        return {"bid": 1.0, "ask": 2.0}

    results = gather_instrument_data(monitored, fetcher)
    assert set(results.keys()) == set(monitored)
    for sym in monitored:
        if sym in failing:
            assert results[sym]["data_available"] is False
        else:
            assert results[sym]["data_available"] is True
