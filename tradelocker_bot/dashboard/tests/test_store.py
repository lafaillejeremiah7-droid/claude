"""Tests for the snapshot builder + store (Req 3-16)."""
from __future__ import annotations

from datetime import datetime, timezone

from dashboard.backend.readers import FileReader
from dashboard.backend.store import SnapshotStore, build_snapshot, snapshot_hash

UTC = timezone.utc


def _snap(bot_dir, mode="live", api_state=None, env=None):
    r = FileReader(bot_dir=bot_dir, mode=mode)
    return build_snapshot(r, api_state=api_state, env=env or {})


def test_snapshot_assembles_expected_fields(live_bot_dir):
    s = _snap(live_bot_dir)
    # equity falls back to daily_stats when API disabled
    assert s["account"]["equity_source"] == "daily_stats_fallback"
    assert s["account"]["equity"] == 10142.30
    assert s["account"]["balance_available"] is False
    # pnl assembled from journal CLOSE entries
    assert s["pnl"]["cumulative_realized"] == 210.0 - 67.70
    assert s["pnl"]["daily_realized"] == 142.30
    assert s["pnl"]["daily_return_pct_available"] is True
    # win stats + streaks
    assert s["win_stats"]["wins"] == 2
    assert s["win_stats"]["losses"] == 1
    # most recent CLOSE is a loss -> loss streak 1, win streak 0
    assert s["win_stats"]["loss_streak"] == 1
    assert s["win_stats"]["win_streak"] == 0
    # confidence + gate
    assert s["confidence"]["gate"] == 8.0
    assert any(c["source"] == "journal" for c in s["confidence"]["recent"])
    # feed newest-first, non-empty
    assert s["feed_empty"] is False
    # positions
    assert len(s["positions"]) == 1
    assert s["positions"][0]["symbol"] == "BTCUSD"
    # equity curve has 2 points (2 valid closes)
    assert s["equity_curve"]["state"] == "ok"
    assert len(s["equity_curve"]["points"]) == 2
    # reports exposed
    assert s["reports"]["daily"]["pnl_usd"] == 142.30
    assert s["reports"]["weekly"]["improvements"]


def test_equity_prefers_api_when_enabled(live_bot_dir):
    api_state = {"equity": 11111.0, "balance": 12000.0, "free_margin": 9000.0}
    s = _snap(live_bot_dir, api_state=api_state, env={"API_READER_ENABLED": "true"})
    assert s["account"]["equity_source"] == "api"
    assert s["account"]["equity"] == 11111.0
    assert s["account"]["balance"] == 12000.0
    assert s["connection"]["api_status"] == "ok"


def test_api_disabled_status(live_bot_dir):
    s = _snap(live_bot_dir, env={})
    assert s["connection"]["api_status"] == "disabled"


def test_empty_dir_degrades_gracefully(tmp_path):
    s = _snap(tmp_path)
    assert s["account"]["equity_source"] == "none"
    assert s["feed_empty"] is True
    assert s["positions"] == []
    assert s["equity_curve"]["state"] == "insufficient_data"
    assert s["bot_status"]["state"] == "initializing"
    assert s["reports"]["daily"] is None


def test_snapshot_contains_no_secret_fields(live_bot_dir):
    env = {"TL_EMAIL": "a@b.com", "TL_PASSWORD": "hunter2", "TL_SERVER": "AQUA",
           "TL_ENVIRONMENT": "live"}
    r = FileReader(bot_dir=live_bot_dir, mode="live")
    s = build_snapshot(r, secret_values=["a@b.com", "hunter2"], env=env)
    blob = repr(s)
    assert "hunter2" not in blob
    assert "TL_PASSWORD" not in blob
    assert "TL_EMAIL" not in blob


def test_snapshot_hash_excludes_server_time(live_bot_dir):
    r = FileReader(bot_dir=live_bot_dir, mode="live")
    s1 = build_snapshot(r, now=datetime(2024, 6, 10, 12, 0, 0, tzinfo=UTC))
    # A snapshot differing ONLY in server_time_utc hashes identically, so the
    # SSE change-detector doesn't push purely because the wall clock advanced.
    s2 = dict(s1)
    s2["server_time_utc"] = "2099-01-01T00:00:00+00:00"
    assert snapshot_hash(s1) == snapshot_hash(s2)
    # And the hash is stable/deterministic for the same content.
    assert snapshot_hash(s1) == snapshot_hash(dict(s1))


def test_store_refresh_and_get(live_bot_dir):
    store = SnapshotStore(reader=FileReader(bot_dir=live_bot_dir, mode="live"), env={})
    s = store.get()
    assert s["mode"] == "live"
    assert store.content_hash is not None
