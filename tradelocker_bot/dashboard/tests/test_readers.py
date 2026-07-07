"""Tests for the MODE-aware file readers (Req 8, 9, 13, 16)."""
from __future__ import annotations

from datetime import datetime, timezone

from dashboard.backend.readers import (
    FileReader,
    api_reader_enabled,
    parse_log_text,
    resolve_bot_dir,
    resolve_mode,
)

UTC = timezone.utc


# ---- mode / env resolution ---------------------------------------------
def test_resolve_mode_defaults_live():
    assert resolve_mode({}) == "live"
    assert resolve_mode({"DASHBOARD_MODE": "paper"}) == "paper"
    assert resolve_mode({"DASHBOARD_MODE": "PAPER"}) == "paper"
    assert resolve_mode({"DASHBOARD_MODE": "weird"}) == "live"


def test_resolve_bot_dir_env(tmp_path):
    assert resolve_bot_dir({"BOT_DIR": str(tmp_path)}) == tmp_path


def test_api_reader_enabled_flag():
    assert api_reader_enabled({}) is False
    assert api_reader_enabled({"API_READER_ENABLED": "true"}) is True
    assert api_reader_enabled({"API_READER_ENABLED": "0"}) is False


# ---- live vs paper file selection --------------------------------------
def test_live_mode_reads_live_files(live_bot_dir):
    r = FileReader(bot_dir=live_bot_dir, mode="live")
    stats = r.read_daily_stats()
    assert stats["daily"]["current_equity"] == 10142.30
    assert "12345" in r.read_positions()


def test_paper_mode_reads_paper_files(paper_bot_dir):
    r = FileReader(bot_dir=paper_bot_dir, mode="paper")
    stats = r.read_daily_stats()
    assert stats is not None
    assert stats["daily"]["current_equity"] == 10142.30
    # A live reader over the paper-only fixture finds nothing.
    live = FileReader(bot_dir=paper_bot_dir, mode="live")
    assert live.read_daily_stats() is None
    assert live.read_positions() == {}


# ---- tolerance to missing / malformed ----------------------------------
def test_missing_files_yield_empty(tmp_path):
    r = FileReader(bot_dir=tmp_path, mode="live")
    assert r.read_daily_stats() is None
    assert r.read_positions() == {}
    assert r.read_adaptive_config() is None
    assert r.read_trade_features() == []
    assert r.read_journal_entries([]) == []
    assert r.read_reports()["daily"] is None


def test_malformed_lines_skipped(live_bot_dir):
    r = FileReader(bot_dir=live_bot_dir, mode="live")
    entries = r.read_journal_entries()
    # 3 valid journal records (1 OPEN + 2 CLOSE); the garbage line is skipped.
    assert len(entries) == 3
    closes = r.read_close_actions()
    assert len(closes) == 2
    # source keys annotated for deterministic ordering
    assert all("file_date" in e and "line_index" in e for e in entries)


def test_reports_read_from_logs_reports(live_bot_dir):
    r = FileReader(bot_dir=live_bot_dir, mode="live")
    reports = r.read_reports()
    assert reports["daily"]["pnl_usd"] == 142.30
    assert reports["weekly"]["improvements"]
    assert reports["monthly"]["month"] == "2024-06"
    assert len(reports["history"]) == 1


# ---- log parsing (pure) -------------------------------------------------
def test_parse_log_text_extracts_scan_and_confidence(live_bot_dir):
    r = FileReader(bot_dir=live_bot_dir, mode="live")
    log = r.read_bot_log_events()
    assert isinstance(log["last_scan_utc"], datetime)
    actions = {e["action"] for e in log["events"]}
    assert "NEAR_MISS" in actions
    assert "APPROVED" in actions
    # best-effort last-known prices parsed from the "Entry: $..." lines
    assert "BTCUSD" in log["prices"]
    assert log["prices"]["BTCUSD"]["price"] == 67100.0


def test_parse_log_text_empty():
    assert parse_log_text("")["last_scan_utc"] is None
    assert parse_log_text(None)["events"] == []
