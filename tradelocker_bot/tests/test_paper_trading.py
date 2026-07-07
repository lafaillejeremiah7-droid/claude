"""
Tests for the paper-trading engine (Feature 2).

Covers:
- Close math: LONG hitting TP (win) and SHORT hitting SL (loss).
- Breakeven-at-1R moves SL to entry.
- Paper daily stats update (wins/losses/realized_pnl/consecutive_losses).
- Locks using PAPER stats: 2 consecutive losses, max trades/day, drawdown.
- Paper files written to a temp dir with correct schemas; live files untouched.
"""
import json
import os

import pytest

from modules.paper_trading import PaperTradeManager
from modules.risk_management import TradeSetup
import modules.risk_management as rm_mod
import modules.trade_manager as tm_mod


def _setup(direction, entry, sl, tp, qty, risk_amount, symbol="BTCUSD"):
    rr = abs(tp - entry) / abs(entry - sl)
    return TradeSetup(
        symbol=symbol,
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        position_size=qty,
        risk_amount=risk_amount,
        risk_reward_ratio=rr,
        sl_distance=abs(entry - sl),
        tp_distance=abs(tp - entry),
        sl_method="manual",
        valid=True,
    )


@pytest.fixture
def paper(tmp_path):
    logs = tmp_path / "logs"
    journal = tmp_path / "journal"
    return PaperTradeManager(
        client=None,
        starting_equity=10_000.0,
        logs_dir=logs,
        journal_dir=journal,
    )


# ============================================================
# CLOSE MATH
# ============================================================

def test_long_hits_tp_is_win(paper):
    # entry 100, SL 90 (dist 10), TP 120, qty 10 -> risk 100, reward 200 (2R)
    setup = _setup("buy", 100.0, 90.0, 120.0, qty=10.0, risk_amount=100.0)
    pid = paper.open_position(setup)
    assert pid is not None

    # Price gaps past TP -> exit at the TP level.
    closed = paper.manage_open_positions(price_overrides={"BTCUSD": 125.0})
    assert len(closed) == 1
    trade = closed[0]
    assert trade["is_win"] is True
    assert trade["exit_price"] == pytest.approx(120.0)
    assert trade["pnl"] == pytest.approx((120.0 - 100.0) * 10.0)  # 200
    assert trade["r_multiple"] == pytest.approx(2.0)
    # Position removed from active tracking.
    assert pid not in paper.active_positions


def test_short_hits_sl_is_loss(paper):
    # short entry 100, SL 110 (dist 10), TP 80, qty 10 -> risk 100
    setup = _setup("sell", 100.0, 110.0, 80.0, qty=10.0, risk_amount=100.0)
    pid = paper.open_position(setup)
    assert pid is not None

    closed = paper.manage_open_positions(price_overrides={"BTCUSD": 115.0})
    assert len(closed) == 1
    trade = closed[0]
    assert trade["is_win"] is False
    assert trade["exit_price"] == pytest.approx(110.0)
    assert trade["pnl"] == pytest.approx((100.0 - 110.0) * 10.0)  # -100
    assert trade["r_multiple"] == pytest.approx(-1.0)


def test_position_stays_open_between_levels(paper):
    setup = _setup("buy", 100.0, 90.0, 120.0, qty=1.0, risk_amount=10.0)
    pid = paper.open_position(setup)
    closed = paper.manage_open_positions(price_overrides={"BTCUSD": 105.0})
    assert closed == []
    assert pid in paper.active_positions


# ============================================================
# BREAKEVEN AT 1R
# ============================================================

def test_breakeven_moves_sl_to_entry(paper):
    # 1R for long entry 100 SL 90 is price 110.
    setup = _setup("buy", 100.0, 90.0, 130.0, qty=1.0, risk_amount=10.0)
    pid = paper.open_position(setup)

    # Price at 111 (>= 1R but below TP) -> breakeven, no close.
    closed = paper.manage_open_positions(price_overrides={"BTCUSD": 111.0})
    assert closed == []
    pos = paper.active_positions[pid]
    assert pos.is_breakeven is True
    # Breakeven SL = entry + spread offset (0.01% of entry = 100 * 0.0001 = 0.01)
    assert pos.stop_loss == pytest.approx(100.01)

    # A dip back to breakeven SL triggers close at the SL level.
    closed = paper.manage_open_positions(price_overrides={"BTCUSD": 100.0})
    assert len(closed) == 1
    # PnL is (exit_at_SL - entry) * qty = (100.01 - 100) * 1 = 0.01
    assert closed[0]["pnl"] == pytest.approx(0.01, abs=0.02)
    assert closed[0]["was_breakeven"] is True


# ============================================================
# PAPER DAILY STATS
# ============================================================

def test_daily_stats_track_win_then_loss(paper):
    # Win first.
    win = _setup("buy", 100.0, 90.0, 120.0, qty=10.0, risk_amount=100.0)
    paper.open_position(win)
    paper.manage_open_positions(price_overrides={"BTCUSD": 121.0})

    stats = paper.risk_manager.daily_stats
    assert stats.wins == 1
    assert stats.losses == 0
    assert stats.consecutive_losses == 0
    assert stats.realized_pnl == pytest.approx(200.0)
    assert paper.current_equity == pytest.approx(10_200.0)

    # Loss second (short hits SL).
    loss = _setup("sell", 100.0, 110.0, 80.0, qty=10.0, risk_amount=100.0)
    paper.open_position(loss)
    paper.manage_open_positions(price_overrides={"BTCUSD": 111.0})

    assert stats.wins == 1
    assert stats.losses == 1
    assert stats.consecutive_losses == 1
    assert stats.realized_pnl == pytest.approx(100.0)  # 200 - 100
    assert paper.current_equity == pytest.approx(10_100.0)


# ============================================================
# LOCKS USING PAPER STATS
# ============================================================

def test_max_trades_per_day_lock(paper):
    """After MAX_TRADES_PER_DAY (2) opens, further opens are blocked."""
    s1 = _setup("buy", 100.0, 90.0, 120.0, qty=1.0, risk_amount=10.0)
    s2 = _setup("buy", 100.0, 90.0, 120.0, qty=1.0, risk_amount=10.0)
    s3 = _setup("buy", 100.0, 90.0, 120.0, qty=1.0, risk_amount=10.0)

    assert paper.open_position(s1) is not None
    assert paper.open_position(s2) is not None
    # Third should be blocked by the max-trades/day lock on PAPER stats.
    assert paper.open_position(s3) is None
    assert paper.risk_manager.daily_stats.is_locked is True


def test_two_consecutive_losses_lock(paper):
    """Two consecutive losses lock trading (independent of max-trades check)."""
    rmgr = paper.risk_manager
    # Prime today's stats, then simulate two consecutive losses.
    rmgr.can_trade(paper.current_equity)
    rmgr.record_trade_closed(-50.0, is_win=False)
    rmgr.record_trade_closed(-50.0, is_win=False)
    assert rmgr.daily_stats.consecutive_losses == 2

    allowed, reason = rmgr.can_trade(paper.current_equity)
    assert allowed is False
    assert "consecutive" in reason.lower()


def test_drawdown_lock(paper):
    """A >=4% paper drawdown locks trading."""
    rmgr = paper.risk_manager
    rmgr.can_trade(10_000.0)  # sets starting equity for the day
    # Equity down 5% -> exceeds the 4% daily drawdown limit.
    allowed, reason = rmgr.can_trade(9_500.0)
    assert allowed is False
    assert "drawdown" in reason.lower()


# ============================================================
# PAPER FILES / SCHEMAS / LIVE UNTOUCHED
# ============================================================

def test_paper_files_written_with_correct_schema(paper, tmp_path):
    setup = _setup("buy", 100.0, 90.0, 120.0, qty=10.0, risk_amount=100.0)
    pid = paper.open_position(setup)
    paper.manage_open_positions(price_overrides={"BTCUSD": 121.0})

    # Paper positions file exists under the temp logs dir.
    assert paper.positions_file.exists()
    assert paper.positions_file.parent == tmp_path / "logs"
    positions = json.loads(paper.positions_file.read_text())
    assert positions == {}  # closed -> empty

    # Paper stats file exists with daily + weekly schema.
    assert paper.stats_file.exists()
    stats = json.loads(paper.stats_file.read_text())
    assert "daily" in stats and "weekly" in stats
    for key in ("date", "trades_taken", "wins", "losses", "realized_pnl",
                "consecutive_losses", "current_equity", "is_locked"):
        assert key in stats["daily"]

    # Paper journal file exists with OPEN + CLOSE entries and required fields.
    journal_files = list((tmp_path / "journal").glob("paper_journal_*.jsonl"))
    assert len(journal_files) == 1
    entries = [json.loads(line) for line in journal_files[0].read_text().splitlines() if line]
    actions = [e["action"] for e in entries]
    assert "OPEN" in actions and "CLOSE" in actions
    close_entry = next(e for e in entries if e["action"] == "CLOSE")
    for key in ("exit_price", "pnl", "is_win", "r_multiple", "was_breakeven"):
        assert key in close_entry
    # Journal entries mirror the live schema fields.
    for key in ("timestamp", "symbol", "direction", "position_id", "entry_price",
                "stop_loss", "take_profit", "quantity", "risk_amount"):
        assert key in close_entry


def test_live_files_untouched(paper, tmp_path):
    """The paper engine must never write to the live schema file paths."""
    live_positions = tm_mod.POSITIONS_FILE
    live_stats = rm_mod.STATS_FILE

    def snapshot(p):
        return p.stat().st_mtime_ns if p.exists() else None

    before = (snapshot(live_positions), snapshot(live_stats))

    setup = _setup("buy", 100.0, 90.0, 120.0, qty=10.0, risk_amount=100.0)
    paper.open_position(setup)
    paper.manage_open_positions(price_overrides={"BTCUSD": 121.0})

    after = (snapshot(live_positions), snapshot(live_stats))
    assert before == after

    # And the paper files live strictly under the temp dir, not the live paths.
    assert paper.positions_file != live_positions
    assert paper.stats_file != live_stats
    assert str(tmp_path) in str(paper.positions_file)
    assert str(tmp_path) in str(paper.stats_file)
