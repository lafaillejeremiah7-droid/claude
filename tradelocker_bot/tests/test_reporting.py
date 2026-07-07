"""
Tests for the Performance Reporting engine (modules/reporting.py).

Hermetic: every test that touches disk uses pytest's tmp_path. Pure functions
are tested directly with crafted data. A handful of property-based tests
(hypothesis) assert universal invariants across many inputs.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make the bot root importable so `modules.reporting` resolves regardless of
# where pytest is invoked from.
BOT_ROOT = Path(__file__).resolve().parent.parent
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from modules.reporting import (  # noqa: E402
    PerformanceReporter,
    ReportState,
    avg_r_multiple,
    compute_due_reports,
    confidence_band_win_rates,
    count_wins_losses,
    day_key,
    extract_best_worst,
    generate_improvement_suggestions,
    hour_bucket_win_rates,
    month_key,
    pattern_win_rates,
    pnl_abs,
    pct_return,
    week_key,
    win_rate_pct,
)

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st
    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover
    HAS_HYPOTHESIS = False


UTC = timezone.utc


# ============================================================
# P&L / RETURN COMPUTATION
# ============================================================
class TestPnlReturn:
    def test_pnl_abs_profit(self):
        assert pnl_abs(10000, 10142.30) == 142.30

    def test_pnl_abs_loss(self):
        assert pnl_abs(10000, 9800) == -200.0

    def test_return_pct_profit(self):
        assert pct_return(10000, 10142.30) == 1.42

    def test_return_pct_loss(self):
        assert pct_return(10000, 9800) == -2.0

    def test_return_pct_starting_zero_is_zero(self):
        # The critical edge case: no division-by-zero, returns 0.00.
        assert pct_return(0, 500) == 0.00

    def test_return_pct_starting_negative_is_zero(self):
        assert pct_return(-100, 500) == 0.00

    def test_pnl_and_return_flat(self):
        assert pnl_abs(5000, 5000) == 0.0
        assert pct_return(5000, 5000) == 0.0


# ============================================================
# WIN RATE
# ============================================================
class TestWinRate:
    def test_basic(self):
        assert win_rate_pct(1, 1) == 50.0

    def test_all_wins(self):
        assert win_rate_pct(4, 0) == 100.0

    def test_no_trades(self):
        assert win_rate_pct(0, 0) == 0.0


# ============================================================
# BEST / WORST / AVG R
# ============================================================
class TestBestWorst:
    def _entries(self):
        return [
            {"action": "CLOSE", "pnl": 210.0, "is_win": True, "r_multiple": 2.1},
            {"action": "CLOSE", "pnl": -68.0, "is_win": False, "r_multiple": -1.0},
            {"action": "CLOSE", "pnl": 45.0, "is_win": True, "r_multiple": 0.5},
        ]

    def test_best_worst(self):
        best, worst = extract_best_worst(self._entries())
        assert best == 210.0
        assert worst == -68.0

    def test_best_worst_empty(self):
        assert extract_best_worst([]) == (0.0, 0.0)

    def test_avg_r(self):
        # (2.1 - 1.0 + 0.5) / 3 = 0.533...
        assert avg_r_multiple(self._entries()) == pytest.approx(0.533, abs=1e-3)

    def test_avg_r_empty(self):
        assert avg_r_multiple([]) == 0.0

    def test_count_wins_losses(self):
        wins, losses = count_wins_losses(self._entries())
        assert (wins, losses) == (2, 1)

    def test_count_excludes_breakeven(self):
        entries = [
            {"result": "win", "pnl_r": 1.0},
            {"result": "breakeven", "pnl_r": 0.0},
            {"result": "loss", "pnl_r": -1.0},
        ]
        assert count_wins_losses(entries) == (1, 1)


# ============================================================
# HOUR-BUCKET WIN RATES
# ============================================================
class TestHourBuckets:
    def test_basic_bucketing(self):
        trades = [
            {"hour_utc": 8, "result": "loss", "pnl_r": -1.0},
            {"hour_utc": 8, "result": "loss", "pnl_r": -1.0},
            {"hour_utc": 8, "result": "win", "pnl_r": 2.0},
            {"hour_utc": 12, "result": "win", "pnl_r": 2.0},
            {"hour_utc": 12, "result": "win", "pnl_r": 1.5},
        ]
        buckets = hour_bucket_win_rates(trades)
        assert buckets[8]["total"] == 3
        assert buckets[8]["wins"] == 1
        assert buckets[8]["win_rate"] == pytest.approx(1 / 3, abs=1e-4)
        assert buckets[12]["win_rate"] == 1.0

    def test_ignores_undecided_and_missing_hour(self):
        trades = [
            {"hour_utc": 9, "result": "breakeven"},
            {"result": "win"},  # no hour
            {"hour_utc": 9, "result": "win", "pnl_r": 1.0},
        ]
        buckets = hour_bucket_win_rates(trades)
        assert buckets[9]["total"] == 1


# ============================================================
# CONFIDENCE-BAND WIN RATES
# ============================================================
class TestConfidenceBands:
    def test_band_assignment(self):
        trades = [
            {"confidence": 8.1, "result": "loss"},
            {"confidence": 8.4, "result": "loss"},
            {"confidence": 8.6, "result": "win"},
            {"confidence": 9.5, "result": "win"},
            {"confidence": 9.9, "result": "win"},
        ]
        bands = confidence_band_win_rates(trades)
        assert bands["[8.0-8.5)"]["total"] == 2
        assert bands["[8.0-8.5)"]["win_rate"] == 0.0
        assert bands["[8.5-9.0)"]["total"] == 1
        assert bands["[9.0-10.0)"]["total"] == 2
        assert bands["[9.0-10.0)"]["win_rate"] == 1.0

    def test_records_without_confidence_ignored(self):
        trades = [{"result": "win"}, {"result": "loss"}]
        bands = confidence_band_win_rates(trades)
        assert all(b["total"] == 0 for b in bands.values())

    def test_upper_bound_ten_included(self):
        trades = [{"confidence": 10.0, "result": "win"}]
        bands = confidence_band_win_rates(trades)
        assert bands["[9.0-10.0)"]["total"] == 1


# ============================================================
# PATTERN WIN RATES
# ============================================================
class TestPatternWinRates:
    def test_basic(self):
        trades = [
            {"candle_pattern": "doji", "result": "loss", "pnl_r": -0.5},
            {"candle_pattern": "doji", "result": "loss", "pnl_r": -0.3},
            {"candle_pattern": "hammer", "result": "win", "pnl_r": 2.0},
        ]
        patterns = pattern_win_rates(trades)
        assert patterns["doji"]["total"] == 2
        assert patterns["doji"]["win_rate"] == 0.0
        assert patterns["doji"]["avg_pnl_r"] == pytest.approx(-0.4, abs=1e-6)
        assert patterns["hammer"]["win_rate"] == 1.0


# ============================================================
# IMPROVEMENT SUGGESTION GENERATOR
# ============================================================
class TestImprovementSuggestions:
    def _crafted_dataset(self):
        """A clearly-bad hour (08:00) and a clearly-bad pattern ('doji')."""
        trades = []
        # Hour 8: 12 trades, ~22% win rate (bad hour).
        for i in range(12):
            trades.append({
                "hour_utc": 8,
                "candle_pattern": "hammer",
                "result": "win" if i < 3 else "loss",
                "pnl_r": 1.5 if i < 3 else -1.0,
                "confidence": 8.7,
            })
        # 'doji' pattern: 9 trades at a good hour but negative avg R (bad pattern).
        for i in range(9):
            trades.append({
                "hour_utc": 13,
                "candle_pattern": "doji",
                "result": "win" if i < 3 else "loss",
                "pnl_r": 0.5 if i < 3 else -0.8,
                "confidence": 8.7,
            })
        return trades

    def test_flags_bad_hour(self):
        suggestions = generate_improvement_suggestions(
            self._crafted_dataset(), adaptive_params={"avoid_hours": []}
        )
        joined = " ".join(suggestions)
        assert "08:00-09:00 UTC" in joined
        assert "avoid_hours" in joined

    def test_flags_bad_pattern(self):
        suggestions = generate_improvement_suggestions(
            self._crafted_dataset(), adaptive_params={"avoid_hours": []}
        )
        joined = " ".join(suggestions)
        assert "'doji'" in joined
        assert "down-weight" in joined

    def test_respects_existing_avoid_hours(self):
        # Hour 8 already avoided -> should not be re-suggested.
        suggestions = generate_improvement_suggestions(
            self._crafted_dataset(), adaptive_params={"avoid_hours": [8]}
        )
        joined = " ".join(suggestions)
        assert "08:00-09:00 UTC" not in joined

    def test_confidence_band_suggestion(self):
        trades = []
        # Low band [8.0-8.5): 10 trades, 20% win.
        for i in range(10):
            trades.append({"confidence": 8.2, "hour_utc": 13,
                           "candle_pattern": "hammer",
                           "result": "win" if i < 2 else "loss",
                           "pnl_r": 1.0 if i < 2 else -1.0})
        # High band [9.0-10.0): 10 trades, 80% win.
        for i in range(10):
            trades.append({"confidence": 9.5, "hour_utc": 13,
                           "candle_pattern": "hammer",
                           "result": "win" if i < 8 else "loss",
                           "pnl_r": 1.0 if i < 8 else -1.0})
        suggestions = generate_improvement_suggestions(
            trades, adaptive_params={"avoid_hours": []}
        )
        joined = " ".join(suggestions)
        assert "min_confidence" in joined

    def test_lock_incident_surfaced(self):
        suggestions = generate_improvement_suggestions(
            [], adaptive_params={"avoid_hours": []},
            lock_incidents=[{"date": "2024-06-11", "lock_reason": "Two consecutive losses"}],
        )
        joined = " ".join(suggestions)
        assert "Two consecutive losses" in joined

    def test_no_weakness_message(self):
        # Small, all-winning sample below min_sample -> no negatives flagged.
        trades = [{"hour_utc": 13, "candle_pattern": "hammer", "result": "win", "pnl_r": 1.0}]
        suggestions = generate_improvement_suggestions(
            trades, adaptive_params={"avoid_hours": []}
        )
        assert any("No statistically significant" in s for s in suggestions)

    def test_adaptive_trend_regression(self):
        suggestions = generate_improvement_suggestions(
            [], adaptive_params={"avoid_hours": [], "current_win_rate": 55.0},
            prior_week_metrics={"current_win_rate": 65.0},
        )
        joined = " ".join(suggestions)
        assert "slipped" in joined


# ============================================================
# ROLLOVER DETECTION
# ============================================================
class TestRollover:
    def test_first_run_seeds_no_emit(self):
        now = datetime(2024, 6, 10, 12, 0, tzinfo=UTC)
        due, new_state = compute_due_reports(ReportState(), now)
        assert due == {"daily": None, "weekly": None, "monthly": None}
        assert new_state.last_daily == "2024-06-10"
        assert new_state.last_weekly == week_key(now)
        assert new_state.last_monthly == "2024-06"

    def test_day_only_rollover(self):
        # Tue 2024-06-11 -> Wed 2024-06-12 (same ISO week, same month).
        state = ReportState(last_daily="2024-06-11", last_weekly="2024-W24", last_monthly="2024-06")
        now = datetime(2024, 6, 12, 0, 5, tzinfo=UTC)
        assert week_key(now) == "2024-W24"  # sanity: still same week
        due, new_state = compute_due_reports(state, now)
        assert due["daily"] == "2024-06-11"
        assert due["weekly"] is None
        assert due["monthly"] is None
        assert new_state.last_daily == "2024-06-12"

    def test_day_and_week_rollover(self):
        # Sunday 2024-06-16 (W24) -> Monday 2024-06-17 (W25).
        state = ReportState(last_daily="2024-06-16", last_weekly="2024-W24", last_monthly="2024-06")
        now = datetime(2024, 6, 17, 0, 1, tzinfo=UTC)
        assert week_key(now) == "2024-W25"
        due, new_state = compute_due_reports(state, now)
        assert due["daily"] == "2024-06-16"
        assert due["weekly"] == "2024-W24"
        assert due["monthly"] is None

    def test_day_week_month_rollover(self):
        # 2024-06-30 (Sun, W26) -> 2024-07-01 (Mon, W27, new month).
        state = ReportState(last_daily="2024-06-30", last_weekly="2024-W26", last_monthly="2024-06")
        now = datetime(2024, 7, 1, 0, 1, tzinfo=UTC)
        due, new_state = compute_due_reports(state, now)
        assert due["daily"] == "2024-06-30"
        assert due["weekly"] == "2024-W26"
        assert due["monthly"] == "2024-06"
        assert new_state.last_daily == "2024-07-01"
        assert new_state.last_weekly == "2024-W27"
        assert new_state.last_monthly == "2024-07"

    def test_no_double_emit_same_period(self):
        state = ReportState(last_daily="2024-06-12", last_weekly="2024-W24", last_monthly="2024-06")
        now = datetime(2024, 6, 12, 18, 0, tzinfo=UTC)
        due, new_state = compute_due_reports(state, now)
        assert due == {"daily": None, "weekly": None, "monthly": None}

    def test_offline_across_boundary_still_emits(self):
        # Bot offline for days; on restart the last-active day differs -> emit.
        state = ReportState(last_daily="2024-06-01", last_weekly="2024-W22", last_monthly="2024-06")
        now = datetime(2024, 6, 10, 9, 0, tzinfo=UTC)
        due, _ = compute_due_reports(state, now)
        assert due["daily"] == "2024-06-01"


# ============================================================
# INTEGRATION: PerformanceReporter end-to-end on tmp dirs
# ============================================================
class TestReporterIntegration:
    def _setup_bot_dir(self, tmp_path: Path, day: str, week_start: str):
        logs = tmp_path / "logs"
        journal = tmp_path / "journal"
        logs.mkdir()
        journal.mkdir()
        # daily_stats.json
        stats = {
            "daily": {
                "date": day,
                "trades_taken": 2,
                "wins": 1,
                "losses": 1,
                "consecutive_losses": 0,
                "realized_pnl": 142.30,
                "starting_equity": 10000.0,
                "current_equity": 10142.30,
                "max_drawdown_pct": 0.5,
                "is_locked": False,
                "lock_reason": "",
            },
            "weekly": {
                "week_start": week_start,
                "starting_equity": 10000.0,
                "current_equity": 10142.30,
                "total_trades": 2,
            },
        }
        (logs / "daily_stats.json").write_text(json.dumps(stats))
        # journal for the day with two CLOSE entries
        journal_lines = [
            {"action": "OPEN", "symbol": "BTCUSD"},
            {"action": "CLOSE", "symbol": "BTCUSD", "pnl": 210.0, "is_win": True, "r_multiple": 2.1},
            {"action": "CLOSE", "symbol": "XAUUSD", "pnl": -67.7, "is_win": False, "r_multiple": -1.0},
        ]
        with open(journal / f"journal_{day}.jsonl", "w") as f:
            for e in journal_lines:
                f.write(json.dumps(e) + "\n")
        return logs, journal

    def test_daily_report_written(self, tmp_path):
        day = "2024-06-10"
        self._setup_bot_dir(tmp_path, day, "2024-06-10")
        reporter = PerformanceReporter(base_dir=tmp_path)
        reporter.emit_daily(day)

        report_file = tmp_path / "logs" / "reports" / f"daily_{day}.json"
        assert report_file.exists()
        payload = json.loads(report_file.read_text())
        assert payload["pnl_usd"] == 142.30
        assert payload["return_pct"] == 1.42
        assert payload["wins"] == 1
        assert payload["losses"] == 1
        assert payload["best_trade_usd"] == 210.0
        assert payload["worst_trade_usd"] == -67.7
        # history appended
        hist = (tmp_path / "logs" / "reports" / "history.jsonl").read_text().strip()
        assert '"type": "daily"' in hist

    def test_maybe_emit_first_run_no_reports(self, tmp_path):
        self._setup_bot_dir(tmp_path, "2024-06-10", "2024-06-10")
        reporter = PerformanceReporter(base_dir=tmp_path)
        emitted = reporter.maybe_emit(datetime(2024, 6, 10, 12, 0, tzinfo=UTC))
        assert emitted == []
        # state file created
        assert (tmp_path / "logs" / "reports" / ".report_state.json").exists()

    def test_maybe_emit_daily_on_rollover(self, tmp_path):
        self._setup_bot_dir(tmp_path, "2024-06-10", "2024-06-10")
        reporter = PerformanceReporter(base_dir=tmp_path)
        # First scan on the 10th -> seed only.
        reporter.maybe_emit(datetime(2024, 6, 10, 23, 0, tzinfo=UTC))
        # Next scan on the 11th -> daily report for the 10th fires.
        emitted = reporter.maybe_emit(datetime(2024, 6, 11, 0, 2, tzinfo=UTC))
        assert "daily" in emitted
        assert (tmp_path / "logs" / "reports" / "daily_2024-06-10.json").exists()
        # A second scan same day does not re-emit.
        emitted2 = reporter.maybe_emit(datetime(2024, 6, 11, 6, 0, tzinfo=UTC))
        assert emitted2 == []

    def test_paper_mode_absent_files_no_crash(self, tmp_path):
        # No stat files at all -> reporter must degrade gracefully.
        (tmp_path / "logs").mkdir()
        reporter = PerformanceReporter(base_dir=tmp_path, mode="paper")
        payload = reporter.build_daily_payload("2024-06-10")
        assert payload["pnl_usd"] == 0.0
        assert payload["return_pct"] == 0.0
        assert payload["trades"] == 0

    def test_weekly_report_includes_improvements(self, tmp_path):
        day = "2024-06-10"
        logs, journal = self._setup_bot_dir(tmp_path, day, "2024-06-10")
        reporter = PerformanceReporter(base_dir=tmp_path, min_sample=3)
        wk = week_key(datetime(2024, 6, 10, tzinfo=UTC))
        reporter.emit_weekly(wk)
        wf = tmp_path / "logs" / "reports" / f"weekly_{wk}.json"
        assert wf.exists()
        payload = json.loads(wf.read_text())
        assert "improvements" in payload
        assert isinstance(payload["improvements"], list)

    def test_monthly_aggregates_history(self, tmp_path):
        reporter = PerformanceReporter(base_dir=tmp_path)
        # Seed history.jsonl with two daily entries in the same month.
        reports = tmp_path / "logs" / "reports"
        reports.mkdir(parents=True)
        entries = [
            {"type": "daily", "date": "2024-06-10", "pnl_usd": 142.30, "trades": 2,
             "wins": 1, "losses": 1, "starting_equity": 10000.0, "current_equity": 10142.30,
             "max_drawdown_pct": 0.5},
            {"type": "daily", "date": "2024-06-11", "pnl_usd": -68.0, "trades": 1,
             "wins": 0, "losses": 1, "starting_equity": 10142.30, "current_equity": 10074.30,
             "max_drawdown_pct": 1.0},
        ]
        with open(reports / "history.jsonl", "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        payload = reporter.build_monthly_payload("2024-06")
        assert payload["pnl_usd"] == pytest.approx(74.30, abs=1e-6)
        assert payload["total_trades"] == 3
        assert payload["best_day"]["date"] == "2024-06-10"
        assert payload["worst_day"]["date"] == "2024-06-11"
        # Equity change: 10000 -> 10074.30
        assert payload["return_pct"] == pytest.approx(0.74, abs=1e-2)


# ============================================================
# PROPERTY-BASED TESTS (hypothesis)
# ============================================================
@pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")
class TestProperties:
    @settings(max_examples=200)
    @given(
        starting=st.floats(min_value=1.0, max_value=1e7, allow_nan=False, allow_infinity=False),
        current=st.floats(min_value=0.0, max_value=1e7, allow_nan=False, allow_infinity=False),
    )
    def test_pct_return_sign_matches_pnl(self, starting, current):
        # Return % has the same sign as absolute P&L when starting > 0.
        pnl = pnl_abs(starting, current)
        ret = pct_return(starting, current)
        if pnl > 0:
            assert ret >= 0
        elif pnl < 0:
            assert ret <= 0

    @settings(max_examples=100)
    @given(st.floats(min_value=-1e7, max_value=1e7, allow_nan=False, allow_infinity=False))
    def test_pct_return_zero_start_always_zero(self, current):
        assert pct_return(0, current) == 0.00

    @settings(max_examples=100)
    @given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
    def test_win_rate_bounded(self, wins, losses):
        wr = win_rate_pct(wins, losses)
        assert 0.0 <= wr <= 100.0

    @settings(max_examples=100)
    @given(
        st.lists(
            st.fixed_dictionaries({
                "action": st.just("CLOSE"),
                "pnl": st.floats(min_value=-5000, max_value=5000, allow_nan=False, allow_infinity=False),
            }),
            min_size=1, max_size=50,
        )
    )
    def test_best_worst_ordering(self, entries):
        best, worst = extract_best_worst(entries)
        assert best >= worst

    @settings(max_examples=100)
    @given(
        st.lists(
            st.fixed_dictionaries({
                "hour_utc": st.integers(min_value=0, max_value=23),
                "result": st.sampled_from(["win", "loss"]),
                "pnl_r": st.floats(min_value=-3, max_value=3, allow_nan=False),
            }),
            min_size=0, max_size=80,
        )
    )
    def test_hour_buckets_consistent(self, trades):
        buckets = hour_bucket_win_rates(trades)
        for b in buckets.values():
            assert b["wins"] + b["losses"] == b["total"]
            assert 0.0 <= b["win_rate"] <= 1.0

    @settings(max_examples=50)
    @given(
        offset_days=st.integers(min_value=0, max_value=400),
    )
    def test_rollover_no_double_emit_when_state_current(self, offset_days):
        # If state already equals the current period keys, nothing is due.
        base = datetime(2024, 1, 1, tzinfo=UTC)
        from datetime import timedelta
        now = base + timedelta(days=offset_days)
        state = ReportState(
            last_daily=day_key(now),
            last_weekly=week_key(now),
            last_monthly=month_key(now),
        )
        due, _ = compute_due_reports(state, now)
        assert due == {"daily": None, "weekly": None, "monthly": None}
