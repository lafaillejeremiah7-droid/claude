"""
Performance Reporting Engine

Produces human-readable summaries AND machine-readable report files so the
dashboard can render them. It is deliberately READ-mostly of the bot's own
state files (daily_stats.json / weekly stats, adaptive_config.json,
trade_features.jsonl, journal/*.jsonl) and only ever WRITES inside a dedicated
``logs/reports/`` directory. It never mutates the bot's live state.

Reports:
- Daily   : emitted on UTC day rollover for the day that just ended.
- Weekly  : emitted on UTC (ISO) week rollover.
- Monthly : emitted on UTC month rollover.

Rollover is detected via a small state file (``.report_state.json``) that
records the last-reported day / week / month keys. ``maybe_emit(now_utc)`` is
called once per scan cycle and fires whatever reports are due -- including
after the bot has been offline across a boundary (it emits on next start).

All times are UTC.

Design goals:
- Pure, side-effect-free computation lives in module-level functions so it is
  trivially unit-testable (no disk, no clock).
- Mode-aware: a ``stats_file`` (and companion paths) can be supplied so the
  same engine reports on LIVE stats or PAPER stats. Missing paper files never
  hard-fail -- the reporter degrades gracefully.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# PERIOD KEY HELPERS (all UTC)
# ============================================================
def day_key(dt: datetime) -> str:
    """Calendar day key, e.g. '2024-06-10'."""
    return dt.strftime("%Y-%m-%d")


def week_key(dt: datetime) -> str:
    """ISO week key, e.g. '2024-W24'. Robust across year boundaries."""
    iso = dt.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def month_key(dt: datetime) -> str:
    """Calendar month key, e.g. '2024-06'."""
    return dt.strftime("%Y-%m")


# ============================================================
# PURE COMPUTATION HELPERS (no I/O, no clock)
# ============================================================
def pnl_abs(starting_equity: float, current_equity: float) -> float:
    """Absolute profit/loss in account currency."""
    return round(float(current_equity) - float(starting_equity), 2)


def pct_return(starting_equity: float, current_equity: float) -> float:
    """
    Percentage return. Returns 0.00 when starting_equity == 0 (or <= 0) to
    avoid division-by-zero blowups on a freshly-initialised day.
    """
    if not starting_equity or starting_equity <= 0:
        return 0.00
    return round((float(current_equity) - float(starting_equity)) / float(starting_equity) * 100, 2)


def win_rate_pct(wins: int, losses: int) -> float:
    """Win rate as a percentage of decided (win/loss) trades. 0.0 if none."""
    total = int(wins) + int(losses)
    if total <= 0:
        return 0.0
    return round(int(wins) / total * 100, 2)


def _is_win(record: Dict[str, Any]) -> Optional[bool]:
    """
    Normalise a trade record's outcome to True/False, or None if undecided.
    Accepts journal CLOSE entries ('is_win') and trade_features ('result').
    """
    if "result" in record and record["result"]:
        res = str(record["result"]).lower()
        if res == "win":
            return True
        if res == "loss":
            return False
        return None  # 'breakeven' / 'pending' -> undecided
    if "is_win" in record and record["is_win"] is not None:
        return bool(record["is_win"])
    return None


def _pnl_of(record: Dict[str, Any]) -> float:
    """Extract a trade's dollar P&L from a journal-style record."""
    return float(record.get("pnl", 0.0) or 0.0)


def _r_of(record: Dict[str, Any]) -> float:
    """Extract a trade's R multiple ('r_multiple' journal, 'pnl_r' features)."""
    if "r_multiple" in record and record["r_multiple"] is not None:
        return float(record["r_multiple"])
    return float(record.get("pnl_r", 0.0) or 0.0)


def extract_best_worst(close_entries: List[Dict[str, Any]]) -> Tuple[float, float]:
    """
    Best (max) and worst (min) trade P&L in $ from a list of CLOSE entries.
    Empty input -> (0.0, 0.0).
    """
    pnls = [_pnl_of(e) for e in close_entries]
    if not pnls:
        return 0.0, 0.0
    return round(max(pnls), 2), round(min(pnls), 2)


def avg_r_multiple(close_entries: List[Dict[str, Any]]) -> float:
    """Average R multiple across CLOSE entries. Empty -> 0.0."""
    rs = [_r_of(e) for e in close_entries]
    if not rs:
        return 0.0
    return round(sum(rs) / len(rs), 3)


def count_wins_losses(close_entries: List[Dict[str, Any]]) -> Tuple[int, int]:
    """Count wins and losses (undecided/breakeven excluded) in CLOSE entries."""
    wins = losses = 0
    for e in close_entries:
        w = _is_win(e)
        if w is True:
            wins += 1
        elif w is False:
            losses += 1
    return wins, losses


def hour_bucket_win_rates(trades: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Group trades by UTC hour and compute win-rate stats per hour.

    Each trade needs an 'hour_utc' key plus a decidable outcome
    ('result' or 'is_win'). Returns:
        { hour: {"wins", "losses", "total", "win_rate", "avg_pnl_r"} }
    """
    buckets: Dict[int, Dict[str, Any]] = {}
    for t in trades:
        if "hour_utc" not in t or t["hour_utc"] is None:
            continue
        outcome = _is_win(t)
        if outcome is None:
            continue
        hour = int(t["hour_utc"])
        b = buckets.setdefault(
            hour, {"wins": 0, "losses": 0, "total": 0, "win_rate": 0.0, "_r_sum": 0.0}
        )
        b["total"] += 1
        b["_r_sum"] += _r_of(t)
        if outcome:
            b["wins"] += 1
        else:
            b["losses"] += 1
    for b in buckets.values():
        b["win_rate"] = round(b["wins"] / b["total"], 4) if b["total"] else 0.0
        b["avg_pnl_r"] = round(b["_r_sum"] / b["total"], 3) if b["total"] else 0.0
        del b["_r_sum"]
    return buckets


def pattern_win_rates(trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Group trades by candle_pattern and compute win-rate + avg R per pattern.
    Returns { pattern: {"wins","losses","total","win_rate","avg_pnl_r"} }.
    """
    buckets: Dict[str, Dict[str, Any]] = {}
    for t in trades:
        pattern = t.get("candle_pattern") or ""
        if not pattern:
            continue
        outcome = _is_win(t)
        if outcome is None:
            continue
        b = buckets.setdefault(
            pattern, {"wins": 0, "losses": 0, "total": 0, "win_rate": 0.0, "_r_sum": 0.0}
        )
        b["total"] += 1
        b["_r_sum"] += _r_of(t)
        if outcome:
            b["wins"] += 1
        else:
            b["losses"] += 1
    for b in buckets.values():
        b["win_rate"] = round(b["wins"] / b["total"], 4) if b["total"] else 0.0
        b["avg_pnl_r"] = round(b["_r_sum"] / b["total"], 3) if b["total"] else 0.0
        del b["_r_sum"]
    return buckets


# Default confidence bands used for confidence-band analysis.
DEFAULT_CONFIDENCE_BANDS: List[Tuple[float, float]] = [(8.0, 8.5), (8.5, 9.0), (9.0, 10.01)]


def _band_label(lo: float, hi: float) -> str:
    # Present the upper bound as a human-friendly closed-ish value.
    hi_disp = 10.0 if hi > 10 else hi
    return f"[{lo:.1f}-{hi_disp:.1f})"


def confidence_band_win_rates(
    records: List[Dict[str, Any]],
    bands: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute win rate per confidence band. Each record needs a 'confidence'
    value (0-10) and a decidable outcome. Records without a confidence value
    are ignored (older feature records may lack it). Bands are half-open
    [lo, hi). Returns { band_label: {"wins","losses","total","win_rate"} }.
    """
    if bands is None:
        bands = DEFAULT_CONFIDENCE_BANDS
    out: Dict[str, Dict[str, Any]] = {
        _band_label(lo, hi): {"wins": 0, "losses": 0, "total": 0, "win_rate": 0.0}
        for lo, hi in bands
    }
    for r in records:
        conf = r.get("confidence")
        if conf is None:
            continue
        outcome = _is_win(r)
        if outcome is None:
            continue
        conf = float(conf)
        for lo, hi in bands:
            if lo <= conf < hi:
                b = out[_band_label(lo, hi)]
                b["total"] += 1
                if outcome:
                    b["wins"] += 1
                else:
                    b["losses"] += 1
                break
    for b in out.values():
        b["win_rate"] = round(b["wins"] / b["total"], 4) if b["total"] else 0.0
    return out


def generate_improvement_suggestions(
    trades: List[Dict[str, Any]],
    adaptive_params: Optional[Dict[str, Any]] = None,
    prior_week_metrics: Optional[Dict[str, Any]] = None,
    lock_incidents: Optional[List[Dict[str, Any]]] = None,
    min_sample: int = 5,
    weak_win_rate: float = 0.40,
) -> List[str]:
    """
    Derive concrete, human-readable self-adaptation suggestions from real data.

    Sources considered:
    - Worst-performing UTC hours -> suggest adding to avoid_hours.
    - Lowest-performing candle pattern (low win rate / negative avg R) ->
      suggest down-weighting.
    - Confidence-band analysis -> if low-confidence trades underperform,
      suggest raising min_confidence.
    - Drawdown / consecutive-loss incidents that tripped locks.
    - Adaptive engine win-rate / avg-R trend vs the prior week.

    Nothing is hard-coded: every bullet is computed from the supplied data and
    only surfaces when the sample size is meaningful (>= ``min_sample``).
    """
    adaptive_params = adaptive_params or {}
    suggestions: List[str] = []
    avoid_hours = set(adaptive_params.get("avoid_hours", []) or [])

    # --- Worst-performing hours ------------------------------------------
    hours = hour_bucket_win_rates(trades)
    weak_hours = [
        (h, s) for h, s in hours.items()
        if s["total"] >= min_sample and s["win_rate"] < weak_win_rate and h not in avoid_hours
    ]
    # Worst first.
    weak_hours.sort(key=lambda x: x[1]["win_rate"])
    for h, s in weak_hours:
        suggestions.append(
            f"Win rate in {h:02d}:00-{(h + 1) % 24:02d}:00 UTC is "
            f"{s['win_rate'] * 100:.0f}% ({s['total']} trades) - consider avoid_hours."
        )

    # --- Worst-performing pattern ----------------------------------------
    patterns = pattern_win_rates(trades)
    weak_patterns = [
        (p, s) for p, s in patterns.items()
        if s["total"] >= min_sample and (s["win_rate"] < weak_win_rate or s["avg_pnl_r"] < 0)
    ]
    weak_patterns.sort(key=lambda x: (x[1]["avg_pnl_r"], x[1]["win_rate"]))
    for p, s in weak_patterns:
        suggestions.append(
            f"'{p}' pattern avg R {s['avg_pnl_r']:+.1f} over {s['total']} trades "
            f"(win rate {s['win_rate'] * 100:.0f}%) - down-weight."
        )

    # --- Confidence-band analysis ----------------------------------------
    bands = confidence_band_win_rates(trades)
    low_band = bands.get(_band_label(8.0, 8.5))
    high_band = bands.get(_band_label(9.0, 10.01))
    if (
        low_band and high_band
        and low_band["total"] >= min_sample and high_band["total"] >= min_sample
        and low_band["win_rate"] + 0.15 < high_band["win_rate"]
    ):
        suggestions.append(
            f"Low-confidence trades [8.0-8.5) win {low_band['win_rate'] * 100:.0f}% "
            f"vs {high_band['win_rate'] * 100:.0f}% for [9.0-10.0) "
            f"({low_band['total']} vs {high_band['total']} trades) - consider raising min_confidence."
        )

    # --- Lock incidents (drawdown / consecutive losses) ------------------
    for inc in (lock_incidents or []):
        reason = inc.get("lock_reason") or ""
        date = inc.get("date") or ""
        if reason:
            suggestions.append(f"Trading locked on {date}: {reason} - review risk exposure.")

    # --- Adaptive trend vs prior week ------------------------------------
    if prior_week_metrics and adaptive_params:
        cur_wr = adaptive_params.get("current_win_rate")
        prev_wr = prior_week_metrics.get("current_win_rate")
        if cur_wr is not None and prev_wr is not None:
            delta = cur_wr - prev_wr
            if delta < 0:
                suggestions.append(
                    f"Adaptive win rate slipped {prev_wr:.1f}% -> {cur_wr:.1f}% "
                    f"vs last week - monitor recent parameter shifts."
                )
            else:
                suggestions.append(
                    f"Adaptive win rate improved {prev_wr:.1f}% -> {cur_wr:.1f}% vs last week."
                )

    if not suggestions:
        suggestions.append("No statistically significant weaknesses detected this period.")
    return suggestions


# ============================================================
# ROLLOVER DETECTION (pure)
# ============================================================
@dataclass
class ReportState:
    """Last-reported period keys, persisted to .report_state.json."""
    last_daily: Optional[str] = None
    last_weekly: Optional[str] = None
    last_monthly: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_daily": self.last_daily,
            "last_weekly": self.last_weekly,
            "last_monthly": self.last_monthly,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReportState":
        return cls(
            last_daily=data.get("last_daily"),
            last_weekly=data.get("last_weekly"),
            last_monthly=data.get("last_monthly"),
        )


def compute_due_reports(
    state: ReportState, now: datetime
) -> Tuple[Dict[str, Optional[str]], ReportState]:
    """
    Given the last-reported state and the current UTC time, decide which
    reports are due and return (due, new_state).

    ``due`` maps 'daily'/'weekly'/'monthly' to the period-key that should be
    reported (the period that JUST ended), or None if nothing is due.

    First-ever run (all state keys None) initialises the state to the current
    period keys and emits NOTHING (the current periods are still in progress).

    Updating the state keys immediately guarantees no double-emit, while still
    firing correctly when the bot was offline across one or more boundaries.
    """
    cur_day = day_key(now)
    cur_week = week_key(now)
    cur_month = month_key(now)

    due: Dict[str, Optional[str]] = {"daily": None, "weekly": None, "monthly": None}
    new_state = ReportState(
        last_daily=state.last_daily,
        last_weekly=state.last_weekly,
        last_monthly=state.last_monthly,
    )

    # First run for any dimension: seed without emitting.
    if state.last_daily is None:
        new_state.last_daily = cur_day
    elif cur_day != state.last_daily:
        due["daily"] = state.last_daily
        new_state.last_daily = cur_day

    if state.last_weekly is None:
        new_state.last_weekly = cur_week
    elif cur_week != state.last_weekly:
        due["weekly"] = state.last_weekly
        new_state.last_weekly = cur_week

    if state.last_monthly is None:
        new_state.last_monthly = cur_month
    elif cur_month != state.last_monthly:
        due["monthly"] = state.last_monthly
        new_state.last_monthly = cur_month

    return due, new_state


# ============================================================
# SMALL I/O HELPERS
# ============================================================
def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"REPORT: failed to read {path}: {e}")
    return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    try:
        if path.exists():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"REPORT: failed to read {path}: {e}")
    return records


# ============================================================
# PERFORMANCE REPORTER
# ============================================================
class PerformanceReporter:
    """
    Orchestrates rollover detection, data gathering, report computation, log
    summaries, and persistence of machine-readable report files.
    """

    def __init__(
        self,
        base_dir: Path,
        reports_dir: Optional[Path] = None,
        stats_file: Optional[Path] = None,
        adaptive_file: Optional[Path] = None,
        features_file: Optional[Path] = None,
        journal_dir: Optional[Path] = None,
        mode: str = "live",
        min_sample: int = 5,
        weak_win_rate: float = 0.40,
        log: Optional[logging.Logger] = None,
    ):
        self.base_dir = Path(base_dir)
        self.mode = mode
        self.min_sample = min_sample
        self.weak_win_rate = weak_win_rate
        self.log = log or logger

        logs_dir = self.base_dir / "logs"
        # Mode-aware default source files (paper uses *_paper suffix).
        suffix = "_paper" if mode == "paper" else ""
        self.stats_file = Path(stats_file) if stats_file else logs_dir / f"daily_stats{suffix}.json"
        self.adaptive_file = (
            Path(adaptive_file) if adaptive_file else logs_dir / f"adaptive_config{suffix}.json"
        )
        self.features_file = (
            Path(features_file) if features_file else logs_dir / f"trade_features{suffix}.jsonl"
        )
        self.journal_dir = Path(journal_dir) if journal_dir else self.base_dir / "journal"

        self.reports_dir = Path(reports_dir) if reports_dir else logs_dir / "reports"
        self.state_file = self.reports_dir / ".report_state.json"
        self.history_file = self.reports_dir / "history.jsonl"

    # ---------------------------------------------------------------
    # State persistence
    # ---------------------------------------------------------------
    def _load_state(self) -> ReportState:
        data = _load_json(self.state_file)
        if data:
            return ReportState.from_dict(data)
        return ReportState()

    def _save_state(self, state: ReportState):
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _write_report(self, filename: str, payload: Dict[str, Any]):
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        with open(self.reports_dir / filename, "w") as f:
            json.dump(payload, f, indent=2)

    def _append_history(self, payload: Dict[str, Any]):
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "a") as f:
            f.write(json.dumps(payload) + "\n")

    # ---------------------------------------------------------------
    # Data gathering
    # ---------------------------------------------------------------
    def _load_stats(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (daily_obj, weekly_obj). Empty dicts if missing (paper mode)."""
        data = _load_json(self.stats_file) or {}
        return data.get("daily", {}) or {}, data.get("weekly", {}) or {}

    def _journal_close_entries(self, date_str: str) -> List[Dict[str, Any]]:
        """CLOSE entries from a specific day's journal file."""
        path = self.journal_dir / f"journal_{date_str}.jsonl"
        return [e for e in _load_jsonl(path) if e.get("action") == "CLOSE"]

    def _history_daily_entries(self) -> List[Dict[str, Any]]:
        return [e for e in _load_jsonl(self.history_file) if e.get("type") == "daily"]

    # ---------------------------------------------------------------
    # Public entry point
    # ---------------------------------------------------------------
    def maybe_emit(self, now_utc: datetime) -> List[str]:
        """
        Check for day/week/month rollover since the last report and emit any
        due reports. Returns the list of report types emitted (for testing /
        logging). Safe to call every scan cycle.
        """
        if now_utc.tzinfo is None:
            now_utc = now_utc.replace(tzinfo=timezone.utc)

        state = self._load_state()
        due, new_state = compute_due_reports(state, now_utc)

        emitted: List[str] = []
        # Order matters: daily first (feeds history), then weekly, then monthly.
        if due["daily"]:
            self.emit_daily(due["daily"])
            emitted.append("daily")
        if due["weekly"]:
            self.emit_weekly(due["weekly"])
            emitted.append("weekly")
        if due["monthly"]:
            self.emit_monthly(due["monthly"])
            emitted.append("monthly")

        # Persist state only after successful emission attempts.
        self._save_state(new_state)
        return emitted

    # ---------------------------------------------------------------
    # Daily report
    # ---------------------------------------------------------------
    def build_daily_payload(self, date_str: str) -> Dict[str, Any]:
        daily, _ = self._load_stats()
        close_entries = self._journal_close_entries(date_str)

        # Prefer equity from the persisted daily stats when it matches the day
        # being reported; otherwise fall back to journal-derived P&L.
        if daily.get("date") == date_str:
            starting = float(daily.get("starting_equity", 0.0) or 0.0)
            current = float(daily.get("current_equity", 0.0) or 0.0)
            trades_taken = int(daily.get("trades_taken", len(close_entries)) or 0)
            max_dd = float(daily.get("max_drawdown_pct", 0.0) or 0.0)
        else:
            starting = 0.0
            current = 0.0
            trades_taken = len(close_entries)
            max_dd = 0.0

        pnl = pnl_abs(starting, current)
        # If equity was unavailable, derive P&L from the journal so the report
        # is still meaningful.
        if starting == 0.0 and current == 0.0 and close_entries:
            pnl = round(sum(_pnl_of(e) for e in close_entries), 2)

        ret = pct_return(starting, current)
        wins, losses = count_wins_losses(close_entries)
        best, worst = extract_best_worst(close_entries)
        avg_r = avg_r_multiple(close_entries)
        wr = win_rate_pct(wins, losses)

        return {
            "type": "daily",
            "mode": self.mode,
            "date": date_str,
            "pnl_usd": pnl,
            "return_pct": ret,
            "trades": trades_taken,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": wr,
            "best_trade_usd": best,
            "worst_trade_usd": worst,
            "avg_r": avg_r,
            "max_drawdown_pct": max_dd,
            "starting_equity": starting,
            "current_equity": current,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def emit_daily(self, date_str: str):
        p = self.build_daily_payload(date_str)
        self.log.info(
            f"=== DAILY REPORT {date_str} UTC ===  "
            f"P&L: {_money(p['pnl_usd'])} ({p['return_pct']:+.2f}%) | "
            f"{p['trades']} trades | {p['wins']}W/{p['losses']}L ({p['win_rate_pct']:.0f}%) | "
            f"Best {_money(p['best_trade_usd'], 0)} | Worst {_money(p['worst_trade_usd'], 0)} | "
            f"Avg R {p['avg_r']:.2f}"
        )
        self._write_report(f"daily_{date_str}.json", p)
        self._append_history(p)

    # ---------------------------------------------------------------
    # Weekly report
    # ---------------------------------------------------------------
    def _week_close_entries(self, wk_key: str) -> List[Dict[str, Any]]:
        """Aggregate CLOSE entries across all journal files in the ISO week."""
        entries: List[Dict[str, Any]] = []
        if not self.journal_dir.exists():
            return entries
        for path in sorted(self.journal_dir.glob("journal_*.jsonl")):
            date_part = path.stem.replace("journal_", "")
            try:
                d = datetime.strptime(date_part, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if week_key(d) == wk_key:
                entries.extend(e for e in _load_jsonl(path) if e.get("action") == "CLOSE")
        return entries

    def build_weekly_payload(self, wk_key: str) -> Dict[str, Any]:
        _, weekly = self._load_stats()
        starting = float(weekly.get("starting_equity", 0.0) or 0.0)
        current = float(weekly.get("current_equity", 0.0) or 0.0)

        close_entries = self._week_close_entries(wk_key)
        wins, losses = count_wins_losses(close_entries)
        total_trades = int(weekly.get("total_trades", len(close_entries)) or 0)
        avg_r = avg_r_multiple(close_entries)

        # Max drawdown across the week's daily reports (if any recorded).
        daily_hist = [
            e for e in self._history_daily_entries()
            if _date_in_week(e.get("date"), wk_key)
        ]
        max_dd = max([float(e.get("max_drawdown_pct", 0.0) or 0.0) for e in daily_hist], default=0.0)

        # Improvement insights from feature/journal + adaptive data.
        features = _load_jsonl(self.features_file)
        insight_trades = features if features else close_entries
        adaptive = _load_json(self.adaptive_file) or {}
        lock_incidents = [
            {"date": e.get("date"), "lock_reason": e.get("lock_reason")}
            for e in daily_hist if e.get("lock_reason")
        ]
        suggestions = generate_improvement_suggestions(
            insight_trades,
            adaptive_params=adaptive,
            lock_incidents=lock_incidents,
            min_sample=self.min_sample,
            weak_win_rate=self.weak_win_rate,
        )

        return {
            "type": "weekly",
            "mode": self.mode,
            "week": wk_key,
            "pnl_usd": pnl_abs(starting, current),
            "return_pct": pct_return(starting, current),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": win_rate_pct(wins, losses),
            "avg_r": avg_r,
            "max_drawdown_pct": round(max_dd, 2),
            "starting_equity": starting,
            "current_equity": current,
            "improvements": suggestions,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def emit_weekly(self, wk_key: str):
        p = self.build_weekly_payload(wk_key)
        self.log.info(
            f"=== WEEKLY REPORT {wk_key} UTC ===  "
            f"P&L: {_money(p['pnl_usd'])} ({p['return_pct']:+.2f}%) | "
            f"{p['total_trades']} trades | {p['wins']}W/{p['losses']}L "
            f"({p['win_rate_pct']:.0f}%) | Avg R {p['avg_r']:.2f} | "
            f"Max DD {p['max_drawdown_pct']:.2f}%"
        )
        for s in p["improvements"]:
            self.log.info(f"IMPROVE: - {s}")
        self._write_report(f"weekly_{wk_key}.json", p)

    # ---------------------------------------------------------------
    # Monthly report
    # ---------------------------------------------------------------
    def build_monthly_payload(self, mo_key: str) -> Dict[str, Any]:
        daily_hist = [
            e for e in self._history_daily_entries()
            if str(e.get("date", "")).startswith(mo_key)
        ]
        daily_hist.sort(key=lambda e: e.get("date", ""))

        total_pnl = round(sum(float(e.get("pnl_usd", 0.0) or 0.0) for e in daily_hist), 2)
        total_trades = sum(int(e.get("trades", 0) or 0) for e in daily_hist)
        wins = sum(int(e.get("wins", 0) or 0) for e in daily_hist)
        losses = sum(int(e.get("losses", 0) or 0) for e in daily_hist)

        # Equity change: first day's starting -> last day's current.
        if daily_hist:
            starting = float(daily_hist[0].get("starting_equity", 0.0) or 0.0)
            current = float(daily_hist[-1].get("current_equity", 0.0) or 0.0)
        else:
            starting = current = 0.0
        ret = pct_return(starting, current)

        best_day = max(
            daily_hist, key=lambda e: float(e.get("pnl_usd", 0.0) or 0.0), default=None
        )
        worst_day = min(
            daily_hist, key=lambda e: float(e.get("pnl_usd", 0.0) or 0.0), default=None
        )

        return {
            "type": "monthly",
            "mode": self.mode,
            "month": mo_key,
            "pnl_usd": total_pnl,
            "return_pct": ret,
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": win_rate_pct(wins, losses),
            "best_day": {"date": best_day.get("date"), "pnl_usd": best_day.get("pnl_usd")}
            if best_day else None,
            "worst_day": {"date": worst_day.get("date"), "pnl_usd": worst_day.get("pnl_usd")}
            if worst_day else None,
            "starting_equity": starting,
            "current_equity": current,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def emit_monthly(self, mo_key: str):
        p = self.build_monthly_payload(mo_key)
        best = p["best_day"]
        worst = p["worst_day"]
        best_str = f"{best['date']} ({_money(best['pnl_usd'], 0)})" if best else "n/a"
        worst_str = f"{worst['date']} ({_money(worst['pnl_usd'], 0)})" if worst else "n/a"
        self.log.info(
            f"=== MONTHLY REPORT {mo_key} UTC ===  "
            f"P&L: {_money(p['pnl_usd'])} ({p['return_pct']:+.2f}%) | "
            f"{p['total_trades']} trades | {p['wins']}W/{p['losses']}L "
            f"({p['win_rate_pct']:.0f}%) | Best day {best_str} | Worst day {worst_str}"
        )
        self._write_report(f"monthly_{mo_key}.json", p)


def _money(value: float, decimals: int = 2) -> str:
    """Format a signed dollar amount, e.g. +$142.30 / -$68.00."""
    sign = "+" if value >= 0 else "-"
    return f"{sign}${abs(value):.{decimals}f}"


def _date_in_week(date_str: Optional[str], wk_key: str) -> bool:
    if not date_str:
        return False
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return week_key(d) == wk_key
