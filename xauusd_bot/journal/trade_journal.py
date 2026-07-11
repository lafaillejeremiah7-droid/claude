"""
Trade Journal System — Track, Learn, Improve.

From the video: "Building an edge is all about answering
the smaller questions using data."

Tracks:
    - Every trade with full detail
    - Confluences that led to the trade
    - MFE (Max Favorable Excursion) — how far did it go in your favor?
    - MAE (Max Adverse Excursion) — how far against before hitting TP?
    - Win rate by confluence combination
    - Win rate by session/time
    - Win rate by condition quality
    - Screenshots (file paths)

This data lets you:
    1. Do more of what works
    2. Stop doing what doesn't work
    3. Identify your highest WR setups
    4. Improve entries (using MAE data)
    5. Improve targets (using MFE data)
"""

import json
import os
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field, asdict


@dataclass
class JournalEntry:
    """A complete trade journal entry."""
    # Trade basics
    id: int = 0
    timestamp: str = ""
    direction: str = ""
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    lot_size: float = 0.0

    # Results
    pnl_pips: float = 0.0
    pnl_dollars: float = 0.0
    result: str = ""           # "WIN" or "LOSS"
    exit_reason: str = ""
    duration_minutes: int = 0

    # Confluences (what justified the trade)
    confluences: dict = field(default_factory=dict)
    confluence_count: int = 0
    condition_quality: float = 0.0

    # Context
    session: str = ""          # "asian", "london", etc.
    hour_of_day: int = 0
    minute_of_hour: int = 0
    day_of_week: str = ""
    adx_at_entry: float = 0.0
    rsi_at_entry: float = 0.0
    atr_at_entry: float = 0.0

    # MFE/MAE (tracked post-entry)
    mfe_pips: float = 0.0      # Max Favorable Excursion
    mfe_rr: float = 0.0        # MFE as R multiple
    mae_pips: float = 0.0      # Max Adverse Excursion
    mae_pct_of_sl: float = 0.0 # MAE as % of stop loss

    # Notes
    screenshot_path: str = ""
    notes: str = ""
    pre_trade_bias: str = ""   # What you expected
    post_trade_lesson: str = "" # What you learned


class TradeJournal:
    """
    Persistent trade journal with analysis capabilities.

    Usage:
        journal = TradeJournal("./journal_data")
        journal.add_trade(entry)
        stats = journal.analyze_by_confluence("rsi_extreme")
        best_time = journal.best_trading_hour()
    """

    def __init__(self, data_dir: str = "./journal_data"):
        self.data_dir = data_dir
        self.entries: list[JournalEntry] = []
        self._next_id = 1
        os.makedirs(data_dir, exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_trade(self, trade: dict, snap=None) -> JournalEntry:
        """Add a completed trade to the journal."""
        entry = JournalEntry(
            id=self._next_id,
            timestamp=trade.get("entry_time", datetime.utcnow()).strftime(
                "%Y-%m-%d %H:%M") if trade.get("entry_time") else "",
            direction=trade.get("direction", ""),
            entry_price=trade.get("entry_price", 0),
            exit_price=trade.get("exit_price", 0),
            stop_loss=trade.get("sl", 0),
            take_profit=trade.get("tp", 0),
            lot_size=trade.get("lot_size", 0),
            pnl_pips=trade.get("pnl_pips", 0),
            pnl_dollars=trade.get("pnl_dollars", 0),
            result=trade.get("result", ""),
            exit_reason=trade.get("reason", ""),
            confluences=trade.get("confluences", {}),
            confluence_count=sum(
                1 for v in trade.get("confluences", {}).values()
                if v is True
            ),
        )

        # Context from timestamp
        if trade.get("entry_time"):
            et = trade["entry_time"]
            entry.hour_of_day = et.hour
            entry.minute_of_hour = et.minute
            entry.day_of_week = et.strftime("%A")

            if 0 <= et.hour < 7:
                entry.session = "asian"
            elif 7 <= et.hour < 12:
                entry.session = "london"
            elif 12 <= et.hour < 16:
                entry.session = "overlap"
            elif 16 <= et.hour < 21:
                entry.session = "new_york"
            else:
                entry.session = "dead_zone"

        # Duration
        if trade.get("entry_time") and trade.get("exit_time"):
            dur = (trade["exit_time"] - trade["entry_time"]).total_seconds()
            entry.duration_minutes = int(dur / 60)

        self.entries.append(entry)
        self._next_id += 1
        self._save()
        return entry

    def update_mfe_mae(self, trade_id: int, mfe_pips: float,
                       mae_pips: float):
        """Update MFE/MAE for a trade after it closes."""
        for entry in self.entries:
            if entry.id == trade_id:
                entry.mfe_pips = mfe_pips
                entry.mae_pips = mae_pips

                # Calculate ratios
                sl_distance = abs(entry.entry_price - entry.stop_loss)
                if sl_distance > 0:
                    entry.mfe_rr = (mfe_pips * 0.01) / sl_distance
                    entry.mae_pct_of_sl = (mae_pips * 0.01) / sl_distance * 100

                self._save()
                return

    # ------------------------------------------------------------------
    # ANALYSIS — Answer the "smaller questions"
    # ------------------------------------------------------------------

    def overall_stats(self) -> dict:
        """Overall performance stats."""
        if not self.entries:
            return {}

        wins = [e for e in self.entries if e.result == "WIN"]
        losses = [e for e in self.entries if e.result == "LOSS"]

        return {
            "total_trades": len(self.entries),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.entries) * 100,
            "avg_winner": sum(e.pnl_dollars for e in wins) / len(wins) if wins else 0,
            "avg_loser": sum(e.pnl_dollars for e in losses) / len(losses) if losses else 0,
            "total_pnl": sum(e.pnl_dollars for e in self.entries),
            "avg_mfe": sum(e.mfe_pips for e in self.entries) / len(self.entries),
            "avg_mae": sum(e.mae_pips for e in self.entries) / len(self.entries),
        }

    def analyze_by_confluence(self, confluence_name: str) -> dict:
        """
        Win rate when a specific confluence is TRUE vs FALSE.
        Answers: "Should I require this confluence?"
        """
        with_conf = [e for e in self.entries
                     if e.confluences.get(confluence_name) is True]
        without_conf = [e for e in self.entries
                        if e.confluences.get(confluence_name) is not True]

        def calc_wr(trades):
            if not trades:
                return 0
            wins = sum(1 for t in trades if t.result == "WIN")
            return wins / len(trades) * 100

        return {
            "confluence": confluence_name,
            "with_confluence": {
                "trades": len(with_conf),
                "win_rate": calc_wr(with_conf),
            },
            "without_confluence": {
                "trades": len(without_conf),
                "win_rate": calc_wr(without_conf),
            },
            "edge_difference": calc_wr(with_conf) - calc_wr(without_conf),
        }

    def analyze_all_confluences(self) -> list[dict]:
        """Analyze every confluence and rank by edge."""
        if not self.entries:
            return []

        # Get all confluence names
        all_confs = set()
        for e in self.entries:
            all_confs.update(e.confluences.keys())

        results = []
        for conf in all_confs:
            analysis = self.analyze_by_confluence(conf)
            results.append(analysis)

        # Sort by edge difference (biggest positive edge first)
        results.sort(key=lambda x: x["edge_difference"], reverse=True)
        return results

    def best_trading_hour(self) -> dict:
        """Win rate by hour of day. Answers: 'When should I trade?'"""
        hours = {}
        for e in self.entries:
            h = e.hour_of_day
            if h not in hours:
                hours[h] = {"trades": 0, "wins": 0}
            hours[h]["trades"] += 1
            if e.result == "WIN":
                hours[h]["wins"] += 1

        for h in hours:
            hours[h]["win_rate"] = (hours[h]["wins"] / hours[h]["trades"] * 100
                                    if hours[h]["trades"] > 0 else 0)

        return dict(sorted(hours.items(), key=lambda x: x[1]["win_rate"], reverse=True))

    def best_session(self) -> dict:
        """Win rate by session."""
        sessions = {}
        for e in self.entries:
            s = e.session
            if s not in sessions:
                sessions[s] = {"trades": 0, "wins": 0, "pnl": 0}
            sessions[s]["trades"] += 1
            sessions[s]["pnl"] += e.pnl_dollars
            if e.result == "WIN":
                sessions[s]["wins"] += 1

        for s in sessions:
            sessions[s]["win_rate"] = (sessions[s]["wins"] / sessions[s]["trades"] * 100
                                       if sessions[s]["trades"] > 0 else 0)

        return sessions

    def analyze_by_confluence_count(self) -> dict:
        """Win rate by number of confluences. More = better?"""
        counts = {}
        for e in self.entries:
            c = e.confluence_count
            if c not in counts:
                counts[c] = {"trades": 0, "wins": 0}
            counts[c]["trades"] += 1
            if e.result == "WIN":
                counts[c]["wins"] += 1

        for c in counts:
            counts[c]["win_rate"] = (counts[c]["wins"] / counts[c]["trades"] * 100
                                     if counts[c]["trades"] > 0 else 0)

        return dict(sorted(counts.items()))

    def mfe_analysis(self) -> dict:
        """
        MFE analysis — could you target a bigger TP?
        If avg MFE >> 1:1, you're leaving money on the table.
        """
        if not self.entries:
            return {}

        mfes = [e.mfe_rr for e in self.entries if e.mfe_rr > 0]
        if not mfes:
            return {"message": "No MFE data tracked yet"}

        return {
            "avg_mfe_rr": sum(mfes) / len(mfes),
            "median_mfe_rr": sorted(mfes)[len(mfes) // 2],
            "trades_that_hit_2r": sum(1 for m in mfes if m >= 2.0),
            "trades_that_hit_3r": sum(1 for m in mfes if m >= 3.0),
            "recommendation": (
                "Consider raising TP" if sum(mfes) / len(mfes) > 1.5
                else "1:1 TP is optimal for this data"
            ),
        }

    def mae_analysis(self) -> dict:
        """
        MAE analysis — could you get better entries?
        If avg MAE is high, you need to refine entry timing.
        """
        if not self.entries:
            return {}

        maes = [e.mae_pct_of_sl for e in self.entries if e.mae_pct_of_sl > 0]
        if not maes:
            return {"message": "No MAE data tracked yet"}

        return {
            "avg_mae_pct_of_sl": sum(maes) / len(maes),
            "trades_with_high_mae": sum(1 for m in maes if m > 70),
            "recommendation": (
                "Entry timing needs work — price going too far against you"
                if sum(maes) / len(maes) > 50
                else "Entry timing is good"
            ),
        }

    # ------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------

    def _save(self):
        """Save journal to JSON file."""
        filepath = os.path.join(self.data_dir, "journal.json")
        data = [asdict(e) for e in self.entries]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        """Load journal from JSON file."""
        filepath = os.path.join(self.data_dir, "journal.json")
        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            data = json.load(f)

        for item in data:
            entry = JournalEntry(**item)
            self.entries.append(entry)
            self._next_id = max(self._next_id, entry.id + 1)

    def export_csv(self, filepath: str = None) -> str:
        """Export journal to CSV for spreadsheet analysis."""
        if filepath is None:
            filepath = os.path.join(self.data_dir, "journal.csv")

        import csv
        headers = [
            "id", "timestamp", "direction", "entry_price", "exit_price",
            "pnl_pips", "pnl_dollars", "result", "session", "hour",
            "confluence_count", "condition_quality", "mfe_pips", "mae_pips",
            "duration_minutes", "exit_reason",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for e in self.entries:
                writer.writerow([
                    e.id, e.timestamp, e.direction, e.entry_price,
                    e.exit_price, e.pnl_pips, e.pnl_dollars, e.result,
                    e.session, e.hour_of_day, e.confluence_count,
                    e.condition_quality, e.mfe_pips, e.mae_pips,
                    e.duration_minutes, e.exit_reason,
                ])

        return filepath
