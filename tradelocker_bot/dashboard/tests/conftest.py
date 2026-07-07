"""Pytest + Hypothesis configuration for the dashboard test suite.

Makes the ``dashboard`` package importable regardless of the invocation cwd and
registers a hypothesis profile that runs at least 100 examples per property.
"""
import sys
from pathlib import Path

from hypothesis import settings

# tradelocker_bot/ (the bot root) must be importable so `dashboard...` resolves.
BOT_ROOT = Path(__file__).resolve().parents[2]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

# >= 100 examples per property test (design Testing Strategy).
settings.register_profile("dashboard", max_examples=100, deadline=None)
settings.load_profile("dashboard")



import json
from datetime import datetime, timezone

import pytest

UTC = timezone.utc


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _write(path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def seed_bot_dir(base, mode: str = "live") -> None:
    """Seed a fixture bot directory with realistic on-disk files.

    Writes both the live-named and paper-named files so MODE-aware readers can be
    exercised against the same fixture directory.
    """
    logs = base / "logs"
    journal = base / "journal"
    reports = logs / "reports"
    today = _today()

    daily_stats = {
        "daily": {
            "date": today, "trades_taken": 3, "wins": 2, "losses": 1,
            "consecutive_losses": 0, "realized_pnl": 142.30,
            "starting_equity": 10000.0, "current_equity": 10142.30,
            "max_drawdown_pct": 0.18, "is_locked": False, "lock_reason": "",
        },
        "weekly": {
            "week_start": today, "starting_equity": 10000.0,
            "current_equity": 10142.30, "total_trades": 3,
            "is_locked": False, "lock_reason": "",
        },
    }
    positions = {
        "12345": {
            "position_id": "12345", "symbol": "BTCUSD", "direction": "buy",
            "entry_price": 67250.5, "stop_loss": 66980.0, "take_profit": 67791.5,
            "quantity": 0.12, "risk_amount": 202.19, "risk_reward_ratio": 2.0,
            "is_breakeven": False, "entry_time": today + "T12:34:56+00:00",
            "entry_reasons": ["Pullback to EMA20", "Confidence: 8.5/10"],
        }
    }
    adaptive = {
        "min_confidence": 8.0, "avoid_hours": [15, 16, 17],
        "current_win_rate": 62.5, "current_avg_r": 0.34,
    }
    journal_lines = [
        json.dumps({
            "timestamp": today + "T10:00:00+00:00", "action": "OPEN",
            "symbol": "BTCUSD", "direction": "buy", "position_id": "1",
            "entry_price": 67000.0, "entry_reasons": ["Confidence: 8.5/10"],
        }),
        "this line is not valid json and must be skipped",
        json.dumps({
            "timestamp": today + "T11:00:00+00:00", "action": "CLOSE",
            "symbol": "BTCUSD", "direction": "buy", "position_id": "1",
            "exit_price": 67300.0, "pnl": 210.0, "is_win": True, "r_multiple": 2.0,
        }),
        json.dumps({
            "timestamp": today + "T12:00:00+00:00", "action": "CLOSE",
            "symbol": "XAUUSD", "direction": "sell", "position_id": "2",
            "exit_price": 2300.0, "pnl": -67.70, "is_win": False, "r_multiple": -1.0,
        }),
    ]
    log_text = (
        f"{today} 09:59:00 | INFO     | main                 | Scanning instruments...\n"
        f"{today} 10:30:00 | INFO     | main                 | BTCUSD: NEAR-MISS (LONG) | Confidence: 7.5/10 (need 8.0)\n"
        f"  Entry: $67100.00 | SL: $66900.00 | TP: $67500.00\n"
        f"{today} 11:45:00 | INFO     | main                 | TRADE SIGNAL APPROVED (Adaptive)\n"
        f"  XAUUSD: signal\n"
        f"  Confidence: 8.5/10 | Est. Win Prob: 62%\n"
        f"  Entry: $2305.00 | SL: $2295.00 | TP: $2325.00\n"
    )
    daily_report = {
        "type": "daily", "mode": mode, "date": today, "pnl_usd": 142.30,
        "return_pct": 1.42, "trades": 3, "wins": 2, "losses": 1,
        "win_rate_pct": 66.7, "best_trade_usd": 210.0, "worst_trade_usd": -67.70,
        "avg_r": 1.0, "max_drawdown_pct": 0.18,
        "starting_equity": 10000.0, "current_equity": 10142.30,
        "generated_at": today + "T23:59:00+00:00",
    }
    weekly_report = {
        "type": "weekly", "mode": mode, "week": "2024-W24", "pnl_usd": 142.30,
        "return_pct": 1.42, "total_trades": 3, "wins": 2, "losses": 1,
        "win_rate_pct": 66.7, "avg_r": 1.0, "max_drawdown_pct": 0.18,
        "starting_equity": 10000.0, "current_equity": 10142.30,
        "improvements": [
            "Win rate in 15:00-16:00 UTC is 20% (6 trades) - consider avoid_hours.",
            "Adaptive win rate improved 55.0% -> 62.5% vs last week.",
        ],
        "generated_at": today + "T23:59:00+00:00",
    }
    monthly_report = {
        "type": "monthly", "mode": mode, "month": "2024-06", "pnl_usd": 512.10,
        "return_pct": 5.12, "total_trades": 20, "wins": 13, "losses": 7,
        "win_rate_pct": 65.0,
    }

    prefix = "paper_" if mode == "paper" else ""
    _write(logs / f"{prefix}daily_stats.json", json.dumps(daily_stats, indent=2))
    _write(logs / f"{prefix}active_positions.json", json.dumps(positions, indent=2))
    _write(logs / "adaptive_config.json", json.dumps(adaptive, indent=2))
    _write(journal / f"{prefix}journal_{today}.jsonl", "\n".join(journal_lines) + "\n")
    _write(logs / f"bot_{today}.log", log_text)
    _write(reports / f"daily_{today}.json", json.dumps(daily_report, indent=2))
    _write(reports / "weekly_2024-W24.json", json.dumps(weekly_report, indent=2))
    _write(reports / "monthly_2024-06.json", json.dumps(monthly_report, indent=2))
    _write(reports / "history.jsonl", json.dumps(daily_report) + "\n")


@pytest.fixture
def live_bot_dir(tmp_path):
    seed_bot_dir(tmp_path, mode="live")
    return tmp_path


@pytest.fixture
def paper_bot_dir(tmp_path):
    seed_bot_dir(tmp_path, mode="paper")
    return tmp_path
