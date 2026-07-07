"""Snapshot assembly + in-memory store (Req 3-16).

``build_snapshot`` composes the full ``DashboardSnapshot`` dict purely from the
readers + pure derivations. The live TradeLocker API is OPTIONAL: when it is
disabled or unavailable, equity falls back to ``daily_stats.current_equity`` and
positions use best-effort last-known prices from the log, so the dashboard RUNS
with no network. Secrets are never placed in the snapshot (``redact_secrets``).
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

from .readers import FileReader, api_reader_enabled, resolve_bot_dir, resolve_mode
from .security import credential_status, redact_secrets
from .derivations import (
    classify_gate,
    compute_countdown,
    compute_streaks,
    cumulative_realized_pnl,
    daily_return_pct,
    data_age,
    derive_position,
    equity_curve,
    hour_in_avoid_hours,
    is_bot_offline,
    monitored_instruments,
    order_closes,
    parse_confidence,
    parse_iso_utc,
    resolve_bot_status,
    select_equity,
    seconds_since,
    total_unrealized,
    build_feed,
)

UTC = timezone.utc

CONFIDENCE_RECENT_CAP = 20


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if isinstance(dt, datetime) else None


def _num(value: object) -> Optional[float]:
    from .derivations import is_number

    return float(value) if is_number(value) else None


def configured_instruments(env: Optional[Dict[str, str]] = None) -> List[str]:
    """Instrument set from ``INSTRUMENTS`` env (default BTCUSD,XAUUSD)."""
    source = env if env is not None else os.environ
    raw = source.get("INSTRUMENTS", "BTCUSD,XAUUSD")
    return [s.strip() for s in raw.split(",") if s.strip()]


def _last_file_mod(reader: FileReader) -> Optional[datetime]:
    """Most recent mtime across monitored bot files (Req 14.1)."""
    paths = [
        reader.daily_stats_path(),
        reader.positions_path(),
        reader.adaptive_config_path(),
        reader.bot_log_path(datetime.now(UTC).strftime("%Y-%m-%d")),
    ]
    latest: Optional[datetime] = None
    for path in paths:
        try:
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
                if latest is None or mtime > latest:
                    latest = mtime
        except OSError:
            continue
    return latest


def _quotes_from(api_state: Optional[dict], log_prices: dict) -> dict:
    """Build a ``symbol -> quote`` map, preferring the API, else log marks."""
    quotes: Dict[str, dict] = {}
    if isinstance(api_state, dict):
        for sym, q in (api_state.get("quotes") or {}).items():
            if isinstance(q, dict):
                quotes[sym] = q
    for sym, mark in (log_prices or {}).items():
        if sym in quotes:
            continue
        price = mark.get("price")
        quotes[sym] = {"bid": price, "ask": price, "_ts": mark.get("_ts")}
    return quotes


def _confidence_recent(journal_entries: List[dict], log_conf: List[dict], gate) -> List[dict]:
    """Assemble the recent-confidence list from journal reasons + log lines."""
    entries: List[dict] = []
    for rec in journal_entries:
        reasons = rec.get("entry_reasons")
        if not isinstance(reasons, list):
            continue
        for reason in reasons:
            value = parse_confidence(reason)
            if value is not None:
                entries.append({
                    "timestamp_utc": rec.get("timestamp"),
                    "symbol": rec.get("symbol"),
                    "value": value,
                    "available": True,
                    "source": "journal",
                })
                break
    for item in log_conf:
        entries.append({
            "timestamp_utc": item.get("timestamp_utc"),
            "symbol": item.get("symbol"),
            "value": item.get("value"),
            "available": True,
            "source": "log",
        })
    for entry in entries:
        entry["classification"] = classify_gate(entry.get("value"), gate)

    def sort_key(e):
        dt = parse_iso_utc(e.get("timestamp_utc"))
        return dt or datetime.min.replace(tzinfo=UTC)

    entries.sort(key=sort_key, reverse=True)
    return entries[:CONFIDENCE_RECENT_CAP]


def build_snapshot(
    reader: FileReader,
    now: Optional[datetime] = None,
    api_state: Optional[dict] = None,
    secret_values: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> dict:
    """Assemble the full DashboardSnapshot dict from readers + derivations."""
    now = now or datetime.now(UTC)

    daily_stats = reader.read_daily_stats() or {}
    daily = daily_stats.get("daily", {}) if isinstance(daily_stats, dict) else {}
    weekly = daily_stats.get("weekly", {}) if isinstance(daily_stats, dict) else {}
    daily = daily if isinstance(daily, dict) else {}
    weekly = weekly if isinstance(weekly, dict) else {}

    positions_map = reader.read_positions()
    adaptive = reader.read_adaptive_config() or {}
    journal_entries = reader.read_journal_entries()
    close_actions = [e for e in journal_entries if e.get("action") == "CLOSE"]
    log = reader.read_bot_log_events()
    reports = reader.read_reports()

    quotes = _quotes_from(api_state, log.get("prices", {}))

    # --- account / equity (Req 3) --------------------------------------
    account = select_equity(api_state, daily)

    # --- pnl (Req 4) ---------------------------------------------------
    starting_equity = daily.get("starting_equity")
    current_equity = account.get("equity")
    ret_pct = daily_return_pct(starting_equity, current_equity)
    positions_list = list(positions_map.values())
    tot_unreal = total_unrealized(positions_list, quotes, now)
    pnl = {
        "daily_realized": _num(daily.get("realized_pnl")),
        "daily_realized_available": _num(daily.get("realized_pnl")) is not None,
        "daily_return_pct": ret_pct,
        "daily_return_pct_available": ret_pct is not None,
        "cumulative_realized": cumulative_realized_pnl(close_actions),
        "total_unrealized": tot_unreal,
        "total_unrealized_available": tot_unreal is not None,
    }

    # --- win stats + streaks (Req 5, 6) --------------------------------
    wins = daily.get("wins")
    losses = daily.get("losses")
    win_total = (wins or 0) + (losses or 0) if (_num(wins) is not None or _num(losses) is not None) else 0
    daily_wr = (float(wins) / win_total * 100.0) if win_total > 0 else None
    streaks = compute_streaks(order_closes(close_actions))
    adaptive_wr = _num(adaptive.get("current_win_rate"))
    win_stats = {
        "daily_win_rate": daily_wr,
        "daily_win_rate_available": daily_wr is not None,
        "adaptive_win_rate": adaptive_wr,
        "adaptive_win_rate_available": adaptive_wr is not None,
        "wins": wins if _num(wins) is not None else 0,
        "losses": losses if _num(losses) is not None else 0,
        "win_streak": streaks.win_streak,
        "loss_streak": streaks.loss_streak,
    }

    # --- confidence (Req 7) --------------------------------------------
    gate = _num(adaptive.get("min_confidence"))
    confidence = {
        "gate": gate,
        "gate_available": gate is not None,
        "recent": _confidence_recent(journal_entries, log.get("confidence", []), gate),
    }

    # --- feed (Req 8) --------------------------------------------------
    feed = build_feed(journal_entries, log.get("events", []))
    feed_out = []
    for e in feed:
        feed_out.append({
            "timestamp_utc": e.get("timestamp"),
            "kind": e.get("kind", "trade"),
            "action": e.get("action"),
            "symbol": e.get("symbol"),
            "direction": e.get("direction"),
            "confidence": e.get("confidence"),
            "pnl": e.get("pnl"),
            "is_win": e.get("is_win"),
        })

    # --- positions (Req 9) ---------------------------------------------
    positions_out = []
    for pos in positions_list:
        quote = quotes.get(pos.get("symbol"))
        derived = derive_position(pos, quote, now)
        if quote is not None:
            derived["live_price"] = quote.get("bid", quote.get("ask"))
        else:
            derived["live_price"] = None
        positions_out.append(derived)

    # --- equity curve (Req 10) -----------------------------------------
    baseline = weekly.get("starting_equity")
    if _num(baseline) is None:
        baseline = starting_equity if _num(starting_equity) is not None else 0.0
    curve = equity_curve(close_actions, baseline)
    curve["points"] = [
        {"timestamp_utc": p.get("timestamp_utc"), "equity": p["equity"]}
        for p in curve["points"]
    ]

    # --- countdown / bot status (Req 11, 14) ---------------------------
    last_scan = log.get("last_scan_utc")
    last_file_mod = _last_file_mod(reader)
    is_locked = bool(daily.get("is_locked")) or bool(weekly.get("is_locked"))
    in_avoid = hour_in_avoid_hours(now.hour, adaptive.get("avoid_hours"))
    offline = is_bot_offline(last_file_mod, last_scan, now)
    initializing = last_file_mod is None and last_scan is None
    status = resolve_bot_status(
        initializing=initializing,
        bot_offline=offline,
        is_locked=is_locked,
        in_avoid_hours=in_avoid,
        out_of_session=False,
        has_scan_activity=last_scan is not None,
    )
    cd = compute_countdown(last_scan, now)
    countdown = {
        "state": status if status not in ("scanning",) else cd["state"],
        "seconds_remaining": cd["seconds_remaining"],
        "next_scan_utc": _iso(cd["next_scan_utc"]),
        "lock_reason": daily.get("lock_reason") or weekly.get("lock_reason") or None,
        "paused_reason": "avoid_hours" if in_avoid else None,
    }

    # --- instruments (Req 15) ------------------------------------------
    configured = configured_instruments(env)
    monitored = monitored_instruments(configured)
    per_instrument = {
        sym: {"data_available": sym in quotes and quotes[sym].get("bid") is not None}
        for sym in monitored
    }
    instruments = {
        "configured": monitored,
        "per_instrument": per_instrument,
        "none_configured": len(monitored) == 0,
    }

    # --- freshness (Req 12, 14) ----------------------------------------
    candidates = [t for t in (last_file_mod, last_scan) if t is not None]
    last_update = max(candidates) if candidates else None
    age = data_age(last_update, now)

    api_on = api_reader_enabled(env)
    api_status = "ok" if (api_on and isinstance(api_state, dict) and api_state.get("equity") is not None) else ("disabled" if not api_on else "error")
    cred = credential_status(env)

    snapshot = {
        "server_time_utc": _iso(now),
        "last_update_utc": _iso(last_update),
        "data_age_seconds": age,
        "mode": reader.mode,
        "connection": {
            "auth_status": cred["status"],
            "config_error_field": cred["config_error_field"],
            "file_status": "ok" if last_file_mod is not None else "error",
            "api_status": api_status,
        },
        "bot_status": {
            "state": status,
            "seconds_since_last_file_mod": seconds_since(last_file_mod, now),
            "last_file_mod_utc": _iso(last_file_mod),
        },
        "account": account,
        "pnl": pnl,
        "win_stats": win_stats,
        "confidence": confidence,
        "feed": feed_out,
        "feed_empty": len(feed_out) == 0,
        "positions": positions_out,
        "positions_error": False,
        "equity_curve": curve,
        "countdown": countdown,
        "instruments": instruments,
        "reports": {
            "daily": reports.get("daily"),
            "weekly": reports.get("weekly"),
            "monthly": reports.get("monthly"),
        },
    }

    # Secrets NEVER leave the server (Req 2.3).
    return redact_secrets(snapshot, secret_values)


def snapshot_hash(snapshot: dict) -> str:
    """Stable content hash used for SSE change detection (Req 12.2)."""
    payload = {k: v for k, v in snapshot.items() if k != "server_time_utc"}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class SnapshotStore:
    """Thread-safe holder of the latest assembled snapshot + its content hash."""

    def __init__(
        self,
        reader: Optional[FileReader] = None,
        secret_values: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.env = env
        self.reader = reader or FileReader(
            bot_dir=resolve_bot_dir(env), mode=resolve_mode(env)
        )
        self.secret_values = secret_values or []
        self._lock = threading.Lock()
        self._snapshot: Optional[dict] = None
        self._hash: Optional[str] = None

    def refresh(self, now: Optional[datetime] = None, api_state: Optional[dict] = None) -> dict:
        """Rebuild the snapshot from disk and store it; returns the new snapshot."""
        snap = build_snapshot(
            self.reader,
            now=now,
            api_state=api_state,
            secret_values=self.secret_values,
            env=self.env,
        )
        with self._lock:
            self._snapshot = snap
            self._hash = snapshot_hash(snap)
        return snap

    def get(self) -> dict:
        """Return the latest snapshot, building one on first access."""
        with self._lock:
            if self._snapshot is not None:
                return self._snapshot
        return self.refresh()

    @property
    def content_hash(self) -> Optional[str]:
        with self._lock:
            return self._hash
