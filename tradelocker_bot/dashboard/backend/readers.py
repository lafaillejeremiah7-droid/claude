"""On-disk file readers for the dashboard (Req 8, 9, 13, 16).

Every read goes THROUGH ``ReadOnlyGuard.open_readonly`` so a write can never
happen by construction, and reuses the PURE parsers in ``derivations`` (
``parse_jsonl``, ``parse_iso_utc``, ``parse_confidence`` ...). All readers are
tolerant: a missing / locked / malformed file yields an empty result and NEVER
raises, so the dashboard keeps serving (Req 13.1, 13.4, 13.5).

MODE-awareness (``DASHBOARD_MODE=live|paper``): in paper mode the readers target
the bot's ``paper_*`` shadow files; live mode targets the primary files. The bot
directory is resolved from ``BOT_DIR`` (default: this repo's ``tradelocker_bot/``).
"""
from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .guard import ReadOnlyGuard
from .derivations import (
    parse_confidence,
    parse_iso_utc,
    parse_jsonl,
)

UTC = timezone.utc

# tradelocker_bot/ is three parents up from this file
# (backend/readers.py -> backend -> dashboard -> tradelocker_bot).
_DEFAULT_BOT_DIR = Path(__file__).resolve().parents[2]

# Log line prefix: "2024-06-10 12:34:56 | LEVEL | logger | message"
_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|")
# "Entry: $67250.50" (used for best-effort last-known prices when API is off).
_ENTRY_PRICE_RE = re.compile(r"Entry:\s*[$=]?\s*(\d+(?:\.\d+)?)")
# A leading "SYMBOL:" token in a log message identifies the instrument.
_SYMBOL_RE = re.compile(r"([A-Z0-9]{3,12}):")
# "Symbol: BTCUSD" continuation line inside an APPROVED block.
_APPROVED_SYMBOL_RE = re.compile(r"Symbol:\s*([A-Z0-9]{3,12})")
# "[PAPER] POSITION OPENED: BUY 0.12 BTCUSD | Entry=..." -> BTCUSD.
_PAPER_OPEN_SYMBOL_RE = re.compile(
    r"POSITION OPENED:\s*[A-Za-z]+\s+[\d.]+\s+([A-Z0-9]{3,12})"
)
# "[PAPER] POSITION CLOSED (WIN): BTCUSD buy | ..." -> BTCUSD.
_PAPER_CLOSE_SYMBOL_RE = re.compile(
    r"POSITION CLOSED\s*\([A-Z]+\):\s*([A-Z0-9]{3,12})"
)

# Notable-event detection, evaluated in priority order against a log block's
# full text. Each entry maps a substring marker to the emitted action label.
_EVENT_MARKERS = (
    ("NEAR-MISS", "NEAR_MISS"),
    ("TRADE SIGNAL APPROVED", "APPROVED"),
    ("POSITION OPENED", "PAPER_OPEN"),
    ("POSITION CLOSED", "PAPER_CLOSE"),
    ("REJECTED", "REJECTED"),
)


def resolve_mode(env: Optional[Dict[str, str]] = None) -> str:
    """Return the dashboard data mode: ``"live"`` (default) or ``"paper"``."""
    source = env if env is not None else os.environ
    mode = (source.get("DASHBOARD_MODE") or "live").strip().lower()
    return "paper" if mode == "paper" else "live"


def resolve_bot_dir(env: Optional[Dict[str, str]] = None) -> Path:
    """Resolve the bot directory from ``BOT_DIR`` (default: bundled bot root)."""
    source = env if env is not None else os.environ
    raw = source.get("BOT_DIR")
    if raw and raw.strip():
        return Path(raw).expanduser()
    return _DEFAULT_BOT_DIR


def api_reader_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Whether the optional live TradeLocker API reader is enabled (default off)."""
    source = env if env is not None else os.environ
    raw = (source.get("API_READER_ENABLED") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


class FileReader:
    """Tolerant, read-only reader over the bot's on-disk state files."""

    def __init__(
        self,
        bot_dir: Optional[Path] = None,
        mode: str = "live",
        guard: Optional[ReadOnlyGuard] = None,
    ) -> None:
        self.bot_dir = Path(bot_dir) if bot_dir else _DEFAULT_BOT_DIR
        self.mode = "paper" if mode == "paper" else "live"
        self.guard = guard or ReadOnlyGuard()
        self.logs_dir = self.bot_dir / "logs"
        self.journal_dir = self.bot_dir / "journal"
        self.reports_dir = self.logs_dir / "reports"

    # -- mode-aware filename helpers ------------------------------------
    def _prefix(self) -> str:
        return "paper_" if self.mode == "paper" else ""

    def daily_stats_path(self) -> Path:
        return self.logs_dir / f"{self._prefix()}daily_stats.json"

    def positions_path(self) -> Path:
        return self.logs_dir / f"{self._prefix()}active_positions.json"

    def adaptive_config_path(self) -> Path:
        # adaptive_config is shared; paper variant used when present.
        paper = self.logs_dir / "paper_adaptive_config.json"
        if self.mode == "paper" and paper.exists():
            return paper
        return self.logs_dir / "adaptive_config.json"

    def trade_features_path(self) -> Path:
        return self.logs_dir / f"{self._prefix()}trade_features.jsonl"

    def journal_path(self, date_str: str) -> Path:
        return self.journal_dir / f"{self._prefix()}journal_{date_str}.jsonl"

    def bot_log_path(self, date_str: str) -> Path:
        return self.logs_dir / f"bot_{date_str}.log"

    # -- low level tolerant IO (always via the guard) -------------------
    def _read_text(self, path: Path) -> str:
        """Read a file's text through the read-only guard; ``""`` on any problem."""
        try:
            if not Path(path).exists():
                return ""
            with self.guard.open_readonly(str(path)) as fh:
                return fh.read()
        except Exception:
            # Missing / locked / unreadable / guard violation -> empty, never crash.
            return ""

    def _read_json(self, path: Path) -> Optional[dict]:
        import json

        text = self._read_text(path)
        if not text.strip():
            return None
        try:
            obj = json.loads(text)
        except (ValueError, TypeError):
            return None
        return obj if isinstance(obj, dict) else None

    # -- typed readers --------------------------------------------------
    def read_daily_stats(self) -> Optional[dict]:
        """Return the parsed daily_stats document, or ``None`` when unavailable."""
        return self._read_json(self.daily_stats_path())

    def read_positions(self) -> Dict[str, dict]:
        """Return the ``position_id -> record`` map; ``{}`` on missing/malformed (Req 9.8)."""
        obj = self._read_json(self.positions_path())
        if not isinstance(obj, dict):
            return {}
        return {k: v for k, v in obj.items() if isinstance(v, dict)}

    def read_adaptive_config(self) -> Optional[dict]:
        return self._read_json(self.adaptive_config_path())

    def read_trade_features(self) -> List[dict]:
        """Parse trade_features.jsonl tolerantly (malformed lines skipped)."""
        return parse_jsonl(self._read_text(self.trade_features_path()))

    def read_journal_entries(self, dates: Optional[List[str]] = None) -> List[dict]:
        """Read one or more dated journal files, annotating source keys.

        Each entry gets ``file_date`` and ``line_index`` for deterministic
        tie-breaking (Req 6.1, 10.2). Defaults to today + yesterday (UTC) so a
        midnight rollover is handled without a restart (Req 13.2, 13.3).
        """
        if dates is None:
            dates = self._recent_dates()
        out: List[dict] = []
        for date_str in dates:
            text = self._read_text(self.journal_path(date_str))
            for line_index, rec in enumerate(parse_jsonl(text)):
                rec = dict(rec)
                rec.setdefault("file_date", date_str)
                rec["line_index"] = line_index
                out.append(rec)
        return out

    def read_close_actions(self, dates: Optional[List[str]] = None) -> List[dict]:
        """Journal entries whose ``action`` is ``CLOSE`` (the PnL/streak input)."""
        return [e for e in self.read_journal_entries(dates) if e.get("action") == "CLOSE"]

    def read_bot_log_events(self, date_str: Optional[str] = None) -> dict:
        """Parse the current bot log for scan activity, confidence, and prices.

        Returns ``{last_scan_utc, events, confidence, prices}`` where:
          - ``last_scan_utc`` is the most recent timestamped log line (Req 11.1),
          - ``events`` are NEAR-MISS / TRADE SIGNAL APPROVED feed items,
          - ``confidence`` is the list of parsed confidence entries,
          - ``prices`` maps ``symbol -> {"price", "_ts"}`` (best-effort marks).
        """
        if date_str is None:
            date_str = self._today()
        text = self._read_text(self.bot_log_path(date_str))
        return parse_log_text(text)

    # -- reports --------------------------------------------------------
    def read_reports(self) -> dict:
        """Read the latest daily/weekly/monthly reports + history (Req from task)."""
        return {
            "daily": self._latest_report("daily_"),
            "weekly": self._latest_report("weekly_"),
            "monthly": self._latest_report("monthly_"),
            "history": parse_jsonl(self._read_text(self.reports_dir / "history.jsonl")),
        }

    def _latest_report(self, prefix: str) -> Optional[dict]:
        """Return the most recent ``<prefix>*.json`` report payload, or ``None``."""
        try:
            if not self.reports_dir.exists():
                return None
            candidates = sorted(
                p for p in self.reports_dir.glob(f"{prefix}*.json") if p.is_file()
            )
        except Exception:
            return None
        if not candidates:
            return None
        # Filenames embed a sortable period key, so the last one is the newest.
        return self._read_json(candidates[-1])

    # -- date helpers ---------------------------------------------------
    def _today(self) -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def _recent_dates(self) -> List[str]:
        now = datetime.now(UTC)
        return [
            now.strftime("%Y-%m-%d"),
            (now - timedelta(days=1)).strftime("%Y-%m-%d"),
        ]


def _classify_action(block_text: str) -> Optional[str]:
    """Return the action label for a log block, or ``None`` if not notable."""
    for marker, action in _EVENT_MARKERS:
        if marker in block_text:
            return action
    return None


def _extract_symbol(block_text: str, action: str) -> Optional[str]:
    """Best-effort instrument symbol for a notable block (tolerant, may be None)."""
    if action == "APPROVED":
        m = _APPROVED_SYMBOL_RE.search(block_text)
        if m:
            return m.group(1)
        # Fixture / alternate format may put "SYMBOL:" on a line of the block.
        m = _SYMBOL_RE.search(block_text)
        return m.group(1) if m else None
    if action == "PAPER_OPEN":
        m = _PAPER_OPEN_SYMBOL_RE.search(block_text)
        return m.group(1) if m else None
    if action == "PAPER_CLOSE":
        m = _PAPER_CLOSE_SYMBOL_RE.search(block_text)
        return m.group(1) if m else None
    # REJECTED / NEAR-MISS: the leading "SYMBOL:" token names the instrument.
    m = _SYMBOL_RE.search(block_text)
    return m.group(1) if m else None


def _clean_message(block_lines: List[str]) -> str:
    """Build a compact human-readable message from a log block's lines.

    Strips the "TS | LEVEL | logger |" prefix off the first line and drops
    purely decorative separator lines (``===`` / ``- -``), collapsing the rest
    into a single ``|``-joined, whitespace-normalised string.
    """
    pieces: List[str] = []
    for idx, raw in enumerate(block_lines):
        line = raw
        if idx == 0:
            parts = line.split(" | ")
            # Drop the "TS | LEVEL | logger |" prefix but KEEP the rest of the
            # message, which may itself contain " | " separators.
            line = " | ".join(parts[3:]) if len(parts) >= 4 else line
        stripped = line.strip()
        if not stripped:
            continue
        # Skip decorative separators made only of '=', '-' and spaces.
        if set(stripped) <= set("=- "):
            continue
        pieces.append(stripped)
    message = " | ".join(pieces)
    return re.sub(r"\s+", " ", message).strip()


def parse_log_text(text: object) -> dict:
    """PURE parser for the bot log's notable lines (testable without I/O).

    Groups the log into blocks (a timestamped line plus its non-timestamped
    continuation lines) and extracts, for the notable ones:
      - the latest scan timestamp across ALL timestamped lines (Req 11.1),
      - feed ``events`` for REJECTED / NEAR-MISS / TRADE SIGNAL APPROVED /
        [PAPER] POSITION OPENED / CLOSED, each with a parsed timestamp, an
        action label, the symbol (when present), and a cleaned message,
      - ``confidence`` entries from NEAR-MISS / APPROVED blocks, and
      - best-effort last-known ``prices`` from ``Entry: $...`` lines.
    Tolerant by construction: unparseable lines / blocks are skipped.
    """
    result = {
        "last_scan_utc": None,
        "events": [],
        "confidence": [],
        "prices": {},
    }
    if not isinstance(text, str) or not text.strip():
        return result

    latest_ts: Optional[datetime] = None
    blocks: List[dict] = []
    current: Optional[dict] = None

    for line in text.splitlines():
        m = _LOG_TS_RE.match(line)
        if m:
            ts_str = m.group(1)
            dt = parse_iso_utc(ts_str.replace(" ", "T"))
            if dt is not None and (latest_ts is None or dt > latest_ts):
                latest_ts = dt
            current = {"ts_str": ts_str, "ts_dt": dt, "lines": [line]}
            blocks.append(current)
        elif current is not None:
            # Continuation line of the current block (e.g. NEAR-MISS/APPROVED body).
            current["lines"].append(line)
        # Lines before the first timestamp are ignored.

    result["last_scan_utc"] = latest_ts

    for block in blocks:
        block_text = "\n".join(block["lines"])
        action = _classify_action(block_text)
        if action is None:
            continue
        symbol = _extract_symbol(block_text, action)
        confidence = parse_confidence(block_text)
        ts_str = block["ts_str"]

        result["events"].append({
            "timestamp": ts_str,
            "kind": "event",
            "action": action,
            "symbol": symbol,
            "direction": "n/a",
            "confidence": confidence,
            "message": _clean_message(block["lines"]),
        })

        if confidence is not None and action in ("NEAR_MISS", "APPROVED"):
            result["confidence"].append({
                "timestamp_utc": ts_str,
                "symbol": symbol,
                "value": confidence,
                "available": True,
                "source": "log",
            })

        if symbol:
            price_m = _ENTRY_PRICE_RE.search(block_text)
            if price_m:
                try:
                    result["prices"][symbol] = {
                        "price": float(price_m.group(1)),
                        "_ts": block["ts_dt"],
                    }
                except (ValueError, TypeError):
                    pass

    return result
