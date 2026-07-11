"""
File_Reader – Reads and parses all bot files on disk.

Responsibilities (Requirements 1, 8, 13, 14, 16):
  - Opens every bot file in READ-ONLY mode.
  - Never acquires write/exclusive locks.
  - Never creates, modifies, deletes, or truncates bot files.
  - Handles missing files (empty result set, no crash).
  - Handles date rotation (new journal/log file at UTC midnight).
  - Skips malformed JSON/JSONL lines gracefully.
  - Handles temporarily locked/unreadable files (retain last values, retry next cycle).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dashboard_backend.config import Settings

logger = logging.getLogger("dashboard_backend.file_reader")


class FileReader:
    """
    Reads bot output files in read-only mode with full robustness.
    All read operations are non-blocking and fault-tolerant.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        # Cache last successful reads for fallback (Req 13.5)
        self._cache: Dict[str, Any] = {}
        self._file_mtimes: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public methods – each returns parsed data or cached fallback
    # ------------------------------------------------------------------

    def read_daily_stats(self) -> Dict[str, Any]:
        """Read tradelocker_bot/logs/daily_stats.json."""
        return self._read_json_file(self._settings.daily_stats_file, "daily_stats")

    def read_positions(self) -> Dict[str, Any]:
        """Read tradelocker_bot/logs/active_positions.json."""
        return self._read_json_file(self._settings.positions_file, "positions")

    def read_adaptive_config(self) -> Dict[str, Any]:
        """Read tradelocker_bot/logs/adaptive_config.json."""
        return self._read_json_file(self._settings.adaptive_config_file, "adaptive_config")

    def read_trade_features(self) -> List[Dict[str, Any]]:
        """Read tradelocker_bot/logs/trade_features.jsonl."""
        return self._read_jsonl_file(self._settings.trade_features_file, "trade_features")

    def read_today_journal(self) -> List[Dict[str, Any]]:
        """Read current-day journal file (journal_YYYY-MM-DD.jsonl)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self._settings.journal_dir / f"journal_{today}.jsonl"
        return self._read_jsonl_file(path, f"journal_{today}")

    def read_all_journals(self) -> List[Dict[str, Any]]:
        """Read all available journal files, sorted by date."""
        journal_dir = self._settings.journal_dir
        if not journal_dir.exists():
            return self._get_cached("all_journals", [])

        entries: List[Dict[str, Any]] = []
        try:
            files = sorted(journal_dir.glob("journal_*.jsonl"))
            for f in files:
                file_entries = self._read_jsonl_file(f, f"journal_{f.stem}")
                entries.extend(file_entries)
        except Exception as e:
            logger.warning(f"Error reading journal directory: {e}")
            return self._get_cached("all_journals", [])

        self._cache["all_journals"] = entries
        return entries

    def read_today_bot_log(self) -> str:
        """Read current-day bot log file (bot_YYYY-MM-DD.log)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self._settings.logs_dir / f"bot_{today}.log"
        return self._read_text_file(path, f"bot_log_{today}")

    def read_all_bot_logs(self) -> str:
        """Read all bot log files concatenated."""
        logs_dir = self._settings.logs_dir
        if not logs_dir.exists():
            return self._get_cached("all_bot_logs", "")

        content_parts: List[str] = []
        try:
            files = sorted(logs_dir.glob("bot_*.log"))
            for f in files:
                text = self._read_text_file(f, f"bot_log_{f.stem}")
                if text:
                    content_parts.append(text)
        except Exception as e:
            logger.warning(f"Error reading bot log directory: {e}")
            return self._get_cached("all_bot_logs", "")

        result = "\n".join(content_parts)
        self._cache["all_bot_logs"] = result
        return result

    # ------------------------------------------------------------------
    # File modification time tracking (for bot offline detection)
    # ------------------------------------------------------------------

    def get_latest_file_mtime(self) -> Optional[float]:
        """
        Return the most recent modification time across all monitored files.
        Used for bot-offline detection (Req 14.1).
        """
        monitored_paths = [
            self._settings.daily_stats_file,
            self._settings.positions_file,
            self._settings.adaptive_config_file,
        ]
        # Add today's journal and log
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        monitored_paths.append(self._settings.journal_dir / f"journal_{today}.jsonl")
        monitored_paths.append(self._settings.logs_dir / f"bot_{today}.log")

        latest: Optional[float] = None
        for path in monitored_paths:
            try:
                if path.exists():
                    mtime = path.stat().st_mtime
                    if latest is None or mtime > latest:
                        latest = mtime
            except (OSError, PermissionError):
                continue

        return latest

    def any_file_ever_existed(self) -> bool:
        """Check if any monitored file has ever existed (for initializing state, Req 14.5)."""
        paths_to_check = [
            self._settings.daily_stats_file,
            self._settings.positions_file,
            self._settings.adaptive_config_file,
            self._settings.trade_features_file,
        ]
        # Check journal dir for any file
        if self._settings.journal_dir.exists():
            try:
                if any(self._settings.journal_dir.glob("journal_*.jsonl")):
                    return True
            except OSError:
                pass

        for p in paths_to_check:
            try:
                if p.exists():
                    return True
            except OSError:
                continue

        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_json_file(self, path: Path, cache_key: str) -> Dict[str, Any]:
        """Read a JSON file safely, returning cached value on failure."""
        if not path.exists():
            # Req 13.1: treat as empty result set
            return self._get_cached(cache_key, {})

        try:
            # Req 1.3: open in read-only mode
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return self._get_cached(cache_key, {})

            data = json.loads(content)
            self._cache[cache_key] = data
            self._file_mtimes[cache_key] = time.time()
            return data

        except json.JSONDecodeError as e:
            # Req 13.4: skip malformed, return last good
            logger.warning(f"Malformed JSON in {path}: {e}")
            return self._get_cached(cache_key, {})
        except (OSError, PermissionError) as e:
            # Req 13.5: retain last values on lock/unreadable
            logger.warning(f"Cannot read {path}: {e}")
            return self._get_cached(cache_key, {})

    def _read_jsonl_file(self, path: Path, cache_key: str) -> List[Dict[str, Any]]:
        """Read a JSONL file safely, skipping malformed lines (Req 13.4)."""
        if not path.exists():
            return self._get_cached(cache_key, [])

        try:
            records: List[Dict[str, Any]] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            records.append(record)
                    except json.JSONDecodeError:
                        # Req 8.8, 13.4: skip invalid line, continue
                        continue

            self._cache[cache_key] = records
            self._file_mtimes[cache_key] = time.time()
            return records

        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot read {path}: {e}")
            return self._get_cached(cache_key, [])

    def _read_text_file(self, path: Path, cache_key: str) -> str:
        """Read a plain text file safely."""
        if not path.exists():
            return self._get_cached(cache_key, "")

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self._cache[cache_key] = content
            self._file_mtimes[cache_key] = time.time()
            return content

        except (OSError, PermissionError) as e:
            logger.warning(f"Cannot read {path}: {e}")
            return self._get_cached(cache_key, "")

    def _get_cached(self, key: str, default: Any) -> Any:
        """Return cached value or default."""
        return self._cache.get(key, default)
