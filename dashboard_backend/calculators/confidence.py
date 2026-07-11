"""
Confidence score extraction from journal and bot log files.

Requirement 7:
  - Extract Confidence_Score (0-10) from journal entry_reasons matching
    "Confidence: <value>/10".
  - Extract near-miss scores from bot log "NEAR-MISS" lines.
  - Extract approved-trade scores from "TRADE SIGNAL APPROVED" lines.
  - Read Confidence_Gate from adaptive_config min_confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Pattern: "Confidence: 8.5/10" or "Confidence: 9/10"
_CONFIDENCE_PATTERN = re.compile(r"Confidence:\s*([\d.]+)\s*/\s*10", re.IGNORECASE)

# Bot log patterns
_NEAR_MISS_PATTERN = re.compile(
    r"NEAR-MISS.*?(?:confidence|score)[:\s]*([\d.]+)",
    re.IGNORECASE,
)
_APPROVED_PATTERN = re.compile(
    r"TRADE SIGNAL APPROVED.*?(?:confidence|score)[:\s]*([\d.]+)",
    re.IGNORECASE,
)
# Alternative: confidence in the same line as NEAR-MISS or APPROVED
_NEAR_MISS_CONF_ALT = re.compile(
    r"NEAR-MISS.*?(\d+\.?\d*)\s*/\s*10",
    re.IGNORECASE,
)
_APPROVED_CONF_ALT = re.compile(
    r"TRADE SIGNAL APPROVED.*?(\d+\.?\d*)\s*/\s*10",
    re.IGNORECASE,
)


@dataclass
class ConfidenceEntry:
    """A single confidence score entry."""
    symbol: str = ""
    score: Optional[float] = None
    gate: Optional[float] = None
    is_approved: bool = False
    is_near_miss: bool = False
    timestamp: str = ""
    source: str = ""  # "journal" or "bot_log"


@dataclass
class ConfidenceData:
    """Aggregated confidence data."""
    gate: Optional[float] = None  # min_confidence from adaptive_config
    entries: List[ConfidenceEntry] = field(default_factory=list)
    gate_unavailable: bool = False


def extract_confidence_gate(adaptive_config: Dict[str, Any]) -> Optional[float]:
    """
    Extract the Confidence_Gate (min_confidence) from adaptive config.
    Returns None if unavailable (Req 7.4).
    """
    if not adaptive_config:
        return None

    min_conf = adaptive_config.get("min_confidence")
    if min_conf is None:
        return None

    try:
        value = float(min_conf)
        if 0 <= value <= 10:
            return value
        return None
    except (TypeError, ValueError):
        return None


def extract_from_journal(
    journal_entries: List[Dict[str, Any]],
    gate: Optional[float] = None,
) -> List[ConfidenceEntry]:
    """
    Extract confidence scores from journal entry_reasons (Req 7.1).
    """
    entries: List[ConfidenceEntry] = []

    for record in journal_entries:
        action = record.get("action", "")
        entry_reasons = record.get("entry_reasons", [])
        symbol = record.get("symbol", "")
        timestamp = record.get("timestamp", "")

        # entry_reasons can be a list of strings or a single string
        if isinstance(entry_reasons, str):
            entry_reasons = [entry_reasons]
        elif not isinstance(entry_reasons, list):
            continue

        score: Optional[float] = None
        for reason in entry_reasons:
            if not isinstance(reason, str):
                continue
            match = _CONFIDENCE_PATTERN.search(reason)
            if match:
                try:
                    val = float(match.group(1))
                    if 0 <= val <= 10:
                        score = val
                        break
                except (ValueError, TypeError):
                    continue

        # Determine if approved or near-miss based on gate
        is_approved = False
        is_near_miss = False
        if score is not None and gate is not None:
            is_approved = score >= gate
            is_near_miss = score < gate

        entry = ConfidenceEntry(
            symbol=symbol,
            score=score,
            gate=gate,
            is_approved=is_approved,
            is_near_miss=is_near_miss,
            timestamp=timestamp,
            source="journal",
        )
        entries.append(entry)

    return entries


def extract_from_bot_log(
    log_content: str,
    gate: Optional[float] = None,
) -> List[ConfidenceEntry]:
    """
    Extract confidence scores from bot log lines (Req 7.2).
    Near-miss and approved trade patterns.
    """
    entries: List[ConfidenceEntry] = []

    if not log_content:
        return entries

    for line in log_content.splitlines():
        score: Optional[float] = None
        is_near_miss = False
        is_approved = False

        # Check for NEAR-MISS
        if "NEAR-MISS" in line.upper() or "NEAR_MISS" in line.upper():
            match = _NEAR_MISS_PATTERN.search(line) or _NEAR_MISS_CONF_ALT.search(line)
            if match:
                try:
                    val = float(match.group(1))
                    if 0 <= val <= 10:
                        score = val
                        is_near_miss = True
                except (ValueError, TypeError):
                    pass

        # Check for TRADE SIGNAL APPROVED
        elif "TRADE SIGNAL APPROVED" in line.upper():
            match = _APPROVED_PATTERN.search(line) or _APPROVED_CONF_ALT.search(line)
            if match:
                try:
                    val = float(match.group(1))
                    if 0 <= val <= 10:
                        score = val
                        is_approved = True
                except (ValueError, TypeError):
                    pass

        if is_near_miss or is_approved:
            # Try to extract timestamp from beginning of line
            timestamp = _extract_log_timestamp(line)
            # Try to extract symbol
            symbol = _extract_symbol(line)

            entries.append(ConfidenceEntry(
                symbol=symbol,
                score=score,
                gate=gate,
                is_approved=is_approved,
                is_near_miss=is_near_miss,
                timestamp=timestamp,
                source="bot_log",
            ))

    return entries


def _extract_log_timestamp(line: str) -> str:
    """Try to extract ISO timestamp from start of log line."""
    # Common patterns: "2024-01-15 14:30:00" or "2024-01-15T14:30:00"
    ts_pattern = re.compile(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})")
    match = ts_pattern.search(line[:30])
    return match.group(1) if match else ""


def _extract_symbol(line: str) -> str:
    """Try to extract instrument symbol from log line."""
    for sym in ("BTCUSD", "XAUUSD"):
        if sym in line.upper():
            return sym
    return ""
