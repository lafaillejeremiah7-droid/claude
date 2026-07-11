"""Tolerant, PURE JSONL parsing helpers (Req 8.4, 8.8, 13.1, 13.4).

The file-touching I/O lives elsewhere; these functions operate on already-read
text/lines so the tolerant parse logic is fully testable. They never raise on
malformed input — malformed records are simply skipped.
"""
from __future__ import annotations

import json
from typing import List, Optional

REQUIRED_FEED_FIELDS = ("timestamp", "action", "symbol", "direction")


def parse_jsonl_line(line: object) -> Optional[dict]:
    """Parse a single JSONL line into a dict, or ``None`` if invalid.

    Non-string input, blank lines, non-JSON text, and JSON that is not an object
    all yield ``None``. Never raises.
    """
    if not isinstance(line, str):
        return None
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def parse_jsonl(blob: object) -> List[dict]:
    """Parse a JSONL blob, returning exactly the successfully parsed object records.

    Malformed / partial / blank lines interleaved with valid ones are skipped.
    Returns an empty list when nothing valid is present. Never raises.
    """
    if not isinstance(blob, str):
        return []
    records: List[dict] = []
    for line in blob.splitlines():
        parsed = parse_jsonl_line(line)
        if parsed is not None:
            records.append(parsed)
    return records


def valid_journal_entries(records: List[dict]) -> List[dict]:
    """Keep only entries with non-empty required feed fields (Req 8.3, 8.4)."""
    out: List[dict] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if all(_non_empty(rec.get(field)) for field in REQUIRED_FEED_FIELDS):
            out.append(rec)
    return out


def _non_empty(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True
