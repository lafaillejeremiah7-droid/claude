"""Confidence-score parsing and gate classification (Req 7)."""
from __future__ import annotations

import re
from typing import Optional

from .common import is_number

# "Confidence: <number>/10" allowing surrounding whitespace around the number
# and the slash.
_CONFIDENCE_RE = re.compile(r"Confidence:\s*(\d+(?:\.\d+)?)\s*/\s*10")


def parse_confidence(text: object) -> Optional[float]:
    """Extract a confidence value in ``[0, 10]`` from *text* (Req 7.1, 7.2).

    Returns the parsed float when the text contains a ``Confidence: <n>/10``
    token whose value is within range; otherwise ``None`` ("unavailable"). Never
    fabricates a value and never raises.
    """
    if not isinstance(text, str):
        return None
    match = _CONFIDENCE_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(1))
    except (ValueError, TypeError):
        return None
    if 0.0 <= value <= 10.0:
        return value
    return None


def classify_gate(confidence: Optional[float], gate: Optional[float]) -> str:
    """Classify a confidence value against the gate (Req 7.5).

    ``"met_gate"`` iff ``confidence >= gate``; ``"near_miss"`` iff
    ``confidence < gate``; ``"unknown"`` when either value is unavailable.
    """
    if not is_number(confidence) or not is_number(gate):
        return "unknown"
    return "met_gate" if float(confidence) >= float(gate) else "near_miss"
