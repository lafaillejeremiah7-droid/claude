"""Small shared helpers used across the pure derivation modules."""
from __future__ import annotations

import math
from typing import Any


def is_number(value: Any) -> bool:
    """Return True iff *value* is a finite real number.

    Booleans are explicitly rejected even though ``bool`` is a subclass of
    ``int`` in Python — a ``True``/``False`` must never be treated as a numeric
    ``pnl`` or price. NaN and infinities are rejected as well.
    """
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False
