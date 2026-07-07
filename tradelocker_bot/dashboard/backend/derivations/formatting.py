"""Presentation formatting helpers (Req 3.2, 4.1, 4.5, 5.1, 4.7, 17.3, 17.4)."""
from __future__ import annotations


def format_money(value: float) -> str:
    """Format a finite numeric value with EXACTLY two decimal places.

    ``float(format_money(v)) == round(v, 2)``. Negative zero is normalised to
    ``0.00`` so a rounded-to-zero value never renders with a leading minus.
    """
    rounded = round(float(value), 2)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.2f}"


def format_win_rate(value: float) -> str:
    """Format a percentage value with EXACTLY one decimal place."""
    rounded = round(float(value), 1)
    if rounded == 0:
        rounded = 0.0
    return f"{rounded:.1f}"


def sign_color(value: float) -> str:
    """Map a signed number to a colour class.

    ``> 0`` -> ``"green"``, ``< 0`` -> ``"red"``, ``== 0`` -> ``"neutral"``.
    Applied identically everywhere a signed number is rendered.
    """
    v = float(value)
    if v > 0:
        return "green"
    if v < 0:
        return "red"
    return "neutral"
