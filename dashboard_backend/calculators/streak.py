"""
Streak_Calculator – Derives win/loss streaks from journal CLOSE actions.

Requirement 6:
  - Order all CLOSE actions from Journal_Files ascending by timestamp (UTC).
  - Ties broken by source file date first, then line position.
  - Classify as win (is_win=true) or loss (is_win=false).
  - Exclude entries with missing/non-boolean is_win.
  - Current win streak = consecutive wins ending at most recent CLOSE.
  - Current loss streak = consecutive losses ending at most recent CLOSE.
  - If no classified CLOSE actions: both streaks = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class StreakResult:
    """Current win and loss streak values."""
    win_streak: int = 0
    loss_streak: int = 0
    total_wins: int = 0
    total_losses: int = 0


def compute_streaks(journal_entries: List[Dict[str, Any]]) -> StreakResult:
    """
    Compute current win/loss streaks from ordered journal entries.

    Args:
        journal_entries: All journal entries from read_all_journals(),
                        already ordered by file date then line position.

    Returns:
        StreakResult with current streaks and totals.
    """
    # Filter to CLOSE actions only
    close_actions = [
        e for e in journal_entries
        if e.get("action") == "CLOSE"
    ]

    if not close_actions:
        return StreakResult()

    # Sort by timestamp ascending (Req 6.1)
    # Journal entries are already in file-order from read_all_journals,
    # but we sort by timestamp for correctness. Ties preserve insertion order
    # because Python's sort is stable.
    close_actions.sort(key=lambda e: e.get("timestamp", ""))

    # Classify wins/losses (Req 6.2, 6.3)
    classified: List[bool] = []  # True = win, False = loss
    total_wins = 0
    total_losses = 0

    for entry in close_actions:
        is_win = entry.get("is_win")
        # Req 6.3: must be strictly boolean
        if is_win is True:
            classified.append(True)
            total_wins += 1
        elif is_win is False:
            classified.append(False)
            total_losses += 1
        # else: exclude (not boolean)

    if not classified:
        return StreakResult()

    # Compute current streaks (Req 6.4, 6.5)
    win_streak = 0
    loss_streak = 0

    # Walk backwards from most recent
    most_recent = classified[-1]
    if most_recent:
        # Count consecutive wins from end
        for result in reversed(classified):
            if result:
                win_streak += 1
            else:
                break
        # Req 6.5: loss streak = 0 when most recent is win
        loss_streak = 0
    else:
        # Count consecutive losses from end
        for result in reversed(classified):
            if not result:
                loss_streak += 1
            else:
                break
        # Req 6.4: win streak = 0 when most recent is loss
        win_streak = 0

    return StreakResult(
        win_streak=win_streak,
        loss_streak=loss_streak,
        total_wins=total_wins,
        total_losses=total_losses,
    )
