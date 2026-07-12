"""Win/loss streak computation over ordered CLOSE actions (Req 6)."""
from __future__ import annotations

from typing import Iterable, List, NamedTuple


class Streaks(NamedTuple):
    win_streak: int
    loss_streak: int


def compute_streaks(ordered_closes: Iterable[dict]) -> Streaks:
    """Compute (win_streak, loss_streak) from CLOSE actions in chronological order.

    - ``is_win is True`` -> win, ``is_win is False`` -> loss.
    - Any action whose ``is_win`` is absent or non-boolean is EXCLUDED and does
      NOT break the consecutive run.
    - The win streak is the length of the trailing run of consecutive wins ending
      at the most recent classified action (0 if that action is a loss); the loss
      streak is symmetric.
    - No classified actions -> ``(0, 0)``.
    - The two streaks are never both non-zero.
    """
    classified: List[bool] = []
    for action in ordered_closes:
        if not isinstance(action, dict):
            continue
        is_win = action.get("is_win")
        if is_win is True or is_win is False:  # strict boolean check
            classified.append(is_win)

    if not classified:
        return Streaks(0, 0)

    last = classified[-1]
    run = 0
    for value in reversed(classified):
        if value == last:
            run += 1
        else:
            break

    if last is True:
        return Streaks(run, 0)
    return Streaks(0, run)
