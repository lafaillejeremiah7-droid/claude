"""Pytest + Hypothesis configuration for the dashboard test suite.

Makes the ``dashboard`` package importable regardless of the invocation cwd and
registers a hypothesis profile that runs at least 100 examples per property.
"""
import sys
from pathlib import Path

from hypothesis import settings

# tradelocker_bot/ (the bot root) must be importable so `dashboard...` resolves.
BOT_ROOT = Path(__file__).resolve().parents[2]
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

# >= 100 examples per property test (design Testing Strategy).
settings.register_profile("dashboard", max_examples=100, deadline=None)
settings.load_profile("dashboard")
