"""Shared pytest fixtures / path setup for the tradelocker_bot test suite."""
import os
import sys

# Ensure the bot package root (tradelocker_bot/) is importable so tests can do
# `import config` and `from modules... import ...` regardless of the CWD.
_BOT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BOT_ROOT not in sys.path:
    sys.path.insert(0, _BOT_ROOT)
