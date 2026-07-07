"""Read-only guarantee: a full poll cycle never mutates bot files (Req 1.4)."""
from __future__ import annotations

import os

from dashboard.backend.readers import FileReader
from dashboard.backend.store import SnapshotStore


def _fingerprint(base):
    """Map of relative path -> (bytes, mtime_ns) for every file under base."""
    fp = {}
    for root, _dirs, files in os.walk(base):
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, base)
            st = os.stat(path)
            with open(path, "rb") as fh:
                fp[rel] = (fh.read(), st.st_mtime_ns)
    return fp


def test_full_poll_cycle_leaves_files_unchanged(live_bot_dir):
    before = _fingerprint(live_bot_dir)
    assert before  # fixture actually wrote files

    store = SnapshotStore(reader=FileReader(bot_dir=live_bot_dir, mode="live"), env={})
    # Several full poll cycles (files read repeatedly through the guard).
    for _ in range(5):
        store.refresh()

    after = _fingerprint(live_bot_dir)
    assert set(after.keys()) == set(before.keys()), "no files added/removed"
    for rel, (data, mtime) in before.items():
        assert after[rel][0] == data, f"bytes changed for {rel}"
        assert after[rel][1] == mtime, f"mtime changed for {rel}"

    # The guard recorded zero write/blocked violations during read-only polling.
    assert store.reader.guard.errors == []


def test_paper_poll_cycle_readonly(paper_bot_dir):
    before = _fingerprint(paper_bot_dir)
    store = SnapshotStore(reader=FileReader(bot_dir=paper_bot_dir, mode="paper"), env={})
    for _ in range(3):
        store.refresh()
    after = _fingerprint(paper_bot_dir)
    assert after == before
