"""
ISAGI News Decoupler feed (Q4 spec):
  - Clean Forex Factory JSON scraper feed (faireconomy).
  - Filter OUT all non-USD events.
  - Filter OUT Low and Medium impact (keep High only).
  - If title matches CPI / NFP / FOMC / GDP -> map exact timestamp to a
    Forced Blackout Window: freeze 2 min BEFORE to 15 min AFTER the release.

Live: pulls the real FF JSON. Backtest: builds the known 2025-26 high-impact USD
schedule for those 4 event families (NFP first-Friday, CPI/GDP monthly at 8:30 ET,
FOMC on the published 2025-26 meeting dates at 14:00 ET).
"""
import requests
import pandas as pd

FF_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
BLACKOUT_KEYS = ("CPI", "NFP", "NON-FARM", "NONFARM", "FOMC", "GDP", "FEDERAL FUNDS")
PRE_MIN, POST_MIN = 2, 15


def _is_target(title):
    t = title.upper()
    return any(k in t for k in BLACKOUT_KEYS)


def fetch_live_blackouts():
    """Live FF feed -> list of (start_utc, end_utc) blackout windows (USD/High/target)."""
    r = requests.get(FF_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    out = []
    for e in r.json():
        if e.get("country") != "USD" or e.get("impact") != "High":
            continue
        if not _is_target(e.get("title", "")):
            continue
        ts = pd.to_datetime(e["date"], utc=True)   # ISO w/ tz -> UTC
        out.append((ts - pd.Timedelta(minutes=PRE_MIN), ts + pd.Timedelta(minutes=POST_MIN),
                    e.get("title")))
    return out


# --- 2025-26 known high-impact USD schedule for the backtest ---
FOMC_2025_26 = [  # statement day, 14:00 ET
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
    "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
]


def _et_to_utc(day, hh, mm):
    # US Eastern -> UTC. DST ~ Mar 2nd Sun to Nov 1st Sun. Approx: -4h (EDT) else -5h (EST).
    d = pd.Timestamp(day)
    edt = (pd.Timestamp(d.year, 3, 8) <= d) & (d < pd.Timestamp(d.year, 11, 1))
    off = 4 if edt else 5
    return pd.Timestamp(d.year, d.month, d.day, hh, mm) + pd.Timedelta(hours=off)


def backtest_blackouts(start="2025-01-01", end="2026-07-13"):
    """Known CPI/NFP/GDP (8:30 ET) + FOMC (14:00 ET) -> blackout windows (UTC)."""
    out = []
    months = pd.date_range(start, end, freq="MS")
    for m in months:
        # NFP: first Friday 8:30 ET
        d = m
        while d.weekday() != 4:
            d += pd.Timedelta(days=1)
        out.append((_et_to_utc(d, 8, 30), "NFP"))
        # CPI: ~mid-month (12th) 8:30 ET ; GDP: ~end-month (27th) 8:30 ET
        out.append((_et_to_utc(m.replace(day=12), 8, 30), "CPI"))
        out.append((_et_to_utc(m.replace(day=27), 8, 30), "GDP"))
    for fd in FOMC_2025_26:
        out.append((_et_to_utc(fd, 14, 0), "FOMC"))
    wins = []
    for ts, tag in out:
        if pd.Timestamp(start) <= ts <= pd.Timestamp(end):
            wins.append((ts - pd.Timedelta(minutes=PRE_MIN), ts + pd.Timedelta(minutes=POST_MIN), tag))
    return sorted(wins)


class NewsDecoupler:
    def __init__(self, windows):
        self.win = windows  # list of (start, end, tag) UTC-naive or UTC

    def is_blackout(self, ts):
        ts = pd.Timestamp(ts)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        for s, e, _ in self.win:
            s2 = s.tz_localize(None) if getattr(s, "tz", None) else s
            e2 = e.tz_localize(None) if getattr(e, "tz", None) else e
            if s2 <= ts <= e2:
                return True
        return False

    def in_exploitation(self, ts):
        """minute 16-120 after a release (post-blackout trend-exploitation state)."""
        ts = pd.Timestamp(ts)
        for s, e, _ in self.win:
            e2 = e.tz_localize(None) if getattr(e, "tz", None) else e
            if e2 < ts <= e2 + pd.Timedelta(minutes=105):
                return True
        return False


if __name__ == "__main__":
    print("LIVE FF feed (USD/High/target events this week):")
    try:
        for s, e, t in fetch_live_blackouts():
            print(f"  {t}: blackout {s} -> {e}")
    except Exception as ex:
        print("  live fetch err:", ex)
    bt = backtest_blackouts()
    print(f"\nBACKTEST schedule 2025-26: {len(bt)} high-impact USD blackout windows")
    for s, e, tag in bt[:6]:
        print(f"  {tag}: {s} -> {e} UTC")
    nd = NewsDecoupler(bt)
    print("  sample is_blackout(2025-06-06 12:31 UTC=NFP):", nd.is_blackout("2025-06-06 12:31"))
