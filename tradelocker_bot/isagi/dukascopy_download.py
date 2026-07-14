"""
Dukascopy downloader — real XAUUSD ticks -> 1-min bars WITH volume + tick-count,
for the ISAGI session window (12-17 UTC) across 2025-01 .. 2026-07.
Resumable (skips done days), robust retries, saves incrementally to parquet.
Gives us: real volume (V_dens) + sub-minute capability (ticks cached per day).
"""
import requests, lzma, struct, time, os
import numpy as np, pandas as pd
from datetime import date, timedelta

import sys
OUT = "data/xau_dukas_session_1m.parquet"
TICKDIR = "data/dukas_ticks"
os.makedirs(TICKDIR, exist_ok=True)
HOURS = [11, 12, 13, 14, 15, 16, 17]   # 12-17 UTC session + 1 warmup hour
_a = sys.argv
START = pd.Timestamp(_a[1]).date() if len(_a) > 1 else date(2025, 1, 1)
END = pd.Timestamp(_a[2]).date() if len(_a) > 2 else date(2026, 7, 13)


def fetch_hour(y, m0, d, h, tries=5):
    url = f"https://datafeed.dukascopy.com/datafeed/XAUUSD/{y}/{m0:02d}/{d:02d}/{h:02d}h_ticks.bi5"
    for _ in range(tries):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=90)
            if r.status_code == 200:
                if len(r.content) <= 20:
                    return b""      # market closed -> no ticks
                return lzma.decompress(r.content)
        except Exception:
            pass
        time.sleep(1.0)
    return None


def hour_to_bars(raw, y, m, d, h):
    if not raw:
        return None
    n = len(raw) // 20
    a = np.array(struct.unpack(">" + "iiiff" * n, raw)).reshape(n, 5)
    ms = a[:, 0]; price = (a[:, 1] + a[:, 2]) / 2 / 1000.0; vol = a[:, 3] + a[:, 4]
    base = pd.Timestamp(y, m, d, h)
    df = pd.DataFrame({"dt": base + pd.to_timedelta(ms, unit="ms"),
                       "price": price, "vol": vol})
    df["minute"] = df["dt"].dt.floor("min")
    bars = df.groupby("minute").agg(o=("price", "first"), h=("price", "max"),
                                    l=("price", "min"), c=("price", "last"),
                                    vol=("vol", "sum"), ticks=("price", "size"))
    return bars.reset_index().rename(columns={"minute": "dt"})


def main():
    done_days = set()
    if os.path.exists(OUT):
        ex = pd.read_parquet(OUT)
        done_days = set(pd.to_datetime(ex["dt"]).dt.date.unique())
        allbars = [ex]
        print(f"resuming: {len(done_days)} days already saved, {len(ex)} bars")
    else:
        allbars = []
    d = START; nday = 0; t0 = time.time()
    while d <= END:
        if d.weekday() >= 5 or d in done_days:   # skip weekends / done
            d += timedelta(days=1); continue
        m0 = d.month - 1  # dukascopy 0-indexed month
        day_bars = []
        for h in HOURS:
            raw = fetch_hour(d.year, m0, d.day, h)
            b = hour_to_bars(raw, d.year, d.month, d.day, h)
            if b is not None and len(b):
                day_bars.append(b)
        if day_bars:
            allbars.append(pd.concat(day_bars, ignore_index=True))
        nday += 1
        if nday % 5 == 0:
            pd.concat(allbars, ignore_index=True).to_parquet(OUT)
            el = time.time() - t0
            print(f"  {d} | days done {nday} | bars {sum(len(x) for x in allbars)} | {el:.0f}s", flush=True)
        d += timedelta(days=1)
    final = pd.concat(allbars, ignore_index=True).drop_duplicates("dt").sort_values("dt")
    final.to_parquet(OUT)
    print(f"DONE: {len(final)} 1-min session bars w/ volume, {final['dt'].min()} -> {final['dt'].max()}")


if __name__ == "__main__":
    main()
