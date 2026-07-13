"""
Download REAL 1-minute XAUUSD data from Forexite for Jan 1 2025 -> today
(Jul 13 2026), parse, and save to a parquet for backtesting/optimization.
Forexite URL: http://www.forexite.com/free_forex_quotes/{YYYY}/{MM}/{DDMMYY}.zip
Format per line: XAUUSD,YYYYMMDD,HHMMSS,O,H,L,C  (1-minute bars)
"""
import io
import zipfile
import datetime as dt
from concurrent.futures import ThreadPoolExecutor
import urllib.request

import pandas as pd

START = dt.date(2025, 1, 1)
END = dt.date(2026, 7, 13)


def fetch_day(d):
    dd = d.strftime("%d%m%y")
    url = f"http://www.forexite.com/free_forex_quotes/{d.year}/{d.month:02d}/{dd}.zip"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=30).read()
        if len(raw) < 200:
            return []
        z = zipfile.ZipFile(io.BytesIO(raw))
        rows = []
        for fn in z.namelist():
            for line in z.open(fn).read().decode("utf-8", "ignore").splitlines():
                if line.startswith("XAUUSD,"):
                    p = line.split(",")
                    if len(p) == 7:
                        rows.append(p)
        return rows
    except Exception:
        return []


def main():
    days = []
    d = START
    while d <= END:
        days.append(d)
        d += dt.timedelta(days=1)
    print(f"Downloading {len(days)} days ({START} -> {END}) from Forexite...")

    all_rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for rows in ex.map(fetch_day, days):
            all_rows.extend(rows)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(days)} days processed | rows so far: {len(all_rows)}")

    print(f"Total XAUUSD 1m rows: {len(all_rows)}")
    if not all_rows:
        print("NO DATA — aborting.")
        return

    df = pd.DataFrame(all_rows, columns=["sym", "D", "Tm", "o", "h", "l", "c"])
    for col in ["o", "h", "l", "c"]:
        df[col] = df[col].astype(float)
    df["dt"] = pd.to_datetime(df["D"] + df["Tm"], format="%Y%m%d%H%M%S", errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt").drop_duplicates("dt")
    df = df[["dt", "o", "h", "l", "c"]].reset_index(drop=True)
    df.to_parquet("data/xau_2025_2026.parquet")
    print(f"Saved data/xau_2025_2026.parquet")
    print(f"  range: {df['dt'].min()} -> {df['dt'].max()}")
    print(f"  bars: {len(df)}")
    print(f"  price range: ${df['c'].min():.2f} -> ${df['c'].max():.2f}")


if __name__ == "__main__":
    main()
