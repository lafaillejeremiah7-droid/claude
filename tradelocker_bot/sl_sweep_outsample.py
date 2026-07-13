"""
Out-of-sample check: sweep SL multiplier on a DIFFERENT year (Dec 2022 -> Nov
2023) than the one used to pick the current 1.0x setting (Nov 2023 -> Nov
2024), to make sure the optimum isn't overfit to a single window.
"""
import pandas as pd
import fullyear_backtest as fb
from sl_sweep import dollar_pnl


def load_data_window(start_str, end_str):
    df1 = pd.read_parquet("data/xau_train.parquet")
    df2 = pd.read_parquet("data/xau_test.parquet")
    df = pd.concat([df1, df2], ignore_index=True)
    dt = pd.to_datetime(
        df["Date"].astype(str).str.zfill(8) + df["Time"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S", errors="coerce")
    df = df.assign(dt=dt).dropna(subset=["dt"])
    df = df[["dt", "Open", "High", "Low", "Close"]].rename(
        columns={"Open": "o", "High": "h", "Low": "l", "Close": "c"})
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    return df.loc[start_str:end_str]


if __name__ == "__main__":
    df1m = load_data_window("2022-12-01", "2023-11-30")
    print(f"OUT-OF-SAMPLE window: {df1m.index.min()} -> {df1m.index.max()} | {len(df1m)} bars")
    print()
    header = f"{'SL_MULT':>8} {'SIG':>5} {'WR%':>6} {'TOT_$':>10} {'AVG_$':>8} {'MAXDD_$':>9}"
    print(header)
    print("-" * len(header))
    for sm in [0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0]:
        res = fb.backtest(df1m, "1h", "15m", "1m", exit_atr_tf="15m", return_signals=True, sl_mult=sm)
        if not res or res["signals"] == 0:
            print(f"{sm:>8.2f}  no signals")
            continue
        recs = res["records"]
        rows = [{"dt": pd.Timestamp(dt), "pnl": dollar_pnl(r, sd, e)[0]} for dt, r, sd, e in recs]
        d = pd.DataFrame(rows).set_index("dt").sort_index()
        eq = 5000 + d["pnl"].cumsum()
        dd = (eq.cummax() - eq).max()
        print(f"{sm:>8.2f} {res['signals']:>5} {res['win_rate']:>5.1f}% "
              f"${d['pnl'].sum():>9.2f} ${d['pnl'].mean():>7.2f} ${dd:>8.2f}")
