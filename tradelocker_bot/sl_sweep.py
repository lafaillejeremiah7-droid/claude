"""
Sweep the SL multiplier (x ATR) to find the true optimal SL/TP distance for
maximum dollar profit, using the live $60-risk-cap position sizing formula
against a full year of real 1-minute XAUUSD data.

TPs scale with the SL distance (1R/2R/3R), so R:R stays fixed at 1:1/2:1/3:1
regardless of how wide/tight the SL is set.
"""
import fullyear_backtest as fb
import pandas as pd


def dollar_pnl(r, sl_dist, entry, equity=5000.0):
    contract = 100.0
    leverage = 10.0
    loss_per_lot = contract * sl_dist
    lots = min(0.12, 60.0 / loss_per_lot)
    lots = max(0.01, round(int(lots * 100) / 100, 2))
    margin = lots * contract * entry / leverage
    if margin > equity * 0.95:
        lots = max(0.01, round(int((equity * 0.95 * leverage / (contract * entry)) * 100) / 100, 2))
    risk = lots * loss_per_lot
    return r * risk, risk


if __name__ == "__main__":
    df1m = fb.load_data()
    print(f"Data: {len(df1m)} bars | {df1m.index.min()} -> {df1m.index.max()}")
    print()

    sl_mults = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    header = f"{'SL_MULT':>8} {'SIG':>5} {'WR%':>6} {'TOT_R':>8} {'AVG_R':>7} {'AVG_$':>8} {'TOT_$':>10} {'MAXDD_$':>9} {'AVG_RISK':>9}"
    print(header)
    print("-" * len(header))

    best = None
    all_rows = []
    for sm in sl_mults:
        res = fb.backtest(df1m, "1h", "15m", "1m", exit_atr_tf="15m", return_signals=True, sl_mult=sm)
        if not res or res["signals"] == 0:
            print(f"{sm:>8.2f}  no signals")
            continue
        recs = res["records"]
        rows = []
        for dt, r, sd, e in recs:
            pnl, risk = dollar_pnl(r, sd, e)
            rows.append({"dt": pd.Timestamp(dt), "pnl": pnl, "risk": risk})
        d = pd.DataFrame(rows).set_index("dt").sort_index()
        eq = 5000 + d["pnl"].cumsum()
        peak = eq.cummax()
        dd = (peak - eq).max()
        total_d = d["pnl"].sum()
        avg_d = d["pnl"].mean()
        avg_risk = d["risk"].mean()
        wr = res["win_rate"]
        print(f"{sm:>8.2f} {res['signals']:>5} {wr:>5.1f}% {res['total_r']:>+8.2f} "
              f"{res['avg_r']:>+7.3f} ${avg_d:>7.2f} ${total_d:>9.2f} ${dd:>8.2f} ${avg_risk:>8.2f}")
        all_rows.append((sm, total_d, avg_d, dd, wr, res["signals"]))
        if best is None or total_d > best[1]:
            best = (sm, total_d, avg_d, dd, wr, res["signals"])

    print()
    print("=" * len(header))
    print(f"BEST BY TOTAL $: sl_mult={best[0]} | total=${best[1]:.2f} | avg/signal=${best[2]:.2f} | "
          f"maxDD=${best[3]:.2f} | WR={best[4]:.1f}% | signals={best[5]}")
    print("=" * len(header))
