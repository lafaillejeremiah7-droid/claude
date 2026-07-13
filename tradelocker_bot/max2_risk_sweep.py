"""Quick risk sweep for the max-2/day config with exhaustion filter."""
import numpy as np
import pandas as pd
import tsai_optimize as T
import phase4_usage as P4
import phase7_candleEV as P7
import fullyear_backtest as fb


def sim(df, max_day, risk, exhaust_thr):
    c = T.build_candidates(df)
    pc = P4.precompute(c)
    pc["dt_sorted"] = c["dt"][pc["order"]]
    weeks = (df.index.max() - df.index.min()).days / 7.0

    b1h = fb.resample_tf(df, 60)
    atr1h = fb.atr(b1h, 14)
    hi20 = b1h["h"].rolling(20).max()
    lo20 = b1h["l"].rolling(20).min()
    c1h_s = b1h["c"]
    dts_sig = pd.to_datetime(c["dt"])
    h1_df = pd.DataFrame({"dt": b1h.index, "c1h": c1h_s.values, "hi20": hi20.values,
                           "lo20": lo20.values, "atr1h": atr1h.values}).dropna().sort_values("dt")
    m = pd.merge_asof(pd.DataFrame({"dt": dts_sig}).sort_values("dt"), h1_df, on="dt", direction="backward")
    exh = np.zeros(len(c["d"]))
    for i in range(len(c["d"])):
        a = m.iloc[i]["atr1h"]
        if a <= 0 or pd.isna(a):
            continue
        if c["d"][i] == -1:
            exh[i] = (m.iloc[i]["hi20"] - m.iloc[i]["c1h"]) / a
        else:
            exh[i] = (m.iloc[i]["c1h"] - m.iloc[i]["lo20"]) / a
    exh = exh[pc["order"]]

    r = P7.per_signal_r(pc, 0.30, 0.40, 0.19)
    sld = pc["sl_dist"]
    ent = pc["entries"]
    day = pc["day_code"]
    dts = pd.to_datetime(pc["dt_sorted"])
    n = pc["n"]

    eq = 5000.0
    peak = 5000.0
    pnls = np.zeros(n)
    cur = -1
    dpnl = 0.0
    blk = False
    dc = 0
    taken = 0
    lt = pd.Timestamp("2000-01-01")

    for k in range(n):
        if day[k] != cur:
            cur = day[k]; dpnl = 0.0; blk = False; dc = 0
        if blk or dc >= max_day:
            continue
        if (dts[k] - lt).total_seconds() < 3600:
            continue
        if exh[k] >= exhaust_thr:
            continue
        lpl = 100 * sld[k]
        lots = min(0.12, risk / lpl)
        lots = max(0.01, np.floor(lots * 100) / 100)
        pnl = r[k] * lots * lpl
        pnls[k] = pnl
        eq += pnl
        dpnl += pnl
        taken += 1
        dc += 1
        lt = dts[k]
        if eq > peak:
            peak = eq
        if dpnl <= -120:
            blk = True

    eqc = 5000 + np.cumsum(pnls)
    ddt = float(np.max(np.maximum.accumulate(eqc) - eqc))
    dsum = np.bincount(day, weights=pnls, minlength=day.max() + 1)
    ddd = float(-dsum.min()) if dsum.min() < 0 else 0.0
    gt = ddt < 350 and ddd < 200
    w = int((pnls[pnls != 0] > 0).sum())
    l = int((pnls[pnls != 0] < 0).sum())
    wk = pnls.sum() / weeks
    pd_ = taken / (weeks * 5)
    wr = w / max(1, w + l) * 100
    return wk, pd_, wr, ddt, ddd, gt


if __name__ == "__main__":
    print("Max 2/day + exhaust<3x + 60min cooldown — risk sweep:")
    print(f"{'risk':>6} | {'IS $/wk':>9} {'sig/d':>6} {'WR':>5} {'DDt':>5} {'gate':>5}"
          f" | {'OOS $/wk':>9} {'DDt':>5} {'gate':>5}")
    print("-" * 72)
    df_is = T.load_data()
    df_oos = fb.load_data()
    for risk in [20, 25, 28, 30, 33, 35, 40, 45]:
        w1, pd1, wr1, d1, dd1, g1 = sim(df_is, 2, risk, 3.0)
        w2, pd2, wr2, d2, dd2, g2 = sim(df_oos, 2, risk, 3.0)
        print(f"  ${risk:<4} | ${w1:>+8.2f} {pd1:>5.1f}/d {wr1:>4.0f}% ${d1:>4.0f} "
              f"{'PASS' if g1 else 'FAIL':>5} | ${w2:>+8.2f} ${d2:>4.0f} {'PASS' if g2 else 'FAIL':>5}")
