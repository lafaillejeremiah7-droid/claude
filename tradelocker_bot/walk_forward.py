"""
Walk-forward test: does periodically re-optimizing beat a fixed config on real
XAU/USD? Re-optimize usage params (allocation/tilt/risk) on a trailing 90 days,
trade the next 30 days with them, roll across Jan2025->Jul2026. Compare to a
STATIC config (only-prior-data) and the current 10/20/70 baseline.

All three use ONLY past information at each point -> a fair test of whether
re-fitting more often actually helps.
"""
import numpy as np
import pandas as pd
import tsai_optimize as T
import phase4_usage as P4

BIG = 1e9


def pnl_idx(pc, idx, a1, a2, tilt, risk_cap):
    """Return per-signal pnl + dt for a subset, so blocks can be stitched into
    one continuous equity curve for a TRUE drawdown measurement."""
    if len(idx) == 0:
        return np.array([]), np.array([])
    strong = pc["strong"][idx]
    A1 = np.where(strong, max(0.02, a1 - tilt * 0.5), a1)
    A2 = np.where(strong, max(0.02, a2 - tilt * 0.5), a2)
    A3 = 1.0 - A1 - A2
    code = pc["code"][idx]; t1 = pc["tp1_R"][idx]; t2 = pc["tp2_R"][idx]; t3 = pc["tp3_R"][idx]
    r = np.select([code == 4, code == 2, code == 3, code == 1],
                  [A1*t1 + A2*t2 + A3*t3, A1*t1 + A2*t2 + A3*t1, A1*t1, -1.0],
                  default=pc["timeout_r"][idx])
    sld = pc["sl_dist"][idx]; ent = pc["entries"][idx]
    lpl = 100.0 * sld
    lots = np.minimum(0.12, risk_cap / lpl)
    lots = np.maximum(0.01, np.floor(lots * 100) / 100)
    margin = lots * 100.0 * ent / 10.0
    over = margin > T.EQUITY0 * 0.95
    if over.any():
        alt = np.maximum(0.01, np.floor((T.EQUITY0*0.95*10.0/(100.0*ent))*100)/100)
        lots = np.where(over, alt, lots)
    pnl = r * (lots * lpl)
    return pnl, pc["dt_sorted"][idx]


def eval_idx(pc, idx, a1, a2, tilt, risk_cap, weeks):
    if len(idx) < 15:
        return None
    pnl, dts = pnl_idx(pc, idx, a1, a2, tilt, risk_cap)
    dc = pd.factorize(pd.to_datetime(dts).date)[0]
    day_sums = np.bincount(dc, weights=pnl, minlength=dc.max()+1 if len(dc) else 1)
    eq = T.EQUITY0 + np.cumsum(day_sums)
    dd_total = float(np.max(np.maximum.accumulate(eq) - eq)) if len(eq) else 0.0
    dd_day = float(-day_sums.min()) if len(day_sums) and day_sums.min() < 0 else 0.0
    r_pos = (pnl > 0).sum(); r_neg = (pnl < 0).sum()
    return {"total": float(pnl.sum()), "wk": float(pnl.sum())/weeks if weeks else 0,
            "wr": r_pos/max(1, r_pos+r_neg), "dd_total": dd_total, "dd_day": dd_day, "n": len(idx)}


def stitched_dd(pnls, dts):
    """True max total DD + worst daily DD across the stitched equity curve."""
    if len(pnls) == 0:
        return 0.0, 0.0
    o = np.argsort(dts)
    p = pnls[o]; d = dts[o]
    eq = T.EQUITY0 + np.cumsum(p)
    dd_total = float(np.max(np.maximum.accumulate(eq) - eq))
    dc = pd.factorize(pd.to_datetime(d).date)[0]
    day_sums = np.bincount(dc, weights=p, minlength=dc.max()+1)
    dd_day = float(-day_sums.min()) if day_sums.min() < 0 else 0.0
    return dd_total, dd_day


def optimize_window(pc, idx, weeks, n=1500, seed=0):
    """Find best usage params on a window (maximize total, gate-ish via DD<350)."""
    rng = np.random.default_rng(seed)
    best = None; best_val = -1e18
    for _ in range(n):
        a1 = rng.uniform(0.05, 0.35); a2 = rng.uniform(0.10, 0.45)
        if a1 + a2 > 0.75:
            continue
        tilt = rng.uniform(0.0, 0.30); risk = rng.uniform(15.0, 35.0)
        m = eval_idx(pc, idx, a1, a2, tilt, risk, weeks)
        if m is None or m["dd_total"] >= 350 or m["dd_day"] >= 200:
            continue
        if m["total"] > best_val:
            best_val = m["total"]; best = (a1, a2, tilt, risk)
    return best


def main():
    df = T.load_data()
    c = T.build_candidates(df)
    pc = P4.precompute(c)
    # store sorted dt for subset day-grouping
    pc["dt_sorted"] = c["dt"][pc["order"]]
    dts = pd.to_datetime(pc["dt_sorted"])
    n = len(dts)
    idx_all = np.arange(n)

    start = dts.min().normalize()
    end = dts.max().normalize()
    train_days = 90
    test_days = 30

    total_weeks = 0.0; blocks = 0
    wf_p = []; wf_d = []; st_p = []; st_d = []; bl_p = []; bl_d = []

    # Two fair comparisons:
    #  - risk-capped walk-forward (risk fixed at $26.7, same as static -> pure adaptation test)
    #  - free walk-forward (risk up to $35 -> shows how much is just bigger size)
    ST = (0.30, 0.40, 0.19, 26.7)   # Phase-4 winner, prior-fit, fixed
    BL = (0.10, 0.20, 0.00, 26.0)   # current baseline

    t = start + pd.Timedelta(days=train_days)
    while t + pd.Timedelta(days=test_days) <= end + pd.Timedelta(days=1):
        tr = idx_all[(dts >= t - pd.Timedelta(days=train_days)) & (dts < t)]
        te = idx_all[(dts >= t) & (dts < t + pd.Timedelta(days=test_days))]
        wks = test_days / 7.0
        if len(te) >= 10 and len(tr) >= 30:
            bp = optimize_window(pc, tr, train_days/7.0, n=1500, seed=blocks)
            if bp is not None:
                # RISK-MATCHED: apply the trained allocation/tilt but hold risk at $26.7
                p, d = pnl_idx(pc, te, bp[0], bp[1], bp[2], 26.7)
                wf_p.append(p); wf_d.append(d)
            sp, sd = pnl_idx(pc, te, *ST); st_p.append(sp); st_d.append(sd)
            bpn, bd = pnl_idx(pc, te, *BL); bl_p.append(bpn); bl_d.append(bd)
            total_weeks += wks; blocks += 1
        t += pd.Timedelta(days=test_days)

    def agg(ps, ds):
        p = np.concatenate(ps) if ps else np.array([])
        d = np.concatenate(ds) if ds else np.array([])
        ddt, ddd = stitched_dd(p, d)
        return float(p.sum()), ddt, ddd

    wf_tot, wf_ddt, wf_ddd = agg(wf_p, wf_d)
    st_tot, st_ddt, st_ddd = agg(st_p, st_d)
    bl_tot, bl_ddt, bl_ddd = agg(bl_p, bl_d)

    print(f"Walk-forward test across {blocks} monthly blocks — RISK-MATCHED at $26.7 "
          f"(pure adaptation test), TRUE stitched drawdown:\n")
    print(f"{'Method':34} {'$/week':>9} {'total$':>10} {'maxDD':>8} {'dayDD':>8} {'gates':>6}")
    print("-" * 78)
    def gate(ddt, ddd):
        return "PASS" if (ddt < 350 and ddd < 200) else "FAIL"
    print(f"{'A) WALK-FWD (re-opt alloc monthly)':34} ${wf_tot/total_weeks:>7.2f} ${wf_tot:>9.0f} "
          f"${wf_ddt:>6.0f} ${wf_ddd:>6.0f} {gate(wf_ddt, wf_ddd):>6}")
    print(f"{'B) STATIC 30/40/31 (fixed)':34} ${st_tot/total_weeks:>7.2f} ${st_tot:>9.0f} "
          f"${st_ddt:>6.0f} ${st_ddd:>6.0f} {gate(st_ddt, st_ddd):>6}")
    print(f"{'C) BASELINE 10/20/70 (current)':34} ${bl_tot/total_weeks:>7.2f} ${bl_tot:>9.0f} "
          f"${bl_ddt:>6.0f} ${bl_ddd:>6.0f} {gate(bl_ddt, bl_ddd):>6}")
    d = wf_tot/total_weeks - st_tot/total_weeks
    print(f"\nWalk-forward vs static (risk-matched): ${d:+.2f}/wk "
          f"({'HELPS' if d > 5 else 'NO MEANINGFUL GAIN' if abs(d) <= 5 else 'HURTS'})")


if __name__ == "__main__":
    main()
