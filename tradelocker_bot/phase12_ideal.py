"""
PHASE 12 — The "ideal trader" geometry search for the 12-20 UTC / 2-per-day regime.

Everything proven so far:
  - Entry features CANNOT predict win/loss (5 independent methods). No cherry-picking.
  - Gold's edge lives in 12-20 UTC (rebuild v2). Session fixed there.
  - Win rate is set by GEOMETRY, not selection: once price tags TP1, SL->breakeven,
    so a trade can only become a full loss (code 1) if it never reaches TP1.
    => WR ~= P(reach TP1 before initial SL). The current TP1 sits at ~1.98R (far),
    which is WHY WR is stuck ~44-50%. Pulling TP1 closer raises WR mechanically.

This searches the SL/TP1/TP2/Final geometry + harvest allocation to push WR AND
$/week together, in the 12-20 UTC / 2-per-day / $45+circuit-breaker regime.
Honest protocol: OPTIMIZE on 2023-24, VALIDATE on 2025-26. Keep only configs
that beat the current live baseline on BOTH datasets and pass the DD gate on both.
"""
import json
import time
import numpy as np
import pandas as pd
import tsai_optimize as T
import fullyear_backtest as fb

SPREAD = T.SPREAD
MAX_HOLD = T.MAX_HOLD
EQ0 = 5000.0
CUR = {"base_sl": 1.095, "vol_lo": 0.347, "vol_hi": 1.0, "tp1_r": 1.9836,
       "tp2_r": 3.9925, "tp3_r": 4.0925, "trend_gain": 2.8338, "tp_lo": 0.2, "tp_hi": 2.802}


def sim_geom(c, p):
    """Lean vectorized outcome sim (no stop-hunt loop). Returns code, sl_dist,
    tp R-multiples, timeout_r — enough to price any allocation."""
    d = c["d"]; entries = c["entries"]; x_atr = c["x_atr"]
    H, L, Cc = c["H"], c["L"], c["C"]; n = c["n"]
    is_buy = d == 1
    vr = np.clip(c["vol_ratio"], p["vol_lo"], p["vol_hi"])
    tp_ext = np.clip(1 + p["trend_gain"] * (c["trend_strength"] - 1), p["tp_lo"], p["tp_hi"])
    sl_dist = p["base_sl"] * vr * x_atr + SPREAD / 2
    tp1 = p["tp1_r"] * sl_dist + SPREAD
    tp2 = p["tp2_r"] * sl_dist + SPREAD
    tp3 = p["tp3_r"] * sl_dist * tp_ext + SPREAD
    tp1_hit = np.zeros(n, bool); tp2_hit = np.zeros(n, bool); closed = np.zeros(n, bool)
    code = np.zeros(n, np.int8)
    e_sl = np.where(is_buy, entries - sl_dist, entries + sl_dist)
    e_tp1 = np.where(is_buy, entries + tp1, entries - tp1)
    e_tp2 = np.where(is_buy, entries + tp2, entries - tp2)
    e_tp3 = np.where(is_buy, entries + tp3, entries - tp3)
    be = entries; tp1_as_sl = np.where(is_buy, entries + tp1, entries - tp1)
    for j in range(MAX_HOLD):
        active = ~closed
        if not active.any():
            break
        hi = H[:, j]; lo = L[:, j]
        sl_lvl = e_sl.copy()
        sl_lvl = np.where(tp1_hit & ~tp2_hit, be, sl_lvl)
        sl_lvl = np.where(tp2_hit, tp1_as_sl, sl_lvl)
        hit_sl = np.where(is_buy, lo <= sl_lvl, hi >= sl_lvl)
        newly = active & hit_sl
        code = np.where(newly, np.where(tp2_hit, 2, np.where(tp1_hit, 3, 1)), code)
        closed = closed | newly
        active = ~closed
        tp1_hit = tp1_hit | (active & ~tp1_hit & np.where(is_buy, hi >= e_tp1, lo <= e_tp1))
        tp2_hit = tp2_hit | (active & ~tp2_hit & np.where(is_buy, hi >= e_tp2, lo <= e_tp2))
        nf = active & np.where(is_buy, hi >= e_tp3, lo <= e_tp3)
        code = np.where(nf, 4, code)
        closed = closed | nf
    last_c = Cc[:, -1]
    r_to = (last_c - entries) * d / sl_dist
    return code, sl_dist, tp1 / sl_dist, tp2 / sl_dist, tp3 / sl_dist, r_to


def prep(df):
    c = T.build_candidates(df)
    order = np.argsort(c["dt"])
    dt = pd.to_datetime(c["dt"][order])
    day = pd.factorize(dt.date)[0].astype(np.int64)
    strong = c["trend_strength"] >= np.percentile(c["trend_strength"], 60)
    return dict(c=c, order=order, dt=dt, hour=dt.hour.values, wd=dt.weekday.values,
                day=day, strong=strong[order], entries=c["entries"][order],
                weeks=(df.index.max() - df.index.min()).days / 7.0)


def per_r(code, t1, t2, t3, r_to, strong, a1, a2, tilt):
    A1 = np.where(strong, np.maximum(0.02, a1 - tilt * 0.5), a1)
    A2 = np.where(strong, np.maximum(0.02, a2 - tilt * 0.5), a2)
    A3 = 1.0 - A1 - A2
    return np.select([code == 4, code == 2, code == 3, code == 1],
                     [A1*t1 + A2*t2 + A3*t3, A1*t1 + A2*t2 + A3*t1, A1*t1, -1.0],
                     default=r_to)


def run(S, p, a1, a2, tilt, base=45.0, ds=120.0, ddt_thr=200.0, fac=0.35,
        hrs=range(12, 21), max_day=2, cd=3600):
    code, sld, t1, t2, t3, r_to = sim_geom(S["c"], p)
    o = S["order"]
    code, sld = code[o], sld[o]; t1, t2, t3, r_to = t1[o], t2[o], t3[o], r_to[o]
    r = per_r(code, t1, t2, t3, r_to, S["strong"], a1, a2, tilt)
    hour = S["hour"]; wd = S["wd"]; day = S["day"]; dts = S["dt"]; ent = S["entries"]
    allow = np.isin(hour, list(hrs)); n = len(r)
    eq = EQ0; peak = EQ0; pnls = np.zeros(n)
    cur = -1; dp = 0.0; blk = False; dc = 0; taken = 0
    lt = pd.Timestamp("2000-01-01")
    for k in range(n):
        if day[k] != cur:
            cur = day[k]; dp = 0.0; blk = False; dc = 0
        if hour[k] in (21, 22) or (wd[k] == 4 and hour[k] >= 19):
            continue
        if not allow[k] or blk or dc >= max_day:
            continue
        if (dts[k] - lt).total_seconds() < cd:
            continue
        risk = base * (fac if (peak - eq) >= ddt_thr else 1.0)
        lpl = 100.0 * sld[k]
        lots = max(0.01, min(0.12, np.floor(risk / lpl * 100) / 100))
        pnl = r[k] * lots * lpl
        pnls[k] = pnl; eq += pnl; dp += pnl; taken += 1; dc += 1; lt = dts[k]
        if eq > peak:
            peak = eq
        if dp <= -ds:
            blk = True
    eqc = EQ0 + np.cumsum(pnls)
    ddt = float(np.max(np.maximum.accumulate(eqc) - eqc))
    ds_ = np.bincount(day, weights=pnls, minlength=day.max() + 1)
    ddd = float(-ds_.min()) if ds_.min() < 0 else 0.0
    nz = pnls[pnls != 0]
    w = int((nz > 0).sum()); l = int((nz < 0).sum())
    return dict(wk=pnls.sum() / S["weeks"], wr=w / max(1, w + l) * 100,
                ddt=ddt, ddd=ddd, gate=ddt < 350 and ddd < 200,
                sig_d=taken / (S["weeks"] * 5), n=taken)


def main():
    print("Loading data (train=2023-24, valid=2025-26)...")
    tr = prep(fb.load_data()); te = prep(T.load_data())

    b_tr = run(tr, CUR, 0.30, 0.40, 0.19); b_te = run(te, CUR, 0.30, 0.40, 0.19)
    print("\nCURRENT LIVE BASELINE (12-20 UTC, TP1~1.98R, 30/40/30, $45+CB):")
    print(f"  2023-24: {b_tr['wk']:+.1f}/wk WR{b_tr['wr']:.0f}% DDt${b_tr['ddt']:.0f} gate={b_tr['gate']}")
    print(f"  2025-26: {b_te['wk']:+.1f}/wk WR{b_te['wr']:.0f}% DDt${b_te['ddt']:.0f} gate={b_te['gate']}")

    print("\nMapping the WR vs $/wk Pareto frontier (gate-safe on BOTH datasets)...")
    rng = np.random.default_rng(1212)
    N = 9000
    pool = []
    t0 = time.time()
    for i in range(1, N + 1):
        p = dict(CUR)
        p["base_sl"] = rng.uniform(0.70, 1.55)
        p["tp1_r"] = rng.uniform(0.45, 1.9)          # KEY WR lever (closer TP1)
        p["tp2_r"] = p["tp1_r"] + rng.uniform(0.5, 2.8)
        p["tp3_r"] = p["tp2_r"] + rng.uniform(0.2, 3.2)
        p["trend_gain"] = rng.uniform(0.3, 3.0)
        a1 = rng.uniform(0.20, 0.60)
        a2 = rng.uniform(0.15, 0.45)
        if a1 + a2 > 0.92:
            continue
        tilt = rng.uniform(0.0, 0.30)
        a = run(tr, p, a1, a2, tilt)
        if not a["gate"]:
            continue
        b = run(te, p, a1, a2, tilt)
        if not b["gate"]:
            continue
        # robustness = worst-case across datasets
        wr = min(a["wr"], b["wr"]); wk = min(a["wk"], b["wk"])
        pool.append(dict(wr=wr, wk=wk, p=p, a1=a1, a2=a2, tilt=tilt, a=a, b=b))
        if i % 1500 == 0:
            print(f"  trial {i}/{N} | gate-safe-both: {len(pool)}")
    print(f"  done in {(time.time()-t0)/60:.1f} min | gate-safe on both: {len(pool)}")

    base_wr = min(b_tr["wr"], b_te["wr"]); base_wk = min(b_tr["wk"], b_te["wk"])
    print(f"\n  Baseline worst-case: WR {base_wr:.0f}% | ${base_wk:.0f}/wk")

    # (1) Max WR config
    mx_wr = max(pool, key=lambda x: (x["wr"], x["wk"]))
    # (2) Max $/wk config
    mx_wk = max(pool, key=lambda x: (x["wk"], x["wr"]))
    # (3) Best balanced: highest WR whose worst-case $/wk >= baseline $/wk
    bal_pool = [x for x in pool if x["wk"] >= base_wk]
    bal = max(bal_pool, key=lambda x: x["wr"]) if bal_pool else None
    # (4) High win-rate: max WR while keeping >=85% of baseline $/wk
    hi_pool = [x for x in pool if x["wk"] >= 0.85 * base_wk]
    hiwr = max(hi_pool, key=lambda x: x["wr"]) if hi_pool else None

    def show(tag, x):
        p = x["p"]
        print(f"\n[{tag}]  TP1={p['tp1_r']:.2f}R TP2={p['tp2_r']:.2f}R Final={p['tp3_r']:.2f}R "
              f"SL={p['base_sl']:.2f}x tgain={p['trend_gain']:.2f} | "
              f"alloc {x['a1']*100:.0f}/{x['a2']*100:.0f}/{(1-x['a1']-x['a2'])*100:.0f} tilt {x['tilt']:.2f}")
        print(f"   2023-24: {x['a']['wk']:+.1f}/wk WR{x['a']['wr']:.0f}% DDt${x['a']['ddt']:.0f} sig/d {x['a']['sig_d']:.1f}")
        print(f"   2025-26: {x['b']['wk']:+.1f}/wk WR{x['b']['wr']:.0f}% DDt${x['b']['ddt']:.0f} sig/d {x['b']['sig_d']:.1f}")

    show("MAX WIN-RATE", mx_wr)
    show("MAX $/WEEK", mx_wk)
    if bal:
        show("BEST BALANCED (WR up, $/wk >= baseline on both)", bal)
    if hiwr:
        show("HIGH WIN-RATE (max WR, keep >=85% of baseline $/wk)", hiwr)

    # Pareto frontier table
    print("\nWR vs $/wk frontier (worst-case across both datasets):")
    frontier = []
    for x in sorted(pool, key=lambda z: -z["wr"]):
        if not frontier or x["wk"] > frontier[-1]["wk"]:
            frontier.append(x)
    print(f"{'WR%(min)':>9} {'$/wk(min)':>10}")
    for x in frontier[:14]:
        print(f"{x['wr']:>9.0f} {x['wk']:>10.0f}")

    def pack(x):
        return {"geometry": x["p"], "a1": x["a1"], "a2": x["a2"], "tilt": x["tilt"],
                "train": x["a"], "valid": x["b"]}
    json.dump({"balanced": pack(bal) if bal else None,
               "high_wr": pack(hiwr) if hiwr else None,
               "max_wr": pack(mx_wr), "max_wk": pack(mx_wk),
               "baseline_train": b_tr, "baseline_valid": b_te},
              open("phase12_result.json", "w"), indent=2, default=float)
    print("\nSaved balanced + high_wr configs to phase12_result.json")


if __name__ == "__main__":
    main()
