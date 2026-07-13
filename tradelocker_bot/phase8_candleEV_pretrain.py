"""
PHASE 8 — Candle-movement EV model with random/permutation pre-training.

"Pre-train against random data to hone skill" done rigorously as a PERMUTATION
NULL TEST: train the EV model many times on RANDOMIZED (shuffled) outcomes to
learn exactly what a skill-less model produces (the null distribution of
out-of-sample profit). Then check whether the model trained on REAL data beats
that null. This is how you separate genuine edge from luck — you can't learn
skill from noise, but you can calibrate the noise floor.

Then backtest the REAL model on the 1-year (2025-2026) and 4.5-year (2020-2024)
windows. Push only if it beats the null AND improves weekly profit OOS.

Splits (no leakage):
  TRAIN  = 2020-05-29 .. 2022-12-31  (HF)
  OOS-4.5yr = 2023-01-01 .. 2024-11-29  (held-out tail of the 4.5yr, HF)
  OOS-1yr   = 2025-01-01 .. 2026-07-13  (Forexite)
"""
import numpy as np
import pandas as pd

import tsai_optimize as T
import phase4_usage as P4
import walk_forward as W
import phase6_candleprob as P6
import phase7_candleEV as P7

ALLOC = dict(a1=0.30, a2=0.40, tilt=0.19, risk=25.0)


def load_hf_window(start, end):
    df1 = pd.read_parquet("data/xau_train.parquet")
    df2 = pd.read_parquet("data/xau_test.parquet")
    df = pd.concat([df1, df2], ignore_index=True)
    dt = pd.to_datetime(df["Date"].astype(str).str.zfill(8) + df["Time"].astype(str).str.zfill(6),
                        format="%Y%m%d%H%M%S", errors="coerce")
    df = df.assign(dt=dt).dropna(subset=["dt"])
    df = df[["dt", "Open", "High", "Low", "Close"]].rename(
        columns={"Open": "o", "High": "h", "Low": "l", "Close": "c"})
    df = df.sort_values("dt").drop_duplicates("dt").set_index("dt")
    return df.loc[start:end]


def prep(df):
    c = T.build_candidates(df)
    pc = P4.precompute(c); pc["dt_sorted"] = c["dt"][pc["order"]]
    r = P7.per_signal_r(pc, ALLOC["a1"], ALLOC["a2"], ALLOC["tilt"])
    F = P6.candle_features(df, c)[pc["order"]]
    weeks = (df.index.max() - df.index.min()).days / 7.0
    return pc, F, r, weeks


def weekly(pc, mask, weeks):
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return None
    p, dts = W.pnl_idx(pc, idx, ALLOC["a1"], ALLOC["a2"], ALLOC["tilt"], ALLOC["risk"])
    ddt, ddd = W.stitched_dd(p, dts)
    return {"wk": float(p.sum())/weeks, "dd_total": ddt, "dd_day": ddd,
            "gate": ddt < 350 and ddd < 200, "n": len(idx)}


def filter_improvement(pc_tr, X_tr, r_tr, pc_te, X_te, w_te, thr_q_train_pred):
    """Fit ridge on (X_tr, r_tr), pick EV threshold at train-quantile, return
    OOS weekly-profit improvement vs take-all baseline."""
    w, mu, sd = P7.ridge_fit(X_tr, r_tr)
    pred_tr = P7.ridge_pred(w, mu, sd, X_tr)
    pred_te = P7.ridge_pred(w, mu, sd, X_te)
    base = weekly(pc_te, np.ones(pc_te["n"], bool), w_te)
    # choose threshold on train to maximize train weekly profit
    best_thr, best_wk = np.quantile(pred_tr, 0.0), -1e18  # default = take-all
    pc_tr_weeks = filter_improvement._trweeks
    for q in np.linspace(0.0, 0.6, 25):
        thr = np.quantile(pred_tr, q)
        m = weekly(pc_tr, pred_tr >= thr, pc_tr_weeks)
        if m and m["gate"] and m["wk"] > best_wk:
            best_wk, best_thr = m["wk"], thr
    m_te = weekly(pc_te, pred_te >= best_thr, w_te)
    corr = float(np.corrcoef(pred_te, weekly.__dict__.get("_rte", pred_te))[0, 1]) if False else None
    return (m_te["wk"] - base["wk"]) if (m_te and m_te["gate"]) else -999, base, m_te


def main():
    print("Loading TRAIN (2020-2022), OOS-4.5yr tail (2023-2024), OOS-1yr (2025-2026)...")
    df_tr = load_hf_window("2020-05-29", "2022-12-31")
    df_o45 = load_hf_window("2023-01-01", "2024-11-29")
    df_1y = T.load_data()

    pc_tr, X_tr, r_tr, w_tr = prep(df_tr)
    pc_45, X_45, r_45, w_45 = prep(df_o45)
    pc_1y, X_1y, r_1y, w_1y = prep(df_1y)
    print(f"  train signals={len(r_tr)} | OOS-4.5yr={len(r_45)} | OOS-1yr={len(r_1y)}")
    filter_improvement._trweeks = w_tr

    # ---- REAL model OOS predictive power + profit impact ----
    def corr(a, b): return float(np.corrcoef(a, b)[0, 1])
    w, mu, sd = P7.ridge_fit(X_tr, r_tr)
    pred_45 = P7.ridge_pred(w, mu, sd, X_45)
    pred_1y = P7.ridge_pred(w, mu, sd, X_1y)
    print(f"\nREAL model OOS corr(pred EV, actual R):")
    print(f"  4.5yr tail = {corr(pred_45, r_45):+.3f} | 1yr = {corr(pred_1y, r_1y):+.3f}  (0 = no skill)")

    real_imp_45, base45, _ = filter_improvement(pc_tr, X_tr, r_tr, pc_45, X_45, w_45, None)
    real_imp_1y, base1y, _ = filter_improvement(pc_tr, X_tr, r_tr, pc_1y, X_1y, w_1y, None)

    # ---- NULL distribution: train on shuffled labels K times ----
    print(f"\nBuilding null distribution (50 shuffled-label trainings)...")
    rng = np.random.default_rng(8)
    null_45, null_1y = [], []
    for _ in range(50):
        rsh = r_tr.copy(); rng.shuffle(rsh)
        imp45, _, _ = filter_improvement(pc_tr, X_tr, rsh, pc_45, X_45, w_45, None)
        imp1y, _, _ = filter_improvement(pc_tr, X_tr, rsh, pc_1y, X_1y, w_1y, None)
        if imp45 > -900: null_45.append(imp45)
        if imp1y > -900: null_1y.append(imp1y)
    null_45 = np.array(null_45); null_1y = np.array(null_1y)

    def pct_rank(val, dist):
        return float((dist < val).mean() * 100) if len(dist) else 0.0

    print("\n" + "=" * 74)
    print(f"{'window':>12} {'baseline $/wk':>14} {'REAL filter Δ':>14} {'null mean Δ':>12} {'null p95':>10} {'percentile':>11}")
    print("-" * 74)
    print(f"{'4.5yr':>12} ${base45['wk']:>12.2f} ${real_imp_45:>+12.2f} ${null_45.mean():>+10.2f} "
          f"${np.percentile(null_45,95):>+8.2f} {pct_rank(real_imp_45,null_45):>9.1f}%")
    print(f"{'1yr':>12} ${base1y['wk']:>12.2f} ${real_imp_1y:>+12.2f} ${null_1y.mean():>+10.2f} "
          f"${np.percentile(null_1y,95):>+8.2f} {pct_rank(real_imp_1y,null_1y):>9.1f}%")
    print("=" * 74)

    sig45 = real_imp_45 > np.percentile(null_45, 95) and real_imp_45 > 0
    sig1y = real_imp_1y > np.percentile(null_1y, 95) and real_imp_1y > 0
    print(f"\nReal beats null(95%) & improves profit:  4.5yr={sig45}  |  1yr={sig1y}")
    if sig45 and sig1y:
        print("VERDICT: candle-EV model shows genuine, significant edge on BOTH -> PUSH.")
    else:
        print("VERDICT: candle-EV model is within the noise floor (not distinguishable from "
              "random) on at least one window -> DO NOT PUSH.")


if __name__ == "__main__":
    main()
