"""
PHASE 7 — Candle-movement EXPECTED-VALUE model (not win-probability).

The win-prob model failed (OOS AUC 0.476) because *whether* a trade wins is
unpredictable at entry. This instead regresses the realized R-multiple
(continuous outcome, incl. the big runner tail) on candle-movement features to
estimate each setup's EXPECTED VALUE, then takes only the highest-EV entries.
Rationale: even if win-rate is flat, some setups may produce larger runners.

Protocol (no look-ahead / no overfit): TRAIN ridge regressor on 2023-2024,
TEST on 2025-2026. Report OOS predictive power (corr of predicted vs actual R).
Threshold chosen on TRAIN weekly profit, applied unchanged to TEST.
Baseline = validated 30/40/31 @ $25. Push only if TEST weekly profit improves.
"""
import numpy as np
import pandas as pd

import tsai_optimize as T
import phase4_usage as P4
import fullyear_backtest as fb
import walk_forward as W
import phase6_candleprob as P6

ALLOC = dict(a1=0.30, a2=0.40, tilt=0.19, risk=25.0)


def per_signal_r(pc, a1, a2, tilt):
    strong = pc["strong"]
    A1 = np.where(strong, max(0.02, a1 - tilt * 0.5), a1)
    A2 = np.where(strong, max(0.02, a2 - tilt * 0.5), a2)
    A3 = 1.0 - A1 - A2
    code = pc["code"]; t1 = pc["tp1_R"]; t2 = pc["tp2_R"]; t3 = pc["tp3_R"]
    return np.select([code == 4, code == 2, code == 3, code == 1],
                     [A1*t1 + A2*t2 + A3*t3, A1*t1 + A2*t2 + A3*t1, A1*t1, -1.0],
                     default=pc["timeout_r"])


def prep(df1m):
    c = T.build_candidates(df1m)
    pc = P4.precompute(c); pc["dt_sorted"] = c["dt"][pc["order"]]
    r = per_signal_r(pc, ALLOC["a1"], ALLOC["a2"], ALLOC["tilt"])   # target EV (R)
    F = P6.candle_features(df1m, c)[pc["order"]]
    return pc, F, r


def ridge_fit(X, y, lam=5.0):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    d = Xs.shape[1]
    A = Xs.T @ Xs + lam * np.eye(d); A[0, 0] -= lam
    w = np.linalg.solve(A, Xs.T @ y)
    return w, mu, sd


def ridge_pred(w, mu, sd, X):
    Xs = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    return Xs @ w


def weekly(pc, mask, weeks):
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return None
    p, dts = W.pnl_idx(pc, idx, ALLOC["a1"], ALLOC["a2"], ALLOC["tilt"], ALLOC["risk"])
    ddt, ddd = W.stitched_dd(p, dts)
    return {"wk": float(p.sum())/weeks, "dd_total": ddt, "dd_day": ddd,
            "gate": ddt < 350 and ddd < 200, "n": len(idx)}


def main():
    print("TRAIN = 2023-2024, TEST = 2025-2026 (true out-of-sample)\n")
    df_tr = fb.load_data(); df_te = T.load_data()
    w_tr = (df_tr.index.max() - df_tr.index.min()).days / 7.0
    w_te = (df_te.index.max() - df_te.index.min()).days / 7.0

    pc_tr, X_tr, r_tr = prep(df_tr)
    pc_te, X_te, r_te = prep(df_te)
    print(f"Train signals: {len(r_tr)} | mean R = {r_tr.mean():.3f}")
    print(f"Test  signals: {len(r_te)} | mean R = {r_te.mean():.3f}")

    w, mu, sd = ridge_fit(X_tr, r_tr)
    pred_tr = ridge_pred(w, mu, sd, X_tr)
    pred_te = ridge_pred(w, mu, sd, X_te)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    print(f"\nCan candle movement predict a setup's EV (R)?")
    print(f"  TRAIN corr(pred, actual R) = {corr(pred_tr, r_tr):+.3f}")
    print(f"  TEST  corr(pred, actual R) = {corr(pred_te, r_te):+.3f}   (0.00 = no predictive power)")
    # decile check: mean actual R of top-EV-predicted decile vs bottom, OOS
    q_hi = np.quantile(pred_te, 0.8); q_lo = np.quantile(pred_te, 0.2)
    print(f"  OOS mean R: top-20% predicted-EV = {r_te[pred_te>=q_hi].mean():+.3f} | "
          f"bottom-20% = {r_te[pred_te<=q_lo].mean():+.3f}")

    base = weekly(pc_te, np.ones(pc_te["n"], bool), w_te)
    print(f"\nBASELINE (all, 30/40/31 @ $25): +${base['wk']:.2f}/wk | "
          f"DDtot=${base['dd_total']:.0f} DDday=${base['dd_day']:.0f} | gate={base['gate']}")

    # choose EV threshold on TRAIN maximizing train weekly profit; apply to TEST
    best_thr = None; best_wk = -1e18
    for q in np.linspace(0.0, 0.7, 36):
        thr = np.quantile(pred_tr, q)
        m = weekly(pc_tr, pred_tr >= thr, w_tr)
        if m and m["gate"] and m["wk"] > best_wk:
            best_wk = m["wk"]; best_thr = thr
    m_te = weekly(pc_te, pred_te >= best_thr, w_te)
    print(f"\nBest TRAIN EV threshold (train +${best_wk:.2f}/wk)")
    if m_te:
        print(f"TEST with EV filter: +${m_te['wk']:.2f}/wk | kept {m_te['n']}/{pc_te['n']} | "
              f"DDtot=${m_te['dd_total']:.0f} DDday=${m_te['dd_day']:.0f} | gate={m_te['gate']}")
        imp = m_te["wk"] - base["wk"]
        print("\n" + "=" * 70)
        if m_te["gate"] and imp > 0:
            print(f"VERDICT: candle-EV model IMPROVES weekly profit ${imp:+.2f}/wk OOS -> PUSH.")
        else:
            print(f"VERDICT: candle-EV model does NOT improve weekly profit (${imp:+.2f}/wk OOS) -> DO NOT PUSH.")
        print("=" * 70)


if __name__ == "__main__":
    main()
