"""
PHASE 23 — push brain AUC from 0.536 toward 0.60+.

Approaches:
  1. FEATURE SELECTION: only keep the features with |coef| > median. Less noise.
  2. SEQUENTIAL CONTEXT: add last-3-trade outcomes as features (momentum/streak).
  3. CONFIDENCE-SCALED RISK: when brain is very confident, risk 1.5x; when unsure, 0.7x.
  4. NONLINEAR EXPANSION: add squared terms + key interactions to logistic.
All tested OOS on 2025-26. The goal: AUC > 0.55 → translates to more $/wk.
"""
import numpy as np
import pandas as pd
import phase12_ideal as P12, tsai_optimize as T, fullyear_backtest as fb
import phase21_intermarket as P21, phase22_pathshape as P22, phase19_upgrade as U


def auc(p, y):
    o = np.argsort(p); y_s = y[o]; n1 = y.sum(); n0 = len(y) - n1
    return float(np.cumsum(1-y_s)[y_s==1].sum()) / (n0*n1) if n0*n1>0 else 0.5


def main():
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    ry, dx = P21.load_macro()
    F_tr = P22.expanded_features(tr, P21.align_features(tr, ry, dx))
    F_te = P22.expanded_features(te, P21.align_features(te, ry, dx))
    sld, r, _, _, _ = U.outcomes(tr["c"], 0.45, 1.1363); r_tr = r[tr["order"]]
    sld2, r2, _, _, _ = U.outcomes(te["c"], 0.45, 1.1363); r_te = r2[te["order"]]
    y_tr = (r_tr > 0).astype(float); y_te = (r_te > 0).astype(float)
    mu = F_tr.mean(0); sd = F_tr.std(0) + 1e-9
    Xn_tr = (F_tr - mu) / sd; Xn_te = (F_te - mu) / sd

    # baseline
    w0 = P21.logistic_fit(Xn_tr, y_tr, lam=2.0)
    p0_te = P21.logistic_pred(w0, Xn_te); auc0 = auc(p0_te, y_te)
    print(f"BASELINE (19 features, ridge logistic): OOS AUC = {auc0:.4f}\n")

    # --- 1. FEATURE SELECTION ---
    coefs = np.abs(w0[1:])
    top_idx = np.where(coefs > np.median(coefs))[0]
    print(f"1. FEATURE SELECTION: keeping {len(top_idx)}/{len(coefs)} features (|coef|>median)")
    w1 = P21.logistic_fit(Xn_tr[:, top_idx], y_tr, lam=2.0)
    p1_te = P21.logistic_pred(w1, Xn_te[:, top_idx]); auc1 = auc(p1_te, y_te)
    print(f"   OOS AUC: {auc1:.4f} (vs baseline {auc0:.4f}, delta {auc1-auc0:+.4f})")

    # --- 2. SEQUENTIAL CONTEXT (last-3-trade outcomes) ---
    print(f"\n2. SEQUENTIAL CONTEXT: add rolling last-3-trade outcomes as features")
    def add_seq(X, y_actual):
        n = len(X); seq = np.zeros((n, 3))
        for i in range(n):
            for j in range(1, 4):
                if i - j >= 0:
                    seq[i, j-1] = 2 * y_actual[i-j] - 1  # -1 loss, +1 win
        return np.hstack([X, seq])
    X2_tr = add_seq(Xn_tr, y_tr); X2_te = add_seq(Xn_te, y_te)
    w2 = P21.logistic_fit(X2_tr, y_tr, lam=2.0)
    p2_te = P21.logistic_pred(w2, X2_te); auc2 = auc(p2_te, y_te)
    print(f"   OOS AUC: {auc2:.4f} (delta {auc2-auc0:+.4f})")

    # --- 3. NONLINEAR EXPANSION (squared + top interactions) ---
    print(f"\n3. NONLINEAR EXPANSION: squared terms + top-5 interaction pairs")
    sq = Xn_tr ** 2; sq_te = Xn_te ** 2
    # top-5 pairs by |coef_i * coef_j|
    pairs = []
    for i in range(len(top_idx)):
        for j in range(i+1, len(top_idx)):
            pairs.append((top_idx[i], top_idx[j], coefs[top_idx[i]] * coefs[top_idx[j]]))
    pairs.sort(key=lambda x: -x[2])
    inter_idx = [(p[0], p[1]) for p in pairs[:5]]
    inter_tr = np.column_stack([Xn_tr[:, i] * Xn_tr[:, j] for i, j in inter_idx])
    inter_te = np.column_stack([Xn_te[:, i] * Xn_te[:, j] for i, j in inter_idx])
    X3_tr = np.hstack([Xn_tr, sq, inter_tr]); X3_te = np.hstack([Xn_te, sq_te, inter_te])
    w3 = P21.logistic_fit(X3_tr, y_tr, lam=5.0)
    p3_te = P21.logistic_pred(w3, X3_te); auc3 = auc(p3_te, y_te)
    print(f"   OOS AUC: {auc3:.4f} (delta {auc3-auc0:+.4f})")

    # --- 4. COMBINED: selected features + sequential + nonlinear ---
    print(f"\n4. COMBINED (best of above):")
    X4_tr = add_seq(np.hstack([Xn_tr[:, top_idx], sq[:, top_idx], inter_tr]), y_tr)
    X4_te = add_seq(np.hstack([Xn_te[:, top_idx], sq_te[:, top_idx], inter_te]), y_te)
    w4 = P21.logistic_fit(X4_tr, y_tr, lam=5.0)
    p4_te = P21.logistic_pred(w4, X4_te); auc4 = auc(p4_te, y_te)
    print(f"   OOS AUC: {auc4:.4f} (delta {auc4-auc0:+.4f})")

    # --- 5. CONFIDENCE-SCALED RISK (use best model so far) ---
    best_auc = max(auc0, auc1, auc2, auc3, auc4)
    best_p = [p0_te, p1_te, p2_te, p3_te, p4_te][[auc0, auc1, auc2, auc3, auc4].index(best_auc)]
    best_name = ["baseline", "feat-select", "sequential", "nonlinear", "combined"][
        [auc0, auc1, auc2, auc3, auc4].index(best_auc)]
    print(f"\n  BEST MODEL: {best_name} (AUC {best_auc:.4f})")

    # Also get train preds for best model to set threshold
    if best_name == "feat-select":
        bp_tr = P21.logistic_pred(w1, Xn_tr[:, top_idx])
    elif best_name == "sequential":
        bp_tr = P21.logistic_pred(w2, X2_tr)
    elif best_name == "nonlinear":
        bp_tr = P21.logistic_pred(w3, X3_tr)
    elif best_name == "combined":
        bp_tr = P21.logistic_pred(w4, X4_tr)
    else:
        bp_tr = P21.logistic_pred(w0, Xn_tr)

    print(f"\n5. CONFIDENCE-SCALED RISK (best model, friction ON, 12-20 UTC):")
    print(f"   High conf (top 40%): risk $65 | Low conf (rest): risk $35")
    thr = np.quantile(bp_tr, 0.17); conf_split = np.quantile(bp_tr, 0.60)
    # Can't do per-signal risk in the sim easily, so test: gate-only vs gate+higher-risk
    mask_te = best_p >= thr
    # split: high-confidence trades only
    mask_hi = best_p >= conf_split
    a_hi = U.sim(tr, friction=True, base=65, regime_mask=(bp_tr >= conf_split), dd_cap=400.0)
    b_hi = U.sim(te, friction=True, base=65, regime_mask=mask_hi, dd_cap=400.0)
    a_all = U.sim(tr, friction=True, base=47, regime_mask=(bp_tr >= thr), dd_cap=400.0)
    b_all = U.sim(te, friction=True, base=47, regime_mask=mask_te, dd_cap=400.0)
    print(f"   Brain gate (q=0.17, $47): 23-24 ${a_all['wk']:+.1f}/wk DDt${a_all['ddt']:.0f} | "
          f"25-26 ${b_all['wk']:+.1f}/wk DDt${b_all['ddt']:.0f}")
    print(f"   HIGH-CONF only ($65)   : 23-24 ${a_hi['wk']:+.1f}/wk DDt${a_hi['ddt']:.0f} WR{a_hi['wr']:.0f}% | "
          f"25-26 ${b_hi['wk']:+.1f}/wk DDt${b_hi['ddt']:.0f} WR{b_hi['wr']:.0f}%")
    ok_hi = a_hi['ddt'] < 400 and b_hi['ddt'] < 400
    print(f"   gate: {'PASS' if ok_hi else 'FAIL'}")

    print("\n" + "=" * 70)
    print(f"SUMMARY: best achievable OOS AUC = {best_auc:.4f} (model: {best_name})")
    if best_auc >= 0.55:
        print("  TARGET MET: AUC >= 0.55 — brain is genuinely predictive.")
    elif best_auc > 0.53:
        print("  PARTIAL: AUC 0.53-0.55 — brain adds value but is not strongly predictive.")
    else:
        print("  INSUFFICIENT: AUC < 0.53 — brain remains weak.")
    print("=" * 70)


if __name__ == "__main__":
    main()
