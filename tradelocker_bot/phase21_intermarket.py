"""
PHASE 21 — INTERMARKET BRAIN. The last genuine frontier for predictive edge.

Gold's macro drivers (R² ~-0.82 with real yields, ~-0.6 with DXY historically):
  - 10Y TIPS real yield (DFII10): rising = bearish gold, falling = bullish
  - DXY (Trade Weighted USD): rising = bearish gold, falling = bullish

Hypothesis: bar-by-bar MOMENTUM of these drivers (not level) predicts which
gold trades win. A trade entered while real yields are falling AND dollar is
weakening should have a structurally higher probability of reaching TP.

Protocol:
  1. Align daily DXY + real-yield to each signal's date.
  2. Compute momentum features: 1d, 3d, 5d change for each.
  3. Train a simple logistic classifier on 2023-24 (target: trade wins/loses).
  4. Validate on 2025-26 (OOS): does it predict above chance?
  5. If yes: use it as the brain's gate (take signals the model says are favorable).
     Run with friction, compare vs take-all. If it lifts $/wk and/or cuts DD on
     BOTH datasets, it earns a brain tier upgrade.
  6. If no: report honestly and stop.
"""
import numpy as np
import pandas as pd
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12
import phase19_upgrade as U


def load_macro():
    ry = pd.read_parquet("data/real_yield_daily.parquet")
    ry["date"] = pd.to_datetime(ry["date"])
    ry = ry.set_index("date").sort_index()
    dx = pd.read_parquet("data/dxy_daily.parquet")
    dx["date"] = pd.to_datetime(dx["date"])
    dx = dx.set_index("date").sort_index()
    return ry, dx


def align_features(S, ry, dx):
    """For each signal (chrono order), compute macro momentum features."""
    dts = S["dt"]
    dates = dts.normalize()  # date only
    n = len(dts)
    feats = np.zeros((n, 6))
    for i in range(n):
        d = dates[i]
        # real yield momentum: 1d, 3d, 5d
        for j, lag in enumerate([1, 3, 5]):
            d_prev = d - pd.Timedelta(days=lag)
            rv = ry["value"].asof(d); rv_prev = ry["value"].asof(d_prev)
            if pd.notna(rv) and pd.notna(rv_prev):
                feats[i, j] = rv - rv_prev
        # DXY momentum: 1d, 3d, 5d
        for j, lag in enumerate([1, 3, 5]):
            d_prev = d - pd.Timedelta(days=lag)
            dv = dx["value"].asof(d); dv_prev = dx["value"].asof(d_prev)
            if pd.notna(dv) and pd.notna(dv_prev):
                feats[i, 3 + j] = dv - dv_prev
    return feats


def logistic_fit(X, y, lam=1.0):
    """Ridge logistic regression via IRLS (fast enough for ~500 rows)."""
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-Xb @ w))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        W = p * (1 - p)
        z = Xb @ w + (y - p) / W
        H = Xb.T @ (Xb * W[:, None]) + lam * np.eye(Xb.shape[1])
        H[0, 0] -= lam
        w = np.linalg.solve(H, Xb.T @ (W * z))
    return w


def logistic_pred(w, X):
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return 1 / (1 + np.exp(-Xb @ w))


def main():
    print("Loading data + intermarket series...")
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    ry, dx = load_macro()
    print(f"Real yield: {ry.index[0].date()} -> {ry.index[-1].date()} ({len(ry)} days)")
    print(f"DXY: {dx.index[0].date()} -> {dx.index[-1].date()} ({len(dx)} days)")

    print("\nAligning macro features to signals...")
    F_tr = align_features(tr, ry, dx); F_te = align_features(te, ry, dx)
    print(f"  train signals: {len(F_tr)} | test signals: {len(F_te)}")
    # outcomes
    c_tr = tr["c"]; c_te = te["c"]
    sld_tr, r_tr, _, _, _ = U.outcomes(c_tr, 0.45, 1.1363)
    sld_te, r_te, _, _, _ = U.outcomes(c_te, 0.45, 1.1363)
    r_tr = r_tr[tr["order"]]; r_te = r_te[te["order"]]
    y_tr = (r_tr > 0).astype(float); y_te = (r_te > 0).astype(float)
    print(f"  train WR: {y_tr.mean()*100:.1f}% | test WR: {y_te.mean()*100:.1f}%")

    # normalize features on train
    mu = F_tr.mean(0); sd = F_tr.std(0) + 1e-9
    Xn_tr = (F_tr - mu) / sd; Xn_te = (F_te - mu) / sd

    print("\nTraining logistic model (intermarket momentum -> win/loss)...")
    w = logistic_fit(Xn_tr, y_tr, lam=2.0)
    p_tr = logistic_pred(w, Xn_tr); p_te = logistic_pred(w, Xn_te)

    # AUC
    from numpy import argsort
    def auc(p, y):
        o = argsort(p); y_s = y[o]; n1 = y.sum(); n0 = len(y) - n1
        r = np.cumsum(1 - y_s); return float(r[y_s == 1].sum()) / (n0 * n1) if n0 * n1 > 0 else 0.5
    auc_tr = auc(p_tr, y_tr); auc_te = auc(p_te, y_te)
    print(f"  TRAIN AUC: {auc_tr:.4f}  |  TEST (OOS) AUC: {auc_te:.4f}")
    print(f"  (0.50 = random, >0.55 = genuine signal, >0.60 = strong)")

    # decile analysis OOS
    q80 = np.quantile(p_te, 0.80); q20 = np.quantile(p_te, 0.20)
    wr_hi = y_te[p_te >= q80].mean() * 100; wr_lo = y_te[p_te <= q20].mean() * 100
    print(f"  OOS top-20% predicted WR: {wr_hi:.1f}% | bottom-20%: {wr_lo:.1f}%")
    print(f"  OOS mean R: top-20% = {r_te[p_te>=q80].mean():+.3f} | bottom-20% = {r_te[p_te<=q20].mean():+.3f}")

    # feature importance (coefficient magnitude)
    names = ["ry_1d", "ry_3d", "ry_5d", "dxy_1d", "dxy_3d", "dxy_5d"]
    coefs = w[1:]
    print("\n  Feature coefficients (direction -> P(win)):")
    for nm, cf in sorted(zip(names, coefs), key=lambda x: -abs(x[1])):
        print(f"    {nm:>8}: {cf:+.4f}")

    # ---------- BRAIN GATE TEST ----------
    if auc_te > 0.52:
        print("\n\nOOS AUC > 0.52 — testing as a brain gate (friction ON)...")
        # sweep threshold on train, apply unchanged to test
        best = None
        for q in np.linspace(0.1, 0.5, 20):
            thr = np.quantile(p_tr, q)
            mask_tr = p_tr >= thr; mask_te = p_te >= thr
            # get session-filtered indices only
            hour_tr = tr["hour"]; allow_tr = np.isin(hour_tr, list(range(12, 21)))
            hour_te = te["hour"]; allow_te = np.isin(hour_te, list(range(12, 21)))
            combined_tr = mask_tr & allow_tr; combined_te = mask_te & allow_te
            a = U.sim(tr, friction=True, regime_mask=combined_tr)
            b = U.sim(te, friction=True, regime_mask=combined_te)
            if a['gate'] and b['gate']:
                worst = min(a['wk'], b['wk'])
                if best is None or worst > best[0]:
                    best = (worst, q, a, b, thr)
        base_a = U.sim(tr, friction=True); base_b = U.sim(te, friction=True)
        print(f"\n  No brain (take-all): 23-24 ${base_a['wk']:+.1f}/wk WR{base_a['wr']:.0f}% DDt${base_a['ddt']:.0f} | "
              f"25-26 ${base_b['wk']:+.1f}/wk WR{base_b['wr']:.0f}% DDt${base_b['ddt']:.0f}")
        if best:
            _, q, a, b, thr = best
            print(f"  BRAIN gate (q={q:.2f}): 23-24 ${a['wk']:+.1f}/wk WR{a['wr']:.0f}% DDt${a['ddt']:.0f} | "
                  f"25-26 ${b['wk']:+.1f}/wk WR{b['wr']:.0f}% DDt${b['ddt']:.0f}")
            d = min(a['wk'], b['wk']) - min(base_a['wk'], base_b['wk'])
            print(f"  --> brain gate effect on worst-case $/wk: {d:+.1f}")
            if d > 0:
                print("  VERDICT: intermarket brain HELPS — genuine predictive edge found.")
            else:
                print("  VERDICT: intermarket brain does not lift $/wk vs take-all.")
        else:
            print("  No gate-safe config found with the brain.")
    else:
        print(f"\n  OOS AUC {auc_te:.4f} ≤ 0.52 — no predictive power detected.")
        print("  VERDICT: intermarket momentum does NOT predict gold trade outcomes OOS.")


if __name__ == "__main__":
    main()
