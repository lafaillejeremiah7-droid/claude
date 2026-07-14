"""
PHASE 26 — run ideas #1 & #2 through the rigorous CV harness (2025-26).
Every result gets a bootstrap AUC confidence interval + calibration error.
An idea only "counts" if its AUC CI is meaningfully ABOVE 0.50.
"""
import numpy as np
import pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import phase21_intermarket as P21, phase19_upgrade as U
import phase24_intraday_wf as P24
import hercules_v2 as H
import cv_harness as CV


def nl_fit(X, y):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    a = (X - mu) / sd
    co = np.abs(P21.logistic_fit(a, y, lam=2.0)[1:])
    ti = np.where(co > np.median(co))[0]
    prs = []
    for i in range(len(ti)):
        for j in range(i + 1, len(ti)):
            prs.append((ti[i], ti[j], co[ti[i]] * co[ti[j]]))
    prs.sort(key=lambda x: -x[2]); pairs = [(p[0], p[1]) for p in prs[:5]]
    ita = np.column_stack([a[:, i] * a[:, j] for i, j in pairs]) if pairs else np.zeros((len(a), 0))
    X3 = np.hstack([a, a ** 2, ita])
    w = P21.logistic_fit(X3, y, lam=5.0)
    return {"mu": mu, "sd": sd, "pairs": pairs, "w": w}


def nl_pred(m, X):
    a = (X - m["mu"]) / m["sd"]
    ita = np.column_stack([a[:, i] * a[:, j] for i, j in m["pairs"]]) if m["pairs"] else np.zeros((len(a), 0))
    return P21.logistic_pred(m["w"], np.hstack([a, a ** 2, ita]))


def main():
    print("=" * 70)
    print("IDEAS #1 & #2 THROUGH THE CV HARNESS (2025-26, purged walk-forward)")
    print("=" * 70)
    te = P12.prep(T.load_data()); c = te["c"]; o = te["order"]
    dts = pd.to_datetime(c["dt"][o])
    times = dts.values.astype("datetime64[ns]").astype(np.int64)
    _, r, _, _, _ = U.outcomes(c, 0.45, 1.1363); r = r[o]
    y_win = (r > 0).astype(float)

    intr = np.column_stack([c["rsi"][o], c["vol_ratio"][o], c["trend_strength"][o],
                            c["pb_dist"][o], te["hour"].astype(float)])
    ry, dx = P21.load_macro()
    daily = P21.align_features(te, ry, dx)
    intraday = P24.intraday_macro_features(c["dt"][o])
    df = T.load_data()
    state = H.world_state(te, df, daily)  # path + macro + intrinsic (16)

    print("\n### IDEA #2 — predicting WIN/LOSS (target = trade wins) ###")
    print(f"{'feature set':>28} | {'OOS AUC':>7} {'95% CI':>16} {'ECE':>5}")
    sets = {
        "intrinsic only": intr,
        "intrinsic + daily macro": np.hstack([intr, daily]),
        "intrinsic + intraday macro": np.hstack([intr, intraday]),
        "intrinsic + BOTH macro": np.hstack([intr, daily, intraday]),
        "full world-state (path+macro)": state,
    }
    for name, X in sets.items():
        res = CV.evaluate(nl_fit, nl_pred, X, y_win, times)
        star = " <-- above 0.50" if res["auc_lo"] > 0.50 else ""
        print(f"{name:>28} | {res['auc']:>7.4f} [{res['auc_lo']:.3f},{res['auc_hi']:.3f}] "
              f"{res['ece']:>5.3f}{star}")

    print("\n### IDEA #1 — predicting the CHOP objective (target = chop) ###")
    lab = H.objective_labels(te)          # in te['order'] order
    y_chop = (lab == 3).astype(float)
    res_chop = CV.evaluate(nl_fit, nl_pred, state, y_chop, times)
    print(f"{'chop prediction':>28} | {res_chop['auc']:>7.4f} "
          f"[{res_chop['auc_lo']:.3f},{res_chop['auc_hi']:.3f}] {res_chop['ece']:>5.3f}"
          f"{'  <-- REAL signal' if res_chop['auc_lo'] > 0.50 else ''}")
    # does chop-probability predict WIN? (the conversion question)
    res_c2w = CV.evaluate(nl_fit, nl_pred, state, y_win, times)  # same features, win target
    print(f"  (same state -> WIN target: AUC {res_c2w['auc']:.4f} "
          f"[{res_c2w['auc_lo']:.3f},{res_c2w['auc_hi']:.3f}])")

    print("\n" + "=" * 70)
    print("PROMOTION GATE vs baseline (take-all, no brain):")
    base = {"auc": 0.50, "ece": 0.05, "pf": 1.0, "sharpe": 0.0, "maxdd": 350}
    best = CV.evaluate(nl_fit, nl_pred, np.hstack([intr, daily, intraday]), y_win, times)
    cand = {"auc": best["auc"], "ece": best["ece"], "pf": 1.0, "sharpe": 0.0, "maxdd": 350}
    ok, why = CV.promotion_gate(base, cand)
    for w_ in why:
        print("  " + w_)
    print(f"\n  PROMOTE new predictive brain? {ok}")
    print("=" * 70)


if __name__ == "__main__":
    main()
