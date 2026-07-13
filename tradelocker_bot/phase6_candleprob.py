"""
PHASE 6 — Candle-movement win-probability entry model.

Idea: read the actual candlestick movement around each signal, learn a
probability that the entry resolves favorably (reaches profit vs full stop),
then take only the highest-probability entries. ONLY push if it improves
weekly profit out-of-sample.

Rigorous protocol (no look-ahead, no overfitting):
  - TRAIN a logistic win-probability model on 2023-2024.
  - TEST on 2025-2026 (true out-of-sample).
  - Pick the probability threshold that maximizes TRAIN weekly profit; apply it
    unchanged to TEST.
  - Baseline = current validated engine (30/40/31 allocation @ $25 risk).
  - Push only if TEST weekly profit improves AND stays gate-safe.
"""
import numpy as np
import pandas as pd

import tsai_optimize as T
import phase4_usage as P4
import fullyear_backtest as fb
import walk_forward as W

ALLOC = dict(a1=0.30, a2=0.40, tilt=0.19, risk=25.0)   # validated baseline


def candle_features(df1m, c):
    """Backward-looking candlestick-movement features per signal (no look-ahead:
    uses only candles up to the trigger candle i-1)."""
    o = df1m["o"].values; h = df1m["h"].values
    l = df1m["l"].values; cl = df1m["c"].values
    idxmap = {ts.value: i for i, ts in enumerate(df1m.index)}
    dts = pd.to_datetime(c["dt"])
    d = c["d"]
    n = len(dts)
    F = np.zeros((n, 9))
    eps = 1e-9
    for k in range(n):
        i = idxmap.get(dts[k].value, None)
        if i is None or i < 6:
            continue
        j = i - 1  # trigger candle (entry is at bar i)
        rng = max(h[j] - l[j], eps)
        body = (cl[j] - o[j])
        F[k, 0] = abs(body) / rng                        # body fraction
        F[k, 1] = (cl[j] - l[j]) / rng                   # close position in range
        F[k, 2] = (h[j] - max(o[j], cl[j])) / rng        # upper wick
        F[k, 3] = (min(o[j], cl[j]) - l[j]) / rng        # lower wick
        # consecutive same-direction closes ending at j (signed by signal dir)
        run = 0
        for b in range(j, max(j - 8, 0), -1):
            up = cl[b] > o[b]
            if (up and d[k] == 1) or ((not up) and d[k] == -1):
                run += 1
            else:
                break
        F[k, 4] = run
        # 3-bar momentum in signal direction, normalized by recent range
        avg_rng = np.mean(h[j-4:j+1] - l[j-4:j+1]) + eps
        F[k, 5] = ((cl[j] - cl[j-3]) * d[k]) / avg_rng
        # trigger-candle range vs recent avg (expansion)
        F[k, 6] = rng / avg_rng
        # body direction agrees with signal (1/-1)
        F[k, 7] = np.sign(body) * d[k]
        # distance of close from bar midpoint, signed
        F[k, 8] = ((cl[j] - (h[j] + l[j]) / 2) * d[k]) / rng
    return F


def win_labels_and_pnl(df1m):
    c = T.build_candidates(df1m)
    pc = P4.precompute(c); pc["dt_sorted"] = c["dt"][pc["order"]]
    # pnl per signal (pc order) under the validated allocation
    idx_all = np.arange(pc["n"])
    p, dts = W.pnl_idx(pc, idx_all, ALLOC["a1"], ALLOC["a2"], ALLOC["tilt"], ALLOC["risk"])
    win = (p > 0).astype(float)
    F_sig = candle_features(df1m, c)          # in c order
    F = F_sig[pc["order"]]                      # reorder to pc order
    return pc, F, win, p


def fit_logistic(X, y, iters=800, lr=0.3, l2=1e-3):
    mu = X.mean(0); sd = X.std(0) + 1e-9
    Xs = (X - mu) / sd
    Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
    w = np.zeros(Xs.shape[1])
    for _ in range(iters):
        z = Xs @ w
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        grad = Xs.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]]
        w -= lr * grad
    return w, mu, sd


def predict(w, mu, sd, X):
    Xs = (X - mu) / sd
    Xs = np.hstack([np.ones((len(Xs), 1)), Xs])
    return 1 / (1 + np.exp(-np.clip(Xs @ w, -30, 30)))


def auc(y, p):
    order = np.argsort(p)
    y = y[order]
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    ranks = np.arange(1, len(y) + 1)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def weekly(pc, mask, weeks):
    idx = np.where(mask)[0]
    if len(idx) < 30:
        return None
    p, dts = W.pnl_idx(pc, idx, ALLOC["a1"], ALLOC["a2"], ALLOC["tilt"], ALLOC["risk"])
    ddt, ddd = W.stitched_dd(p, dts)
    return {"wk": float(p.sum()) / weeks, "dd_total": ddt, "dd_day": ddd,
            "gate": ddt < 350 and ddd < 200, "n": len(idx)}


def main():
    print("TRAIN = 2023-2024, TEST = 2025-2026 (true out-of-sample)\n")
    df_tr = fb.load_data()
    df_te = T.load_data()
    w_tr = (df_tr.index.max() - df_tr.index.min()).days / 7.0
    w_te = (df_te.index.max() - df_te.index.min()).days / 7.0

    pc_tr, X_tr, y_tr, p_tr = win_labels_and_pnl(df_tr)
    pc_te, X_te, y_te, p_te = win_labels_and_pnl(df_te)
    print(f"Train signals: {len(y_tr)} (win rate {y_tr.mean()*100:.1f}%)")
    print(f"Test  signals: {len(y_te)} (win rate {y_te.mean()*100:.1f}%)")

    w, mu, sd = fit_logistic(X_tr, y_tr)
    p_hat_tr = predict(w, mu, sd, X_tr)
    p_hat_te = predict(w, mu, sd, X_te)
    print(f"\nModel predictive power (does candle movement predict the win?):")
    print(f"  TRAIN AUC = {auc(y_tr, p_hat_tr):.3f}")
    print(f"  TEST  AUC = {auc(y_te, p_hat_te):.3f}   (0.50 = no predictive power)")

    # baseline weekly profit (take all signals)
    base_te = weekly(pc_te, np.ones(pc_te["n"], bool), w_te)
    print(f"\nBASELINE (all signals, 30/40/31 @ $25): +${base_te['wk']:.2f}/wk | "
          f"DDtot=${base_te['dd_total']:.0f} DDday=${base_te['dd_day']:.0f} | gate={base_te['gate']}")

    # choose threshold on TRAIN that maximizes train weekly profit, apply to TEST
    best_thr = None; best_tr_wk = -1e18
    for thr in np.linspace(0.30, 0.70, 41):
        m = weekly(pc_tr, p_hat_tr >= thr, w_tr)
        if m and m["gate"] and m["wk"] > best_tr_wk:
            best_tr_wk = m["wk"]; best_thr = thr
    print(f"\nBest TRAIN threshold: P(win) >= {best_thr:.2f} (train +${best_tr_wk:.2f}/wk)")

    test_filt = weekly(pc_te, p_hat_te >= best_thr, w_te)
    if test_filt is None:
        print("Filtered test set too small — reject.")
        return
    print(f"TEST with candle-prob filter: +${test_filt['wk']:.2f}/wk | kept {test_filt['n']}/{pc_te['n']} | "
          f"DDtot=${test_filt['dd_total']:.0f} DDday=${test_filt['dd_day']:.0f} | gate={test_filt['gate']}")

    improve = test_filt["wk"] - base_te["wk"]
    print("\n" + "=" * 70)
    if test_filt["gate"] and improve > 0:
        print(f"VERDICT: candle-prob model IMPROVES weekly profit by ${improve:+.2f}/wk "
              f"out-of-sample -> PUSH.")
    else:
        print(f"VERDICT: candle-prob model does NOT improve weekly profit "
              f"(${improve:+.2f}/wk OOS) -> DO NOT PUSH.")
    print("=" * 70)


if __name__ == "__main__":
    main()
