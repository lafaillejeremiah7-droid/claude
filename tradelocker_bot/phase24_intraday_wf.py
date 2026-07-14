"""
PHASE 24 — IDEA #2: intraday intermarket signal, walk-forward WITHIN 2025-26.

Focus on the now (last ~18 months). Full intraday macro coverage (^TNX, DXY,
EURUSD hourly). Proper expanding-window walk-forward with embargo. Honest test:
does HOURLY macro momentum predict XAU trade wins out-of-sample, where DAILY
macro failed (AUC 0.493)?
"""
import numpy as np
import pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import phase21_intermarket as P21, phase19_upgrade as U


def asof_vals(times, vals, query_ns):
    """Value as-of (last known at/before) each query time. times sorted ns int."""
    pos = np.searchsorted(times, query_ns, side="right") - 1
    pos = np.clip(pos, 0, len(vals) - 1)
    out = vals[pos]
    out[query_ns < times[0]] = np.nan
    return out


def intraday_macro_features(sig_dts):
    """9 momentum features: {tnx,dxy,eurusd} x {1h,4h,24h} change."""
    q = pd.to_datetime(sig_dts).values.astype("datetime64[ns]").astype(np.int64)
    feats = []
    H = np.int64(3600 * 1e9)
    for fn in ["tnx", "dxy", "eurusd"]:
        s = pd.read_parquet(f"data/intraday_{fn}.parquet").sort_index()
        t = s.index.values.astype("datetime64[ns]").astype(np.int64)
        v = s["value"].values.astype(float)
        now = asof_vals(t, v, q)
        for lag in [1, 4, 24]:
            prev = asof_vals(t, v, q - lag * H)
            feats.append(now - prev)
    F = np.column_stack(feats)
    return np.nan_to_num(F, nan=0.0)


def auc(p, y):
    o = np.argsort(p); ys = y[o]; n1 = y.sum(); n0 = len(y) - n1
    return float(np.cumsum(1-ys)[ys == 1].sum())/(n0*n1) if n0*n1 > 0 else .5


def walk_forward_auc(X, y, dts, n_folds=5, embargo=20, tag=""):
    """Expanding-window walk-forward. Returns pooled OOS AUC."""
    order = np.argsort(dts)
    X, y = X[order], y[order]
    n = len(y); fold = n // n_folds
    oos_p = np.full(n, np.nan)
    for k in range(1, n_folds):
        tr_end = k * fold
        te_start = tr_end + embargo
        te_end = (k + 1) * fold if k < n_folds - 1 else n
        if te_start >= te_end:
            continue
        Xtr, ytr = X[:tr_end], y[:tr_end]
        Xte = X[te_start:te_end]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
        a = (Xtr - mu) / sd; b = (Xte - mu) / sd
        # nonlinear expansion (squared + top interaction pairs)
        co = np.abs(P21.logistic_fit(a, ytr, lam=2.0)[1:])
        ti = np.where(co > np.median(co))[0]
        prs = []
        for i in range(len(ti)):
            for j in range(i+1, len(ti)):
                prs.append((ti[i], ti[j], co[ti[i]]*co[ti[j]]))
        prs.sort(key=lambda x: -x[2]); ii = [(p[0], p[1]) for p in prs[:5]]
        ita = np.column_stack([a[:, i]*a[:, j] for i, j in ii]) if ii else np.zeros((len(a), 0))
        itb = np.column_stack([b[:, i]*b[:, j] for i, j in ii]) if ii else np.zeros((len(b), 0))
        X3a = np.hstack([a, a**2, ita]); X3b = np.hstack([b, b**2, itb])
        w = P21.logistic_fit(X3a, ytr, lam=5.0)
        oos_p[te_start:te_end] = P21.logistic_pred(w, X3b)
    m = ~np.isnan(oos_p)
    a = auc(oos_p[m], y[m])
    print(f"  {tag}: pooled OOS AUC = {a:.4f}  (n_oos={m.sum()})")
    return a, oos_p, order, m


def main():
    print("Focus: 2025-26 only. Walk-forward within period (train past -> test future).\n")
    te = P12.prep(T.load_data())
    c = te["c"]; o = te["order"]
    dts = pd.to_datetime(c["dt"][o])
    dts_ns = dts.values.astype("datetime64[ns]").astype(np.int64)

    _, r, _, _, _ = U.outcomes(c, 0.45, 1.1363); r = r[o]
    y = (r > 0).astype(float)
    print(f"Signals: {len(y)} | WR: {y.mean()*100:.1f}%\n")

    # feature sets
    intr = np.column_stack([c["rsi"][o], c["vol_ratio"][o], c["trend_strength"][o],
                            c["pb_dist"][o], te["hour"].astype(float)])
    ry, dx = P21.load_macro()
    daily_macro = P21.align_features(te, ry, dx)                    # 6 daily features
    intraday_macro = intraday_macro_features(c["dt"][o])            # 9 hourly features

    print("Walk-forward OOS AUC by feature set (5 folds, embargo 20):")
    walk_forward_auc(intr, y, dts_ns, tag="intrinsic only          ")
    walk_forward_auc(np.hstack([intr, daily_macro]), y, dts_ns, tag="intrinsic + DAILY macro ")
    walk_forward_auc(np.hstack([intr, intraday_macro]), y, dts_ns, tag="intrinsic + INTRADAY mac")
    a_all, oos_p, order, m = walk_forward_auc(
        np.hstack([intr, intraday_macro, daily_macro]), y, dts_ns, tag="intrinsic + BOTH macro  ")

    print("\n" + "=" * 60)
    if a_all >= 0.55:
        print(f"INTRADAY MACRO ADDS REAL EDGE (AUC {a_all:.4f}). Proceed to backtest.")
    elif a_all >= 0.53:
        print(f"WEAK-POSITIVE (AUC {a_all:.4f}). Marginal — test $/wk conversion.")
    else:
        print(f"NO EDGE (best AUC {a_all:.4f}). Intraday macro doesn't beat noise OOS.")
    print("=" * 60)


if __name__ == "__main__":
    main()
