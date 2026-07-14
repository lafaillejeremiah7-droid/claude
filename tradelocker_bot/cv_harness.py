"""
CV HARNESS (Idea #3 foundation) — rigorous validation backbone.

Provides:
  - purged_walk_forward: expanding-window folds with embargo (no adjacent leak)
  - isotonic_calibrate: PAVA probability calibration (fit on train only)
  - bootstrap_auc_ci: AUC with a bootstrap confidence interval
  - expected_calibration_error, brier
  - evaluate: pooled OOS predictions + metrics for any (fit, predict) model
  - promotion_gate: promote a new model ONLY if it improves AUC + calibration +
    profit factor + Sharpe without materially worsening max DD (Phase-9 discipline;
    this rule auto-rejects "too good" leaked models).

numpy-only. Reusable by ideas #1 and #2.
"""
import numpy as np


# ---------------------------------------------------------------------------
def purged_walk_forward(times, n_folds=5, embargo=20):
    """Yield (train_idx, test_idx) expanding-window, sorted by time, embargo gap."""
    order = np.argsort(times)
    n = len(times); fold = n // n_folds
    for k in range(1, n_folds):
        tr_end = k * fold
        te_start = tr_end + embargo
        te_end = (k + 1) * fold if k < n_folds - 1 else n
        if te_start >= te_end:
            continue
        yield order[:tr_end], order[te_start:te_end]


# ---------------------------------------------------------------------------
def _pava(y, w):
    """Pool-adjacent-violators: non-decreasing isotonic fit over already-x-sorted y."""
    val, wt, cnt = [], [], []
    for j in range(len(y)):
        v, ww, cc = float(y[j]), float(w[j]), 1
        while val and val[-1] >= v:
            pv, pw, pc = val.pop(), wt.pop(), cnt.pop()
            v = (v * ww + pv * pw) / (ww + pw); ww += pw; cc += pc
        val.append(v); wt.append(ww); cnt.append(cc)
    out = np.empty(len(y)); idx = 0
    for v, c in zip(val, cnt):
        out[idx:idx + c] = v; idx += c
    return out


def isotonic_calibrate(p_tr, y_tr):
    """Return a calibrator function raw_prob -> calibrated_prob (fit on train only)."""
    o = np.argsort(p_tr)
    xs = p_tr[o]; fitted = _pava(y_tr[o], np.ones(len(y_tr)))
    xu, idx = np.unique(xs, return_index=True)
    fu = fitted[idx]
    if len(xu) < 2:
        mean = float(y_tr.mean())
        return lambda p: np.full_like(np.asarray(p, float), mean)
    return lambda p: np.interp(p, xu, fu, left=fu[0], right=fu[-1])


# ---------------------------------------------------------------------------
def auc(p, y):
    o = np.argsort(p); ys = y[o]; n1 = y.sum(); n0 = len(y) - n1
    return float(np.cumsum(1 - ys)[ys == 1].sum()) / (n0 * n1) if n0 * n1 > 0 else 0.5


def bootstrap_auc_ci(p, y, n_boot=500, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed); n = len(y); aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, len(idx)):
            continue
        aucs.append(auc(p[idx], y[idx]))
    aucs = np.array(aucs)
    return auc(p, y), float(np.quantile(aucs, alpha / 2)), float(np.quantile(aucs, 1 - alpha / 2))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1); ece = 0.0; n = len(y)
    for b in range(bins):
        m = (p >= edges[b]) & (p < edges[b + 1] if b < bins - 1 else p <= edges[b + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(p[m].mean() - y[m].mean())
    return float(ece)


# ---------------------------------------------------------------------------
def evaluate(fit_fn, predict_fn, X, y, times, n_folds=5, embargo=20, calibrate=True):
    """Pooled OOS eval with purged walk-forward + per-fold train-only calibration."""
    n = len(y); oos = np.full(n, np.nan)
    for tr, te in purged_walk_forward(times, n_folds, embargo):
        model = fit_fn(X[tr], y[tr])
        p_te = predict_fn(model, X[te])
        if calibrate:
            p_tr = predict_fn(model, X[tr])
            cal = isotonic_calibrate(p_tr, y[tr])
            p_te = cal(p_te)
        oos[te] = p_te
    m = ~np.isnan(oos)
    a, lo, hi = bootstrap_auc_ci(oos[m], y[m])
    return {"oos": oos, "mask": m, "auc": a, "auc_lo": lo, "auc_hi": hi,
            "brier": brier(oos[m], y[m]), "ece": expected_calibration_error(oos[m], y[m]),
            "n": int(m.sum())}


# ---------------------------------------------------------------------------
def promotion_gate(old, new, dd_tol=1.10):
    """Phase-9: promote new model only if it improves AUC, calibration (lower ECE),
    profit factor, Sharpe, and does NOT worsen max DD beyond dd_tol. Returns
    (promote: bool, reasons: list). Also flags 'too good to be true' (leak guard)."""
    reasons = []
    if new["auc"] > 0.75:
        reasons.append(f"REJECT: AUC {new['auc']:.3f} implausibly high — likely leakage")
        return False, reasons
    checks = [
        ("AUC", new.get("auc", 0) > old.get("auc", 0)),
        ("calibration(ECE lower)", new.get("ece", 1) < old.get("ece", 1)),
        ("profit_factor", new.get("pf", 0) > old.get("pf", 0)),
        ("sharpe", new.get("sharpe", -9) > old.get("sharpe", -9)),
        ("maxDD not worse", new.get("maxdd", 1e9) <= old.get("maxdd", 1e9) * dd_tol),
    ]
    for name, ok in checks:
        reasons.append(f"{'PASS' if ok else 'FAIL'}: {name}")
    return all(ok for _, ok in checks), reasons


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # self-test: random features must give AUC ~0.5 with CI straddling 0.5
    rng = np.random.default_rng(1)
    n = 2000; X = rng.normal(size=(n, 5)); y = (rng.random(n) < 0.6).astype(float)
    times = np.arange(n)
    import phase21_intermarket as P21
    def fit(Xt, yt): return P21.logistic_fit((Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9), yt, lam=2.0)
    def pred(m, Xt): return P21.logistic_pred(m, (Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9))
    r = evaluate(fit, pred, X, y, times)
    print(f"SELF-TEST (random features): AUC {r['auc']:.3f} [{r['auc_lo']:.3f}, {r['auc_hi']:.3f}] "
          f"ECE {r['ece']:.3f} n={r['n']}")
    print("  Expect AUC~0.50 and CI straddling 0.50 -> harness is honest." )
    # promotion gate demo
    old = {"auc": 0.52, "ece": 0.05, "pf": 1.1, "sharpe": 0.3, "maxdd": 300}
    leak = {"auc": 0.82, "ece": 0.02, "pf": 3.0, "sharpe": 2.0, "maxdd": 150}
    ok, why = promotion_gate(old, leak)
    print(f"  leak-model promoted? {ok} -> {why[0]}")
