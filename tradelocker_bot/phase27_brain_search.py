"""
PHASE 27 — 300k-trial creative search to improve the PREDICTIVE BRAIN.

Honest protocol (3-way TIME split, no leakage):
  TRAIN (first 50%)  -> fit each candidate model
  SEARCH (next 25%)  -> used to DECIDE if a trial "improves" (greedy, compounding)
  CONFIRM (last 25%) -> NEVER used for selection; final honesty check

A trial "improves" when its SEARCH-OOS AUC beats the running best (compounding
greedy search over engineered features + model hyperparameters). We then also
measure the CONFIRM-OOS AUC of the final compounded model. If SEARCH creeps up
while CONFIRM stays ~0.50, the "improvements" were overfitting — which is the
honest answer about whether a real edge exists.

Reports: trials run, # improvements (out of 300k), final SEARCH AUC, and the
CONFIRM-OOS AUC (the only number that matters).
"""
import time
import numpy as np
import pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import phase21_intermarket as P21, phase19_upgrade as U
import phase24_intraday_wf as P24
import hercules_v2 as H


def build_feature_pool(te):
    c = te["c"]; o = te["order"]
    df = T.load_data(); ry, dx = P21.load_macro()
    base = {}
    base["rsi"] = c["rsi"][o]; base["vol_ratio"] = c["vol_ratio"][o]
    base["trend"] = c["trend_strength"][o]; base["pb"] = c["pb_dist"][o]
    base["hour"] = te["hour"].astype(float); base["atr5"] = c["atr5"][o]
    dm = P21.align_features(te, ry, dx)
    for i, nm in enumerate(["ry1", "ry3", "ry5", "dx1", "dx3", "dx5"]):
        base[nm] = dm[:, i]
    im = P24.intraday_macro_features(c["dt"][o])
    for i, nm in enumerate(["tnx1h", "tnx4h", "tnx24h", "dxy1h", "dxy4h", "dxy24h",
                            "eur1h", "eur4h", "eur24h"]):
        base[nm] = im[:, i]
    pp = H.pre_entry_path(te, df)
    for i, nm in enumerate(["pslope", "pcurv", "pcompress", "pmom", "pvoltrend"]):
        base[nm] = pp[:, i]
    # time cyclical
    base["hsin"] = np.sin(2*np.pi*base["hour"]/24); base["hcos"] = np.cos(2*np.pi*base["hour"]/24)
    # derived: squares + a few products/ratios (all entry-time, no leak)
    names = list(base.keys()); pool = dict(base)
    for nm in names:
        v = base[nm]
        pool[nm+"_sq"] = v*v
    prods = [("ry5", "vol_ratio"), ("dx5", "trend"), ("tnx24h", "vol_ratio"),
             ("dxy24h", "trend"), ("pmom", "vol_ratio"), ("rsi", "trend"),
             ("pslope", "pmom"), ("ry1", "dx1"), ("tnx1h", "dxy1h"), ("hour", "vol_ratio")]
    for a, b in prods:
        if a in base and b in base:
            pool[f"{a}x{b}"] = base[a]*base[b]
    keys = list(pool.keys())
    M = np.column_stack([np.nan_to_num(pool[k], nan=0.0) for k in keys])
    return keys, M


def ridge_auc(Xtr, ytr, Xev, yev, lam):
    """Closed-form ridge, return AUC on eval set. Fast."""
    n, k = Xtr.shape
    A = Xtr.T @ Xtr + lam*np.eye(k)
    try:
        w = np.linalg.solve(A, Xtr.T @ ytr)
    except np.linalg.LinAlgError:
        return 0.5
    p = Xev @ w
    o = np.argsort(p); ys = yev[o]; n1 = yev.sum(); n0 = len(yev)-n1
    if n0*n1 == 0:
        return 0.5
    return float(np.cumsum(1-ys)[ys == 1].sum())/(n0*n1)


def main():
    print("Building feature pool (all entry-time, no leakage)...")
    te = P12.prep(T.load_data()); c = te["c"]; o = te["order"]
    dts = pd.to_datetime(c["dt"][o]).values.astype("datetime64[ns]").astype(np.int64)
    _, r, _, _, _ = U.outcomes(c, 0.45, 1.1363); r = r[o]
    y = (r > 0).astype(float)

    keys, M = build_feature_pool(te)
    # time-order
    order = np.argsort(dts); M = M[order]; y = y[order]
    # standardize on full (only for scale; ridge handles it, and split is time-based)
    n = len(y); i1 = n//2; i2 = i1 + n//4
    # standardize using TRAIN stats only (honest)
    mu = M[:i1].mean(0); sd = M[:i1].std(0)+1e-9
    Ms = (M-mu)/sd
    Xtr, ytr = Ms[:i1], y[:i1]
    Xse, yse = Ms[i1:i2], y[i1:i2]      # SEARCH
    Xcf, ycf = Ms[i2:], y[i2:]          # CONFIRM (untouched)
    print(f"features in pool: {len(keys)} | TRAIN {i1} SEARCH {i2-i1} CONFIRM {n-i2}\n")

    rng = np.random.default_rng(27)
    K = len(keys)
    # start: single best-on-search single feature
    cur = [int(rng.integers(K))]
    def ev(cols, lam):
        return ridge_auc(Xtr[:, cols], ytr, Xse[:, cols], yse, lam)
    best_lam = 5.0
    best_auc = ev(cur, best_lam)
    improvements = 0
    TARGET = 300_000
    t0 = time.time(); trial = 0
    confirm_trace = []
    while trial < TARGET:
        trial += 1
        move = rng.random()
        cand = list(cur)
        if move < 0.45 or len(cand) < 2:      # add
            cand.append(int(rng.integers(K)))
        elif move < 0.75:                      # swap
            cand[int(rng.integers(len(cand)))] = int(rng.integers(K))
        else:                                  # drop
            cand.pop(int(rng.integers(len(cand))))
        cand = list(dict.fromkeys(cand))       # unique
        if not cand or len(cand) > 16:
            continue
        lam = float(rng.choice([0.5, 1, 2, 5, 10, 20]))
        a = ridge_auc(Xtr[:, cand], ytr, Xse[:, cand], yse, lam)
        if a > best_auc + 1e-5:
            best_auc = a; cur = cand; best_lam = lam; improvements += 1
            if improvements % 25 == 0:
                cfa = ridge_auc(Xtr[:, cur], ytr, Xcf[:, cur], ycf, best_lam)
                confirm_trace.append((improvements, best_auc, cfa))
        if trial % 50000 == 0:
            el = time.time()-t0
            print(f"  trial {trial:>7}/{TARGET} | improvements {improvements:>4} | "
                  f"best SEARCH AUC {best_auc:.4f} | {el:.0f}s")
        if time.time()-t0 > 900:   # wall cap
            print(f"  [wall cap at trial {trial}]")
            break

    confirm_auc = ridge_auc(Xtr[:, cur], ytr, Xcf[:, cur], ycf, best_lam)
    print("\n" + "=" * 68)
    print(f"BRAIN SEARCH DONE: {trial} trials")
    print(f"  IMPROVEMENTS (beat running-best on SEARCH-OOS): {improvements}")
    print(f"  final SEARCH-OOS AUC:  {best_auc:.4f}  ({len(cur)} features, lam={best_lam})")
    print(f"  final CONFIRM-OOS AUC: {confirm_auc:.4f}  <-- the ONLY number that matters")
    print("=" * 68)
    if confirm_trace:
        print("  SEARCH vs CONFIRM as 'improvements' accumulated (overfit check):")
        print(f"  {'#impr':>6} {'SEARCH':>7} {'CONFIRM':>7}")
        for imp, sa, ca in confirm_trace[::max(1, len(confirm_trace)//10)]:
            print(f"  {imp:>6} {sa:>7.4f} {ca:>7.4f}")
    print()
    if confirm_auc > 0.55:
        print(f"VERDICT: real edge found (CONFIRM AUC {confirm_auc:.3f} > 0.55).")
    else:
        print(f"VERDICT: despite {improvements} 'improvements' on SEARCH, CONFIRM AUC")
        print(f"is {confirm_auc:.3f} ~ 0.50. The improvements were OVERFITTING, not edge.")
        print("This is the honest proof: you cannot search your way to an edge that")
        print("isn't in the data. (This is exactly why the CV harness exists.)")


if __name__ == "__main__":
    main()
