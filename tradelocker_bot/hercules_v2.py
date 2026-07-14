"""
HERCULES v2 — Adaptive World Model Decision Engine (clean, no look-ahead).

Built to the user's 9-phase spec. STRICT rule: every FEATURE uses only
information available at/before the signal bar. Labels for training the
objective model come from the forward window (legitimate supervised learning),
but are NEVER used as features at decision time.

This script's job: honestly test whether Phase 2 (inferring the market's next
objective from entry-time state) adds out-of-sample predictive value above the
~0.51 AUC entry-time baseline. If it does, HERCULES has real edge. If not, it's
an honest, robust framework on a bounded edge — reported truthfully either way.
"""
import numpy as np
import pandas as pd
import phase12_ideal as P12
import tsai_optimize as T
import fullyear_backtest as fb
import phase21_intermarket as P21
import phase19_upgrade as U


# ---------------------------------------------------------------------------
# PHASE 1 — World Model: entry-time state features (CLEAN, no forward bars)
# ---------------------------------------------------------------------------
def pre_entry_path(S, df):
    """Path shape from the 10 bars BEFORE the signal (zero leakage)."""
    c = S["c"]; o = S["order"]
    dts = pd.to_datetime(c["dt"]); idx = df.index
    pos = idx.searchsorted(dts)
    H = df["h"].values; L = df["l"].values; C = df["c"].values
    n = len(dts); feat = np.zeros((n, 5))
    for i in range(n):
        p = pos[i]
        if p < 12:
            continue
        h10 = H[p-11:p-1]; l10 = L[p-11:p-1]; c10 = C[p-11:p-1]
        if len(c10) < 10 or c10.std() == 0:
            continue
        x = np.arange(10, dtype=float); x -= x.mean()
        cn = (c10 - c10.mean()) / (c10.std() + 1e-9)
        slope = np.polyfit(x, cn, 1)[0]; curv = np.polyfit(x, cn, 2)[0]
        rf = (h10[:3]-l10[:3]).mean(); rl = (h10[-3:]-l10[-3:]).mean()
        compress = rl/(rf+1e-9); mom = (c10[-1]-c10[0])/(c10.std()+1e-9)
        vt = np.polyfit(x, h10-l10, 1)[0]
        feat[i] = [slope, curv, compress, mom, vt]
    return feat[o]


def world_state(S, df, macro):
    """Full entry-time state vector (Phase 1)."""
    c = S["c"]; o = S["order"]
    path = pre_entry_path(S, df)
    intrinsic = np.column_stack([
        c["rsi"][o], c["vol_ratio"][o], c["trend_strength"][o],
        c["pb_dist"][o], S["hour"].astype(float),
    ])
    return np.hstack([path, macro, intrinsic])  # 5 + 6 + 5 = 16 features


# ---------------------------------------------------------------------------
# PHASE 2 — Market Objectives: label from forward window (for training only)
# ---------------------------------------------------------------------------
def objective_labels(S):
    """Operational, mutually-exclusive objective for each signal, derived from
    the forward window. Used ONLY as training labels, never as a feature.
      0 CONTINUATION  : ran favorably >=2R, little adverse first
      1 SWEEP_THEN_GO : dipped adverse >=0.5R early, then ran favorably >=1.5R
      2 REVERSAL      : went adverse >=2R, little favorable
      3 CHOP          : neither side reached ~1R (range)
    """
    c = S["c"]; o = S["order"]
    is_buy = c["d"] == 1
    H = c["H"]; L = c["L"]; entries = c["entries"]; atr = c["x_atr"]
    W = 40
    n = c["n"]; lab = np.full(n, 3, np.int8)  # default CHOP
    for i in range(n):
        a = atr[i]
        if a <= 0:
            continue
        h = H[i, :W]; l = L[i, :W]
        if is_buy[i]:
            fav = (np.maximum.accumulate(h) - entries[i]) / a
            adv = (entries[i] - np.minimum.accumulate(l)) / a
        else:
            fav = (entries[i] - np.minimum.accumulate(l)) / a
            adv = (np.maximum.accumulate(h) - entries[i]) / a
        fav_max = fav[-1]; adv_max = adv[-1]
        # early adverse before first favorable 1.5R
        first_fav15 = np.argmax(fav >= 1.5) if (fav >= 1.5).any() else W
        early_adv = adv[:max(1, first_fav15)].max() if first_fav15 > 0 else 0.0
        if fav_max >= 2.0 and adv_max < 1.0:
            lab[i] = 0
        elif fav_max >= 1.5 and early_adv >= 0.5:
            lab[i] = 1
        elif adv_max >= 2.0 and fav_max < 1.0:
            lab[i] = 2
        else:
            lab[i] = 3
    return lab[o]


def softmax_ovr_fit(X, y, n_classes, lam=2.0):
    """One-vs-rest logistic, returns list of weight vectors."""
    ws = []
    for k in range(n_classes):
        yk = (y == k).astype(float)
        ws.append(P21.logistic_fit(X, yk, lam=lam))
    return ws


def softmax_ovr_pred(ws, X):
    ps = np.column_stack([P21.logistic_pred(w, X) for w in ws])
    ps = ps / (ps.sum(1, keepdims=True) + 1e-9)
    return ps


def auc(p, y):
    o = np.argsort(p); ys = y[o]; n1 = y.sum(); n0 = len(y) - n1
    return float(np.cumsum(1-ys)[ys == 1].sum())/(n0*n1) if n0*n1 > 0 else .5


def main():
    print("Loading real data...")
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    df_tr = fb.load_data(); df_te = T.load_data()
    ry, dx = P21.load_macro()
    Fm_tr = P21.align_features(tr, ry, dx); Fm_te = P21.align_features(te, ry, dx)

    # PHASE 1: state
    print("PHASE 1 — building world-state (entry-time only)...")
    St_tr = world_state(tr, df_tr, Fm_tr); St_te = world_state(te, df_te, Fm_te)

    # outcomes (win label)
    _, r_tr, _, _, _ = U.outcomes(tr["c"], 0.45, 1.1363); r_tr = r_tr[tr["order"]]
    _, r_te, _, _, _ = U.outcomes(te["c"], 0.45, 1.1363); r_te = r_te[te["order"]]
    y_tr = (r_tr > 0).astype(float); y_te = (r_te > 0).astype(float)

    # PHASE 2: objective labels + model
    print("PHASE 2 — inferring market objectives...")
    lab_tr = objective_labels(tr); lab_te = objective_labels(te)
    names = ["CONTINUATION", "SWEEP_THEN_GO", "REVERSAL", "CHOP"]
    print("  objective distribution (train):",
          {names[k]: int((lab_tr == k).sum()) for k in range(4)})
    print("  objective distribution (test): ",
          {names[k]: int((lab_te == k).sum()) for k in range(4)})

    mu = St_tr.mean(0); sd = St_tr.std(0) + 1e-9
    Xtr = (St_tr - mu) / sd; Xte = (St_te - mu) / sd
    ws = softmax_ovr_fit(Xtr, lab_tr, 4, lam=2.0)
    P_te = softmax_ovr_pred(ws, Xte)  # OOS objective probabilities

    # How well can entry-state predict the objective, OOS?
    print("\n  OOS AUC of each objective (entry-state -> objective):")
    for k in range(4):
        yk = (lab_te == k).astype(float)
        print(f"    {names[k]:>14}: AUC {auc(P_te[:, k], yk):.4f}")

    # Intent alignment = P(favorable objective) = P(CONTINUATION) + P(SWEEP_THEN_GO)
    intent_te = P_te[:, 0] + P_te[:, 1]
    print(f"\n  INTENT ALIGNMENT (P(continuation)+P(sweep-then-go)) vs actual WIN:")
    print(f"    OOS AUC = {auc(intent_te, y_te):.4f}   (0.51 baseline = no edge)")
    q80 = np.quantile(intent_te, 0.8); q20 = np.quantile(intent_te, 0.2)
    print(f"    top-20% intent WR: {y_te[intent_te>=q80].mean()*100:.1f}% | "
          f"bottom-20%: {y_te[intent_te<=q20].mean()*100:.1f}%")

    print("\n" + "=" * 66)
    a = auc(intent_te, y_te)
    if a >= 0.55:
        print(f"PHASE 2 ADDS REAL EDGE (AUC {a:.4f}). HERCULES has predictive value.")
        print("-> proceed to full 9-phase decision engine + backtest.")
    elif a >= 0.53:
        print(f"PHASE 2 WEAK-POSITIVE (AUC {a:.4f}). Marginal; test in backtest.")
    else:
        print(f"PHASE 2 NO EDGE (AUC {a:.4f}). Objective inference doesn't beat noise")
        print("out-of-sample — same ceiling as direct win-prediction. HERCULES would be")
        print("an honest, robust FRAMEWORK on a bounded edge, not a source of alpha.")
    print("=" * 66)


if __name__ == "__main__":
    main()
