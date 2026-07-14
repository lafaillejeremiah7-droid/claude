"""
PHASE 22b — test the weak signal (AUC 0.536) as a live brain gate.
If it lifts $/wk or cuts DD vs take-all, WITH friction, on BOTH datasets → ship it.
"""
import numpy as np
import pandas as pd
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12
import phase19_upgrade as U
import phase21_intermarket as P21
import phase22_pathshape as P22


def main():
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    ry, dx = P21.load_macro()
    F_tr = P22.expanded_features(tr, P21.align_features(tr, ry, dx))
    F_te = P22.expanded_features(te, P21.align_features(te, ry, dx))
    sld_tr, r_tr, _, _, _ = U.outcomes(tr["c"], 0.45, 1.1363)
    sld_te, r_te, _, _, _ = U.outcomes(te["c"], 0.45, 1.1363)
    r_tr = r_tr[tr["order"]]; r_te = r_te[te["order"]]
    y_tr = (r_tr > 0).astype(float)

    mu = F_tr.mean(0); sd = F_tr.std(0) + 1e-9
    Xn_tr = (F_tr - mu) / sd; Xn_te = (F_te - mu) / sd
    w = P21.logistic_fit(Xn_tr, y_tr, lam=2.0)
    p_tr = P21.logistic_pred(w, Xn_tr); p_te = P21.logistic_pred(w, Xn_te)

    base_a = U.sim(tr, friction=True); base_b = U.sim(te, friction=True)
    print(f"NO brain (take-all, friction ON):")
    print(f"  23-24: ${base_a['wk']:+.1f}/wk WR{base_a['wr']:.0f}% DDt${base_a['ddt']:.0f} DDd${base_a['ddd']:.0f} sig/d {base_a['sig_d']:.1f}")
    print(f"  25-26: ${base_b['wk']:+.1f}/wk WR{base_b['wr']:.0f}% DDt${base_b['ddt']:.0f} DDd${base_b['ddd']:.0f} sig/d {base_b['sig_d']:.1f}")

    print(f"\nBRAIN GATE sweep (take only signals where model predicts P(win) >= threshold):")
    print(f"  {'q-thr':>6} | {'23-24 $/wk':>10} {'WR':>4} {'DDt':>5} {'DDd':>5} {'s/d':>4} | "
          f"{'25-26 $/wk':>10} {'WR':>4} {'DDt':>5} {'DDd':>5} {'s/d':>4} | {'vs base':>7}")
    best = None
    for q in np.linspace(0.05, 0.50, 19):
        thr_tr = np.quantile(p_tr, q)
        mask_tr = p_tr >= thr_tr; mask_te = p_te >= thr_tr  # same absolute threshold
        a = U.sim(tr, friction=True, regime_mask=mask_tr)
        b = U.sim(te, friction=True, regime_mask=mask_te)
        worst = min(a['wk'], b['wk']); d = worst - min(base_a['wk'], base_b['wk'])
        ok = a['gate'] and b['gate'] and a['ddd'] < 120 and b['ddd'] < 120
        if ok and (best is None or worst > best[0]):
            best = (worst, q, a, b)
        print(f"  {q:>6.2f} | ${a['wk']:>+9.1f} {a['wr']:>3.0f}% ${a['ddt']:>4.0f} ${a['ddd']:>4.0f} "
              f"{a['sig_d']:>3.1f} | ${b['wk']:>+9.1f} {b['wr']:>3.0f}% ${b['ddt']:>4.0f} ${b['ddd']:>4.0f} "
              f"{b['sig_d']:>3.1f} | {d:>+6.1f}")

    print("\n" + "=" * 70)
    if best and best[0] > min(base_a['wk'], base_b['wk']):
        _, q, a, b = best
        print(f"BRAIN GATE HELPS. Best q={q:.2f}:")
        print(f"  23-24: ${a['wk']:+.1f}/wk WR{a['wr']:.0f}% DDt${a['ddt']:.0f} DDd${a['ddd']:.0f}")
        print(f"  25-26: ${b['wk']:+.1f}/wk WR{b['wr']:.0f}% DDt${b['ddt']:.0f} DDd${b['ddd']:.0f}")
        print(f"  vs no-brain: {best[0]-min(base_a['wk'],base_b['wk']):+.1f}/wk worst-case")
    else:
        print("BRAIN GATE DOES NOT HELP (no threshold improves worst-case $/wk vs take-all).")
    print("=" * 70)


if __name__ == "__main__":
    main()
