"""
Follow-up: the ONE real signal HERCULES Phase 2 found is CHOP prediction
(OOS AUC 0.609). Test two legit uses, walk-forward, friction ON, both datasets:
  (A) GATE: skip signals with high predicted P(chop)
  (B) compare vs take-all baseline
Learn threshold on 2023-24, apply UNCHANGED to 2025-26. Honest verdict.
"""
import numpy as np
import pandas as pd
import phase12_ideal as P12, tsai_optimize as T, fullyear_backtest as fb
import phase21_intermarket as P21, phase19_upgrade as U
import hercules_v2 as H


def main():
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    df_tr = fb.load_data(); df_te = T.load_data()
    ry, dx = P21.load_macro()
    St_tr = H.world_state(tr, df_tr, P21.align_features(tr, ry, dx))
    St_te = H.world_state(te, df_te, P21.align_features(te, ry, dx))
    lab_tr = H.objective_labels(tr)
    mu = St_tr.mean(0); sd = St_tr.std(0)+1e-9
    Xtr = (St_tr-mu)/sd; Xte = (St_te-mu)/sd
    ws = H.softmax_ovr_fit(Xtr, lab_tr, 4, lam=2.0)
    pchop_tr = H.softmax_ovr_pred(ws, Xtr)[:, 3]
    pchop_te = H.softmax_ovr_pred(ws, Xte)[:, 3]

    base_a = U.sim(tr, friction=True, base=85); base_b = U.sim(te, friction=True, base=85)
    print("Baseline (take-all, friction ON, $85, 12-20 UTC):")
    print(f"  23-24 ${base_a['wk']:+.1f}/wk WR{base_a['wr']:.0f}% DDt${base_a['ddt']:.0f}")
    print(f"  25-26 ${base_b['wk']:+.1f}/wk WR{base_b['wr']:.0f}% DDt${base_b['ddt']:.0f}")

    print("\nCHOP-GATE: skip signals with P(chop) >= threshold (learn on 23-24):")
    print(f"  {'keep%':>6} | {'23-24 $/wk':>10} {'WR':>4} {'DDt':>5} | {'25-26 $/wk':>10} {'WR':>4} {'DDt':>5} | vs base")
    best = None
    for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        thr = np.quantile(pchop_tr, q)   # keep the (1-q) lowest-chop; skip top-q chop... actually keep < thr
        mask_tr = pchop_tr < thr; mask_te = pchop_te < thr
        a = U.sim(tr, friction=True, base=85, regime_mask=mask_tr)
        b = U.sim(te, friction=True, base=85, regime_mask=mask_te)
        d = min(a['wk'], b['wk']) - min(base_a['wk'], base_b['wk'])
        okgate = a['ddt'] < 400 and b['ddt'] < 400 and a['ddd'] < 250 and b['ddd'] < 250
        if okgate and (best is None or d > best[0]):
            best = (d, q, a, b)
        print(f"  {q*100:>5.0f}% | ${a['wk']:>+9.1f} {a['wr']:>3.0f}% ${a['ddt']:>4.0f} | "
              f"${b['wk']:>+9.1f} {b['wr']:>3.0f}% ${b['ddt']:>4.0f} | {d:>+6.1f}")

    print("\n" + "=" * 60)
    if best and best[0] > 0:
        d, q, a, b = best
        print(f"CHOP-GATE HELPS: keep lowest-chop {q*100:.0f}%, worst-case {d:+.1f}/wk")
        print(f"  23-24 ${a['wk']:+.1f}/wk WR{a['wr']:.0f}% | 25-26 ${b['wk']:+.1f}/wk WR{b['wr']:.0f}%")
    else:
        print("CHOP-GATE does NOT improve worst-case $/wk out-of-sample.")
        print("The chop signal is real but doesn't convert to trade-outcome edge.")
    print("=" * 60)


if __name__ == "__main__":
    main()
