"""
PHASE 26b — does the REAL chop signal (OOS AUC 0.65) have MONETARY value?
Bucket OOS trades by predicted chop-probability; look at realized runner-R,
win-rate, and R-magnitude per bucket. If high-chop trades have systematically
lower R (runner dies), the signal is monetizable via geometry/sizing.
Purged walk-forward OOS predictions only. Honest.
"""
import numpy as np
import pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import phase21_intermarket as P21, phase19_upgrade as U
import hercules_v2 as H
import cv_harness as CV
import phase26_ideas_through_cv as P26


def main():
    te = P12.prep(T.load_data()); c = te["c"]; o = te["order"]
    dts = pd.to_datetime(c["dt"][o])
    times = dts.values.astype("datetime64[ns]").astype(np.int64)
    _, r_run, _, _, _ = U.outcomes(c, 0.45, 1.1363); r_run = r_run[o]
    y_win = (r_run > 0).astype(float)
    df = T.load_data(); ry, dx = P21.load_macro()
    state = H.world_state(te, df, P21.align_features(te, ry, dx))
    lab = H.objective_labels(te); y_chop = (lab == 3).astype(float)

    # OOS chop predictions via harness
    res = CV.evaluate(P26.nl_fit, P26.nl_pred, state, y_chop, times)
    p = res["oos"]; m = res["mask"]
    pc = p[m]; rr = r_run[m]; yw = y_win[m]

    print(f"OOS trades: {m.sum()} | chop-pred AUC {res['auc']:.3f} [{res['auc_lo']:.3f},{res['auc_hi']:.3f}]\n")
    print("Runner-R bucketed by predicted chop-probability (quintiles):")
    print(f"{'bucket':>18} {'n':>5} {'winrate':>8} {'mean R':>8} {'sum R':>8} {'mean|R|win':>10}")
    qs = np.quantile(pc, [0, .2, .4, .6, .8, 1.0])
    for b in range(5):
        lo, hi = qs[b], qs[b + 1]
        mask = (pc >= lo) & (pc <= hi if b == 4 else pc < hi)
        if mask.sum() == 0:
            continue
        rb = rr[mask]; wb = yw[mask]
        winr = wb.mean() * 100
        mean_r = rb.mean()
        win_mag = rb[rb > 0].mean() if (rb > 0).any() else 0
        label = f"{'LOW chop' if b==0 else 'HIGH chop' if b==4 else f'Q{b+1}'}"
        print(f"{label:>18} {mask.sum():>5} {winr:>7.1f}% {mean_r:>+8.3f} {rb.sum():>+8.1f} {win_mag:>+10.3f}")

    # correlation
    corr = np.corrcoef(pc, rr)[0, 1]
    print(f"\n  corr(chop_prob, realized runner-R) = {corr:+.4f}")
    # low-chop vs high-chop mean R
    lo_mask = pc <= np.quantile(pc, 0.4); hi_mask = pc >= np.quantile(pc, 0.6)
    print(f"  LOW-chop mean R = {rr[lo_mask].mean():+.3f} | HIGH-chop mean R = {rr[hi_mask].mean():+.3f}")
    diff = rr[lo_mask].mean() - rr[hi_mask].mean()
    print(f"  edge from favoring low-chop = {diff:+.3f} R/trade")
    print("\n" + "=" * 60)
    if corr < -0.06 and diff > 0.05:
        print(f"CHOP SIGNAL HAS MONETARY VALUE: low-chop trades earn {diff:+.3f}R more.")
        print("-> size up in low-chop, down in high-chop. Worth building.")
    else:
        print(f"CHOP SIGNAL: real for classification but NO monetary value")
        print(f"(corr {corr:+.3f}, R-diff {diff:+.3f}). Doesn't separate profitable trades.")
    print("=" * 60)


if __name__ == "__main__":
    main()
