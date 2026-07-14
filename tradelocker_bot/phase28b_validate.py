"""
PHASE 28b — rigorously validate the phase-28 search winner before believing it.
Full 2025-26, real friction sim, risk sweep, gate-checked, vs phase-15 baseline.
Also a 4-fold walk-forward to check it's not confirm-window luck.
"""
import numpy as np, pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import phase28_strategy_search as P28
import phase25_adaptive_geom as P25

WINNER = (1.481, 0.5938, 1.5187, 0.4656, 2.0532, 0.7363, 1.8521, 1.234)
BASE = (1.1342, 0.346, 1.2653, 0.45, 6.26, 0.45, 1.1363, 2.509)


def main():
    te = P12.prep(T.load_data()); P = P28.precompute(te)
    wd = P["dt"].weekday.values
    SPREAD = P28.SPREAD
    allsig = np.ones(P["n"], bool)

    def run(cfg, mask, risk):
        r, sl_atr = P28.fast_R(P, *cfg)
        sldp = sl_atr * P["atr"] + SPREAD/2
        return P25.fric_sim(P["dt"], P["day"], P["hour"], wd, r, sldp, mask,
                            base=risk, tp1hit=(r > 0), a1=cfg[5])

    print("FULL 2025-26, real friction sim, risk sweep (gate = DDt<400, DDd<250):")
    print(f"{'risk':>5} | {'BASELINE $/wk':>13} {'WR':>4} {'DDt':>5} {'g':>4} | "
          f"{'WINNER $/wk':>12} {'WR':>4} {'DDt':>5} {'g':>4}")
    for risk in [40, 47, 55, 60, 70, 85]:
        b = run(BASE, allsig, risk); w = run(WINNER, allsig, risk)
        print(f"  ${risk:>3} | ${b['wk']:>+12.1f} {b['wr']:>3.0f}% ${b['ddt']:>4.0f} "
              f"{'P' if b['gate'] else 'F':>4} | ${w['wk']:>+11.1f} {w['wr']:>3.0f}% "
              f"${w['ddt']:>4.0f} {'P' if w['gate'] else 'F':>4}")

    # 4-fold walk-forward robustness (each quarter as its own OOS account)
    print("\n4-fold time-slice robustness at $47 (winner vs baseline, gate-safe):")
    order = np.argsort(P["dt"].values.astype('datetime64[ns]').astype(np.int64))
    n = P["n"]; q = n // 4
    for f in range(4):
        seg = np.zeros(n, bool); seg[order[f*q:(f+1)*q if f < 3 else n]] = True
        b = run(BASE, seg, 47); w = run(WINNER, seg, 47)
        print(f"  Q{f+1}: baseline ${b['wk']:+.1f}/wk (DDt${b['ddt']:.0f} {'P' if b['gate'] else 'F'}) | "
              f"winner ${w['wk']:+.1f}/wk (DDt${w['ddt']:.0f} {'P' if w['gate'] else 'F'})")


if __name__ == "__main__":
    main()
