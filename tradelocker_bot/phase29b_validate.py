"""
PHASE 29b — validate the phase-29 finding (later TP1/breakeven ~0.8R vs current
0.45R) with the ACCURATE bar-by-bar outcome sim + FULL sequential friction sim
(circuit-breaker, cooldown, lot-cap). Full 2025-26 + 4-quarter robustness.
No proxy. If the $254/wk was real, it survives here; if it was a scoring
artifact, it collapses.
"""
import numpy as np, pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import phase28c_accurate as ACC
import phase25_adaptive_geom as P25


def main():
    te = P12.prep(T.load_data()); c = te["c"]; o = te["order"]
    dt = pd.to_datetime(c["dt"][o]); day = te["day"]; hour = te["hour"]; wd = dt.weekday.values
    n = c["n"]; allsig = np.ones(n, bool)

    def build(tp1):
        r, sld, h = ACC.accurate_outcomes(c, 1.1342, 0.346, 1.2653,
                                          [tp1, 3.16, 6.26], [0.45, 0.16, 0.39], 1.1363, 0.849, 2.509)
        return r[o], sld[o], h[o]

    def sim(r, sld, h, mask, risk):
        return P25.fric_sim(dt, day, hour, wd, r, sld, mask, base=risk, tp1hit=h, a1=0.45)

    print("ACCURATE full sim — first-partial level (TP1) sweep, full 2025-26:")
    print(f"{'TP1':>5} | {'$/wk':>8} {'WR':>4} {'DDt':>5} {'DDd':>5} {'gate':>4}")
    cache = {}
    for tp1 in [0.45, 0.55, 0.65, 0.8, 1.0]:
        r, sld, h = build(tp1); cache[tp1] = (r, sld, h)
        for risk in [47]:
            m = sim(r, sld, h, allsig, risk)
            tag = " <-- current" if tp1 == 0.45 else ""
            print(f"  {tp1:>4.2f} | ${m['wk']:>+7.1f} {m['wr']:>3.0f}% ${m['ddt']:>4.0f} "
                  f"${m['ddd']:>4.0f} {'PASS' if m['gate'] else 'FAIL':>4}{tag}")

    print("\n4-quarter robustness (TP1 0.45 vs 0.8) at $47:")
    order = np.argsort(dt.values.astype('datetime64[ns]').astype(np.int64)); q = n//4
    r45, s45, h45 = cache[0.45]; r80, s80, h80 = cache[0.8]
    for f in range(4):
        seg = np.zeros(n, bool); seg[order[f*q:(f+1)*q if f < 3 else n]] = True
        a = sim(r45, s45, h45, seg, 47); b = sim(r80, s80, h80, seg, 47)
        print(f"  Q{f+1}: TP1=0.45 ${a['wk']:+.1f}/wk (DDt${a['ddt']:.0f} {'P' if a['gate'] else 'F'}) | "
              f"TP1=0.80 ${b['wk']:+.1f}/wk (DDt${b['ddt']:.0f} {'P' if b['gate'] else 'F'})")


if __name__ == "__main__":
    main()
