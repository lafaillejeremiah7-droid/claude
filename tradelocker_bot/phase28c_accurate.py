"""
PHASE 28c — DECISIVE honest validation: winner (2-leg) vs REAL phase-15 (3-leg),
both via the ACCURATE bar-by-bar outcome sim (real BE trailing, real timeout),
then the full friction sim. Full 2025-26 + 4-quarter robustness. No fast proxy.
"""
import numpy as np, pandas as pd
import phase12_ideal as P12, tsai_optimize as T
import phase25_adaptive_geom as P25

SPREAD = T.SPREAD; MAX_HOLD = T.MAX_HOLD


def accurate_outcomes(c, base_sl, vlo, vhi, tgts_R, alloc, tgain, tp_lo, tp_hi):
    n = c["n"]; is_buy = c["d"] == 1
    H, L, Cc = c["H"], c["L"], c["C"]; entries = c["entries"]
    vrc = np.clip(c["vol_ratio"], vlo, vhi)
    sld = base_sl * vrc * c["x_atr"] + SPREAD/2
    ext = np.clip(1 + tgain*(c["trend_strength"]-1), tp_lo, tp_hi)
    m_ = len(tgts_R)
    tpx = []
    for i, tr in enumerate(tgts_R):
        scale = ext if i == m_-1 else 1.0
        tpx.append(np.where(is_buy, entries + tr*scale*sld + SPREAD,
                            entries - tr*scale*sld - SPREAD))
    e_sl0 = np.where(is_buy, entries - sld, entries + sld)
    res = np.zeros(n); level = np.zeros(n, int); done = np.zeros(n, bool); be = np.zeros(n, bool)
    cumA = np.concatenate([[0.0], np.cumsum(alloc)])
    tp1hit = np.zeros(n, bool)
    for j in range(MAX_HOLD):
        if done.all(): break
        hi = H[:, j]; lo = L[:, j]
        sl_lvl = np.where(be, entries, e_sl0)
        hitsl = (~done) & np.where(is_buy, lo <= sl_lvl, hi >= sl_lvl)
        res = np.where(hitsl & ~be, -1.0, res); done = done | hitsl
        for i in range(m_):
            tp = tpx[i]
            reach = (~done) & (level == i) & np.where(is_buy, hi >= tp, lo <= tp)
            scale = ext if i == m_-1 else 1.0
            res = np.where(reach, res + alloc[i]*tgts_R[i]*scale, res)
            level = np.where(reach, i+1, level)
            if i == 0: tp1hit = tp1hit | reach
            if m_ > 1: be = np.where(reach, True, be)
            if i == m_-1: done = done | reach
    if not done.all():
        rC = (Cc[:, -1]-entries)*np.where(is_buy, 1.0, -1.0)/sld
        res = np.where(~done, res + (1.0-cumA[level])*rC, res)
    return res, sld, tp1hit


def main():
    te = P12.prep(T.load_data()); c = te["c"]; o = te["order"]
    wd = pd.to_datetime(c["dt"][o]).weekday.values
    dt = pd.to_datetime(c["dt"][o]); day = te["day"]; hour = te["hour"]; n = c["n"]

    # REAL phase-15 (3-leg) and WINNER (2-leg), accurate
    r15, s15, h15 = accurate_outcomes(c, 1.1342, 0.346, 1.2653,
                                      [0.45, 3.16, 6.26], [0.45, 0.16, 0.39], 1.1363, 0.849, 2.509)
    rW, sW, hW = accurate_outcomes(c, 1.481, 0.5938, 1.5187,
                                   [0.4656, 2.0532], [0.7363, 0.2637], 1.8521, 0.5, 1.234)
    r15, s15, h15 = r15[o], s15[o], h15[o]
    rW, sW, hW = rW[o], sW[o], hW[o]
    allsig = np.ones(n, bool)

    def sim(r, sld, tp1, mask, risk, a1):
        return P25.fric_sim(dt, day, hour, wd, r, sld, mask, base=risk, tp1hit=tp1, a1=a1)

    print("ACCURATE outcome sim (real BE trailing + timeout). Full 2025-26, risk sweep:")
    print(f"{'risk':>5} | {'phase-15 3leg $/wk':>18} {'WR':>4} {'DDt':>5} {'g':>4} | "
          f"{'WINNER 2leg $/wk':>16} {'WR':>4} {'DDt':>5} {'g':>4}")
    for risk in [40, 47, 55, 60, 70]:
        b = sim(r15, s15, h15, allsig, risk, 0.45)
        w = sim(rW, sW, hW, allsig, risk, 0.7363)
        print(f"  ${risk:>3} | ${b['wk']:>+17.1f} {b['wr']:>3.0f}% ${b['ddt']:>4.0f} "
              f"{'P' if b['gate'] else 'F':>4} | ${w['wk']:>+15.1f} {w['wr']:>3.0f}% "
              f"${w['ddt']:>4.0f} {'P' if w['gate'] else 'F':>4}")

    print("\n4-quarter robustness at $47 (accurate):")
    order = np.argsort(dt.values.astype('datetime64[ns]').astype(np.int64)); q = n//4
    for f in range(4):
        seg = np.zeros(n, bool); seg[order[f*q:(f+1)*q if f < 3 else n]] = True
        b = sim(r15, s15, h15, seg, 47, 0.45); w = sim(rW, sW, hW, seg, 47, 0.7363)
        print(f"  Q{f+1}: phase-15 ${b['wk']:+.1f}/wk (DDt${b['ddt']:.0f} {'P' if b['gate'] else 'F'}) | "
              f"winner ${w['wk']:+.1f}/wk (DDt${w['ddt']:.0f} {'P' if w['gate'] else 'F'})")


if __name__ == "__main__":
    main()
