"""
PHASE 3 — rebuild the brain as a MACRO REGIME GATE (Expansion vs Chop on/off).
Features (computable on BOTH datasets; 2025-26 has no volume):
  - ATR Z-score: rolling z of the volatility-expansion ratio (vol_ratio)
  - Density proxy: rolling z of |pullback distance| inverted — near-value (tight,
    high "density") vs void (stretched). True volume unavailable on 2025-26.
Classify Expansion when vol is expanding AND structure is directional; else Chop.
If Chop -> global pause (bot OFF). Learn threshold on 2023-24, validate on 2025-26.
The brain no longer scores individual trades (proven useless) — it only decides
WHEN the whole strategy is allowed to run.
"""
import numpy as np


def regime_features(S):
    c = S["c"]; o = S["order"]
    vr = c["vol_ratio"][o]              # ATR expansion ratio, chronological
    ts = c["trend_strength"][o]         # structural trend
    pb = np.abs(c["pb_dist"][o])        # distance from value (EMA)
    W = 50
    def zroll(x):
        z = np.zeros(len(x))
        for i in range(len(x)):
            lo = max(0, i - W); win = x[lo:i]
            if len(win) >= 10:
                m = win.mean(); s = win.std() or 1.0
                z[i] = (x[i] - m) / s
        return z
    return dict(atr_z=zroll(vr), ts=ts, dens_z=-zroll(pb))  # dens_z high = tight/value


def phase3(tr, te, sim):
    print("=" * 82)
    print("PHASE 3 — MACRO REGIME GATE (Expansion=ON / Chop=OFF), friction ON, $47")
    print("=" * 82)
    ftr = regime_features(tr); fte = regime_features(te)

    def mask(f, atr_thr, ts_thr):
        # Expansion = volatility expanding OR a genuinely strong trend
        return (f["atr_z"] >= atr_thr) | (f["ts"] >= ts_thr)

    base_tr = sim(tr, friction=True); base_te = sim(te, friction=True)
    print(f"\nNo regime gate (take all in-session):")
    print(f"  23-24 {base_tr['wk']:+.1f}/wk WR{base_tr['wr']:.0f}% DDt${base_tr['ddt']:.0f} sig/d {base_tr['sig_d']:.1f}")
    print(f"  25-26 {base_te['wk']:+.1f}/wk WR{base_te['wr']:.0f}% DDt${base_te['ddt']:.0f} sig/d {base_te['sig_d']:.1f}")

    print(f"\nLearn threshold on 2023-24, apply UNCHANGED to 2025-26:")
    print(f"  {'atr_z>=':>8} {'ts>=':>5} | {'23-24 $/wk':>10} {'WR':>4} {'DDt':>5} {'s/d':>4} "
          f"| {'25-26 $/wk':>10} {'WR':>4} {'DDt':>5} {'s/d':>4}")
    results = []
    for atr_thr in [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5]:
        for ts_thr in [0.8, 1.0, 1.5, 99]:   # 99 = disable the trend-OR clause
            mtr = mask(ftr, atr_thr, ts_thr); mte = mask(fte, atr_thr, ts_thr)
            a = sim(tr, friction=True, regime_mask=mtr)
            b = sim(te, friction=True, regime_mask=mte)
            results.append((atr_thr, ts_thr, a, b))
    # rank by TRAIN gate-safe $/wk, then show top few + their OOS
    ranked = sorted([r for r in results if r[2]['gate']], key=lambda r: -r[2]['wk'])
    print("  -- top configs by 2023-24 $/wk (that pass 2023-24 gate) --")
    for atr_thr, ts_thr, a, b in ranked[:6]:
        tsl = "off" if ts_thr == 99 else f"{ts_thr:.1f}"
        print(f"  {atr_thr:>8.2f} {tsl:>5} | ${a['wk']:>+9.1f} {a['wr']:>3.0f}% ${a['ddt']:>4.0f} "
              f"{a['sig_d']:>3.1f} | ${b['wk']:>+9.1f} {b['wr']:>3.0f}% ${b['ddt']:>4.0f} {b['sig_d']:>3.1f}")

    # honest verdict: does the BEST train config improve OOS $/wk or DD vs no-gate?
    if ranked:
        atr_thr, ts_thr, a, b = ranked[0]
        print(f"\n  Best-on-train config OOS effect (2025-26):")
        print(f"    $/wk {b['wk']-base_te['wk']:+.1f} | DDt {b['ddt']-base_te['ddt']:+.0f} | "
              f"WR {b['wr']-base_te['wr']:+.0f}pts")
        print("    (regime gate helps only if it cuts DD or lifts $/wk OOS, not just volume)")

    # Also: does regime-gating + higher risk beat baseline? (DD headroom -> more risk)
    print("\n  Can the DD saved by the gate fund higher risk? (best gate config, risk sweep)")
    if ranked:
        atr_thr, ts_thr, _, _ = ranked[0]
        mtr = mask(ftr, atr_thr, ts_thr); mte = mask(fte, atr_thr, ts_thr)
        for base in [47, 55, 65, 75]:
            a = sim(tr, friction=True, regime_mask=mtr, base=base)
            b = sim(te, friction=True, regime_mask=mte, base=base)
            ok = a['gate'] and b['gate']
            print(f"    ${base}: 23-24 ${a['wk']:+.1f}/wk DDt${a['ddt']:.0f} | "
                  f"25-26 ${b['wk']:+.1f}/wk DDt${b['ddt']:.0f} | {'PASS' if ok else 'FAIL'}")
