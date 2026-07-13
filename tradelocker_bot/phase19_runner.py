"""
PHASE 4 — runner trend-strength coefficient sweep.
(A) Global: sweep trend_gain 1.0-2.5, friction ON. Does a deeper runner help?
(B) NY-only: apply the higher coefficient ONLY to 13-17 UTC trades (peak volume),
    keep base 1.1363 elsewhere. Higher $/wk without degrading WR?
Validated on BOTH datasets.
"""


def phase4(tr, te, sim):
    print("=" * 82)
    print("PHASE 4 — RUNNER TREND-COEFFICIENT SWEEP (friction ON, $47, TP1 0.45R)")
    print("=" * 82)
    b_tr = sim(tr, friction=True); b_te = sim(te, friction=True)
    print(f"\nBaseline (trend_gain 1.1363):")
    print(f"  23-24 {b_tr['wk']:+.1f}/wk WR{b_tr['wr']:.0f}% DDt${b_tr['ddt']:.0f}")
    print(f"  25-26 {b_te['wk']:+.1f}/wk WR{b_te['wr']:.0f}% DDt${b_te['ddt']:.0f}")

    print("\n(A) GLOBAL trend_gain sweep:")
    print(f"  {'gain':>5} | {'23-24 $/wk':>10} {'WR':>4} {'DDt':>5} {'g':>3} | "
          f"{'25-26 $/wk':>10} {'WR':>4} {'DDt':>5} {'g':>3}")
    for g in [1.0, 1.1363, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]:
        a = sim(tr, trend_gain=g, friction=True); b = sim(te, trend_gain=g, friction=True)
        print(f"  {g:>5.2f} | ${a['wk']:>+9.1f} {a['wr']:>3.0f}% ${a['ddt']:>4.0f} "
              f"{'P' if a['gate'] else 'F':>3} | ${b['wk']:>+9.1f} {b['wr']:>3.0f}% "
              f"${b['ddt']:>4.0f} {'P' if b['gate'] else 'F':>3}")

    print("\n(B) NY-ONLY deeper runner (13-17 UTC uses ny_gain, else 1.1363):")
    print(f"  {'ny_gain':>7} | {'23-24 $/wk':>10} {'WR':>4} {'DDt':>5} {'g':>3} | "
          f"{'25-26 $/wk':>10} {'WR':>4} {'DDt':>5} {'g':>3}")
    best = None
    for g in [1.1363, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5]:
        a = sim(tr, ny_gain=g, friction=True); b = sim(te, ny_gain=g, friction=True)
        worst = min(a['wk'], b['wk']); wr_ok = a['wr'] >= b_tr['wr']-1 and b['wr'] >= b_te['wr']-1
        if a['gate'] and b['gate'] and wr_ok and (best is None or worst > best[0]):
            best = (worst, g)
        flag = " <==" if best and best[1] == g else ""
        print(f"  {g:>7.2f} | ${a['wk']:>+9.1f} {a['wr']:>3.0f}% ${a['ddt']:>4.0f} "
              f"{'P' if a['gate'] else 'F':>3} | ${b['wk']:>+9.1f} {b['wr']:>3.0f}% "
              f"${b['ddt']:>4.0f} {'P' if b['gate'] else 'F':>3}{flag}")
    if best:
        print(f"\n  BEST NY-only runner gain: {best[1]:.2f} (worst-case ${best[0]:.1f}/wk, "
              f"WR not degraded, gate-safe both)")
    else:
        print("\n  No NY-only runner gain beats baseline without degrading WR / breaking gate.")
