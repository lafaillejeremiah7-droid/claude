"""
PHASE 20 — S-TIER BOUNDARY TEST.
Absolute constraints (must hold on BOTH 2023-24 AND 2025-26, friction ON):
  (1) net $/wk > $50   (significantly)
  (2) max total drawdown  < $200
  (3) worst single-day loss < $120  (daily circuit-breaker never triggers)
Optimize risk (defensive sizing) + macro regime gate to satisfy ALL THREE.
Reject anything that breaches even once. Report the achievable frontier honestly.
"""
import numpy as np
import phase12_ideal as P12
import tsai_optimize as T
import fullyear_backtest as fb
import phase19_upgrade as U
import phase19_regime as RG

TARGET_WK = 50.0
MAX_DDT = 200.0
MAX_DDD = 120.0


def ok(a, b):
    return (a['wk'] > TARGET_WK and b['wk'] > TARGET_WK and
            a['ddt'] < MAX_DDT and b['ddt'] < MAX_DDT and
            a['ddd'] < MAX_DDD and b['ddd'] < MAX_DDD)


def main():
    print("Loading real data...")
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    ftr = RG.regime_features(tr); fte = RG.regime_features(te)

    def rmask(f, atr_thr):
        return f["atr_z"] >= atr_thr

    print(f"\nConstraints: $/wk>${TARGET_WK} AND DDt<${MAX_DDT} AND worst-day<${MAX_DDD}, BOTH sets\n")
    print(f"{'regime':>10} {'maxD':>4} {'risk':>5} | {'23-24 $/wk':>9} {'DDt':>4} {'DDd':>4} | "
          f"{'25-26 $/wk':>9} {'DDt':>4} {'DDd':>4} | {'ALL?':>5}")
    print("-" * 92)
    winners = []
    # defensive sizing: sweep risk low->high; regime: none, and a few ATR-z thresholds; maxday 1,2
    for atr_thr in [None, -0.5, 0.0, 0.25, 0.5]:
        mtr = None if atr_thr is None else rmask(ftr, atr_thr)
        mte = None if atr_thr is None else rmask(fte, atr_thr)
        for md in [1, 2]:
            for risk in [25, 30, 35, 40, 45, 50, 55, 60]:
                a = U.sim(tr, friction=True, base=risk, regime_mask=mtr, max_day=md)
                b = U.sim(te, friction=True, base=risk, regime_mask=mte, max_day=md)
                good = ok(a, b)
                if good:
                    winners.append((min(a['wk'], b['wk']), atr_thr, md, risk, a, b))
                # only print rows that are close (either hits $/wk or is a winner) to keep concise
                if good or (a['wk'] > 40 and b['wk'] > 40 and max(a['ddt'], b['ddt']) < 260):
                    rn = "none" if atr_thr is None else f"z>={atr_thr}"
                    print(f"{rn:>10} {md:>4} ${risk:>4} | ${a['wk']:>+8.1f} {a['ddt']:>4.0f} {a['ddd']:>4.0f} | "
                          f"${b['wk']:>+8.1f} {b['ddt']:>4.0f} {b['ddd']:>4.0f} | {'YES' if good else 'no':>5}")

    print("\n" + "=" * 60)
    if winners:
        winners.sort(key=lambda x: -x[0])
        w = winners[0]
        print(f"S-TIER BOUNDARY MET. Best: regime={w[1]} maxday={w[2]} risk=${w[3]}")
        print(f"  23-24: ${w[4]['wk']:.1f}/wk DDt${w[4]['ddt']:.0f} DDd${w[4]['ddd']:.0f}")
        print(f"  25-26: ${w[5]['wk']:.1f}/wk DDt${w[5]['ddt']:.0f} DDd${w[5]['ddd']:.0f}")
    else:
        print("NO configuration satisfies all three constraints on both datasets.")
        print("The $50/wk yield and the $200 DD ceiling are mutually exclusive after friction.")


if __name__ == "__main__":
    main()
