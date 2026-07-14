"""
PHASE 16b — best achievable with a REALISTIC fixed final ratio (<= 3:1) and sane
SL/TP distances. Optimizes: structure (single / near+final / mid+final), the near
partial's R and weight, the final ratio (1.5-3.0), and base SL — to maximize
gate-safe $/wk on BOTH datasets. Then compares honestly to the shipped 6R config.
"""
import numpy as np
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12
import phase16_fixed_ratio as P16


def main():
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    import sys
    KMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    rng = np.random.default_rng(1616); N = 6000; pool = []
    for i in range(1, N + 1):
        base_sl = rng.uniform(0.80, 1.40)
        k = rng.uniform(1.5, KMAX)                 # final reward:risk
        two_leg = rng.random() < 0.85
        if two_leg:
            near = rng.uniform(0.4, min(1.4, k - 0.2))
            a1 = rng.uniform(0.15, 0.60)
            tgts = [near, k]; alloc = [a1, 1 - a1]
        else:
            tgts = [k]; alloc = [1.0]
        if any(tgts[j] >= tgts[j + 1] for j in range(len(tgts) - 1)):
            continue
        r_tr, sld_tr = P16.ladder_R(tr["c"], base_sl, tgts, alloc)
        r_te, sld_te = P16.ladder_R(te["c"], base_sl, tgts, alloc)
        a = P16.cb_sim(tr, r_tr, sld_tr); b = P16.cb_sim(te, r_te, sld_te)
        if not (a["gate"] and b["gate"]):
            continue
        pool.append(dict(wk=min(a["wk"], b["wk"]), wr=min(a["wr"], b["wr"]),
                         base_sl=base_sl, k=k, tgts=tgts, alloc=alloc,
                         medsl=float(np.median(sld_te)), a=a, b=b))
    print(f"gate-safe (both) realistic-ratio configs: {len(pool)}\n")
    if not pool:
        print("NONE pass the gate on both datasets with a <=3:1 final target.")
        print("=> a realistic fixed ratio cannot hold the drawdown gate here.")
        return

    def show(t, x):
        legs = " ".join(f"{a*100:.0f}%@{g:.2f}R" for g, a in zip(x["tgts"], x["alloc"]))
        print(f"[{t}] SL={x['base_sl']:.2f}xATR (~${x['medsl']:.1f})  final={x['k']:.2f}:1  legs: {legs}")
        print(f"   23-24: {x['a']['wk']:+.1f}/wk WR{x['a']['wr']:.0f}% DDt${x['a']['ddt']:.0f} sig/d {x['a']['sig_d']:.1f}")
        print(f"   25-26: {x['b']['wk']:+.1f}/wk WR{x['b']['wr']:.0f}% DDt${x['b']['ddt']:.0f} sig/d {x['b']['sig_d']:.1f}")

    show("MAX $/WEEK (realistic ratio)", max(pool, key=lambda z: z["wk"]))
    show("MAX WIN-RATE (realistic ratio)", max(pool, key=lambda z: (z["wr"], z["wk"])))
    # best balanced: max wr*wk
    show("BALANCED (max WR x $/wk)", max(pool, key=lambda z: z["wr"] * z["wk"]))
    print("\nReference — shipped phase-15 (6R runner): 25-26 +$54/wk WR70% | 23-24 +$55/wk WR64%")


if __name__ == "__main__":
    main()
