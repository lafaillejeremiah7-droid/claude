"""
PHASE 15 — MAXIMIZE DOLLARS. Full parameter space (vol-clip + geometry + alloc +
base risk), 12-20 UTC / 2-per-day / circuit-breaker. Objective: max worst-case
$/week across BOTH datasets, subject to the DD gate passing on BOTH.
Win rate is whatever it lands at — dollars first.
"""
import json, time
import numpy as np
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12

# current shipped phase-14 config (the thing to beat on $/wk)
CUR = {"base_sl": 1.158, "vol_lo": 0.8559, "vol_hi": 1.0723, "tp1_r": 0.50,
       "tp2_r": 2.75, "tp3_r": 4.70, "trend_gain": 1.82, "tp_lo": 0.833, "tp_hi": 1.333}


def main():
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    ca = P12.run(tr, CUR, 0.49, 0.30, 0.09, base=45.0)
    cb = P12.run(te, CUR, 0.49, 0.30, 0.09, base=45.0)
    print(f"PHASE-14 SHIPPED: 23-24 {ca['wk']:+.0f}/wk WR{ca['wr']:.0f}% DDt${ca['ddt']:.0f} | "
          f"25-26 {cb['wk']:+.0f}/wk WR{cb['wr']:.0f}% DDt${cb['ddt']:.0f}")
    base_wk = min(ca['wk'], cb['wk'])

    rng = np.random.default_rng(1515); N = 14000; pool = []
    t0 = time.time()
    for i in range(1, N + 1):
        p = {}
        p["base_sl"] = rng.uniform(0.45, 1.55)
        p["vol_lo"] = rng.uniform(0.20, 0.90)
        p["vol_hi"] = rng.uniform(max(0.95, p["vol_lo"] + 0.1), 1.45)
        p["tp1_r"] = rng.uniform(0.40, 2.2)
        p["tp2_r"] = p["tp1_r"] + rng.uniform(0.5, 3.0)
        p["tp3_r"] = p["tp2_r"] + rng.uniform(0.3, 3.5)
        p["trend_gain"] = rng.uniform(0.5, 3.0)
        # FREED: runner-extension range — lets the Final TP stretch far in strong
        # trends to capture gold's big multi-R moves (the real dollar lever).
        p["tp_lo"] = rng.uniform(0.60, 0.95)
        p["tp_hi"] = rng.uniform(1.20, 3.00)
        # STRICT TP ORDERING: realized Final (even in the weakest trend) must stay
        # clearly above TP2, which must stay above TP1. No degenerate ladders.
        if not (p["tp1_r"] < p["tp2_r"] and p["tp3_r"] * p["tp_lo"] >= p["tp2_r"] * 1.05):
            continue
        a1 = rng.uniform(0.10, 0.55); a2 = rng.uniform(0.15, 0.45)
        if a1 + a2 > 0.92: continue
        tilt = rng.uniform(0.0, 0.30)
        risk = rng.uniform(40.0, 80.0)
        x = P12.run(tr, p, a1, a2, tilt, base=risk)
        if not x["gate"]: continue
        y = P12.run(te, p, a1, a2, tilt, base=risk)
        if not y["gate"]: continue
        wk = min(x['wk'], y['wk'])
        pool.append(dict(wk=wk, wr=min(x['wr'], y['wr']), p=p, a1=a1, a2=a2,
                         tilt=tilt, risk=risk, a=x, b=y))
        if i % 3000 == 0: print(f"  {i}/{N} gate-safe-both={len(pool)} best_wk=${max([q['wk'] for q in pool], default=0):.0f}")
    print(f"  done {(time.time()-t0)/60:.1f} min | pool={len(pool)}")

    def show(t, x):
        p = x['p']
        print(f"\n[{t}] risk=${x['risk']:.0f} SL={p['base_sl']:.3f}x volclip[{p['vol_lo']:.2f},{p['vol_hi']:.2f}] "
              f"TP1={p['tp1_r']:.2f}R TP2={p['tp2_r']:.2f}R Final={p['tp3_r']:.2f}R tgain={p['trend_gain']:.2f} "
              f"alloc {x['a1']*100:.0f}/{x['a2']*100:.0f}/{(1-x['a1']-x['a2'])*100:.0f} tilt {x['tilt']:.2f}")
        print(f"   23-24: {x['a']['wk']:+.1f}/wk WR{x['a']['wr']:.0f}% DDt${x['a']['ddt']:.0f} DDd${x['a']['ddd']:.0f} sig/d {x['a']['sig_d']:.1f}")
        print(f"   25-26: {x['b']['wk']:+.1f}/wk WR{x['b']['wr']:.0f}% DDt${x['b']['ddt']:.0f} DDd${x['b']['ddd']:.0f} sig/d {x['b']['sig_d']:.1f}")

    pool.sort(key=lambda z: -z['wk'])
    print(f"\nTop dollar-makers (gate-safe on BOTH, worst-case $/wk):")
    for x in pool[:6]:
        show("max$", x)
    # also the best that keeps WR >= 55 (dollars with decent WR)
    dec = [x for x in pool if x['wr'] >= 55]
    if dec:
        show("MAX$ with WR>=55%", max(dec, key=lambda z: z['wk']))
    best = pool[0]
    json.dump({"max_dollars": {"geometry": best['p'], "a1": best['a1'], "a2": best['a2'],
               "tilt": best['tilt'], "risk": best['risk'], "train": best['a'], "valid": best['b']},
               "cur_tr": ca, "cur_te": cb}, open("phase15_result.json", "w"), indent=2, default=float)
    print("\nSaved -> phase15_result.json")


if __name__ == "__main__":
    main()
