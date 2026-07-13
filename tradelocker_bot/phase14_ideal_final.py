"""
PHASE 14 — final ideal-trader config. Full geometry range, pick the balanced
point (max WR x $/wk product, worst-case across both datasets), plus the
max-WR and max-$/wk corners. Ship the balanced one to live.
"""
import json, time
import numpy as np
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12

LIVE = {"base_sl": 0.8939, "vol_lo": 0.856, "vol_hi": 1.072, "tp1_r": 1.0,
        "tp2_r": 2.0, "tp3_r": 3.0, "trend_gain": 1.7875, "tp_lo": 0.833, "tp_hi": 1.333}


def main():
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())
    la = P12.run(tr, LIVE, 0.30, 0.40, 0.19); lb = P12.run(te, LIVE, 0.30, 0.40, 0.19)
    print(f"TRUE LIVE: 23-24 {la['wk']:+.0f}/wk WR{la['wr']:.0f}% DDt${la['ddt']:.0f} g={la['gate']} | "
          f"25-26 {lb['wk']:+.0f}/wk WR{lb['wr']:.0f}% DDt${lb['ddt']:.0f} g={lb['gate']}")

    rng = np.random.default_rng(1414); N = 9000; pool = []
    t0 = time.time()
    for i in range(1, N + 1):
        p = dict(LIVE)
        p["base_sl"] = rng.uniform(0.75, 1.45)
        p["tp1_r"] = rng.uniform(0.40, 2.0)
        p["tp2_r"] = p["tp1_r"] + rng.uniform(0.5, 2.6)
        p["tp3_r"] = p["tp2_r"] + rng.uniform(0.3, 3.0)
        p["trend_gain"] = rng.uniform(0.5, 3.0)
        a1 = rng.uniform(0.20, 0.55); a2 = rng.uniform(0.20, 0.45)
        if a1 + a2 > 0.9: continue
        tilt = rng.uniform(0.0, 0.30)
        x = P12.run(tr, p, a1, a2, tilt)
        if not x["gate"]: continue
        y = P12.run(te, p, a1, a2, tilt)
        if not y["gate"]: continue
        wr = min(x['wr'], y['wr']); wk = min(x['wk'], y['wk'])
        if wk <= 0: continue
        pool.append(dict(wr=wr, wk=wk, prod=wr * wk, p=p, a1=a1, a2=a2, tilt=tilt, a=x, b=y))
        if i % 2000 == 0: print(f"  {i}/{N} pool={len(pool)}")
    print(f"  done {(time.time()-t0)/60:.1f} min | pool={len(pool)}")

    def show(t, x):
        p = x['p']
        print(f"\n[{t}] TP1={p['tp1_r']:.2f}R TP2={p['tp2_r']:.2f}R Final={p['tp3_r']:.2f}R "
              f"SL={p['base_sl']:.3f}x tgain={p['trend_gain']:.2f} "
              f"alloc {x['a1']*100:.0f}/{x['a2']*100:.0f}/{(1-x['a1']-x['a2'])*100:.0f} tilt {x['tilt']:.2f}")
        print(f"   23-24: {x['a']['wk']:+.1f}/wk WR{x['a']['wr']:.0f}% DDt${x['a']['ddt']:.0f} DDd${x['a']['ddd']:.0f} sig/d {x['a']['sig_d']:.1f}")
        print(f"   25-26: {x['b']['wk']:+.1f}/wk WR{x['b']['wr']:.0f}% DDt${x['b']['ddt']:.0f} DDd${x['b']['ddd']:.0f} sig/d {x['b']['sig_d']:.1f}")

    balanced = max(pool, key=lambda z: z['prod'])
    max_wr = max(pool, key=lambda z: (z['wr'], z['wk']))
    max_wk = max(pool, key=lambda z: (z['wk'], z['wr']))
    show("MAX $/WEEK", max_wk)
    show("MAX WIN-RATE", max_wr)
    show("BALANCED (max WR x $/wk)  <== SHIP THIS", balanced)

    def pack(x):
        return {"geometry": x['p'], "a1": x['a1'], "a2": x['a2'], "tilt": x['tilt'],
                "train": x['a'], "valid": x['b']}
    json.dump({"balanced": pack(balanced), "max_wr": pack(max_wr), "max_wk": pack(max_wk),
               "live_tr": la, "live_te": lb}, open("phase14_result.json", "w"), indent=2, default=float)
    print("\nSaved -> phase14_result.json")


if __name__ == "__main__":
    main()
