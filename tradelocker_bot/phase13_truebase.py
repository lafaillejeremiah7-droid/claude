"""
PHASE 13 — measure the TRUE live geometry, then optimize honestly FROM it.

Discovery: the live bot uses SL=0.8939x, TP1=1.0R, TP2=2.0R, Final 2.5-4R,
30/40/30. All prior backtests used the phase3 BEST geometry (TP1~1.98R). So the
phase12 result was measured against the wrong baseline. This measures the real
live geometry with the session filter + 2/day + CB on BOTH datasets, then
searches for a geometry that genuinely beats IT on WR and $/wk.
"""
import json, time
import numpy as np
import tsai_optimize as T
import fullyear_backtest as fb
import phase12_ideal as P12

# TRUE live geometry (matches live_terminal.py)
LIVE = {"base_sl": 0.8939, "vol_lo": 0.856, "vol_hi": 1.072, "tp1_r": 1.0,
        "tp2_r": 2.0, "tp3_r": 3.0, "trend_gain": 1.7875, "tp_lo": 0.833, "tp_hi": 1.333}


def main():
    print("Loading (train=2023-24, valid=2025-26)...")
    tr = P12.prep(fb.load_data()); te = P12.prep(T.load_data())

    a = P12.run(tr, LIVE, 0.30, 0.40, 0.19); b = P12.run(te, LIVE, 0.30, 0.40, 0.19)
    print("\nTRUE LIVE BASELINE (SL 0.89x, TP1 1R, TP2 2R, Final 2.5-4R, 30/40/30, 12-20 UTC, $45+CB):")
    print(f"  2023-24: {a['wk']:+.1f}/wk WR{a['wr']:.0f}% DDt${a['ddt']:.0f} DDd${a['ddd']:.0f} sig/d {a['sig_d']:.1f} gate={a['gate']}")
    print(f"  2025-26: {b['wk']:+.1f}/wk WR{b['wr']:.0f}% DDt${b['ddt']:.0f} DDd${b['ddd']:.0f} sig/d {b['sig_d']:.1f} gate={b['gate']}")
    b_wr = min(a['wr'], b['wr']); b_wk = min(a['wk'], b['wk'])
    print(f"  worst-case: WR {b_wr:.0f}% | ${b_wk:.0f}/wk")

    print("\nSearching for geometry that beats TRUE LIVE on WR AND $/wk (both sets, gate-safe)...")
    rng = np.random.default_rng(1313); N = 6000; pool = []
    t0 = time.time()
    for i in range(1, N + 1):
        p = dict(LIVE)
        p["base_sl"] = rng.uniform(0.70, 1.40)
        p["tp1_r"] = rng.uniform(0.40, 1.30)     # WR lever: <=1R region
        p["tp2_r"] = p["tp1_r"] + rng.uniform(0.5, 2.2)
        p["tp3_r"] = p["tp2_r"] + rng.uniform(0.3, 2.5)
        p["trend_gain"] = rng.uniform(0.5, 3.0)
        a1 = rng.uniform(0.20, 0.55); a2 = rng.uniform(0.20, 0.45)
        if a1 + a2 > 0.9: continue
        tilt = rng.uniform(0.0, 0.30)
        x = P12.run(tr, p, a1, a2, tilt)
        if not x["gate"]: continue
        y = P12.run(te, p, a1, a2, tilt)
        if not y["gate"]: continue
        pool.append(dict(wr=min(x['wr'],y['wr']), wk=min(x['wk'],y['wk']),
                         p=p, a1=a1, a2=a2, tilt=tilt, a=x, b=y))
        if i % 1500 == 0:
            print(f"  {i}/{N} gate-safe-both={len(pool)}")
    print(f"  done {(time.time()-t0)/60:.1f} min | gate-safe both: {len(pool)}")

    # strict Pareto over LIVE: WR up AND $/wk >= live, both datasets
    strict = [x for x in pool if x['wr'] >= b_wr + 2 and x['wk'] >= b_wk]
    print(f"\nConfigs beating LIVE on WR(+2) AND $/wk (both sets): {len(strict)}")
    def show(t,x):
        p=x['p']
        print(f"[{t}] TP1={p['tp1_r']:.2f}R TP2={p['tp2_r']:.2f}R Final={p['tp3_r']:.2f}R SL={p['base_sl']:.2f}x "
              f"tgain={p['trend_gain']:.2f} alloc {x['a1']*100:.0f}/{x['a2']*100:.0f}/{(1-x['a1']-x['a2'])*100:.0f} tilt {x['tilt']:.2f}")
        print(f"   23-24: {x['a']['wk']:+.1f}/wk WR{x['a']['wr']:.0f}% DDt${x['a']['ddt']:.0f}")
        print(f"   25-26: {x['b']['wk']:+.1f}/wk WR{x['b']['wr']:.0f}% DDt${x['b']['ddt']:.0f}")
    if strict:
        for x in sorted(strict, key=lambda z:-(z['wr']+0.1*z['wk']))[:5]: show("beats-live", x)
    else:
        print("  none — the live 1R/2R geometry is already near the WR+profit ceiling.")
    # also show max-WR at >=85% live $/wk, and best balanced
    hi = [x for x in pool if x['wk'] >= 0.85*b_wk]
    if hi: show("MAX-WR @ >=85% live $/wk", max(hi, key=lambda z:z['wr']))
    bal = [x for x in pool if x['wk'] >= b_wk]
    if bal: show("BEST BALANCED (>= live $/wk)", max(bal, key=lambda z:z['wr']))
    json.dump({"live_base_tr":a,"live_base_te":b,
               "n_beat":len(strict)}, open("phase13_result.json","w"), indent=2, default=float)


if __name__ == "__main__":
    main()
