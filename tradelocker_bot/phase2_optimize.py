"""
Phase 2: bar starts at Phase 1's peak. A progression counts ONLY if a valid
(gate-passing) config beats the current bar by >= +0.10% (compounding). Find
as many as the real Jan2025->Jul2026 data can support above the ceiling.

python3 phase2_optimize.py [time_budget_s]
"""
import sys
import json
import time
import os
import numpy as np

import tsai_optimize as T
import phase_optimize as P

THRESH = 1.001   # +0.10% per progression, compounding
CKPT = "phase2_ckpt.json"


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1500.0
    df1m = T.load_data()
    weeks = (df1m.index.max() - df1m.index.min()).days / 7.0
    c = T.build_candidates(df1m)

    # Phase 1 peak = starting bar + starting search point
    p1 = json.load(open("phase1_ckpt.json"))
    start_params = p1["best"]
    mt0, ts0, valid0 = P.evaluate(c, start_params)
    phase1_peak = mt0["total_d"]

    rng = np.random.default_rng(7777)

    # resume phase 2 if checkpoint exists
    best = dict(start_params)
    bar = phase1_peak            # current peak to beat (rises by >=0.10% each progression)
    abs_best_val = phase1_peak
    abs_best = dict(start_params)
    progressions = 0
    trials = 0
    ladder = []
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT))
        best = ck["best"]; bar = ck["bar"]; abs_best_val = ck["abs_best_val"]
        abs_best = ck["abs_best"]; progressions = ck["progressions"]
        trials = ck["trials"]; ladder = ck.get("ladder", [])
        print(f"[resume] phase2: progressions={progressions}, trials={trials}, "
              f"bar=${bar:.2f}, abs_best=${abs_best_val:.2f}")

    print(f"Phase 1 peak (starting bar): ${phase1_peak:.2f}  (+{phase1_peak/weeks:.2f}/wk)")
    print(f"Each progression must clear +0.10% over the last peak.")
    print(f"  next target to count progression #1: ${bar*THRESH:.2f}\n")

    t0 = time.time()
    since = 0
    while (time.time() - t0) < budget:
        # search: fine local around abs_best, occasional random / restart-from-P1
        rr = rng.random()
        if rr < 0.15 or since > 8000:
            cand = P.random_params(rng)
            since = 0
        elif rr < 0.30:
            cand = dict(start_params)  # restart from Phase 1 optimum
            k = rng.choice(P.KEYS)
            lo, hi = P.BOUNDS[k]; cand[k] += rng.normal(0, 0.05 * (hi - lo))
            cand = P.clip_params(cand)
        else:
            cand = dict(abs_best)
            ksub = rng.choice(P.KEYS, size=int(rng.integers(1, 4)), replace=False)
            s = 0.003 + 0.025 * rng.random()
            for k in ksub:
                lo, hi = P.BOUNDS[k]; cand[k] += rng.normal(0, s * (hi - lo))
            cand = P.clip_params(cand)

        trials += 1
        mt, ts, valid = P.evaluate(c, cand)
        if not valid:
            since += 1
            continue
        # track absolute best (for continued local search)
        if mt["total_d"] > abs_best_val:
            abs_best_val = mt["total_d"]; abs_best = cand
        # count a progression only if it clears the +0.10% compounding bar
        if mt["total_d"] >= bar * THRESH:
            progressions += 1
            bar = mt["total_d"]
            best = cand
            since = 0
            ladder.append({"prog": progressions, "trial": trials,
                           "total_d": round(mt["total_d"], 2),
                           "wk": round(mt["total_d"] / weeks, 2),
                           "tsai": round(ts["TSAI"], 3),
                           "dd_total": round(mt["dd_total"], 1),
                           "dd_day": round(mt["dd_day"], 1)})
            print(f"  progression {progressions} | trial {trials} | ${mt['total_d']:+.2f} "
                  f"(+{mt['total_d']/weeks:.2f}/wk) | next bar ${bar*THRESH:.2f} | TSAI={ts['TSAI']:.3f}")
        else:
            since += 1

    json.dump({"best": best, "bar": bar, "abs_best": abs_best, "abs_best_val": abs_best_val,
               "progressions": progressions, "trials": trials, "ladder": ladder,
               "phase1_peak": phase1_peak, "weeks": weeks},
              open(CKPT, "w"), indent=2, default=float)

    print()
    print("=" * 90)
    print(f"PHASE 2: +0.10% progressions above Phase-1 peak = {progressions} | trials={trials} "
          f"| {(time.time()-t0)/60:.1f} min")
    print(f"  Phase 1 peak: ${phase1_peak:.2f}")
    print(f"  Phase 2 abs best: ${abs_best_val:.2f} (+{abs_best_val/weeks:.2f}/wk)  "
          f"[+{(abs_best_val/phase1_peak-1)*100:.3f}% over Phase 1]")
    if progressions > 0:
        print("  PARAMS(best):", json.dumps({k: round(best[k], 4) for k in P.KEYS}))
    print("=" * 90)


if __name__ == "__main__":
    main()
